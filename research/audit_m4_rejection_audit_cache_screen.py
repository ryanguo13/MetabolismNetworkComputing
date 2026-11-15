"""Independent evidence audit for the paired rejection-audit-cache screen."""
import json
import math
import statistics

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m3_continuation import ROOT, digest, prepare
from metabolism.m2_experiments import error

OUT = ROOT / "results/m4-rejection-audit-cache-screen"
TARGETS = (1e-3, 1e-5)
REPEATS = (-1, 0, 1, 2)
MODES = ("fresh", "cached")


def close(a, b, rtol=1e-9, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def main():
    registration = json.loads((OUT / "registration.json").read_text())
    rows = json.loads((OUT / "runs.json").read_text())["rows"]
    assert len(rows) == len(TARGETS) * len(REPEATS) * len(MODES), len(rows)
    for path, expected in registration["source_sha256"].items():
        assert digest(ROOT / path) == expected, f"source hash mismatch: {path}"
    expected_keys = []
    for target in TARGETS:
        for repeat in REPEATS:
            modes = list(MODES)
            if repeat >= 0 and repeat % 2:
                modes.reverse()
            expected_keys.extend((target, repeat, mode) for mode in modes)
    observed_keys = [(r["target"], r["repeat"], r["mode"]) for r in rows]
    assert observed_keys == expected_keys, "missing, duplicate or reordered rows"

    case = prepare("kecoli")
    with np.load(case["reference_path"]) as archive:
        y0 = archive["concentrations"][:, registration["state_index"]].copy()
        archive_time = float(archive["times"][registration["state_index"]])
    assert archive_time == registration["archive_time"]
    window = registration["interval"][1]
    grid = window * np.array(registration["query_fractions"])
    truth_result = solve_ivp(
        lambda t, y: case["system"].S @ case["model"].flux(t, y),
        (0., window), y0, method="Radau", t_eval=grid[1:], rtol=1e-13,
        atol=np.maximum(case["scale"] * 1e-15, 1e-18),
        jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
    assert truth_result.success
    truth = np.column_stack((y0, truth_result.y))
    with np.load(OUT / "reference.npz") as archived_truth:
        close(archived_truth["initial"], y0)
        close(archived_truth["values"], truth)
    flux_truth = np.column_stack([case["model"].flux(t, y) for t, y in zip(grid, truth.T)])
    local_case = dict(case, times=grid, reference=truth, flux_reference=flux_truth)

    options = case["options"]
    scales = case["scale"]
    pools = options["pools"]
    initial_pool = pools @ y0
    pool_denominator = max(1e-12, float(np.max(np.abs(initial_pool))))
    def recompute_metrics(values, target, stats):
        flux_values = np.column_stack([case["model"].flux(t, y) for t, y in zip(grid, values.T)])
        out = dict(
            concentration_error=error(values, truth, scales, target),
            flux_error=error(flux_values, flux_truth, options["flux_scale"], target),
            invariant_error=float(np.max(np.abs(pools @ values - initial_pool[:, None]))) / pool_denominator,
            significant_negative_output=bool(np.any(values < -1e-12 * scales[:, None])),
            finite=bool(np.isfinite(values).all()),
            relative_ledger_error=0.0)
        if "net_map" in options:
            out["net_flux_error"] = error(options["net_map"] @ flux_values,
                options["net_map"] @ flux_truth, options["net_scale"], 1e-3)
        out["passes"] = (max(out["concentration_error"], out["flux_error"],
            out.get("net_flux_error", 0.0)) <= 1 and out["invariant_error"] <= 1e-10
            and out["relative_ledger_error"] <= 1e-10 and out["finite"]
            and not out["significant_negative_output"])
        return out

    details = []
    metric_record_mismatches = []
    by_key = {}
    for row in rows:
        key = (row["target"], row["repeat"], row["mode"])
        by_key[key] = row
        assert row["status"] == "success", (key, row.get("reason"))
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
        assert row["warmup"] is (row["repeat"] == -1)
        with np.load(OUT / f"{row['mode']}-{row['target']}-{row['repeat']}.npz") as saved:
            values = saved["values"]
        assert values.shape == truth.shape and np.isfinite(values).all()
        recomputed = recompute_metrics(values, row["target"], row["stats"])
        assert recomputed["passes"] is True, (key, recomputed)
        for name, value in recomputed.items():
            recorded = row["metrics"].get(name)
            if isinstance(value, (bool, np.bool_)):
                matches = recorded is bool(value)
            else:
                matches = recorded is not None and np.isclose(recorded, value, rtol=1e-9, atol=1e-12)
            if not matches:
                metric_record_mismatches.append(dict(key=key, metric=name,
                    recorded=recorded, independently_recomputed=value))
                if name != "invariant_error":
                    raise AssertionError((key, name, recorded, value))
        stats = row["stats"]
        attempts = stats["attempts"]
        assert row["every_attempt_audited"] is True
        assert stats["audit_calls"] == len(attempts)
        assert stats["full_calls"] == len(attempts)
        assert stats["audit_solver_calls"] + stats["audit_cache_hits"] == stats["audit_calls"]
        assert stats["rejected_steps"] == stats["rollback_checks"]
        assert stats["max_pool_error"] <= 1e-10
        assert all(max(a["audit_max"], a["audit_flux_max"], a["audit_net_flux_max"])
                   <= 1 for a in attempts if a["status"] == "accepted")
        assert row["all_rejections_rolled_back"] is True
        assert all(a.get("rollback_verified") is True for a in attempts
                   if a["status"] == "rejected")
        if row["mode"] == "fresh":
            assert stats["audit_cache_hits"] == 0
            assert stats["audit_solver_calls"] == stats["audit_calls"]
        else:
            assert stats["audit_cache_hits"] > 0
            assert stats["audit_solver_calls"] < stats["audit_calls"]
        details.append(dict(target=row["target"], repeat=row["repeat"], mode=row["mode"],
                            seconds=row["seconds"], solver_calls=stats["audit_solver_calls"],
                            cache_hits=stats["audit_cache_hits"], attempts=len(attempts),
                            rejected=stats["rejected_steps"], coverage=stats["split_coverage"],
                            truth_error=row["state_truth_error"], metrics_pass=recomputed["passes"]))

    paired = []
    for target in TARGETS:
        for repeat in REPEATS:
            fresh = by_key[(target, repeat, "fresh")]
            cached = by_key[(target, repeat, "cached")]
            with np.load(OUT / f"fresh-{target}-{repeat}.npz") as a, \
                    np.load(OUT / f"cached-{target}-{repeat}.npz") as b:
                close(a["values"], b["values"], rtol=1e-8, atol=1e-12)
            paired.append(dict(target=target, repeat=repeat,
                fresh_seconds=fresh["seconds"], cached_seconds=cached["seconds"],
                speedup=fresh["seconds"] / cached["seconds"],
                fresh_attempts=len(fresh["stats"]["attempts"]),
                cached_attempts=len(cached["stats"]["attempts"]),
                cached_hits=cached["stats"]["audit_cache_hits"]))
    summaries = []
    for target in TARGETS:
        formal = [p for p in paired if p["target"] == target and p["repeat"] >= 0]
        ratios = [p["speedup"] for p in formal]
        summaries.append(dict(target=target,
            fresh_median=statistics.median(p["fresh_seconds"] for p in formal),
            cached_median=statistics.median(p["cached_seconds"] for p in formal),
            paired_speedup_median=statistics.median(ratios),
            paired_speedup_range=[min(ratios), max(ratios)],
            total_cache_hits=sum(p["cached_hits"] for p in formal),
            total_retry_attempts=sum(p["cached_attempts"] for p in formal)))
    report = dict(evidence_integrity=("pass" if not metric_record_mismatches else "pass_with_instrumentation_mismatch"),
                  rows=len(rows), source_hashes="pass",
                  metric_record_mismatches=metric_record_mismatches,
                  all_attempts_audited=True, all_rejections_rolled_back=True,
                  scope=registration["scope"], summaries=summaries,
                  paired=paired, row_details=details)
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
