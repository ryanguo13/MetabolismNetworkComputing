"""Independent raw-record audit for the prospective audit-backend screen."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m3_continuation import ROOT, digest, prepare

OUT = ROOT / "results/m4-audit-backend-screen"
METHODS = ("CVODE-KLU", "BDF")
RTOLS = (1e-9, 1e-10, 1e-11)
REPEATS = (-1, 0, 1, 2)
TARGETS = (1e-3, 1e-5)
H = 1e-6


def close(a, b, *, rtol=1e-10, atol=1e-13):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def main():
    registration = json.loads((OUT / "registration.json").read_text())
    data = json.loads((OUT / "runs.json").read_text())
    rows = data["rows"]
    assert registration["methods"] == list(METHODS)
    assert registration["rtols"] == list(RTOLS)
    assert registration["repeats"] == list(REPEATS)
    assert registration["targets"] == list(TARGETS)
    for path, expected in registration["source_sha256"].items():
        assert digest(ROOT / path) == expected, f"source hash mismatch: {path}"
    expected_keys = set(itertools.product(registration["state_indices"], REPEATS, METHODS, RTOLS))
    actual_keys = [(r["state_index"], r["repeat"], r["method"], r["rtol"]) for r in rows]
    assert len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys

    case = prepare("kecoli")
    query = H * np.array(registration["query_fractions"])
    errors = []
    for state_index in registration["state_indices"]:
        with np.load(case["reference_path"]) as reference:
            y0 = reference["concentrations"][:, state_index].copy()
            state_time = float(reference["times"][state_index])
        assert state_time == registration["state_times"][registration["state_indices"].index(state_index)]
        truth_result = solve_ivp(lambda t, y: case["system"].S @ case["model"].flux(t, y),
            (0.0, H), y0, method="Radau", t_eval=query, rtol=1e-13,
            atol=np.maximum(case["scale"] * 1e-15, 1e-18),
            jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
        assert truth_result.success
        truth = truth_result.y
        with np.load(OUT / f"truth-{state_index}.npz") as saved:
            close(saved["initial"], y0)
            close(saved["values"], truth, rtol=1e-9, atol=1e-12)
        pool = case["options"]["pools"]
        flux_scale = case["options"]["flux_scale"]
        net_map = case["options"]["net_map"]
        net_scale = case["options"]["net_scale"]
        truth_flux = np.column_stack([case["model"].flux(0.0, y) for y in truth.T])
        state_rows = [r for r in rows if r["state_index"] == state_index]
        expected_order = []
        schedule = list(itertools.product(METHODS, RTOLS))
        for repeat in REPEATS:
            offset = max(0, repeat) % len(schedule)
            expected_order.extend((repeat, method, rtol) for method, rtol in schedule[offset:] + schedule[:offset])
        observed_order = [(r["repeat"], r["method"], r["rtol"]) for r in state_rows]
        assert observed_order == expected_order, f"execution order mismatch for state {state_index}"
        for row in state_rows:
            assert row["status"] == "success", row.get("reason")
            assert math.isfinite(row["seconds"]) and row["seconds"] > 0
            filename = f"state-{state_index}-{row['repeat']}-{row['method']}-{row['rtol']}.npz"
            with np.load(OUT / filename) as saved:
                values = saved["values"]
            assert values.shape == truth.shape and np.isfinite(values).all()
            observed_hash = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            assert observed_hash == row["value_sha256"]
            flux = np.column_stack([case["model"].flux(0.0, y) for y in values.T])
            metrics = {}
            for target in TARGETS:
                conc = float(np.max(np.abs(values - truth) / (case["scale"][:, None] * target)))
                one_way = float(np.max(np.abs(flux - truth_flux) / (flux_scale[:, None] * target)))
                net = float(np.max(np.abs(net_map @ (flux - truth_flux)) /
                                    (net_scale[:, None] * 1e-3)))
                pool_error = float(np.max(np.abs(pool @ (values - y0[:, None]))) /
                    max(1e-12, float(np.max(np.abs(pool @ y0)))))
                passed = (max(conc, one_way, net) <= .05 and pool_error <= 1e-10
                          and np.all(values >= -1e-12 * case["scale"][:, None]))
                actual = row["metrics"][str(target)]
                for key, value in (("concentration", conc), ("flux", one_way),
                                   ("net_flux", net), ("pool_error", pool_error)):
                    close(actual[key], value, rtol=1e-9, atol=1e-12)
                assert actual["finite"] is True
                assert actual["nonnegative"] is bool(np.all(values >= -1e-12 * case["scale"][:, None]))
                assert actual["passes"] is bool(passed)
                metrics[str(target)] = dict(concentration=conc, flux=one_way, net_flux=net,
                                            pool_error=pool_error, passes=bool(passed))
            errors.append(dict(state_index=state_index, repeat=row["repeat"], method=row["method"],
                               rtol=row["rtol"], seconds=row["seconds"], metrics=metrics))

    summaries = []
    for state_index, method, rtol in itertools.product(registration["state_indices"], METHODS, RTOLS):
        samples = [r for r in errors if r["state_index"] == state_index and r["method"] == method
                   and r["rtol"] == rtol and r["repeat"] >= 0]
        seconds = [r["seconds"] for r in samples]
        summaries.append(dict(state_index=state_index, method=method, rtol=rtol,
            repeats=len(samples), all_qualified=len(samples) == 3 and
            all(v["passes"] for row in samples for v in row["metrics"].values()),
            median_seconds=float(np.median(seconds)),
            iqr_seconds=float(np.percentile(seconds, 75) - np.percentile(seconds, 25)),
            max_concentration_error=max(v["concentration"] for r in samples for v in r["metrics"].values()),
            max_flux_error=max(v["flux"] for r in samples for v in r["metrics"].values()),
            max_net_flux_error=max(v["net_flux"] for r in samples for v in r["metrics"].values())))
    report = dict(evidence_integrity="pass", rows=len(rows), summaries=summaries,
        limitation="Two early Kecoli states only; local backend calibration, not an M4 audit-policy qualification.")
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
