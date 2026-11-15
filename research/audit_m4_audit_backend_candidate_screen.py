"""Independent raw-record audit of the controller-level backend screen."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m3_continuation import ROOT, digest, metrics, prepare

OUT = ROOT / "results/m4-audit-backend-candidate-screen"
METHODS = ("CVODE-KLU", "BDF")
TARGETS = (1e-3, 1e-5)
REPEATS = (-1, 0, 1)
WINDOW = 1e-6
STATE_INDEX = 61


def close(a, b, *, rtol=1e-9, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def main():
    registration = json.loads((OUT / "registration.json").read_text())
    rows = json.loads((OUT / "runs.json").read_text())["rows"]
    assert registration["state_index"] == STATE_INDEX
    assert registration["targets"] == list(TARGETS)
    assert registration["methods"] == ["AdaptiveSplit", *METHODS]
    for path, expected in registration["source_sha256"].items():
        assert digest(ROOT / path) == expected, f"source hash mismatch: {path}"
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as source:
        y0 = source["concentrations"][:, STATE_INDEX].copy()
        start_time = float(source["times"][STATE_INDEX])
    assert start_time == registration["start_time"]
    grid = np.r_[start_time, start_time + WINDOW*np.array([.25, .5, .75, 1.])]
    truth_result = solve_ivp(lambda t, y: case["system"].S @ case["model"].flux(t, y),
        (start_time, grid[-1]), y0, method="Radau", t_eval=grid[1:], rtol=1e-13,
        atol=np.maximum(case["scale"]*1e-15, 1e-18),
        jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
    assert truth_result.success
    truth = np.column_stack((y0, truth_result.y))
    with np.load(OUT / "reference.npz") as stored:
        close(stored["initial"], y0)
        close(stored["values"], truth)
    flux_reference = np.column_stack([case["model"].flux(t, y) for t, y in zip(grid, truth.T)])
    local_case = dict(case, times=grid, reference=truth, flux_reference=flux_reference)

    expected_keys = set()
    expected_order = []
    for target in TARGETS:
        sequence = [("baseline", None, -1), ("candidate", METHODS[0], -1),
                    ("candidate", METHODS[1], -1)]
        for repeat in (0, 1):
            sequence.extend([("baseline", None, repeat), ("candidate", METHODS[0], repeat),
                             ("candidate", METHODS[1], repeat)])
        for kind, method, repeat in sequence:
            expected_keys.add((target, kind, method, repeat))
            expected_order.append((target, kind, method, repeat))
    actual = [(r["target"], r["kind"], r["audit_method"], r["repeat"]) for r in rows]
    assert actual == expected_order, "missing, duplicate, or reordered screen records"

    checked = []
    for row in rows:
        assert row["status"] == "success", row.get("reason")
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
        target, kind, method, repeat = (row["target"], row["kind"], row["audit_method"], row["repeat"])
        assert row["warmup"] is (repeat == -1)
        expected_parameters = registration["candidate"].copy()
        expected_parameters.update(target=target, audit_method=method, timeout=240.)
        if kind == "baseline":
            assert method is None
            expected_parameters = dict(target=target, initial_step=1e-8, max_step=WINDOW,
                max_attempts=256, max_rejections=16, timeout=240.,
                fallback_method="CVODE-KLU", fallback_rtol=1e-11)
        else:
            assert method in METHODS
            expected_parameters.update(audit_rtol=1e-9, fallback_method="CVODE-KLU",
                                       fallback_rtol=1e-11, local_fraction=1.,
                                       audit_control_only=True)
        assert row["parameters"] == expected_parameters
        filename = f"{kind}-{target}-{method or 'all'}-{repeat}.npz"
        with np.load(OUT / filename) as stored:
            values = stored["values"]
        assert values.shape == truth.shape and np.isfinite(values).all()
        actual_metrics = metrics(values, local_case, target, row["stats"])
        for key, value in actual_metrics.items():
            if isinstance(value, (bool, np.bool_)):
                assert row["metrics"][key] is bool(value)
            else:
                close(row["metrics"][key], value)
        stats = row["stats"]
        assert stats["clipping_count"] == 0 and stats["projection_count"] == 0
        assert stats["full_calls"] == len(stats["attempts"]) if kind == "candidate" else True
        assert row["every_attempt_audited"] is (kind == "baseline" or
            stats["full_calls"] == len(stats["attempts"]))
        if kind == "candidate":
            coverage = sum(a["h"] for a in stats["attempts"] if a["status"] == "accepted") / WINDOW
            close(row["coverage_recomputed"], coverage, atol=1e-14)
            for attempt in stats["attempts"]:
                if attempt["status"] == "rejected":
                    assert attempt["rollback_verified"] is True
        checked.append(dict(target=target, kind=kind, method=method, repeat=repeat,
                            seconds=row["seconds"], metrics=actual_metrics,
                            coverage=row["coverage_recomputed"]))

    summary=[]
    for target in TARGETS:
        base=[r["seconds"] for r in rows if r["target"]==target and r["kind"]=="baseline" and r["repeat"]>=0]
        for method in METHODS:
            cand=[r["seconds"] for r in rows if r["target"]==target and r["kind"]=="candidate"
                  and r["audit_method"]==method and r["repeat"]>=0]
            cm=[r for r in rows if r["target"]==target and r["kind"]=="candidate"
                and r["audit_method"]==method and r["repeat"]>=0]
            summary.append(dict(target=target, audit_method=method,
                baseline_median=float(np.median(base)), candidate_median=float(np.median(cand)),
                speedup=float(np.median(base)/np.median(cand)),
                candidate_iqr=float(np.percentile(cand,75)-np.percentile(cand,25)),
                coverage=[r["coverage_recomputed"] for r in cm],
                all_metrics_pass=all(r["metrics"]["passes"] for r in cm),
                all_attempts_audited=all(r["every_attempt_audited"] for r in cm)))
    report=dict(evidence_integrity="pass",rows=len(rows),summary=summary,
        scope="Short-window Kecoli diagnostic only; no M4 qualification or audit-policy promotion.")
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
