"""Short-prefix screen for overlapping split generation with independent audits.

Both jobs start from the same committed state.  The candidate is committed only
after the full nonlinear audit completes, so overlap changes wall time only and
does not relax the audit or rollback policy.
"""
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import metrics, prepare
from metabolism.splitting import SplitFailure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m4-overlap-audit-screen-v3"
HORIZONS = {"teusink": 1e-3, "kecoli": 1e-4}
STATE_INDEX = {"teusink": 1, "kecoli": 61}
REPEATS = (0, 1, 2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    source_files = (
        "research/m4_overlap_audit_screen.py", "metabolism/continuation.py",
        "metabolism/adaptive.py", "metabolism/splitting.py",
        "metabolism/kecoli.py", "metabolism/m3_continuation.py",
    )
    registration = dict(
        date="2026-11-16",
        scope="Short-prefix overlap screen; no full-window, seven-repeat, or M4 qualification.",
        hypothesis="Concurrent candidate generation and independent complete ODE audit reduce wall time when both CPU kernels overlap.",
        horizons=HORIZONS, state_indices=STATE_INDEX, repeats=list(REPEATS), warmups=1,
        semantics=dict(candidate_and_audit_same_committed_state=True,
                       commit_after_both_complete=True, full_ode_audit_every_trial=True,
                       no_audit_cache=True, no_clipping=True, rollback=True),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    save(OUT / "registration.json", registration)
    rows = []
    save(OUT / "runs.json", {"runs": rows})
    for name in ("teusink", "kecoli"):
        case = prepare(name)
        horizon = HORIZONS[name]
        state = np.load(case["reference_path"])["concentrations"][:, STATE_INDEX[name]].copy()
        grid = np.array([0.0, horizon])
        reference_options = dict(method="Radau", t_eval=grid,
                                 rtol=1e-13, atol=np.maximum(case["scale"] * 1e-15, 1e-18))
        if case["system"].flux_jacobian is not None:
            reference_options["jac"] = lambda t, y: (case["system"].S @ case["model"].flux_jacobian(0.0, y)).tocsc()
        else:
            reference_options["jac_sparsity"] = (abs(case["system"].S) @ case["system"].dependencies).astype(bool)
        truth_result = solve_ivp(
            lambda t, y: case["system"].S @ case["model"].flux(0.0, y),
            (0.0, horizon), state, **reference_options,
        )
        if not truth_result.success:
            raise RuntimeError(truth_result.message)
        truth = truth_result.y
        truth_flux = np.column_stack([case["model"].flux(0.0, y) for y in truth.T])
        local = dict(case, times=grid, reference=truth, flux_reference=truth_flux)
        large = name == "kecoli"
        for overlap in (False, True):
            for repeat in (-1,) + REPEATS:
                config = ContinuationConfig(
                    target=1e-3, initial_step=horizon, max_step=horizon,
                    classification_step_floor=horizon, enter=1e-4, leave=3e-5,
                    group_every=1, max_attempts=32, max_rejections=8,
                    slow_bound_limit=1e6, audit_control_only=False,
                    audit_method="BDF" if large else "Radau", audit_rtol=1e-9,
                    fallback_method="CVODE-KLU" if large else "BDF",
                    fallback_rtol=1e-11, timeout=180., overlap_audit=overlap,
                )
                started = time.perf_counter()
                try:
                    controller = MacroSplit(case["system"], case["scale"], config, **case["options"])
                    values, stats = controller.integrate(grid, y0=state)
                    row = dict(model=name, overlap=overlap, repeat=repeat,
                               status="success", seconds=time.perf_counter() - started,
                               parameters=asdict(config), stats=stats,
                               metrics=metrics(values, local, 1e-3, stats),
                               every_attempt_audited=stats["full_calls"] == len(stats["attempts"]),
                               split_coverage=sum(a.get("h", 0.) for a in stats["attempts"]
                                                  if a.get("status") == "accepted") / horizon)
                    np.savez_compressed(OUT / f"{name}-overlap-{int(overlap)}-repeat-{repeat}.npz",
                                        initial=state, values=values, truth=truth,
                                        attempts=np.array(len(stats["attempts"])),
                                        overlap=np.array(overlap), repeat=np.array(repeat))
                except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                    row = dict(model=name, overlap=overlap, repeat=repeat,
                               status="failed", seconds=time.perf_counter() - started,
                               reason=str(exc), stats=getattr(exc, "stats", None))
                rows.append(row)
                save(OUT / "runs.json", {"runs": rows})
                print(name, overlap, repeat, row["status"], round(row["seconds"], 6), flush=True)
    summaries = []
    for name in HORIZONS:
        for overlap in (False, True):
            formal = [r for r in rows if r["model"] == name and r["overlap"] == overlap and r["repeat"] >= 0]
            vals = [r["seconds"] for r in formal if r["status"] == "success"]
            summaries.append(dict(model=name, overlap=overlap, repeats=len(vals),
                                  median_seconds=float(np.median(vals)) if vals else None,
                                  all_metrics_pass=bool(vals) and all(r.get("metrics", {}).get("passes", False) for r in formal),
                                  all_attempts_audited=bool(vals) and all(r.get("every_attempt_audited", False) for r in formal)))
    by_model = {}
    for name in HORIZONS:
        seq = next((s for s in summaries if s["model"] == name and not s["overlap"]), None)
        par = next((s for s in summaries if s["model"] == name and s["overlap"]), None)
        by_model[name] = dict(sequential=seq, overlap=par,
                              wall_speedup=(seq["median_seconds"] / par["median_seconds"]
                                            if seq and par and seq["median_seconds"] and par["median_seconds"] else None))
    summary = dict(registration=registration, summaries=summaries, by_model=by_model,
                   M4_admission=False,
                   interpretation="Overlap preserves independent full nonlinear audits and can only reduce wall time when candidate and audit CPU work overlap. This short-prefix screen is not a formal M4 qualification.")
    save(OUT / "summary.json", summary)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
