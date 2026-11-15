"""Paired real-Kecoli short-window rejection-audit-cache diagnostic."""
import json
import platform
import time
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, digest, metrics, prepare
from metabolism.m2_experiments import save
from metabolism.splitting import SplitFailure

OUT = ROOT / "results/m4-rejection-audit-cache-screen"
STATE_INDEX = 61
WINDOW = 1e-6
TARGETS = (1e-3, 1e-5)
REPEATS = 3


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as source:
        y0 = source["concentrations"][:, STATE_INDEX].copy()
        archive_time = float(source["times"][STATE_INDEX])
    grid = WINDOW * np.array([0., .25, .5, .75, 1.])
    truth_result = solve_ivp(
        lambda t, y: case["system"].S @ case["model"].flux(t, y),
        (0., WINDOW), y0, method="Radau", t_eval=grid[1:], rtol=1e-13,
        atol=np.maximum(case["scale"] * 1e-15, 1e-18),
        jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
    if not truth_result.success:
        raise RuntimeError(truth_result.message)
    reference = np.column_stack((y0, truth_result.y))
    flux_reference = np.column_stack([case["model"].flux(t, y) for t, y in zip(grid, reference.T)])
    local_case = dict(case, times=grid, reference=reference, flux_reference=flux_reference)
    common = dict(target=None, initial_step=WINDOW, max_step=WINDOW,
                  slow_bound_limit=1e6, max_rejections=16, max_attempts=256,
                  audit_control_only=True, audit_method="Radau", audit_rtol=1e-10,
                  fallback_method="Radau", fallback_rtol=1e-11, timeout=240.)
    files = ["research/m4_rejection_audit_cache_screen.py",
             "metabolism/continuation.py", "metabolism/adaptive.py",
             "metabolism/splitting.py", "metabolism/kecoli.py",
             "metabolism/m3_continuation.py"]
    registration = dict(date="2026-11-15", model="kecoli", state_index=STATE_INDEX,
        archive_time=archive_time, interval=[0., WINDOW], query_fractions=[0., .25, .5, .75, 1.],
        targets=list(TARGETS), repeats=REPEATS, warmups=1,
        common_config={**common, "target": "per target"},
        modes=["fresh", "rejection_audit_cache"],
        truth={"method": "Radau", "rtol": 1e-13},
        gates={"audit_per_attempt": True, "no_clipping": True,
               "transactional_rollback": True, "concentration_flux_netflux_checks": True},
        scope="Short local Kecoli diagnostic only; not M4 qualification.",
        source_sha256={p: digest(ROOT / p) for p in files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / "registration.json", registration)
    np.savez_compressed(OUT / "reference.npz", initial=y0, values=reference)
    rows = []
    for target in TARGETS:
        for repeat in [-1, *range(REPEATS)]:
            modes = ["fresh", "cached"]
            if repeat >= 0 and repeat % 2:
                modes.reverse()
            for mode in modes:
                params = dict(common, target=target, rejection_audit_cache=(mode == "cached"))
                config = ContinuationConfig(**params)
                started = time.perf_counter()
                row = dict(target=target, repeat=repeat, warmup=repeat == -1,
                           mode=mode, parameters=asdict(config))
                try:
                    values, stats = MacroSplit(case["system"], case["scale"], config,
                                               **case["options"]).integrate(grid, y0=y0)
                    row.update(status="success", seconds=time.perf_counter()-started,
                        metrics=metrics(values, local_case, target, stats), stats=stats,
                        state_truth_error=float(np.max(np.abs(values-reference) /
                            np.maximum(case["scale"][:, None], 1e-300))),
                        every_attempt_audited=(stats["audit_calls"] == len(stats["attempts"])
                            and stats["full_calls"] == len(stats["attempts"])),
                        all_rejections_rolled_back=(stats["rejected_steps"] == stats["rollback_checks"]
                            and all(a.get("rollback_verified") for a in stats["attempts"]
                                    if a["status"] == "rejected")))
                    np.savez_compressed(OUT / f"{mode}-{target}-{repeat}.npz", values=values)
                except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                    row.update(status="failed", seconds=time.perf_counter()-started,
                               reason=str(exc), stats=getattr(exc, "stats", None))
                rows.append(row)
                save(OUT / "runs.json", dict(rows=rows))
                print(target, mode, repeat, row["status"], round(row["seconds"], 4),
                      (row.get("stats") or {}).get("audit_cache_hits"), flush=True)
    print("complete", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
