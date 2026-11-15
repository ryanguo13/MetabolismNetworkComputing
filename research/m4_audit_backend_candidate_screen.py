"""Compare calibrated audit backends inside the same short split candidate."""
import itertools
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, digest, metrics, prepare
from metabolism.m2_experiments import save
from metabolism.splitting import SplitFailure

OUT = ROOT / "results/m4-audit-backend-candidate-screen"
METHODS = ("CVODE-KLU", "BDF")
TARGETS = (1e-3, 1e-5)
REPEATS = (-1, 0, 1)
STATE_INDEX = 61
WINDOW = 1e-6


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as saved:
        initial = saved["concentrations"][:, STATE_INDEX].copy()
        start_time = float(saved["times"][STATE_INDEX])
    grid = np.r_[start_time, start_time + WINDOW*np.array([.25, .5, .75, 1.])]
    truth_result = solve_ivp(lambda t, y: case["system"].S @ case["model"].flux(t, y),
        (start_time, grid[-1]), initial, method="Radau", t_eval=grid[1:], rtol=1e-13,
        atol=np.maximum(case["scale"]*1e-15, 1e-18),
        jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
    if not truth_result.success:
        raise RuntimeError(truth_result.message)
    reference = np.column_stack((initial, truth_result.y))
    flux_reference = np.column_stack([case["model"].flux(t, y) for t, y in zip(grid, reference.T)])
    local_case = dict(case, times=grid, reference=reference, flux_reference=flux_reference)
    source_files = ["research/m4_audit_backend_candidate_screen.py", "metabolism/continuation.py",
        "metabolism/adaptive.py", "metabolism/splitting.py", "metabolism/kecoli.py",
        "metabolism/cvode.py", "metabolism/m3_continuation.py"]
    registration = dict(date="2026-11-15", model="kecoli", state_index=STATE_INDEX,
        start_time=start_time, interval=WINDOW, query_fractions=[0., .25, .5, .75, 1.],
        methods=["AdaptiveSplit", *METHODS], targets=list(TARGETS), repeats=list(REPEATS),
        candidate=dict(audit_control_only=True, local_fraction=1., group_every=1,
            classification_step_floor=WINDOW, growth_safety=.9, initial_step=1e-8,
            max_step=WINDOW, max_attempts=256, max_rejections=16,
            audit_rtol=1e-9, fallback_method="CVODE-KLU", fallback_rtol=1e-11),
        baseline=dict(method="AdaptiveSplit", fixed_all_reactions=True,
            fallback_method="CVODE-KLU", fallback_rtol=1e-11),
        reference=dict(method="Radau", rtol=1e-13),
        scope="One Kecoli early-state short interval; controller-level backend diagnostic only, not full-window or M4 evidence.",
        source_sha256={p:digest(ROOT/p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT/"registration.json", registration)
    np.savez_compressed(OUT/"reference.npz", initial=initial, values=reference)
    rows=[]
    for target in TARGETS:
        sequence = [("baseline", None, -1), ("candidate", METHODS[0], -1),
                    ("candidate", METHODS[1], -1)]
        for repeat in (0, 1):
            sequence.extend([("baseline", None, repeat), ("candidate", METHODS[0], repeat),
                             ("candidate", METHODS[1], repeat)])
        for kind, method, repeat in sequence:
            if kind == "baseline":
                config = AdaptiveConfig(target=target, initial_step=1e-8, max_step=WINDOW,
                    max_attempts=256, max_rejections=16, timeout=240.,
                    fallback_method="CVODE-KLU", fallback_rtol=1e-11)
            else:
                config = ContinuationConfig(target=target, initial_step=1e-8, max_step=WINDOW,
                    group_every=1, max_attempts=256, max_rejections=16, timeout=240.,
                    audit_method=method, audit_rtol=1e-9, fallback_method="CVODE-KLU",
                    fallback_rtol=1e-11, local_fraction=1., audit_control_only=True,
                    classification_step_floor=WINDOW, growth_safety=.9)
            row=dict(target=target, kind=kind, audit_method=method, repeat=repeat,
                     warmup=repeat==-1, parameters=asdict(config))
            started=time.perf_counter()
            try:
                controller=(AdaptiveSplit(case["system"],case["scale"],config,
                    fixed_groups=np.arange(len(case["system"].groups)),**case["options"])
                    if kind == "baseline" else
                    MacroSplit(case["system"],case["scale"],config,**case["options"]))
                values,stats=controller.integrate(grid,y0=initial)
                row.update(status="success",seconds=time.perf_counter()-started,
                    metrics=metrics(values,local_case,target,stats),stats=stats,
                    every_attempt_audited=(kind == "baseline" or
                        stats["full_calls"]==len(stats["attempts"])),
                    coverage_recomputed=(0. if kind == "baseline" else
                        sum(a["h"] for a in stats["attempts"] if a["status"]=="accepted")/WINDOW))
                np.savez_compressed(OUT/f"{kind}-{target}-{method or 'all'}-{repeat}.npz",values=values)
            except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                row.update(status="failed",seconds=time.perf_counter()-started,reason=str(exc),
                           stats=getattr(exc,"stats",None))
            rows.append(row)
            save(OUT/"runs.json",dict(rows=rows))
            print(target,kind,method,repeat,row["status"],round(row["seconds"],4),flush=True)
    print("complete",len(rows),"rows",flush=True)


if __name__=="__main__":
    main()
