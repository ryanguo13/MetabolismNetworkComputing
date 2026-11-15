"""Balanced short-window screen of intermediate audited step-growth factors."""
import json
import platform
import time
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, digest, metrics, prepare
from metabolism.m2_experiments import save
from metabolism.splitting import SplitFailure

OUT = ROOT / "results/m4-growth-factor-screen-v3"
SAFETIES = (.9, 1.1, 1.3, 1.5)
TARGETS = (1e-3, 1e-5)
REPEATS = 2
STATE_INDEX = 61
WINDOW = 1e-6


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as source:
        y0 = source["concentrations"][:, STATE_INDEX].copy()
        archive_time = float(source["times"][STATE_INDEX])
    grid = WINDOW * np.array([0., .25, .5, .75, 1.])
    truth_result = solve_ivp(lambda t,y: case["system"].S @ case["model"].flux(t,y),
        (0., WINDOW), y0, method="Radau", t_eval=grid[1:], rtol=1e-13,
        atol=np.maximum(case["scale"]*1e-15, 1e-18),
        jac=lambda t,y: case["system"].S @ case["model"].flux_jacobian(t,y))
    if not truth_result.success:
        raise RuntimeError(truth_result.message)
    reference = np.column_stack((y0, truth_result.y))
    flux_reference = np.column_stack([case["model"].flux(t,y) for t,y in zip(grid,reference.T)])
    local_case = dict(case,times=grid,reference=reference,flux_reference=flux_reference)
    candidate = dict(audit_control_only=True,local_fraction=1.,group_every=1,
        classification_step_floor=WINDOW,initial_step=1e-8,max_step=WINDOW,
        max_attempts=256,max_rejections=16,audit_method="CVODE-KLU",audit_rtol=1e-9,
        fallback_method="CVODE-KLU",fallback_rtol=1e-11,timeout=240.)
    source_files=["research/m4_growth_factor_screen_v3.py","metabolism/continuation.py",
        "metabolism/adaptive.py","metabolism/splitting.py","metabolism/kecoli.py",
        "metabolism/cvode.py","metabolism/m3_continuation.py"]
    registration=dict(date="2026-11-15",model="kecoli",state_index=STATE_INDEX,
        archive_time=archive_time,interval=[0.,WINDOW],query_fractions=[0.,.25,.5,.75,1.],
        targets=list(TARGETS),safeties=list(SAFETIES),warmups=1,repeats=REPEATS,
        candidate=candidate,baseline=dict(initial_step=1e-8,max_step=WINDOW,
            group_every=5,enter=1.,leave=.3,inner_fraction=.05,local_fraction=.1,
            max_rejections=16,max_attempts=256,slow_bound_limit=2.,
            fallback_method="CVODE-KLU",audit_rtol=1e-11,fallback_rtol=1e-11,
            estimator_ratio_limit=2.,timeout=240.),
        truth=dict(method="Radau",rtol=1e-13),
        gates=dict(full_ode_audit_every_candidate_attempt=True,no_clipping=True,
            transactional_rollback=True,concentration_flux_netflux_and_physical_checks=True),
        scope="One early Kecoli local interval; diagnostic only, not full-window or M4 qualification.",
        source_sha256={p:digest(ROOT/p) for p in source_files},
        environment=dict(python=platform.python_version(),numpy=np.__version__))
    save(OUT/"registration.json",registration)
    np.savez_compressed(OUT/"reference.npz",initial=y0,values=reference)
    rows=[]
    for target in TARGETS:
        base=AdaptiveConfig(target=target,initial_step=1e-8,max_step=WINDOW,
            group_every=5,enter=1.,leave=.3,inner_fraction=.05,local_fraction=.1,
            max_rejections=16,max_attempts=256,slow_bound_limit=2.,
            fallback_method="CVODE-KLU",audit_rtol=1e-11,fallback_rtol=1e-11,
            estimator_ratio_limit=2.,timeout=240.)
        sequence=[("baseline",None,-1)] + [("candidate",s,-1) for s in SAFETIES]
        for rep in range(REPEATS):
            sequence += [("baseline",None,rep)] + [("candidate",s,rep) for s in SAFETIES]
        for kind,safety,repeat in sequence:
            if kind=="baseline":
                config=base
                controller=lambda:AdaptiveSplit(case["system"],case["scale"],config,
                    fixed_groups=np.arange(len(case["system"].groups)),**case["options"])
            else:
                config=ContinuationConfig(**dict(candidate,target=target,growth_safety=safety))
                controller=lambda:MacroSplit(case["system"],case["scale"],config,**case["options"])
            row=dict(target=target,kind=kind,growth_safety=safety,repeat=repeat,
                warmup=repeat==-1,parameters=asdict(config))
            started=time.perf_counter()
            try:
                values,stats=controller().integrate(grid,y0=y0)
                row.update(status="success",seconds=time.perf_counter()-started,
                    metrics=metrics(values,local_case,target,stats),stats=stats,
                    every_attempt_audited=(kind=="baseline" or stats["full_calls"]==len(stats["attempts"])),
                    coverage_recomputed=(0. if kind=="baseline" else
                        sum(a["h"] for a in stats["attempts"] if a["status"]=="accepted")/WINDOW))
                np.savez_compressed(OUT/f"{kind}-{target}-{safety or 'all'}-{repeat}.npz",values=values)
            except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                row.update(status="failed",seconds=time.perf_counter()-started,reason=str(exc),
                           stats=getattr(exc,"stats",None))
            rows.append(row)
            save(OUT/"runs.json",dict(rows=rows))
            print(target,kind,safety,repeat,row["status"],round(row["seconds"],4),
                  (row.get("stats") or {}).get("split_coverage"),flush=True)
    print("complete",len(rows),"rows",flush=True)


if __name__=="__main__": main()
