"""Separate macro coupling and inner tolerance at frozen budget-stall states.

Local diagnostic only: estimates start at zero because every candidate has
an independent same-state local reference. No reset is used on a trajectory.
"""
import json
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.linear_context_row import LinearContextROW
from research.linear_error_output import LinearStreaming
from research.extrapolated_flow import pack
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-budget-attribution'
OLD=ROOT/'results/m3-streaming-budget'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    reg=dict(date='2026-11-08',models=['teusink','kecoli'],states=['origin','final'],
        base_steps=[[.0003125,.0003125],[.003/1024,.0005779533758707202]],
        divisors=[1,2,4],tolerances=[1e-6,1e-10],grids=[1,2],queries=65,
        purpose='24 local same-state candidates distinguish macro defect from inner tolerance; no global acceptance or speed claim.',
        reference='Independent full ODE pair1e-12/1e-11; strict budget<=.05.',
        source_sha256={p:digest(ROOT/p) for p in ['research/m3_budget_attribution.py','research/linear_context_row.py','research/streaming_defect.py','research/linear_error_output.py','research/inner_dense.py','research/m3_growth_study.py','research/m3_extrapolated_paths.py']},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OLD/'runs.json',OLD/'registration.json',OLD/'path-0.npz',OLD/'path-1.npz']})
    save(OUT/'registration.json',reg);rows=[]
    for mi,name in enumerate(reg['models']):
        case=prepare(name)
        with np.load(OLD/f'path-{mi}.npz') as src:
            groups=src['groups'];states=[src[k] for k in reg['states']]
        kernel=LinearContextROW(case,groups)
        for si,y in enumerate(states):
            for divisor in reg['divisors']:
                h=reg['base_steps'][mi][si]/divisor;q=np.linspace(0.,1.,reg['queries'])
                ref,budget,seconds=audit(case,y,q,h)
                for tol in reg['tolerances']:
                    key=f'{mi}-{si}-{divisor}-{tol}';row=dict(key=key,model=name,state=reg['states'][si],h=h,tolerance=tol,reference_budget=budget,audit_seconds=seconds)
                    tick=time.perf_counter()
                    try:
                        trial=kernel.trial(y,h,tol);curve=trial['pieces'][0];estimate=LinearStreaming(case,y,reg['grids']).trial(trial,q)
                        guard,targets=qualified(case,curve,curve(q),ref,trial['stats'])
                        for target in targets.values():target['passes'] &= max(budget.values())<=.05
                        row.update(status='success',stats=trial['stats'],estimate=dict(status=estimate['status'],targets=estimate['targets'],stats=estimate['stats'],seconds=estimate['seconds']),guard=guard,targets=targets)
                        np.savez_compressed(OUT/f'{key}.npz',initial=y,groups=groups,reference=ref,endpoint=trial['fine'],**pack(curve))
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',reason=str(exc),seconds=time.perf_counter()-tick)
                    rows.append(row);save(OUT/'runs.json',dict(runs=rows))
                print(name,reg['states'][si],divisor,[(r['tolerance'],r.get('stats',{}).get('physical_ledger_residual'),r.get('targets',{}).get('1e-06')) for r in rows[-2:]],flush=True)
        kernel.close()


if __name__=='__main__':run()
