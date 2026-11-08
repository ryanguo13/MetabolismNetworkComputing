"""Prospective linear-action cost/accuracy screen; no nonlinear coverage credit."""
import itertools
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze
from research.rational_affine import action
from research.m3_priority import metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-rational-affine'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    reg=dict(date='2026-11-08',models=['teusink','kecoli'],steps=[.003,.01,.03],dimensions=[4,8,12,20,32],shifts=[.25,1.],
        construction='Scale by fixed species scales; inverse Arnoldi from f0 with(I-shift*H*J)^-1, two-pass Gram-Schmidt, directly project J. Small augmented exponential advances requested query intervals; cache only bitwise-equal gaps.',
        scope='60 frozen-linear diagnostics, dimensions capped by actual dimension. No nonlinear stage or reaction fast correction; sampled positivity/pools only. No physical/modified ledger or continuous positivity certification; no M4 eligibility.',
        reference='Use frozen affine independent full Radau/CVODE-validated reference from m3-affine-attribution; same query,initial and gradient; verify manifest before comparison. Reference budgets<=.05 strict.',
        timing='Each measured call includes current analytic reaction gradient and J formation, full initial rates, inverse factorization, basis solves/projection, small exponentials, reconstruction and sampled structure. Setup outside; no full candidate or speedup claim.',
        source_sha256={p:digest(ROOT/p) for p in ('research/rational_affine.py','research/test_rational_affine.py','research/m3_rational_affine_study.py','research/affine_partition.py','research/m3_priority.py','research/m3_ordered_cost.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-affine-attribution/manifest.json','results/m3-affine-attribution/runs.json','results/m3-merb-predictor/registration.json','results/m3-merb-predictor/runs.json')})
    save(OUT/'registration.json',reg);rows=[]
    for mi,name in enumerate(reg['models']):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale'])
        for hi,h in enumerate(reg['steps']):
            oldkey=f'{mi}-{hi+1}'
            with np.load(ROOT/f'results/m3-affine-attribution/reference-{oldkey}.npz') as a:y=a['initial'];q=a['query'];reference=a['affine_reference']
            frozen,ctx=freeze(case,y)
            for dimension,shift in itertools.product(reg['dimensions'],reg['shifts']):
                key=f'{mi}-{hi}-{dimension}-{shift}';row=dict(key=key,model=name,h=h,oldkey=oldkey,dimension=dimension,shift=shift);tick=time.perf_counter()
                try:
                    gradient=screen.gradient(y);j=(case['system'].S@gradient).tocsc();rhs=case['system'].S@case['system'].flux(0,y);setup=time.perf_counter()-tick
                    r=action(j,rhs,y,case['scale'],h,q,dimension,shift);guard=structural(frozen,r['values']);seconds=time.perf_counter()-tick;actual=metrics(frozen,r['values'],reference,1e-6)
                    row.update(status='success',seconds=seconds,formation_seconds=setup,stats=r['stats'],guard=guard,actual=actual,sampled_accurate=bool(max(actual.values())<=1),M4_eligible=False)
                    np.savez_compressed(OUT/f'{key}.npz',**{k:r[k] for k in ('values','derivatives','basis','projection','drive')})
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),M4_eligible=False)
                rows.append(row);save(OUT/'runs.json',dict(runs=rows))
            print(name,h,[(r['dimension'],r['shift'],r['status'],r.get('actual',{}).get('concentration'),round(r['seconds'],5)) for r in rows[-10:]],flush=True)


if __name__=='__main__':run()
