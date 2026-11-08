"""Prospective bounded predictor-tolerance challenge after correction screen."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.collocation_correction import CorrectedROW
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-collocation-tolerance-fixed';OLD=ROOT/'results/m3-collocation-correction'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    reg=dict(date='2026-11-08',models=[dict(name='teusink',reference_key='0-2',stages=3),dict(name='kecoli',reference_key='1-2',stages=5)],h=.03,sweeps=2,predictor_tolerances=[1e-8,1e-4,1e-2],repeats=[-1,0,1,2],
        methods=['mri-1e-08','mri-0.0001','mri-0.01','row','radau-1e-06-auto','radau-1e-06-whole','radau-1e-08-auto','radau-1e-08-whole'],
        order='Rotate method list by repeat+1; repeat-1 is retained warmup, exclude it from medians.',
        selection='H=.03 and minimum stages passing strict original local screen under two sweeps; same unperturbed prefix and fixed groups. Development follow-up, not independent holdout or formal seven-repeat benchmark.',
        scope='Predictor tolerance only changes internal work; original local external guards and independent full references unchanged. Full ROW gets same correction; mature Radau uninterrupted. No error controller or global claim.',
        repair='First tolerance batch preserved with32 control-dispatch parse failures; this independent rerun repairs scientific-notation method parsing only. All candidate and control timings rerun.',
        reference='Independent complete ODE1e-12/1e-11 pair, budget<=.05, same original65+public grid.',
        source_sha256={p:digest(ROOT/p) for p in ('research/m3_collocation_tolerance_fixed.py','research/collocation_correction.py','research/linear_context_row.py','research/m3_growth_study.py','research/inner_dense.py','research/m3_extrapolated_paths.py')},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OLD/'registration.json',OLD/'runs.json',OLD/'reference-0-2.npz',OLD/'reference-1-2.npz',ROOT/'results/m3-collocation-tolerance/registration.json',ROOT/'results/m3-collocation-tolerance/runs.json']})
    save(OUT/'registration.json',reg);rows=[];references=[]
    for mi,spec in enumerate(reg['models']):
        case=prepare(spec['name']);h=reg['h'];tick=time.perf_counter()
        with np.load(OLD/f"reference-{spec['reference_key']}.npz") as a:y=a['initial'];groups=a['groups'];q=a['query']
        kernel=CorrectedROW(case,groups);screen=GroupScreen(case['system'],case['scale']);setup=time.perf_counter()-tick;ref,budget,seconds=audit(case,y,q,h);assert max(budget.values())<=.05
        references.append(dict(model=spec['name'],budget=budget,seconds=seconds,setup_seconds=setup));np.savez_compressed(OUT/f'reference-{mi}.npz',initial=y,groups=groups,query=q,reference=ref)
        for rep in reg['repeats']:
            shift=(rep+1)%len(reg['methods']);methods=reg['methods'][shift:]+reg['methods'][:shift]
            for position,method in enumerate(methods):
                key=f'{mi}-{rep}-{method}';row=dict(key=key,model=spec['name'],index=mi,repeat=rep,position=position,method=method);tick=time.perf_counter()
                try:
                    if method.startswith('radau'):
                        tol,initial=method.removeprefix('radau-').rsplit('-',1);tol=float(tol);r=(fullstep_curve if initial=='whole' else full_curve)(case,y,h,tol,screen)
                    else:
                        tol=float(method.removeprefix('mri-')) if method.startswith('mri-') else 1e-8
                        r=kernel.trial(y,h,tol,spec['stages'],reg['sweeps'],method=='row')
                    c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                    row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets)
                    # All curves use one uniform power-polynomial file format.
                    if method.startswith('radau'):
                        from research.extrapolated_flow import DensePower
                        from research.inner_dense import InnerCurve
                        c=InnerCurve([DensePower(np.column_stack([p.d,p.c,p.b,p.a]),p.h) for p in c.pieces])
                    np.savez_compressed(OUT/f'{key}.npz',endpoint=r['fine'],**pack(c))
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',reason=str(exc),seconds=time.perf_counter()-tick,stats=getattr(exc,'stats',{}))
                rows.append(row);save(OUT/'runs.json',dict(runs=rows,references=references))
            print(spec['name'],rep,[(r['method'],r['status'],r.get('targets',{}).get('1e-06',{}).get('passes'),round(r['seconds'],5)) for r in rows[-8:]],flush=True)
        kernel.close()


if __name__=='__main__':run()
