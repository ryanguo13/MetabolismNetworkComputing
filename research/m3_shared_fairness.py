"""Challenge promising stage-sharing cost with equally aggressive full starts."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.history_polynomial import HistoryPolynomial
from research.shared_radau import SharedMRIEndpoint, shared_full_curve
from research.fullstep_radau import FullStepMRIEndpoint, fullstep_curve
from research.inner_dense import InnerCurve, physical_ledger, full_curve
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT, digest

OUT = ROOT/'results/m3-shared-radau/fairness'
OLD = ROOT/'results/m3-inner-dense'
METHODS = ['shared', 'shared_fullstep', 'full', 'full_shared', 'full_fullstep']


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    sources = ['research/m3_shared_fairness.py', 'research/fullstep_radau.py', 'research/test_fullstep_radau.py',
               'research/shared_radau.py', 'research/mri_endpoint.py', 'research/ordered_radau.py',
               'research/inner_dense.py', 'research/history_polynomial.py', 'research/m3_ordered_cost.py', 'research/m3_priority.py']
    inputs = [OLD/'history.json', ROOT/'results/m3-shared-radau/runs.json']+[OLD/f'history-{i}.npz' for i in (2,3,5)]
    reg = dict(date='2026-11-08',paths=[2,3,5],step=4,degree=5,tolerances=[1e-6,1e-7,1e-8,1e-10],
               methods=METHODS,repeats=3,warmups=1,
               selection='Adaptive development follow-up: both large paths with first-stage study speedup >=1.5 and the nonconstant small-model path; not independent validation.',
               hypothesis='Previous gain may depend on automatic initial step causing full solver to use two steps. Give both methods full-interval initial attempts and broader tolerance search.',
               checks='No changed Newton/error thresholds; rejected first attempts shrink normally. Same polynomial positivity, pool and physical/modified ledger limits.',
               timing='Same guards as prior study; free actual history, cached setup; independent actual error outside timer. No estimator, full audit, startup or grouping cost.',
               stopping='No full-window implementation unless qualified candidate has >=1.5 speedup against fastest qualified full option and audit/estimator budget is separately established.',
               source_sha256={p:digest(ROOT/p) for p in sources},
               input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in inputs})
    registration=OUT/'registration.json'
    if registration.exists():
        assert json.loads(registration.read_text())==reg
        if (OUT/'runs.json').exists(): raise FileExistsError('Preserve existing evidence')
    else: save(registration,reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[]
    for index in reg['paths']:
        oldrow=next(r for r in old if r['shard']==f'history-{index}' and r['step']==4)
        saved=np.load(OLD/f'history-{index}.npz');case=prepare(oldrow['model'])
        y,h=saved['endpoints'][:,4],oldrow['h'];query,ref=saved['4-query'],saved['4-reference']
        screen=GroupScreen(case['system'],case['scale'])
        kernels={'shared':SharedMRIEndpoint(case,saved['groups']),'shared_fullstep':FullStepMRIEndpoint(case,saved['groups'])}
        fulls={'full':full_curve,'full_shared':shared_full_curve,'full_fullstep':fullstep_curve}
        arrays={};times,previous=saved['endpoint_times'][:6],saved['endpoints'][:,:5]
        for repeat in (-1,0,1,2):
            offset=(repeat+1+index)%len(METHODS);order=METHODS[offset:]+METHODS[:offset]
            for method in order:
                for tol in (reg['tolerances'] if repeat%2==0 else reg['tolerances'][::-1]):
                    key=f'{method}-{tol}-{repeat}'
                    row=dict(key=key,source_shard=f'history-{index}',model=oldrow['model'],h=h,threshold=oldrow['threshold'],
                             method=method,tolerance=tol,repeat=repeat,
                             setup_seconds=kernels[method].setup_seconds if method in kernels else None)
                    tick=time.perf_counter()
                    try:
                        if method in kernels:
                            trial=kernels[method].trial(y,h,tol,timeout=20.);stats=trial['stats']
                            poly=HistoryPolynomial(times,np.column_stack([previous,trial['fine']]))
                            curve=InnerCurve([poly]);stats.update(physical_ledger(case,curve,y,trial['fine']))
                            stats['continuous_nonnegative']=bool(np.all(poly.minima()>=-1e-12*case['scale']))
                        else:
                            trial=fulls[method](case,y,h,tol,screen);curve,stats=trial['pieces'][0],trial['stats']
                        values=curve(query);guard=structural(case,values);seconds=time.perf_counter()-tick
                        valid=stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                        valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
                        targets={}
                        for target in (1e-4,1e-6):
                            actual=metrics(case,values,ref,target) if stats['continuous_nonnegative'] else None
                            targets[str(target)]=dict(actual=actual,passes=bool(valid and max(actual.values())<=1))
                        row.update(status='success',seconds=seconds,stats=stats,guard=guard,targets=targets)
                        arrays[key+'-values']=values;arrays[key+'-endpoint']=trial['fine']
                        if method in kernels: arrays[key+'-coefficients']=poly.coefficients
                        else:
                            for k,v in curve.pack().items(): arrays[key+'-'+k]=v
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row);save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
            print(index,repeat,'done',flush=True)
        for kernel in kernels.values(): kernel.close()
        tmp=OUT/f'cost-{index}.tmp.npz';np.savez_compressed(tmp,**arrays)
        with np.load(tmp) as check:
            for key in check.files: check[key]
        tmp.replace(OUT/f'cost-{index}.npz')


if __name__=='__main__': run()
