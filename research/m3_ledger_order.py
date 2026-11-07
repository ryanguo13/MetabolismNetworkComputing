"""Freeze-step diagnosis of nonconstant slow-forcing ledger and endpoint errors."""
import json
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_single import SingleMRI
from research.mri_third import MRI3
from research.mri_kernel import GAUSS,Hermite
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_ordered_cost import structural
from research.m3_priority import full,metrics
from research.ledger_diagnostic import split_ledger,flux_integral
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-ledger-order'
OLD=ROOT/'results/m3-inner-dense'
SOURCES=['research/m3_ledger_order.py','research/ledger_diagnostic.py','research/test_ledger_diagnostic.py',
         'research/mri_single.py','research/mri_third.py','research/mri_kernel.py','research/inner_dense.py',
         'research/m3_priority.py','research/m3_ordered_cost.py']


def run():
    OUT.mkdir(exist_ok=True)
    reg=dict(date='2026-11-08',model='teusink',states={'early':2,'late':3},
        state_times={'early':.001,'late':1.},partitions={'constant':0,'nonconstant':2},
        h_values=[.008/(2**i) for i in range(9)],methods=['MRI2','MRI3'],inner_tolerances=[1e-10,1e-12],
        reference_tolerances=[1e-12,1e-13],reference_budget=.05,local_targets=[1e-4,1e-6],ledger_limit=1e-10,
        scope='Frozen existing groups at two known prefix states; one-step endpoint/Hermite error diagnosis, not an online policy or full-window validation.',
        decomposition='signed ledger = weighted endpoint error + reference signed ledger - integral bias along output; same Gauss quadrature on candidate/reference.',
        oracle='Hermite with accurate reference endpoint is only an output-error diagnostic; never a candidate or free history.',
        stopping='Determine whether nonconstant-ledger failure is dominated by endpoint coupling or output before implementing fewer-stage cost candidates.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [ROOT/'results/m3-priority/p0.npz',
                        OLD/'history-0.npz',OLD/'history-2.npz',ROOT/'results/m3-shared-radau/manifest.json']})
    if (OUT/'registration.json').exists():
        assert json.loads((OUT/'registration.json').read_text())==reg
        if (OUT/'runs.json').exists():raise FileExistsError('Do not overwrite evidence')
    else:save(OUT/'registration.json',reg)
    case=prepare('teusink');system=case['system'];weights=case['options']['ledger_weights']
    prefix=np.load(ROOT/'results/m3-priority/p0.npz')['teusink-prefix']
    query=np.unique(np.r_[np.linspace(0,1,65),*( (GAUSS[n][0]+1)/2 for n in (8,16))])
    positions={n:np.searchsorted(query,(GAUSS[n][0]+1)/2) for n in (8,16)}
    rows=[];references=[];partitions={}
    for state,column in reg['states'].items():
        y=prefix[:,column];arrays={'initial':y,'query':query}
        for hi,h in enumerate(reg['h_values']):
            ref=full(case,y,h*query,1e-13);loose=full(case,y,h*query,1e-12)
            budget=metrics(case,loose,ref,1e-6)
            ref_integrals={n:flux_integral(case,ref[:,positions[n]],h,n) for n in (8,16)}
            den=max(1.,abs(float(weights@y)))
            ref_residual=abs(float(weights@(ref[:,-1]-y))-ref_integrals[16])/den
            ref_quad=abs(ref_integrals[16]-ref_integrals[8])/den
            reference_ok=max(budget.values())<=reg['reference_budget'] and max(ref_residual,ref_quad)<=1e-12
            references.append(dict(state=state,h_index=hi,h=h,budget=budget,ledger_residual=ref_residual,
                ledger_quadrature=ref_quad,reference_ok=reference_ok,
                weighted_endpoint_uncertainty=abs(float(weights@(loose[:,-1]-ref[:,-1])))/den))
            arrays[f'{hi}-reference']=ref;arrays[f'{hi}-loose']=loose
            oracle=Hermite(y,ref[:,-1],system.S@system.flux(0.,y),
                           system.S@system.flux(0.,ref[:,-1]),h)
            oracle_ledger=physical_ledger(case,InnerCurve([oracle]),y,ref[:,-1])
            for partition,index in reg['partitions'].items():
                groups=np.load(OLD/f'history-{index}.npz')['groups']
                for method,cls in (('MRI2',SingleMRI),('MRI3',MRI3)):
                    kernel=cls(case,groups)
                    partitions[partition]=dict(groups=groups.tolist(),active=kernel.active.tolist(),fast_columns=kernel.cols.tolist(),
                        slow_columns=np.setdiff1d(np.arange(system.S.shape[1]),kernel.cols).tolist())
                    try:
                        for tol in reg['inner_tolerances']:
                            key=f'{hi}-{partition}-{method}-{tol}';tick=time.perf_counter()
                            row=dict(key=key,state=state,h_index=hi,h=h,partition=partition,method=method,tolerance=tol,
                                     reference_ok=reference_ok,oracle_hermite_ledger=oracle_ledger)
                            try:
                                trial=kernel.trial(y,h,tol);curve=trial['pieces'][0];end=trial['fine'];stats=trial['stats']
                                values=curve(query);guard=structural(case,values)
                                integers={n:flux_integral(case,curve((GAUSS[n][0]+1)/2),h,n) for n in (8,16)}
                                decomposition=split_ledger(weights,y,end,ref[:,-1],integers[16],ref_integrals[16])
                                assert abs(decomposition['decomposition_roundoff'])<1e-13
                                valid=reference_ok and stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                                valid &= all(stats.get(k,np.inf)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
                                targets={}
                                for target in reg['local_targets']:
                                    error=metrics(case,values,ref,target) if stats['continuous_nonnegative'] else None
                                    endpoint=metrics(case,end[:,None],ref[:,-1:],target)
                                    targets[str(target)]=dict(actual=error,endpoint=endpoint,passes=bool(valid and max(error.values())<=1))
                                row.update(status='success',stats=stats,guard=guard,targets=targets,decomposition=decomposition,
                                           endpoint_scaled_error=float(np.max(abs(end-ref[:,-1])/case['scale'])),seconds=time.perf_counter()-tick)
                                arrays[key+'-endpoint']=end;arrays[key+'-values']=values
                                arrays[key+'-coefficients']=np.stack([curve.a,curve.b,curve.c,curve.d])
                            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                                row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}),seconds=time.perf_counter()-tick)
                            rows.append(row)
                    finally:kernel.close()
            save(OUT/'runs.json',dict(runs=rows,references=references,partitions=partitions))
            print(state,h,'done',flush=True)
        np.savez_compressed(OUT/f'{state}.npz',**arrays)


if __name__=='__main__':run()
