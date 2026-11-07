"""Registered self-generated paths: fused guards and current-system LU reuse."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.row_continuous import ROWContinuous
from research.cached_row import CachedROW
from research.cached_full import full_curve as refined_full
from research.inner_dense import InnerCurve,physical_ledger,full_curve
from research.fullstep_radau import fullstep_curve
from research.extrapolated_flow import pack
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-cached-row'
VARIANTS=['original','shared','cached','row-shared','row-cached']


def aggregate(case,local,pieces,origin,end,query,reference,complete):
    curve=InnerCurve(pieces);stats=dict(pool_error=max(r['stats']['pool_error'] for r in local),
        continuous_nonnegative=all(r['stats']['continuous_nonnegative'] for r in local))
    weights=case['options'].get('ledger_weights');den=1. if weights is None else max(1.,abs(float(weights@origin)))
    for k in ('modified_ledger_residual','modified_ledger_quadrature'):
        stats[k]=sum(r['stats'].get(k,0.)*r['ledger_denominator'] for r in local)/den
    stats.update(physical_ledger(case,curve,origin,end));guard,targets=qualified(case,curve,curve(query),reference,stats)
    for target,v in targets.items():v['passes'] &= complete and all(r['targets'][target]['passes'] for r in local)
    return stats,guard,targets


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/'+p for p in ('cached_row.py','refined_lu.py','shared_ledger.py','cached_full.py','test_cached_row.py','m3_cached_row_study.py',
        'row_continuous.py','row_spc.py','shared_radau.py','ordered_radau.py','inner_dense.py','m3_growth_study.py','m3_extrapolated_paths.py')]
    reg=dict(date='2026-11-08',steps=16,paths=[dict(model=m,start=1. if m=='teusink' else .1,h=h,threshold=g)
        for m,h,g in [('teusink',.003,3.),('teusink',.01,3.),('kecoli',.003,1.),('kecoli',.01,.01)]],variants=VARIANTS,tolerance=1e-6,
        selection='Post-ROW screen: strict-qualified partitions at H=.003/.01, one per model/step. Not an independent confirmation or full original window.',
        cache='Current Jacobian every macro; old LU only preconditions CURRENT matrices. At most3 corrections, relative increment correction<=1e-12 with1e-14*concentration-scale floor. Predictor residual/scale<=1e-10; fast residual uses weighted current-matrix row norm. Failure refreshes current LU. No stale-Jacobian formula.',
        transactions='Fresh local cache fork per trial. Commit only after independent strict local accuracy and structural gates; failed attempt discarded and path stops. Never mark failed steps as accepted coverage.',
        guards='Batch only admitted generated SBML expression assembly, same scalar rates and complex-step derivative. Modified Gauss4/8 and physical Gauss8/16 share Gauss8 rates; all residual and quadrature checks retained. Aggregate modified defects are absolute sums.',
        baselines='Uninterrupted mature Radau over the same16-step duration, original/shared/refined, tolerance1e-6/1e-8/1e-10, automatic/whole initial step. Full ROW receives both shared and refined options on identical macro grid.',
        audit='Original scalar full ODE1e-12/1e-11 locally from actual candidate state and globally from prefix.65 points per macro plus public observations, budget<=.05. No reference history passed to algorithm.',
        timing='Kernel+observation guards measured; cached grouping/setup separate. Optimistic cost excludes local audits; audited cost adds every local reference pair, setup and grouping. Global reference/accuracy scoring outside both. Single runs only, no formal repeats.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ('results/m3-row-spc/manifest.json','results/m3-row-continuous/runs.json','results/m3-priority/p0.npz')})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);prefix=np.load(ROOT/'results/m3-priority/p0.npz');rows=[];paths=[];controls=[];references=[]
    def persist():save(OUT/'runs.json',dict(steps=rows,paths=paths,controls=controls,references=references))
    for index,spec in enumerate(reg['paths']):
        case=prepare(spec['model']);origin=prefix[spec['model']+'-prefix'][:,3];h=spec['h'];duration=reg['steps']*h;start=spec['start']
        grid=np.unique(np.r_[np.arange(reg['steps']*64+1)*h/64,case['times'][(case['times']>start)&(case['times']<start+duration)]-start])
        reference,budget,audit_seconds=audit(case,origin,grid/duration,duration);assert max(budget.values())<=.05
        gtick=time.perf_counter();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(origin);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);group_seconds=time.perf_counter()-gtick
        references.append(dict(path=index,budget=budget,reference_ok=True,audit_seconds=audit_seconds,group_seconds=group_seconds))
        np.savez_compressed(OUT/f'path-{index}.npz',origin=origin,grid=grid,groups=groups)
        for method in ('original','shared','refined'):
            for tol in (1e-6,1e-8,1e-10):
                for whole in (False,True):
                    key=f'control-{index}-{method}-{tol}-{whole}';row=dict(key=key,path=index,method=method,tolerance=tol,whole=whole);tick=time.perf_counter()
                    try:
                        trial=(fullstep_curve if whole else full_curve)(case,origin,duration,tol,screen) if method=='original' else refined_full(case,origin,duration,tol,whole,reuse=method=='refined')
                        curve=trial['pieces'][0];values=curve(grid/duration);structural(case,values);elapsed=time.perf_counter()-tick
                        guard,targets=qualified(case,curve,values,reference,trial['stats']);row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                        arrays=curve.pack() if method=='original' else pack(curve);arrays['endpoint']=trial['fine'];np.savez_compressed(OUT/f'{key}.npz',**arrays)
                    except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    controls.append(row);persist()
        for variant in reg['variants']:
            key=f'{index}-{variant}';base_only=variant.startswith('row-');chosen=np.array([],dtype=int) if base_only else groups
            setup=time.perf_counter();kernel=ROWContinuous(case,chosen) if variant=='original' else CachedROW(case,chosen,reuse=variant.endswith('cached'));setup_seconds=time.perf_counter()-setup
            y=origin.copy();pieces=[];accepted=[];arrays={};spent=0.;audit_spent=0.;stop=None
            for step in range(reg['steps']):
                left=step*h;right=(step+1)*h;mask=np.flatnonzero((grid>=left-1e-14)&(grid<=right+1e-14));query=np.clip((grid[mask]-left)/h,0,1)
                row=dict(key=key,path=index,variant=variant,step=step,h=h,query=query.tolist(),ledger_denominator=max(1.,abs(float(kernel.weights@y))) if kernel.weights is not None else 1.)
                before=kernel.cache_state() if variant!='original' else None;tick=time.perf_counter();trial=None
                try:
                    trial=kernel.trial(y,h,reg['tolerance'],base_only=base_only);curve=trial['pieces'][0];values=curve(query);structural(case,values);seconds=time.perf_counter()-tick;spent+=seconds
                    local_ref,local_budget,local_audit=audit(case,y,query,h);audit_spent+=local_audit
                    guard,targets=qualified(case,curve,values,local_ref,trial['stats'])
                    for v in targets.values():v['passes'] &= max(local_budget.values())<=.05
                    accept=targets['1e-06']['passes'];row.update(status='success',accepted=accept,seconds=seconds,stats=trial['stats'],guard=guard,targets=targets,reference_budget=local_budget,audit_seconds=local_audit)
                    arrays[f'{step}-initial']=y;arrays[f'{step}-endpoint']=trial['fine']
                    for k,v in pack(curve).items():arrays[f'{step}-{k}']=v
                    for k in ('increments','slow_increments','stages','rates','full'):arrays[f'{step}-{k}']=trial[k]
                    if not base_only:arrays[f'{step}-forcing']=trial['forcing_coefficients']
                    if variant!='original':
                        if accept:kernel.commit(trial)
                        else:kernel.discard(trial)
                        row.update(cache_before=before,cache_after=kernel.cache_state())
                        if not accept:assert before==kernel.cache_state()
                    if accept:accepted.append(row);pieces.extend(curve.pieces);y=trial['fine'].copy()
                    else:stop='strict local accuracy or structural gate';row['stop_reason']=stop
                except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                    seconds=time.perf_counter()-tick;spent+=seconds;row.update(status='failed',accepted=False,seconds=seconds,reason=str(exc),stats=getattr(exc,'stats',{}));stop=str(exc)
                    if variant!='original':
                        if trial is not None:kernel.discard(trial)
                        assert kernel.cache_state()==before;row.update(cache_before=before,cache_after=kernel.cache_state())
                rows.append(row);persist()
                if stop:break
            summary=dict(key=key,path=index,variant=variant,accepted_steps=len(accepted),complete=len(accepted)==reg['steps'],stop_reason=stop,
                optimistic_seconds=spent,audit_seconds=audit_spent,setup_seconds=setup_seconds,group_seconds=group_seconds if not base_only else 0.,
                audited_seconds=spent+audit_spent+setup_seconds+(group_seconds if not base_only else 0.))
            if pieces:
                endtime=len(accepted)*h;indices=np.flatnonzero(grid<=endtime+1e-14);q=np.clip(grid[indices]/endtime,0,1)
                stats,guard,targets=aggregate(case,accepted,pieces,origin,y,q,reference[:,indices],summary['complete']);summary.update(stats=stats,guard=guard,targets=targets)
            arrays['accepted_endpoint']=y;np.savez_compressed(OUT/f'{key}.npz',**arrays);paths.append(summary);kernel.close();persist()
            print(key,'accepted',len(accepted),'strict',summary.get('targets',{}).get('1e-06',{}).get('passes'),'seconds',spent,flush=True)


if __name__=='__main__':run()
