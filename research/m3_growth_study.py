"""Fixed bounded-growth diagnostic; keeps full local and global audits."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.extrapolated_flow import ExtrapolatedFlow,pack
from research.growth_history import schedule,bootstrap
from research.inner_dense import InnerCurve,physical_ledger,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_priority import full,metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-growth-history';OLD=ROOT/'results/m3-inner-dense'


def query_grid(case,start,times):
    observed=case['times'][(case['times']>start)&(case['times']<start+times[-1])]-start
    return np.unique(np.r_[observed,*[np.linspace(a,b,65) for a,b in zip(times[:-1],times[1:])]])


def audit(case,y,query,h):
    times=np.unique(np.r_[0.,h*query]);indices=np.searchsorted(times,h*query);tick=time.perf_counter()
    ref=full(case,y,times,1e-12)[:,indices];loose=full(case,y,times,1e-11)[:,indices]
    seconds=time.perf_counter()-tick;budget=metrics(case,loose,ref,1e-6)
    return ref,budget,seconds


def final_assessment(case,pieces,origin,end,local,grid,reference):
    curve=InnerCurve(pieces);values=curve(grid/curve.h);guard=structural(case,values)
    stats=dict(pool_error=max(r['stats']['pool_error'] for r in local),
        continuous_nonnegative=all(r['stats']['continuous_nonnegative'] for r in local),
        modified_ledger_residual=max(r['stats'].get('modified_ledger_residual',0.) for r in local),
        modified_ledger_quadrature=max(r['stats'].get('modified_ledger_quadrature',0.) for r in local))
    stats.update(physical_ledger(case,curve,origin,end))
    guard,targets=qualified(case,curve,values,reference,stats)
    for t,v in targets.items():v['passes'] &= all(r['targets'][t]['passes'] for r in local)
    return stats,guard,targets


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/m3_growth_study.py','research/growth_history.py','research/test_growth_history.py',
        'research/extrapolated_flow.py','research/fullstep_radau.py','research/inner_dense.py','research/m3_extrapolated_paths.py',
        'research/m3_priority.py','research/m3_ordered_cost.py']
    reg=dict(date='2026-11-08',paths=[2,4,5],duration_initial_steps=16,orders=[3,4],growth=[1.,1.5,2.],tolerances=[1e-6,1e-8],
        startup='One guarded full solve over (q-1)*h0; sample its actual dense output at q history points, exact final solver state. Include solve/history slow evaluation cost.',
        schedule='After startup, increase h by ratio in [1,2], shorten terminal step to hit same fixed window. No error-based acceptance or reference-dependent step selection.',
        scope='36 forced development trajectories; failed accuracy/ledger steps are never accepted coverage. Negative curves/integration failure stop path.',
        audits='Local and whole-window independent full1e-12/1e-11, 65 samples per macro plus public observation points. Reference budget<=.05. All original output/ledger/pool/positivity limits.',
        timing='Cached grouping. Sum startup/candidate/output-guard times; independent audits and accuracy scoring outside timer. Final cumulative assessment outside timer on both sides. Optimistic candidate cost only.',
        baseline='Uninterrupted full Radau over same window and a common grid covering every candidate query; tolerances1e-6/1e-8/1e-10, automatic/whole first step. Single runs, not formal timings.',
        storage='Save every numerical output polynomial, endpoint, forcing and decision. Do not duplicate large independent reference arrays: default verifier reintegrates both reference tolerances from saved initial states and frozen code.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in
            [ROOT/'results/m3-extrapolated-flow/manifest.json',OLD/'history.json']+[OLD/f'history-{i}.npz' for i in (2,4,5)]})
    if (OUT/'registration.json').exists():
        assert json.loads((OUT/'registration.json').read_text())==reg
        if (OUT/'runs.json').exists():raise FileExistsError('Preserve prior evidence')
    else:save(OUT/'registration.json',reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[];trajectories=[];baselines=[];references=[]
    for index in reg['paths']:
        original=next(r for r in old if r['shard']==f'history-{index}' and r['step']==0)
        source=np.load(OLD/f'history-{index}.npz');case=prepare(original['model']);origin=source['initial'];h0=original['h'];duration=16*h0
        screen=GroupScreen(case['system'],case['scale'])
        grids=[query_grid(case,original['start'],schedule(h0,duration,q,r)) for q in reg['orders'] for r in reg['growth']]
        grid=np.unique(np.concatenate(grids));global_ref,budget,ref_seconds=audit(case,origin,grid/duration,duration)
        assert max(budget.values())<=.05
        references.append(dict(path=index,budget=budget,seconds=ref_seconds))
        arrays=dict(initial=origin,groups=source['groups'],global_times=grid)
        for tol in (1e-6,1e-8,1e-10):
            for whole in (False,True):
                key=f'{tol}-{whole}';tick=time.perf_counter();row=dict(path=index,key=key,tolerance=tol,whole=whole)
                try:
                    trial=(fullstep_curve if whole else full_curve)(case,origin,duration,tol,screen)
                    curve=trial['pieces'][0];values=curve(grid/duration);structural(case,values);elapsed=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,values,global_ref,trial['stats'])
                    row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                    arrays[key+'-endpoint']=trial['fine']
                    for k,v in curve.pack().items():arrays[key+'-'+k]=v
                except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                baselines.append(row)
        np.savez_compressed(OUT/f'reference-{index}.npz',**arrays)
        for order in reg['orders']:
            for ratio in reg['growth']:
                planned=schedule(h0,duration,order,ratio)
                for tol in reg['tolerances']:
                    name=f'{index}-{order}-{ratio}-{tol}';kernel=ExtrapolatedFlow(case,source['groups'])
                    y=origin.copy();local=[];pieces=[];total=0.;arrays={'schedule':planned};hist_times=None;hist_states=None;slow=None
                    for step,(left,right) in enumerate(zip(planned[:-1],planned[1:])):
                        h=right-left;tick=time.perf_counter();row=dict(name=name,path=index,order=order,ratio=ratio,tolerance=tol,
                            step=step,left=left,right=right,h=h,startup=step==0,model=original['model'])
                        try:
                            if step==0:
                                trial,ht,hs,ss=bootstrap(case,y,order,h0,tol,screen,kernel)
                                hist_times=list(ht);hist_states=list(hs.T);slow=list(ss.T)
                                arrays['bootstrap_times']=ht;arrays['bootstrap_states']=hs;arrays['bootstrap_slow']=ss
                            else:
                                trial=kernel.trial(y,h,hist_times[-order:],np.column_stack(slow[-(order-1):]),tol,True)
                                slow.append(trial['forcing_coefficients'][:,0].copy())
                                hist_times.append(right);hist_states.append(trial['fine'].copy())
                            curve=trial['pieces'][0];stats=trial['stats']
                            pos=np.flatnonzero((grid>=left-1e-13*h0)&(grid<=right+1e-13*h0));query=np.clip((grid[pos]-left)/h,0,1)
                            values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick;total+=elapsed
                            reference,local_budget,audit_seconds=audit(case,y,query,h)
                            guard,targets=qualified(case,curve,values,reference,stats)
                            for t,v in targets.items():v['passes'] &= max(local_budget.values())<=.05
                            row.update(status='success',seconds=elapsed,stats=stats,guard=guard,targets=targets,reference_budget=local_budget,audit_seconds=audit_seconds)
                            key=str(step);arrays[key+'-query']=query;arrays[key+'-initial']=y;arrays[key+'-endpoint']=trial['fine']
                            for k,v in (curve.pack() if step==0 else pack(curve)).items():arrays[key+'-'+k]=v
                            if step:arrays[key+'-forcing']=trial['forcing_coefficients']
                            if not stats['continuous_nonnegative']:raise ValueError('Negative output; stop forced diagnostic')
                            pieces.extend(curve.pieces);y=trial['fine'].copy()
                        except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                            if 'seconds' not in row:
                                elapsed=time.perf_counter()-tick;total+=elapsed
                            row.update(status='failed',seconds=row.get('seconds',elapsed),reason=str(exc),stats=row.get('stats',getattr(exc,'stats',{})))
                        rows.append(row);local.append(row)
                        if row['status']=='failed':break
                    summary=dict(name=name,path=index,order=order,ratio=ratio,tolerance=tol,steps=len(local),planned_steps=len(planned)-1,
                        complete=len(local)==len(planned)-1 and all(r['status']=='success' for r in local),seconds=total,setup_seconds=kernel.setup_seconds)
                    if summary['complete']:
                        stats,guard,targets=final_assessment(case,pieces,origin,y,local,grid,global_ref)
                        summary.update(stats=stats,guard=guard,targets=targets,
                            local_passes={t:sum(r['targets'][t]['passes'] for r in local) for t in ('0.0001','1e-06')},audit_seconds=sum(r['audit_seconds'] for r in local))
                    trajectories.append(summary);kernel.close();np.savez_compressed(OUT/f'{name}.npz',**arrays)
                    save(OUT/'runs.json',dict(runs=rows,trajectories=trajectories,baselines=baselines,references=references))
                    print(name,summary['complete'],summary.get('local_passes'),flush=True)


if __name__=='__main__':run()
