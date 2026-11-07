"""Self-generated extrapolation history and uninterrupted full-system controls."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.extrapolated_flow import ExtrapolatedFlow,pack
from research.inner_dense import InnerCurve,physical_ledger,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_priority import full,metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-extrapolated-flow';OLD=ROOT/'results/m3-inner-dense'


def qualified(case,curve,values,reference,stats):
    guard=structural(case,values)
    valid=stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
    valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
    targets={}
    for target in (1e-4,1e-6):
        error=metrics(case,values,reference,target) if stats['continuous_nonnegative'] else None
        targets[str(target)]=dict(actual=error,passes=bool(valid and max(error.values())<=1))
    return guard,targets


def run():
    paths=[2,4,5]
    sources=['research/m3_extrapolated_paths.py','research/extrapolated_flow.py','research/inner_dense.py',
             'research/fullstep_radau.py','research/shared_radau.py','research/m3_priority.py','research/m3_ordered_cost.py']
    reg=dict(date='2026-11-08',paths=paths,steps=16,orders=[2,3,4],tolerance=1e-6,whole_step=True,
        baselines=dict(tolerances=[1e-6,1e-8,1e-10],initial_steps=['automatic','whole']),
        selection='After local screen: nonconstant small model plus large H=.001/.003. All orders retained, not independent confirmation.',
        history='First q-1 intervals solved with actual guarded full Radau; thereafter own computed endpoints/slow RHS only. Startup, current slow RHS and output guards counted.',
        scope='Forced 16-step diagnostic: failed accuracy steps are not accepted coverage. Stop on negative output/integration failure. No formal performance claim from one run.',
        reference='Independent local and whole-path full references at1e-12/1e-11; error budget<=.05 at strict local target. 65 uniform points per macro plus public observation points.',
        timing='Candidate includes startup and all guards, excludes independent audits/error scoring; full controls integrate entire same window without macro-boundary restarts. Group setup separately recorded.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'runs.json',OLD/'history.json']+[OLD/f'history-{i}.npz' for i in paths]})
    registration=OUT/'path-registration.json'
    if registration.exists():
        assert json.loads(registration.read_text())==reg
        if (OUT/'paths.json').exists():raise FileExistsError('Preserve existing evidence')
    else:save(registration,reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[];summaries=[];baselines=[];references=[]
    for index in paths:
        original=next(r for r in old if r['shard']==f'history-{index}' and r['step']==0)
        src=np.load(OLD/f'history-{index}.npz');case=prepare(original['model']);origin=src['initial'];h=original['h'];duration=16*h
        observed=case['times'][(case['times']>original['start'])&(case['times']<original['start']+duration)]-original['start']
        grid=np.unique(np.r_[np.arange(16*64+1)*h/64,observed])
        ref=full(case,origin,grid,1e-12);loose=full(case,origin,grid,1e-11);budget=metrics(case,loose,ref,1e-6)
        assert max(budget.values())<=.05
        references.append(dict(path=index,global_budget=budget))
        a=dict(initial=origin,times=grid,reference=ref,loose=loose,groups=src['groups'])
        screen=GroupScreen(case['system'],case['scale'])
        for tol in reg['baselines']['tolerances']:
            for whole in (False,True):
                key=f'full-{tol}-{whole}';tick=time.perf_counter()
                row=dict(path=index,key=key,tolerance=tol,whole_step=whole)
                try:
                    trial=(fullstep_curve if whole else full_curve)(case,origin,duration,tol,screen)
                    curve=trial['pieces'][0];values=curve(grid/duration);structural(case,values);seconds=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,values,ref,trial['stats'])
                    row.update(status='success',seconds=seconds,stats=trial['stats'],guard=guard,targets=targets)
                    for k,v in curve.pack().items():a[key+'-'+k]=v
                    a[key+'-endpoint']=trial['fine']
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                baselines.append(row)
        np.savez_compressed(OUT/f'path-reference-{index}.npz',**a)
        for order in reg['orders']:
            kernel=ExtrapolatedFlow(case,src['groups']);states=[origin.copy()];times=[0.];slow=[];arrays={};pieces=[];total=0.;local=[]
            for step in range(16):
                y=states[-1];key=str(step);tick=time.perf_counter();startup=step<order-1
                row=dict(path=index,order=order,step=step,startup=startup,key=key,model=original['model'],h=h)
                try:
                    if startup:
                        slow.append(kernel.slow(y))
                        trial=fullstep_curve(case,y,h,1e-6,screen)
                    else:
                        trial=kernel.trial(y,h,times[-order:],np.column_stack(slow[-(order-1):]),1e-6,True)
                        slow.append(trial['forcing_coefficients'][:,0].copy())
                    curve=trial['pieces'][0];stats=trial['stats'];position=np.flatnonzero((grid>=step*h-1e-13*h)&(grid<=(step+1)*h+1e-13*h))
                    query=np.clip((grid[position]-step*h)/h,0,1);values=curve(query);structural(case,values)
                    # Recording history and coefficients is outside algorithm timing.
                    elapsed=time.perf_counter()-tick;total+=elapsed
                    rtick=time.perf_counter();atimes=np.unique(np.r_[0.,query*h]);indices=np.searchsorted(atimes,query*h)
                    local_ref=full(case,y,atimes,1e-12)[:,indices];local_loose=full(case,y,atimes,1e-11)[:,indices]
                    audit_seconds=time.perf_counter()-rtick;local_budget=metrics(case,local_loose,local_ref,1e-6)
                    guard,targets=qualified(case,curve,values,local_ref,stats)
                    for record in targets.values():record['passes'] &= max(local_budget.values())<=.05
                    row.update(status='success',seconds=elapsed,stats=stats,guard=guard,targets=targets,reference_budget=local_budget,
                               audit_pair_seconds=audit_seconds)
                    arrays[key+'-query']=query;arrays[key+'-reference']=local_ref;arrays[key+'-loose']=local_loose
                    arrays[key+'-endpoint']=trial['fine']
                    for k,v in (curve.pack() if startup else pack(curve)).items():arrays[key+'-'+k]=v
                    if not startup:arrays[key+'-forcing']=trial['forcing_coefficients']
                    if not stats['continuous_nonnegative']:raise ValueError('Negative output; stop diagnostic path')
                    pieces.extend(curve.pieces);states.append(trial['fine'].copy());times.append((step+1)*h)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    if 'seconds' not in row:
                        elapsed=time.perf_counter()-tick;total+=elapsed
                    row.update(status='failed',seconds=row.get('seconds',elapsed),reason=str(exc))
                rows.append(row);local.append(row)
                save(OUT/'paths.json',dict(runs=rows,trajectories=summaries,baselines=baselines,references=references))
                if row['status']=='failed':break
            summary=dict(path=index,order=order,steps=len(states)-1,seconds=total,setup_seconds=kernel.setup_seconds,
                         startup_steps=order-1,complete=len(states)==17)
            if summary['complete']:
                curve=InnerCurve(pieces);values=curve(grid/duration);guard=structural(case,values)
                stats=dict(pool_error=max(r['stats']['pool_error'] for r in local),
                    continuous_nonnegative=all(r['stats']['continuous_nonnegative'] for r in local),
                    modified_ledger_residual=max(r['stats'].get('modified_ledger_residual',0) for r in local),
                    modified_ledger_quadrature=max(r['stats'].get('modified_ledger_quadrature',0) for r in local))
                stats.update(physical_ledger(case,curve,origin,states[-1]));guard,targets=qualified(case,curve,values,ref,stats)
                for t,record in targets.items():record['passes'] &= all(r['targets'][t]['passes'] for r in local)
                summary.update(stats=stats,guard=guard,targets=targets,
                    local_passes={t:sum(r['targets'][t]['passes'] for r in local) for t in ('0.0001','1e-06')},
                    total_reference_pair_seconds=sum(r['audit_pair_seconds'] for r in local))
            arrays['endpoints']=np.column_stack(states);arrays['endpoint_times']=np.array(times)
            np.savez_compressed(OUT/f'trajectory-{index}-{order}.npz',**arrays);summaries.append(summary);kernel.close()
            save(OUT/'paths.json',dict(runs=rows,trajectories=summaries,baselines=baselines,references=references))
            print(index,order,summary['complete'],summary.get('local_passes'),flush=True)


if __name__=='__main__':run()
