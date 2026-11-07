"""Bounded prospective controller stress test with mandatory full audits.

Step decisions use cumulative estimator/ledger budgets only. Independent local
and global full ODE checks can veto and stop, never silently tune the controller.
"""
import json
import time
import hashlib
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.linear_context_row import LinearContextROW
from research.linear_error_output import LinearStreaming
from research.extrapolated_flow import pack
from research.inner_dense import InnerCurve,physical_ledger,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_priority import full,metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-streaming-controller'
LEDGERS=('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature')


def state_signature(estimator,y,left,ledger):
    d=hashlib.sha256();d.update(y.tobytes());d.update(np.array([left,estimator.generation],dtype=float).tobytes())
    for e in estimator.errors:d.update(e.tobytes())
    d.update(json.dumps(dict(scores=estimator.scores,ledger=ledger),sort_keys=True).encode());return d.hexdigest()


def proposal_budget(scores,ledger,end,duration):
    fraction=end/duration
    return max(max(scores.values(),default=np.inf)/fraction,max(ledger.values())/(1e-10*fraction))


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/'+p for p in ('m3_streaming_controller.py','linear_context_row.py','streaming_defect.py','linear_error_output.py','m3_priority.py','m3_growth_study.py','m3_extrapolated_paths.py')]
    reg=dict(date='2026-11-08',paths=[dict(model='teusink',start=1.,h0=.01,duration=.08,threshold=3.),dict(model='kecoli',start=.1,h0=.003,duration=.024,threshold=1.)],
        perturbation='First strictly positive prefix concentration multiplied by1.001; initial pools rebased to perturbed state. Same perturbation in all references and mature controls.',
        grids=[1,2],inner_tolerance=1e-6,target='1e-06',max_attempts=64,min_step_ratio=1e-5,
        acceptance='Require running factor4 error score<=elapsed/duration and absolute sums of each physical/modified residual/quadrature<=1e-10*elapsed/duration, plus original positivity/pool/individual-ledger gates. No reference used in proposals.',
        step_rule='Start h0. Reject halves H; accepted budget ratio<.25 grows1.5 capped2*h0, otherwise retain H. Shorten final H to exact endpoint. No reference-triggered retries: any estimator/structural proposal accepted but independent strict local/global check failed stops and records a false acceptance.',
        scope='Two prospectively registered perturbed short paths; original full-window/accuracy/coverage/repeat performance gates unchanged. Every attempt audited locally and globally regardless of estimator decision; hard64-attempt cap.',
        timing='Kernel,estimator,query guard,decision/commit timed separately; include setup/grouping and all local/global pair audits in audited sum. Reference error scoring/global final evaluation outside; no formal performance claim.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ('results/m3-priority/p0.npz','results/m3-streaming-linear/runs.json','results/m3-streaming-linear/registration.json')})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);rows=[];paths=[];controls=[]
    def persist():save(OUT/'runs.json',dict(attempts=rows,paths=paths,controls=controls))
    for index,spec in enumerate(reg['paths']):
        case=prepare(spec['model']);origin=np.load(ROOT/'results/m3-priority/p0.npz')[spec['model']+'-prefix'][:,3].copy();perturbed=int(np.flatnonzero(origin>0)[0]);origin[perturbed]*=1.001
        setup=time.perf_counter();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(origin);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);kernel=LinearContextROW(case,groups);est=LinearStreaming(case,origin,reg['grids']);setup_seconds=time.perf_counter()-setup
        duration=spec['duration'];h=spec['h0'];y=origin.copy();left=0.;ledger={k:0. for k in LEDGERS};weights=case['options'].get('ledger_weights');den0=1. if weights is None else max(1.,abs(float(weights@origin)))
        arrays=dict(origin=origin,groups=groups);pieces=[];accepted=[];accepted_grid=[np.array([0.])];max_actual={k:0. for k in ('concentration','flux')+ (('net_flux',) if 'net_map' in case['options'] else ())};reason=None
        for attempt in range(reg['max_attempts']):
            h=min(h,duration-left);right=left+h;public=case['times'][(case['times']>spec['start']+left)&(case['times']<spec['start']+right)]-spec['start']
            q=np.unique(np.r_[np.linspace(0,1,65),(public-left)/h]);before=state_signature(est,y,left,ledger);tick=time.perf_counter();trial=None;estimate=None
            row=dict(path=index,attempt=attempt,left=left,h=h,right=right,query=q.tolist(),before=before)
            try:
                trial=kernel.trial(y,h,reg['inner_tolerance']);curve=trial['pieces'][0];values=curve(q);structural(case,values);candidate_seconds=time.perf_counter()-tick
                estimate=est.trial(trial,q);decision_start=time.perf_counter()
                den=1. if weights is None else max(1.,abs(float(weights@y)));pending={k:ledger[k]+trial['stats'].get(k,0.)*den/den0 for k in LEDGERS}
                score=estimate['targets'].get(reg['target'],{}).get('running_score',{});budget=proposal_budget(score,pending,right,duration)
                structural_ok=trial['stats']['continuous_nonnegative'] and trial['stats']['pool_error']<=1e-10 and all(trial['stats'].get(k,0.)<=1e-10 for k in LEDGERS)
                proposed=bool(estimate['status']=='success' and structural_ok and budget<=1)
                decision_seconds=time.perf_counter()-decision_start
                local_ref,local_budget,local_seconds=audit(case,y,q,h)
                global_times=np.unique(np.r_[0.,left+h*q]);indices=np.searchsorted(global_times,left+h*q);atick=time.perf_counter()
                global_ref=full(case,origin,global_times,1e-12)[:,indices];global_loose=full(case,origin,global_times,1e-11)[:,indices];global_seconds=time.perf_counter()-atick
                global_budget=metrics(case,global_loose,global_ref,1e-6);guard,targets=qualified(case,curve,values,local_ref,trial['stats'])
                actual=metrics(case,values,global_ref,1e-6);running_actual={k:max(max_actual[k],actual[k]) for k in actual};ref_ok=max([*local_budget.values(),*global_budget.values()])<=.05
                actual_ok=ref_ok and targets[reg['target']]['passes'] and max(running_actual.values())<=1
                false_accept=bool(proposed and not actual_ok);accept=bool(proposed and actual_ok)
                row.update(status='success',candidate_seconds=candidate_seconds,estimate_seconds=estimate['seconds'],decision_seconds=decision_seconds,
                    local_audit_seconds=local_seconds,global_audit_seconds=global_seconds,local_budget=local_budget,global_budget=global_budget,guard=guard,local_targets=targets,
                    actual=actual,running_actual=running_actual,stats=trial['stats'],estimate=dict(status=estimate['status'],targets=estimate['targets'],stats=estimate['stats']),
                    pending_ledger=pending,budget_ratio=budget,proposed=proposed,accepted=accept,false_accept=false_accept,reference_ok=ref_ok)
                arrays[f'{attempt}-initial']=y.copy();arrays[f'{attempt}-endpoint']=trial['fine'];arrays[f'{attempt}-forcing']=trial['forcing_coefficients']
                for k,v in pack(curve).items():arrays[f'{attempt}-{k}']=v
                for j,ec in enumerate(estimate['error_curves']):
                    for k,v in pack(ec).items():arrays[f'{attempt}-error-{j}-{k}']=v
                commit_start=time.perf_counter()
                if accept:
                    est.commit(estimate);y=trial['fine'].copy();left=right;ledger=pending;max_actual=running_actual;pieces.extend(curve.pieces);accepted.append(row);accepted_grid.append(row['left']+h*q)
                    if budget<.25:h=min(1.5*h,2*spec['h0'])
                else:
                    est.discard(estimate);assert state_signature(est,y,left,ledger)==before;h*=.5
                row['decision_seconds']+=time.perf_counter()-commit_start;row['after']=state_signature(est,y,left,ledger)
                if false_accept:reason='independent audit vetoed controller proposal'
                elif not ref_ok:reason='reference budget failed'
            except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                if estimate is not None:est.discard(estimate)
                assert state_signature(est,y,left,ledger)==before
                row.update(status='failed',reason=str(exc),accepted=False,seconds=time.perf_counter()-tick,after=before,stats=getattr(exc,'stats',{}));h*=.5
            rows.append(row);persist()
            if reason or left>=duration:break
            if h<spec['h0']*reg['min_step_ratio']:reason='minimum step reached';break
        complete=left>=duration
        if not complete and reason is None:reason='attempt budget exhausted'
        grid=np.unique(np.concatenate(accepted_grid));summary=dict(path=index,model=spec['model'],perturbed_component=perturbed,complete=complete,reason=reason,accepted_steps=len(accepted),duration=left,planned_duration=duration,setup_seconds=setup_seconds,ledger=ledger)
        if pieces:
            reference,budget,secs=audit(case,origin,grid/left,left);curve=InnerCurve(pieces);stats=dict(pool_error=max(r['stats']['pool_error'] for r in accepted),continuous_nonnegative=all(r['stats']['continuous_nonnegative'] for r in accepted))
            stats.update(physical_ledger(case,curve,origin,y));stats.update({k:ledger[k] for k in LEDGERS if k.startswith('modified')});guard,targets=qualified(case,curve,curve(grid/left),reference,stats)
            for v in targets.values():v['passes'] &= complete and max(budget.values())<=.05
            summary.update(stats=stats,guard=guard,targets=targets,final_reference_budget=budget,final_audit_seconds=secs)
        arrays.update(final=y,grid=grid);np.savez_compressed(OUT/f'path-{index}.npz',**arrays)
        subset=[r for r in rows if r['path']==index];summary['optimistic_seconds']=setup_seconds+sum(r.get('candidate_seconds',r.get('seconds',0.))+r.get('estimate_seconds',0.)+r.get('decision_seconds',0.) for r in subset)
        summary['audited_seconds']=summary['optimistic_seconds']+sum(r.get('local_audit_seconds',0.)+r.get('global_audit_seconds',0.) for r in subset)+summary.get('final_audit_seconds',0.)
        # Controls always cover the entire planned window, and no ratio is
        # assigned to incomplete candidates.
        control_grid=np.unique(np.r_[np.linspace(0,duration,513),grid]);ref,budget,_=audit(case,origin,control_grid/duration,duration);assert max(budget.values())<=.05
        np.savez_compressed(OUT/f'control-reference-{index}.npz',grid=control_grid,reference=ref)
        for tol in (1e-6,1e-8,1e-10):
            for whole in (False,True):
                key=f'control-{index}-{tol}-{whole}';tick=time.perf_counter();r=(fullstep_curve if whole else full_curve)(case,origin,duration,tol,screen);c=r['pieces'][0];v=c(control_grid/duration);structural(case,v);seconds=time.perf_counter()-tick
                guard,targets=qualified(case,c,v,ref,r['stats']);controls.append(dict(key=key,path=index,tolerance=tol,whole=whole,status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets))
                np.savez_compressed(OUT/f'{key}.npz',endpoint=r['fine'],**c.pack())
        paths.append(summary);kernel.close();persist();print(index,'complete',complete,'steps',len(accepted),'attempts',len(subset),'reason',reason,'targets',summary.get('targets'),'seconds',summary['optimistic_seconds'],flush=True)


if __name__=='__main__':run()
