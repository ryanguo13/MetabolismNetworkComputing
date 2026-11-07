"""Small-model nonlinear cost/accuracy falsification of endpoint iteration."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.implicit_flow import EndpointImplicitFlow
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-implicit-history'


def run():
    sources=['research/implicit_flow.py','research/test_implicit_flow.py','research/implicit_history.py',
             'research/m3_implicit_local.py','research/fullstep_radau.py','research/inner_dense.py',
             'research/m3_extrapolated_paths.py','research/m3_growth_study.py']
    reg=dict(date='2026-11-08',model='teusink',origin_time=1.,steps=[.001,.003,.01],thresholds=[.3,3.,10.],
        orders=[2,3,4],iterations=['fixed','newton'],tolerances=[1e-8,1e-10],maxiter=12,residual_tolerance=1e-11,
        scope='108 local candidates and18 complete controls; one-shot development timings, no M4 coverage or speed claim.',
        history='One real guarded full solve over2h at1e-10 from known t=1 prefix; sample actual curve at0,h,2h. All orders use same current state. History generation excluded from local timings and recorded separately; no reference history injection.',
        iteration='Euler endpoint predictor (fallback to current state if invalid); fixed point or frozen slow-only Jacobian I-beta*h*J_s, positivity damping, convergence required. All repeated fast solves counted.',
        audits='Independent full1e-12/1e-11 on65 points plus public observations; strict reference budget<=.05. Original positivity/pools/physical and modified ledger/net flux gates.',
        cost='Includes current and future slow RHS, endpoint factorization/solves, all fast solves and continuous guards; excludes cached grouping/history, independent audits and accuracy scoring. Final ledger audited on returned curve, not rejected iterates.',
        failures='Preserve all failures and iteration residuals. Work counters on interrupted inner solves may be partial; their measured time is retained and failures cannot qualify.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={f'results/{p}':digest(ROOT/'results'/p) for p in
            ('m3-priority/p0.npz','m3-implicit-history/stability.json','m3-growth-history/manifest.json')})
    if (OUT/'local-registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'local-registration.json',reg)
    case=prepare('teusink');screen=GroupScreen(case['system'],case['scale'])
    origin=np.load(ROOT/'results/m3-priority/p0.npz')['teusink-prefix'][:,3];rows=[];baselines=[];histories=[]
    for h in reg['steps']:
        tick=time.perf_counter();boot=fullstep_curve(case,origin,2*h,1e-10,screen);history=boot['pieces'][0](np.array([0.,.5,1.]));history[:,-1]=boot['fine']
        history_seconds=time.perf_counter()-tick;y=history[:,-1];start=1+2*h
        observed=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h;query=np.unique(np.r_[np.linspace(0,1,65),observed])
        ref,budget,audit_seconds=audit(case,y,query,h);assert max(budget.values())<=.05
        arrays=dict(origin=origin,history=history,initial=y,query=query,reference=ref)
        for k,v in boot['pieces'][0].pack().items():arrays['bootstrap-'+k]=v
        histories.append(dict(h=h,history_seconds=history_seconds,stats=boot['stats'],budget=budget,audit_pair_seconds=audit_seconds))
        for tol in (1e-6,1e-8,1e-10):
            for whole in (False,True):
                key=f'full-{tol}-{whole}';row=dict(h=h,key=key,tolerance=tol,whole=whole);tick=time.perf_counter()
                try:
                    trial=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen);curve=trial['pieces'][0]
                    values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,values,ref,trial['stats'])
                    row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                    arrays[key+'-endpoint']=trial['fine']
                    for k,v in curve.pack().items():arrays[key+'-'+k]=v
                except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                baselines.append(row)
        np.savez_compressed(OUT/f'reference-{h}.npz',**arrays)
        kappa,_=screen.evaluate(y)
        for threshold in reg['thresholds']:
            groups=np.flatnonzero(1e-3*kappa>=threshold);kernel=EndpointImplicitFlow(case,groups)
            # History RHS evaluation is preparation, recorded outside local timings.
            tick=time.perf_counter();old_slow=np.column_stack([kernel.slow(z) for z in history[:,:-1].T]);prep=time.perf_counter()-tick
            for order in reg['orders']:
                times=np.arange(-(order-2),1)*h;past=old_slow[:,3-order:] if order>2 else np.empty((len(y),0))
                for method in reg['iterations']:
                    for tol in reg['tolerances']:
                        name=f'{h}-{threshold}-{order}-{method}-{tol}';tick=time.perf_counter()
                        row=dict(name=name,h=h,threshold=threshold,order=order,iteration=method,tolerance=tol,
                                 active=len(kernel.active),groups=groups.tolist(),setup_seconds=kernel.setup_seconds,history_rhs_seconds=prep)
                        arrays=dict(initial=y,history_times=times,past_slow=past,groups=groups)
                        try:
                            trial=kernel.trial(y,h,times,past,tol,method,residual_tol=reg['residual_tolerance'],maxiter=reg['maxiter'])
                            curve=trial['pieces'][0];values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick
                            guard,targets=qualified(case,curve,values,ref,trial['stats'])
                            row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                            arrays.update(endpoint=trial['fine'],forcing=trial['forcing_coefficients'],implicit_guess=trial['implicit_guess'],**pack(curve))
                        except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                            row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                        rows.append(row);np.savez_compressed(OUT/f'{name}.npz',**arrays)
                        print(name,row['status'],row['stats'].get('subflows'),[t for t,v in row.get('targets',{}).items() if v['passes']],flush=True)
            kernel.close()
        save(OUT/'local.json',dict(runs=rows,baselines=baselines,histories=histories))


if __name__=='__main__':run()
