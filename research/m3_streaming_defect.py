"""Preregistered streaming cumulative error diagnostic on frozen own paths."""
import json
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from metabolism.adaptive import GroupScreen
from research.streaming_defect import StreamingDefect
from research.linear_context_row import LinearContextROW
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-streaming-defect';OLD=ROOT/'results/m3-cached-row'


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/'+p for p in ('streaming_defect.py','linear_context_row.py','test_streaming_defect.py','m3_streaming_defect.py','defect_fast.py','row_continuous.py','m3_priority.py','m3_growth_study.py','m3_extrapolated_paths.py')]
    reg=dict(date='2026-11-08',paths=[0,1,2,3],grids=[[1,2],[2,4],[4,8]],inner_tolerance=1e-6,
        formula='e prime=J_n e+f(P)-P prime; freeze fresh predictor J at macro start. Third-order3-stage SDIRK with gamma=ROS34PW2 diagonal. Respect native curve boundaries. Only exactly matching H/J reuses the actual live predictor factor; all new factors counted.',
        score='Same factor4*(central+refinement) as prior shadow; independently propagate both error endpoints, Hermite error output per estimator substep. Running maximum over all observed points, no macro resets.',
        guards='Check candidate continuous positivity, corrected query-state positivity and all original ledgers separately. Accumulate absolute local ledger defects using initial physical denominator, retaining quadrature separately.',
        paths_policy='Replay previously accepted original16-step paths or11-step prefix exactly. Continue forced error-state propagation even when score rejects; never claim those steps as prospective accepted coverage. No reference state/history injected.',
        audits='Independent full ODE1e-12/1e-11 globally and locally.65 points per macro plus original observations. Reference budget<=.05. Compare running estimator scores to actual accumulated error; no threshold tuning after results.',
        timing='Actual predictor+fast kernel, estimator costs and local audit pair measured separately. Setup/grouping separate; controls uninterrupted full Radau tol1e-6/1e-8/1e-10 automatic/whole. Single development run only.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ['results/m3-cached-row/manifest.json','results/m3-cached-row/runs.json','results/m3-cached-row/registration.json']+[f'results/m3-cached-row/{i}-original.npz' for i in range(4)]})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);old= json.loads((OLD/'runs.json').read_text());specs=json.loads((OLD/'registration.json').read_text())['paths'];rows=[];paths=[];controls=[];refs=[]
    def persist():save(OUT/'runs.json',dict(steps=rows,paths=paths,controls=controls,references=refs))
    for index in reg['paths']:
        spec=specs[index];case=prepare(spec['model']);h=spec['h'];summary=next(r for r in old['paths'] if r['key']==f'{index}-original');count=summary['accepted_steps'];duration=count*h
        with np.load(OLD/f'path-{index}.npz') as a:origin=a['origin'];groups=a['groups'];grid=a['grid'];grid=grid[grid<=duration+1e-14]
        query=np.clip(grid/duration,0,1);reference,budget,ref_seconds=audit(case,origin,query,duration);assert max(budget.values())<=.05
        refs.append(dict(path=index,model=spec['model'],steps=count,duration=duration,budget=budget,seconds=ref_seconds,complete_original_path=summary['complete']))
        np.savez_compressed(OUT/f'reference-{index}.npz',initial=origin,groups=groups,grid=grid,reference=reference)
        screen=GroupScreen(case['system'],case['scale'])
        for tol in (1e-6,1e-8,1e-10):
            for whole in (False,True):
                key=f'control-{index}-{tol}-{whole}';row=dict(key=key,path=index,tolerance=tol,whole=whole);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(case,origin,duration,tol,screen);curve=r['pieces'][0];v=curve(query);structural(case,v);elapsed=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,v,reference,r['stats']);row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets)
                    np.savez_compressed(OUT/f'{key}.npz',endpoint=r['fine'],**curve.pack())
                except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                controls.append(row);persist()
        setup=time.perf_counter();kernel=LinearContextROW(case,groups);estimators=[StreamingDefect(case,origin,g) for g in reg['grids']];setup_seconds=time.perf_counter()-setup
        path_arrays={};candidate_seconds=0.;audit_seconds=0.;est_seconds=[0.]*len(estimators);ledger={k:0. for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature')}
        weights=case['options'].get('ledger_weights');initial_den=1. if weights is None else max(1.,abs(float(weights@origin)));y=origin.copy();seen_values=[];seen_refs=[]
        with np.load(OLD/f'{index}-original.npz') as src:
            for step in range(count):
                mask=np.flatnonzero((grid>=step*h-1e-14)&(grid<=(step+1)*h+1e-14));q=np.clip((grid[mask]-step*h)/h,0,1)
                tick=time.perf_counter();trial=kernel.trial(y,h,reg['inner_tolerance']);values=trial['pieces'][0](q);structural(case,values);elapsed=time.perf_counter()-tick;candidate_seconds+=elapsed
                np.testing.assert_array_equal(trial['fine'],src[f'{step}-endpoint']);np.testing.assert_array_equal(y,src[f'{step}-initial'])
                local_ref,local_budget,local_seconds=audit(case,y,q,h);audit_seconds+=local_seconds;assert max(local_budget.values())<=.05
                local_guard,local_targets=qualified(case,trial['pieces'][0],values,local_ref,trial['stats']);assert local_targets['1e-06']['passes']
                seen_values.append(values);seen_refs.append(reference[:,mask]);accumulated={str(t):metrics(case,np.column_stack(seen_values),np.column_stack(seen_refs),t) for t in (1e-4,1e-6)}
                local_den=1. if weights is None else max(1.,abs(float(weights@y)))
                for k in ledger:ledger[k]+=trial['stats'].get(k,0.)*local_den/initial_den
                row=dict(path=index,step=step,query=q.tolist(),candidate_seconds=elapsed,candidate_stats=trial['stats'],local_budget=local_budget,local_audit_seconds=local_seconds,local_guard=local_guard,local_targets=local_targets,
                    actual_accumulated=accumulated,absolute_ledger_budgets=ledger.copy(),estimates=[])
                for gi,estimator in enumerate(estimators):
                    result=estimator.trial(trial,q);est_seconds[gi]+=result['seconds'];comparison={}
                    for target,actual in accumulated.items():
                        predicted=result['status']=='success' and result['targets'][target]['would_accept'];actual_ok=max(actual.values())<=1
                        comparison[target]=dict(actual_error_pass=actual_ok,would_accept=predicted,false_accept=bool(predicted and not actual_ok),false_reject=bool(not predicted and actual_ok),
                            combined_would_accept=bool(predicted and max(ledger.values())<=1e-10))
                    row['estimates'].append(dict(grids=list(estimator.grids),status=result['status'],targets=result['targets'],stats=result['stats'],seconds=result['seconds'],comparison=comparison))
                    for j,(endpoint,curve) in enumerate(zip(result['errors'],result['error_curves'])):
                        path_arrays[f'{step}-{gi}-{j}-endpoint']=endpoint
                        for k,v in pack(curve).items():path_arrays[f'{step}-{gi}-{j}-{k}']=v
                    if result['status']!='success':raise RuntimeError('Registered diagnostic error-state guard failure')
                    estimator.commit(result)
                rows.append(row);persist();y=trial['fine'].copy()
        kernel.close();np.savez_compressed(OUT/f'errors-{index}.npz',**path_arrays)
        paths.append(dict(path=index,steps=count,setup_seconds=setup_seconds,candidate_seconds=candidate_seconds,local_audit_seconds=audit_seconds,estimator_seconds=est_seconds,
            actual_accumulated=rows[-1]['actual_accumulated'],absolute_ledger_budgets=ledger,final_estimates=rows[-1]['estimates'],complete_original_path=summary['complete']))
        persist();print(index,'steps',count,'candidate',candidate_seconds,'estimators',est_seconds,'strict',[(e['grids'],e['comparison']['1e-06']) for e in rows[-1]['estimates']],flush=True)


if __name__=='__main__':run()
