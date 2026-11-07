"""Forced-path stress experiment; never a step-accepting controller."""
import argparse
import json
import time
import numpy as np

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.defect_fast import estimate_fast
from research.mri_kernel import MRI,output
from research.m3_priority import full,metrics,assess
from research.m3_shadow_study import ROOT,digest,compare

OUT=ROOT/'results/m3-shadow-stress'
SPECS={'teusink':dict(start=.02,steps=[1e-4,1e-3]),'kecoli':dict(start=.03,steps=[1e-4,1e-3])}
SOURCES=['research/m3_shadow_stress.py','research/defect_fast.py','research/defect_shadow.py',
         'research/mri_kernel.py','research/mri_native.py','research/m3_priority.py','research/m3_shadow_study.py',
         'metabolism/adaptive.py','metabolism/m3_continuation.py']

def registration():
    return dict(date='2026-11-07',specs=SPECS,steps_per_path=8,tolerance=1e-10,
        scenarios=['fixed','switch','perturbed'],perturbation='Positive first state multiplied by 1.01; references use the same perturbed start, no claim of original-pool preservation across perturbation.',
        switch='Recompute groups every step, alternating kappa thresholds .01 and .1 with multiplier 1e-3. State is continuous.',
        estimator_grid=17,independent_grid=65,public_observation_points=True,
        advance='Forced diagnostic candidate path even on error rejection; stop on negative curve or solver failure. Never a production accepted path.',
        false_accept_stop='Any numerical or structural-policy false acceptance prohibits controller promotion; retain all other planned diagnostic paths.',
        cumulative='Global full reference from the same path initial state. Local scores are NOT global acceptance guarantees.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES})

def register():
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/'registration.json'
    if path.exists():assert json.loads(path.read_text())==registration()
    else:save(path,registration())

def run():
    reg=json.loads((OUT/'registration.json').read_text());assert reg==registration()
    rows=[];arrays={}
    for name,spec in SPECS.items():
        case=prepare(name);screen=GroupScreen(case['system'],case['scale'])
        initial=full(case,case['model'].y0,np.array([0.,spec['start']]),1e-12)[:,-1]
        for scenario in reg['scenarios']:
            for hi,h in enumerate(spec['steps']):
                path=f'{name}-{scenario}-{hi}';y=initial.copy()
                if scenario=='perturbed':y[np.flatnonzero(y>0)[0]]*=1.01
                origin=y.copy();arrays[path+'-initial']=origin
                public=case['times'][(case['times']>spec['start'])&(case['times']<spec['start']+8*h)]-spec['start']
                global_times=np.unique(np.r_[np.arange(8*64+1)*h/64,public])
                global_ref=full(case,origin,global_times,1e-12)
                global_loose=full(case,origin,global_times,1e-11)
                budget=metrics(case,global_loose,global_ref,1e-6)
                arrays[path+'-global-times']=global_times;arrays[path+'-global-ref']=global_ref
                arrays[path+'-global-loose']=global_loose
                kappa,_=screen.evaluate(y);fixed=np.flatnonzero(1e-3*kappa>=.01)
                previous=None;all_local_policy={t:True for t in ('0.0001','1e-06')}
                for step in range(8):
                    key=f'{path}-{step}';start=step*h
                    observed=public[(public>start)&(public<start+h)]
                    # Use global query coordinates, avoiding duplicate floating-point endpoints.
                    positions=np.flatnonzero((global_times>=start-1e-14*h)&(global_times<=start+h+1e-14*h))
                    dense_query=np.clip((global_times[positions]-start)/h,0,1)
                    query=np.unique(np.r_[np.linspace(0,1,17),(observed-start)/h])
                    groups=fixed
                    if scenario=='switch':
                        kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=(.01 if step%2==0 else .1))
                    row=dict(key=key,model=name,scenario=scenario,h=h,step=step,groups=groups.tolist(),
                        changed_groups=0 if previous is None else len(np.setxor1d(previous,groups)),global_reference_budget=budget)
                    previous=groups.copy();solver=MRI(case,groups,backend='radau' if name=='teusink' else 'native')
                    try:
                        trial=solver.trial(y,h,reg['tolerance'],timeout=20.)
                        dense=output(trial,dense_query);values=output(trial,query)
                        audit_times=np.unique(np.r_[0.,h*dense_query])
                        refall=full(case,y,audit_times,1e-12);looseall=full(case,y,audit_times,1e-11)
                        indices=np.searchsorted(audit_times,h*dense_query)
                        ref,loose=refall[:,indices],looseall[:,indices]
                        truth,_=assess(case,trial,dense_query,ref)
                        shadow,evidence=estimate_fast(case,trial['pieces'],values,query)
                        comparison=compare(truth,shadow)
                        local_budget=metrics(case,loose,ref,1e-6)
                        assert max(budget.values())<=.05 and max(local_budget.values())<=.05
                        global_errors={t:metrics(case,dense,global_ref[:,positions],float(t)) for t in all_local_policy}
                        for t in all_local_policy:
                            all_local_policy[t] &= comparison.get(t,{}).get('policy_would_accept',False)
                        row.update(status='success',truth=truth,shadow=shadow,comparison=comparison,
                            local_reference_budget=local_budget,global_errors=global_errors,
                            all_previous_policy_accept=all_local_policy.copy(),
                            accumulated_exceedance={t:all_local_policy[t] and max(global_errors[t].values())>1 for t in all_local_policy})
                        for label,array in dict(nodes=np.column_stack([y,trial['mid'],trial['fine']]),values=values,
                            dense=dense,query=query,dense_query=dense_query,reference=ref,loose=loose,global_positions=positions,**evidence).items():
                            arrays[key+'-'+label]=array
                        if trial['stats']['continuous_nonnegative']:y=trial['fine'].copy()
                        else:row['stop']='negative_curve'
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',reason=str(exc),stop='solver_or_reference_failure')
                    finally:solver.close()
                    rows.append(row)
                    save(OUT/'runs.json',dict(scope=reg['advance'],runs=rows,source_sha256=reg['source_sha256']))
                    np.savez_compressed(OUT/'curves.npz',**arrays)
                    print(key,row['status'],{t:c['false_accept'] for t,c in row.get('comparison',{}).items()},row.get('accumulated_exceedance'),flush=True)
                    if 'stop' in row:break
    result=dict(paths=12,steps=len(rows),failed=sum(r['status']=='failed' for r in rows),
        numerical_false_accepts=sum(c['false_accept'] for r in rows for c in r.get('comparison',{}).values()),
        policy_false_accepts=sum(c['policy_false_accept'] for r in rows for c in r.get('comparison',{}).values()),
        accumulated_exceedances=sum(v for r in rows for v in r.get('accumulated_exceedance',{}).values()),
        switched_group_changes=sum(r['changed_groups'] for r in rows),M4_admission=False)
    save(OUT/'summary.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args()
    {'register':register,'run':run}[a.stage]()
