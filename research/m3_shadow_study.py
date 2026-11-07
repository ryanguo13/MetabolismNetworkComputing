"""Registered shadow study: estimates have no authority to accept a step."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_priority import full,metrics,assess
from research.mri_kernel import MRI,Hermite
from research.defect_shadow import estimate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/m3-shadow'
DEV=ROOT/'results/m3-priority'
NEW_STATES={'teusink':[3e-6,.02,.3],'kecoli':[3e-8,.003,.03]}
NEW_STEPS={'teusink':[1e-7,1e-4,.01],'kecoli':[1e-8,1e-4,.01]}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths=[Path(__file__),ROOT/'research/defect_shadow.py',ROOT/'research/test_defect_shadow.py',
           ROOT/'research/mri_kernel.py',ROOT/'research/mri_native.py',ROOT/'research/m3_priority.py',
           ROOT/'metabolism/adaptive.py',ROOT/'metabolism/m3_continuation.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    registration=dict(date='2026-11-07',stage='shadow_only',
        rationale='The independent full ODE audit alone exceeds the local qualified baseline in saved large-model timings. Test estimator reliability before any reduction of audits.',
        development='All 68 saved local cases in m3-priority; no estimator tuning permitted after seeing the new-state results.',
        new_states=NEW_STATES,new_steps=NEW_STEPS,tolerances=[1e-8,1e-10],
        estimator='Frozen full Jacobian at each Hermite piece midpoint; L-stable SDIRK2 with 8 and 16 microsteps/piece; score=4*(estimated observable error+refinement difference).',
        scope='New states of the same known models, not independent models, parameters or trajectories.',
        local_targets=[1e-4,1e-6],net_local_target=1e-4,
        promotion_gate=dict(no_false_accepts=True,median_estimator_over_audit_max=.25,
            true_accept_recall_min=.5,guard_policy='Existing positivity, pools and physical/modified ledger checks remain compulsory'),
        stopping='A new-state false acceptance blocks controller promotion. Save all failures. No replacement of independent audits in this study.',
        source_sha256=source_hashes(),
        development_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in DEV.glob('*') if p.suffix in ('.json','.npz')})
    path=OUT/'registration.json'
    if path.exists():
        assert json.loads(path.read_text())==registration,'Frozen registration differs; use a new study directory'
    else:save(path,registration)


def verify_registration():
    reg=json.loads((OUT/'registration.json').read_text())
    assert reg['source_sha256']==source_hashes(),'Source changed since registration'
    for path,sha in reg['development_sha256'].items():assert digest(ROOT/path)==sha
    return reg


def build_pieces(case,nodes,h):
    derivative=[case['system'].S@case['system'].flux(0.,z) for z in nodes.T]
    return [Hermite(nodes[:,i],nodes[:,i+1],derivative[i],derivative[i+1],h/2) for i in (0,1)]


def compare(row,shadow):
    result={}
    for target,actual in row.get('targets',{}).items():
        predicted=shadow.get('targets',{}).get(target,{})
        accept=predicted.get('would_accept',False)
        structural=(max(row['stats']['pool_error'],row.get('output_pool_error',0.))<=1e-10
                    and row['stats']['continuous_nonnegative']
                    and all(row['stats'].get(k,0.)<=1e-10 for k in
                        ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature')))
        result[target]=dict(actual=actual['actual'],actual_numerical_pass=actual['numerical_pass'],
            actual_all_checks_pass=actual['passes'],structural_pass=structural,
            would_accept=accept,policy_would_accept=accept and structural,
            false_accept=accept and not actual['numerical_pass'],
            policy_false_accept=accept and structural and not actual['passes'],
            false_reject=not accept and actual['passes'],
            effectivity=(max(predicted['score'].values())/max(max(actual['actual'].values()),1e-300)) if predicted else None)
    return result


def dev():
    verify_registration()
    cases={n:prepare(n) for n in NEW_STATES}
    screens={n:GroupScreen(c['system'],c['scale']) for n,c in cases.items()}
    records,arrays=[],{}
    for dataset in ('p0','startup','startup-early','growth'):
        data=json.loads((DEV/f'{dataset}.json').read_text())
        saved=np.load(DEV/f'{dataset}.npz')
        for index,row in enumerate(data['runs']):
            name=row['model']
            case=cases[name]
            key=f'{dataset}-{index}'
            if dataset=='p0':
                si=list((0.,1e-6,.001,1.) if name=='teusink' else (0.,1e-8,.001,.1)).index(row['start'])
                hi=list((1e-6,1e-4,.01) if name=='teusink' else (1e-8,1e-5,.001)).index(row['h'])
                refkey=f'{name}-{si}-{hi}'
            else:refkey=row['key']
            if row['status']=='failed':
                records.append(dict(key=key,model=name,status='upstream_failed',source_dataset=dataset,source_key=row['key']))
                continue
            pieces=build_pieces(case,saved[row['key']+'-nodes'],row['h'])
            shadow,evidence=estimate(case,pieces,saved[row['key']],saved[refkey+'-fractions'],screens[name])
            record=dict(key=key,model=name,start=row['start'],h=row['h'],tolerance=row['tolerance'],
                source_dataset=dataset,source_key=row['key'],shadow=shadow,comparison=compare(row,shadow))
            records.append(record)
            for label,value in evidence.items():arrays[key+'-'+label]=value
            save(OUT/'development.json',dict(runs=records,source_sha256=source_hashes()))
            np.savez_compressed(OUT/'development.npz',**arrays)
            print('dev',name,row['start'],row['h'],row['tolerance'],shadow['status'],
                  [t for t,v in record['comparison'].items() if v['false_accept']],flush=True)


def new_states():
    verify_registration()
    rows,arrays=[],{}
    for name in NEW_STATES:
        case=prepare(name)
        prefix=full(case,case['model'].y0,np.r_[0.,NEW_STATES[name]],1e-12)
        arrays[name+'-prefix']=prefix
        screen=GroupScreen(case['system'],case['scale'])
        for si,start in enumerate(NEW_STATES[name],1):
            y=prefix[:,si]
            kappa,_=screen.evaluate(y)
            groups=np.flatnonzero(1e-3*kappa>=.01)
            solver=MRI(case,groups,backend='radau' if name=='teusink' else 'native')
            try:
                for hi,h in enumerate(NEW_STEPS[name]):
                    observed=case['times'][(case['times']>start)&(case['times']<start+h)]
                    fractions=np.unique(np.r_[np.linspace(0,1,17),(observed-start)/h])
                    stamp=time.perf_counter()
                    ref=full(case,y,h*fractions,1e-12)
                    audit_seconds=time.perf_counter()-stamp
                    loose=full(case,y,h*fractions,1e-11)
                    budget=metrics(case,loose,ref,1e-6)
                    refkey=f'{name}-{si}-{hi}'
                    arrays[refkey+'-reference']=ref
                    arrays[refkey+'-loose']=loose
                    arrays[refkey+'-fractions']=fractions
                    for tol in (1e-8,1e-10):
                        key=refkey+f'-{tol}'
                        row=dict(key=key,model=name,start=start,h=h,tolerance=tol,groups=groups.tolist(),
                                 reference_tightening=budget,reference_pass=max(budget.values())<=.05,audit_seconds=audit_seconds)
                        try:
                            trial=solver.trial(y,h,tol,timeout=15.)
                            truth,values=assess(case,trial,fractions,ref)
                            shadow,evidence=estimate(case,trial['pieces'],values,fractions,screen)
                            row.update(status=truth['status'],truth=truth,shadow=shadow,comparison=compare(truth,shadow))
                            arrays[key]=values
                            arrays[key+'-nodes']=np.column_stack([y,trial['mid'],trial['fine']])
                            for label,value in evidence.items():arrays[key+'-'+label]=value
                        except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                            row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                        rows.append(row)
                        save(OUT/'new-states.json',dict(runs=rows,source_sha256=source_hashes(),scope='Registered new states; all decisions remain shadow-only.'))
                        np.savez_compressed(OUT/'new-states.npz',**arrays)
                        print('new',name,start,h,tol,row['status'],row.get('shadow',{}).get('status'),
                              [t for t,v in row.get('comparison',{}).items() if v['false_accept']],flush=True)
            finally:solver.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['register','dev','new'])
    stage=parser.parse_args().stage
    {'register':register,'dev':dev,'new':new_states}[stage]()
