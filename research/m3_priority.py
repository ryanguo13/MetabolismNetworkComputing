"""Execute registered P0 correctness and P1 cost gates before P2/P3."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from metabolism.adaptive import GroupScreen,AdaptiveConfig,AdaptiveSplit
from metabolism.cvode import solve
from metabolism.m2_experiments import error,save
from metabolism.m3_continuation import prepare,registration
from research.mri_kernel import MRI,output

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/m3-priority'
GRID=np.linspace(0,1,17)
STATES={'teusink':(0.,1e-6,.001,1.),'kecoli':(0.,1e-8,.001,.1)}
STEPS={'teusink':(1e-6,1e-4,.01),'kecoli':(1e-8,1e-5,1e-3)}


def hashes():
    paths=[Path(__file__),ROOT/'research/mri_kernel.py',ROOT/'research/mri_native.py',ROOT/'research/test_mri_kernel.py']
    paths += [ROOT/'metabolism'/p for p in ('__init__.py','adaptive.py','splitting.py','cvode.py','sbml.py','kecoli.py','m3_continuation.py','m2_experiments.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def full(case,y,times,tol):
    system,scale=case['system'],case['scale']
    if system.flux_jacobian is not None:
        model=SimpleNamespace(y0=y,rhs=lambda t,z:system.S@system.flux(t,z),
            jacobian=lambda t,z:system.S@system.flux_jacobian(t,z),
            sparsity=(abs(system.S)@system.dependencies).astype(bool).tocsc())
        return solve(model,times,tol,.01*tol*scale)[0]
    result=solve_ivp(lambda t,z:system.S@system.flux(t,z),(0.,times[-1]),y,
                     t_eval=times,rtol=tol,atol=.01*tol*scale,method='Radau')
    if not result.success:raise RuntimeError(result.message)
    return result.y


def metrics(case,values,ref,target):
    flux=lambda v:np.column_stack([case['system'].flux(0.,z) for z in v.T])
    vf,vr=flux(values),flux(ref)
    result=dict(concentration=error(values,ref,case['scale'],target),
                flux=error(vf,vr,case['options']['flux_scale'],target))
    if 'net_map' in case['options']:
        result['net_flux']=error(case['options']['net_map']@vf,case['options']['net_map']@vr,case['options']['net_scale'],1e-4)
    return result


def assess(case,trial,fractions,ref):
    stats=trial['stats']
    values=output(trial,fractions)
    coarse=trial['coarse_piece'](fractions)
    result=dict(status='success',stats=stats)
    if not stats['continuous_nonnegative']:
        result.update(status='rejected',reason='Hermite internal negative extremum')
        return result,values
    pool=case['options']['pools']
    denom=max(1e-12,float(np.max(abs(pool@values[:,0]))))
    result['output_pool_error']=float(np.max(abs(pool@(values-values[:,0:1]))))/denom
    result['targets']={}
    for target in (1e-4,1e-6):
        actual=metrics(case,values,ref,target)
        estimate=None
        if np.isfinite(coarse).all() and np.all(coarse >= -1e-12*case['scale'][:,None]):
            estimate={k:v/3 for k,v in metrics(case,values,coarse,target).items()}
        structural=max(stats['pool_error'],result['output_pool_error']) <= 1e-10
        ledger=all(stats.get(k,0.) <= 1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature',
                                                     'physical_ledger_residual','physical_ledger_quadrature'))
        result['targets'][str(target)]=dict(actual=actual,estimate=estimate,
            numerical_pass=max(actual.values()) <= 1,
            passes=max(actual.values()) <= 1 and structural and ledger,
            estimator_false_accept=estimate is not None and max(estimate.values()) <= 1 and max(actual.values()) > 1)
    return result,values


def p0():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,refs,registrations,partitions,arrays=[],[],{},[],{}
    # Frozen before executing any candidates; budgets/failures are retained.
    save(OUT/'registration.json',dict(states=STATES,steps=STEPS,tolerances=[1e-8,1e-10],
        local_targets=[1e-4,1e-6],net_local_target=1e-4,reference_budget=.05,
        audit='17 uniform points plus all original observation points in interval',
        groups='frozen per starting state: 1e-3*kappa>=.01',timeout_per_attempt=15,
        P1_stop='No local throughput prospect against same-state full solver; no formal P3 repeats without measured prospect',source_sha256=hashes()))
    for name in ('teusink','kecoli'):
        case=prepare(name)
        registrations[name]=registration(case)
        prefix=full(case,case['model'].y0,np.array(STATES[name]),1e-12)
        arrays[name+'-prefix']=prefix
        screen=GroupScreen(case['system'],case['scale'])
        for si,start in enumerate(STATES[name]):
            y=prefix[:,si]
            stamp=time.perf_counter()
            kappa,_=screen.evaluate(y)
            groups=np.flatnonzero(1e-3*kappa >= .01)
            classify=time.perf_counter()-stamp
            solver=MRI(case,groups,backend='radau' if name=='teusink' else 'native')
            partitions.append(dict(model=name,start=start,groups=groups.tolist(),active=len(solver.active),
                                   classification_seconds=classify,setup_seconds=solver.setup_seconds))
            try:
                for hi,h in enumerate(STEPS[name]):
                    observed=case['times'][(case['times']>start)&(case['times']<start+h)]
                    fractions=np.unique(np.r_[GRID,(observed-start)/h])
                    stamp=time.perf_counter()
                    ref=full(case,y,h*fractions,1e-12)
                    reference_seconds=time.perf_counter()-stamp
                    loose=full(case,y,h*fractions,1e-11)
                    budget=metrics(case,loose,ref,1e-6)
                    refkey=f'{name}-{si}-{hi}'
                    arrays[refkey+'-reference']=ref
                    arrays[refkey+'-loose']=loose
                    arrays[refkey+'-fractions']=fractions
                    refs.append(dict(key=refkey,tightening=budget,seconds=reference_seconds))
                    for tol in (1e-8,1e-10):
                        key=refkey+f'-{tol}'
                        row=dict(key=key,model=name,start=start,h=h,tolerance=tol,
                                 reference_pass=max(budget.values()) <= .05)
                        try:
                            trial=solver.trial(y,h,tol,timeout=15.)
                            assessed,values=assess(case,trial,fractions,ref)
                            row.update(assessed)
                            arrays[key]=values
                            arrays[key+'-coarse']=trial['coarse_piece'](fractions)
                            arrays[key+'-nodes']=np.column_stack([y,trial['mid'],trial['fine']])
                        except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                            row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                        rows.append(row)
                        save(OUT/'p0.json',dict(registration=registrations,partitions=partitions,references=refs,runs=rows,source_sha256=hashes()))
                        np.savez_compressed(OUT/'p0.npz',**arrays)
                        print(name,start,h,tol,row['status'],row.get('targets',{}).get('0.0001',{}).get('passes'),flush=True)
            finally:solver.close()


def p1(cached=False,growth=False):
    data=json.loads((OUT/'p0.json').read_text())
    arrays=np.load(OUT/'p0.npz')
    records,selections=[],[]
    for name in ('teusink','kecoli'):
        case=prepare(name)
        cached_screen=GroupScreen(case['system'],case['scale']) if cached else None
        # Later state, largest qualified h at loose target; all selection inputs saved.
        candidates=[r for r in data['runs'] if r['model']==name and r['start']==STATES[name][-1]
                    and r['reference_pass'] and r.get('targets',{}).get('0.0001',{}).get('passes')]
        if growth:
            larger=json.loads((OUT/'growth.json').read_text())['runs']
            larger=[r for r in larger if r['model']==name and r['reference_pass']
                    and r.get('targets',{}).get('0.0001',{}).get('passes')]
            if not larger or max(r['h'] for r in larger)<=max((r['h'] for r in candidates),default=0):
                continue
            candidates=larger
        if not candidates:
            selections.append(dict(model=name,status='no_P0_candidate'))
            continue
        chosen=max(candidates,key=lambda r:(r['h'],r['tolerance']))
        selections.append(dict(model=name,key=chosen['key'],h=chosen['h'],tolerance=chosen['tolerance']))
        y=arrays[name+'-prefix'][:,-1]
        part=next(p for p in data['partitions'] if p['model']==name and p['start']==chosen['start'])
        if growth:
            extra=np.load(OUT/'growth.npz')
            fractions=extra[chosen['key']+'-fractions']
            ref=extra[chosen['key']+'-reference']
        else:
            refkey=f"{name}-{STATES[name].index(chosen['start'])}-{STEPS[name].index(chosen['h'])}"
            fractions=arrays[refkey+'-fractions']
            ref=arrays[refkey+'-reference']
        variants=([('radau-full','radau',False,False),('radau-reduced','radau',True,False)]
                  if name=='teusink' else [('native-full-fresh','native',False,False),
                      ('native-reduced-fresh','native',True,False),('native-reduced-reused','native',True,True),
                      ('radau-reduced','radau',True,False)])
        solvers={label:MRI(case,part['groups'],backend,reduced,reuse) for label,backend,reduced,reuse in variants}
        full_controller=AdaptiveSplit(case['system'],case['scale'],
            AdaptiveConfig(target=1e-3,fallback_rtol=1e-10,
                           fallback_method='Radau' if name=='teusink' else 'CVODE-KLU'),
            fixed_groups=np.arange(len(case['system'].groups)),**case['options']) if cached else None
        try:
            for repeat in range(-1,3):
                for label in np.roll([v[0] for v in variants],repeat+1):
                    solver=solvers[label]
                    row=dict(model=name,variant=str(label),repeat=repeat,h=chosen['h'],tolerance=chosen['tolerance'],setup_seconds=solver.setup_seconds)
                    tick=time.perf_counter()
                    try:
                        # Count classification, output and independent audit on every attempt.
                        (cached_screen or GroupScreen(case['system'],case['scale'])).evaluate(y)
                        trial=solver.trial(y,chosen['h'],chosen['tolerance'],timeout=15.)
                        stamp=time.perf_counter()
                        audit=full(case,y,chosen['h']*fractions,1e-12)
                        row['audit_seconds']=time.perf_counter()-stamp
                        assessed,_=assess(case,trial,fractions,audit)
                        row.update(assessed)
                        row['seconds']=time.perf_counter()-tick
                    except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                        row.update(status='failed',reason=str(exc),seconds=time.perf_counter()-tick,stats=getattr(exc,'stats',{}))
                    records.append(row)
                    print('P1',name,label,repeat,row['status'],round(row['seconds'],4),flush=True)
                # Complete-system control with same queries and external accuracy checks.
                tick=time.perf_counter()
                if cached:
                    values,control_stats=full_controller.integrate(chosen['h']*fractions,y0=y)
                else:
                    values=full(case,y,chosen['h']*fractions,1e-10)
                    control_stats={}
                elapsed=time.perf_counter()-tick
                actual=metrics(case,values,ref,1e-4)
                records.append(dict(model=name,variant='full-local',repeat=repeat,seconds=elapsed,
                                    actual=actual,passes=max(actual.values())<=1,status='success',stats=control_stats))
                save(OUT/('p1-growth.json' if growth else 'p1-cached.json' if cached else 'p1.json'),dict(scope='One warmup and 3 interleaved DEVELOPMENT repeats, not full-window qualification.',cached_group_structure=cached,
                    control_diagnostics='Historical full-only wrapper, pools/negative/open ledger' if cached else 'bare full solve',
                    selection=selections,runs=records,source_sha256=hashes()))
        finally:
            for solver in solvers.values():solver.close()
    save(OUT/('p1-growth.json' if growth else 'p1-cached.json' if cached else 'p1.json'),dict(scope='One warmup and 3 interleaved DEVELOPMENT repeats, not full-window qualification.',cached_group_structure=cached,
        control_diagnostics='Historical full-only wrapper, pools/negative/open ledger' if cached else 'bare full solve',
        selection=selections,runs=records,source_sha256=hashes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['p0','p1','p1-cached','p1-growth'])
    stage=parser.parse_args().stage
    if stage=='p1-growth':p1(cached=True,growth=True)
    elif stage=='p1-cached':p1(cached=True)
    else:globals()[stage]()
