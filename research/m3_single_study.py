"""Registered single-macro MRI2 and compact-defect development experiments."""
import argparse
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_priority import full,metrics
from research.m3_shadow_study import ROOT,digest,compare
from research.mri_single import SingleMRI
from research.defect_compact import estimate_compact
from research.ordered_radau import OrderedRadau
from research.m3_ordered_cost import full_radau,structural

OUT=ROOT/'results/m3-single'
STATES={'teusink':[0.,1e-6,.001,1.],'kecoli':[0.,1e-8,.001,.1]}
STEPS={'teusink':[1e-7,1e-4,.01],'kecoli':[1e-8,1e-4,.001]}
GRIDS=[(2,4),(4,8),(8,16)]
SOURCES=['research/mri_single.py','research/test_mri_single.py','research/defect_compact.py','research/m3_single_study.py',
         'research/mri_kernel.py','research/ordered_radau.py','research/mri_native.py','research/defect_fast.py','research/defect_shadow.py',
         'research/m3_priority.py','research/m3_shadow_study.py','research/m3_ordered_cost.py','metabolism/m3_continuation.py','metabolism/adaptive.py']

def register():
    reg=dict(date='2026-11-08',scope='Shadow-only single MRI2 macro; full independent audits remain.',
        states=STATES,steps=STEPS,grids=GRIDS,tolerances=[1e-8,1e-10],
        formula='One original MRI2 step (two fast subflows), one full-RHS endpoint Hermite curve. No coarse/fine estimator.',
        targets=[1e-4,1e-6],net_target=1e-4,score='4*(central+refinement), unchanged',
        grids_scope='All three preset pairs are development variants; selection requires fresh validation before any promotion.',
        audit='65 uniform points plus original public observation points; estimator sees 17 plus public points; references at 1e-11 and 1e-12.',
        cost='Known intervals: Teusink start1 H.01; kecoli start.1 H.001 and .01. Cached groups and partition; setup separately recorded; 1 warmup +3 rotating repeats. Accuracy postprocessing outside both candidate and baseline time.',
        stopping='Any shadow false accept blocks that estimator variant. No full-window performance qualification without local prospect. No gate relaxation.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES})
    OUT.mkdir(exist_ok=True,parents=True);path=OUT/'registration.json';reg=json.loads(json.dumps(reg))
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)

def verify_registration():
    reg=json.loads((OUT/'registration.json').read_text())
    for p,sha in reg['source_sha256'].items():assert digest(ROOT/p)==sha,p
    return reg

def assess_single(case,trial,query,ref):
    stats=trial['stats'];values=trial['pieces'][0](query)
    result=dict(status='success',stats=stats)
    if not stats['continuous_nonnegative']:
        result.update(status='rejected',reason='Hermite internal negative extremum');return result,values
    guard=structural(case,values);result['output_pool_error']=guard['pool'];result['targets']={}
    for t in (1e-4,1e-6):
        actual=metrics(case,values,ref,t)
        valid=(guard['nonnegative'] and max(guard['pool'],stats['pool_error'])<=1e-10
               and all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature')))
        result['targets'][str(t)]=dict(actual=actual,numerical_pass=max(actual.values())<=1,passes=max(actual.values())<=1 and valid)
    return result,values

def queries(case,start,h):
    public=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h
    return np.unique(np.r_[np.linspace(0,1,65),public]),np.unique(np.r_[np.linspace(0,1,17),public])

def references(case,y,h,query):
    tick=time.perf_counter()
    optimized=solve_ivp(lambda t,z:case['system'].S@case['system'].flux(0.,z),(0,h),y,t_eval=h*query,method=OrderedRadau,
        jac=lambda t,z:case['system'].S@GroupScreen.gradient(case['screen'],z),rtol=1e-12,atol=1e-14*case['scale'])
    audit_seconds=time.perf_counter()-tick
    assert optimized.success
    # Independent original full solver: CVODE on large, original Radau on small.
    ref=full(case,y,h*query,1e-12);loose=full(case,y,h*query,1e-11)
    budgets=[metrics(case,loose,ref,1e-6),metrics(case,optimized.y,ref,1e-6)]
    assert max(v for b in budgets for v in b.values())<=.05
    return ref,loose,optimized.y,budgets,audit_seconds

def dev():
    reg=verify_registration();rows=[]
    for name,states in STATES.items():
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);case['screen']=screen
        prefix=full(case,case['model'].y0,np.array(states),1e-12)
        for si,start in enumerate(states):
            y=prefix[:,si];kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=.01)
            solver=SingleMRI(case,groups,backend='radau');arrays={'initial':y,'groups':groups}
            for hi,h in enumerate(STEPS[name]):
                query,eq=queries(case,start,h)
                ref,loose,optimized,budgets,audit_seconds=references(case,y,h,query)
                refkey=str(hi)
                for label,value in dict(reference=ref,loose=loose,optimized=optimized,query=query,estimate_query=eq).items():arrays[refkey+'-'+label]=value
                for tol in reg['tolerances']:
                    key=f'{hi}-{tol}';row=dict(key=key,shard=f'dev-{name}-{si}',model=name,start=start,h=h,tolerance=tol,
                        setup_seconds=solver.setup_seconds,audit_seconds=audit_seconds,reference_budgets=budgets)
                    try:
                        trial=solver.trial(y,h,tol,timeout=20.)
                        truth,values=assess_single(case,trial,query,ref);estvalues=trial['pieces'][0](eq)
                        shadows={};comparisons={}
                        for grid in GRIDS:
                            label=f'{grid[0]}-{grid[1]}'
                            shadow,evidence=estimate_compact(case,trial['pieces'],estvalues,eq,grid)
                            shadows[label]=shadow;comparisons[label]=compare(truth,shadow)
                            for k,v in evidence.items():arrays[key+'-'+label+'-'+k]=v
                        row.update(status=truth['status'],truth=truth,shadows=shadows,comparisons=comparisons)
                        arrays[key+'-values']=values;arrays[key+'-estimate_values']=estvalues
                        arrays[key+'-endpoints']=np.column_stack([y,trial['fine']])
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row);save(OUT/'development.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                    print(name,start,h,tol,row['status'],{g:sum(c['false_accept'] for c in v.values()) for g,v in row.get('comparisons',{}).items()},flush=True)
            solver.close();np.savez_compressed(OUT/f'dev-{name}-{si}.npz',**arrays)


def cost():
    reg=verify_registration();rows=[]
    for index,(name,start,h) in enumerate([('teusink',1.,.01),('kecoli',.1,.001),('kecoli',.1,.01)]):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);case['screen']=screen
        y=full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        tick=time.perf_counter();kappa,_=screen.evaluate(y);classification=time.perf_counter()-tick
        groups=np.flatnonzero(1e-3*kappa>=.01);solver=SingleMRI(case,groups)
        query,eq=queries(case,start,h);ref,loose,optimized,budgets,audit_seconds=references(case,y,h,query)
        arrays=dict(initial=y,query=query,estimate_query=eq,reference=ref,loose=loose,optimized=optimized)
        for repeat in (-1,0,1,2):
            methods=['single','full'] if repeat%2==0 else ['full','single']
            for method in methods:
                for tol in reg['tolerances']:
                    key=f'{method}-{tol}-{repeat}';row=dict(key=key,shard=f'cost-{index}',model=name,start=start,h=h,method=method,
                        tolerance=tol,repeat=repeat,classification_seconds=classification,setup_seconds=solver.setup_seconds,
                        reference_budgets=budgets,audit_seconds=audit_seconds)
                    try:
                        tick=time.perf_counter()
                        if method=='single':
                            trial=solver.trial(y,h,tol,timeout=20.)
                            values=trial['pieces'][0](query);ev=trial['pieces'][0](eq);guard=structural(case,values)
                            base_seconds=time.perf_counter()-tick;shadows={}
                            # Rotating preset estimator order; candidate base is shared and timed once.
                            order=GRIDS[repeat%3:]+GRIDS[:repeat%3]
                            for grid in order:
                                label=f'{grid[0]}-{grid[1]}';shadow,_=estimate_compact(case,trial['pieces'],ev,eq,grid)
                                shadows[label]=shadow
                            truth,_=assess_single(case,trial,query,ref)
                            row.update(base_seconds=base_seconds,shadows=shadows,truth=truth,
                                comparisons={g:compare(truth,v) for g,v in shadows.items()})
                        else:
                            values,counts=full_radau(case,y,h,tol,OrderedRadau,screen,query);guard=structural(case,values)
                            elapsed=time.perf_counter()-tick
                            actual={str(t):metrics(case,values,ref,t) for t in (1e-4,1e-6)}
                            row.update(seconds=elapsed,actual=actual,counts=counts)
                        row.update(status='success',guard=guard);arrays[key]=values
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',reason=str(exc))
                    rows.append(row);save(OUT/'cost.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                    print(name,h,key,row['status'],row.get('base_seconds',row.get('seconds')),flush=True)
        solver.close();np.savez_compressed(OUT/f'cost-{index}.npz',**arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','dev','cost']);args=p.parse_args()
    {'register':register,'dev':dev,'cost':cost}[args.stage]()
