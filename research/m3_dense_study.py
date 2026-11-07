"""Registered inner-dense diagnostic: qualify output, not just endpoints."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_priority import full,metrics
from research.m3_single_study import references,assess_single
from research.m3_shadow_study import ROOT,digest
from research.m3_ordered_cost import structural
from research.mri_dense import DenseMRI
from research.inner_dense import InnerCurve,physical_ledger,full_curve

OUT=ROOT/'results/m3-inner-dense'
SPECS=[('teusink',1.,h,.01) for h in (.01,.03,.1,.3)]
SPECS += [('teusink',1.,h,.1) for h in (.001,.01)]+[('teusink',0.,1e-7,.01)]
SPECS += [('kecoli',.1,h,.01) for h in (.001,.003,.01)]+[('kecoli',.1,.001,.1),('kecoli',0.,1e-8,.01)]
SOURCES=['research/inner_dense.py','research/mri_dense.py','research/test_inner_dense.py','research/m3_dense_study.py',
         'research/mri_third.py','research/mri_single.py','research/ordered_radau.py','research/m3_single_study.py','research/m3_ordered_cost.py']

def run():
    OUT.mkdir(exist_ok=True,parents=True)
    reg=dict(date='2026-11-08',specs=SPECS,tolerances=[1e-8,1e-10],
        output='Inner Radau cubics plus exact inactive-coordinate forcing integral; exact polynomial extrema and joins checked.',
        independent_grid='65 uniform points + original observation points + all inner piece ends and midpoints from both candidate tolerances, rounded to 14 fractional decimals to deduplicate roundoff.',
        control='Equally ordered full Radau with the same polynomial nonnegativity, joins, physical ledger and query grid; no shadow estimator or reduced audits.',
        scope='Known states and two startup probes; single development timings only. Constant-slow small-model grouping explicitly marked degenerate.',
        stopping='No output or M4 admission unless original concentration/flux, pool, positivity and physical ledger all pass. Retain every failure.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES})
    reg=json.loads(json.dumps(reg));path=OUT/'registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    rows=[]
    for index,(name,start,h,threshold) in enumerate(SPECS):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);case['screen']=screen
        y=case['model'].y0.copy() if start==0 else full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=threshold);solver=DenseMRI(case,groups)
        slow=np.setdiff1d(np.arange(case['system'].S.shape[1]),solver.cols)
        constant=case['system'].dependencies[slow].nnz==0 and not case['system'].time_dependent
        trials={};base={};arrays=dict(initial=y,groups=groups)
        observed=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h
        query=list(np.r_[np.linspace(0,1,65),observed])
        for tol in reg['tolerances']:
            row=dict(shard=str(index),model=name,start=start,h=h,threshold=threshold,tolerance=tol,groups=groups.tolist(),
                constant_slow=constant,setup_seconds=solver.setup_seconds)
            try:
                trial=solver.trial(y,h,tol,timeout=20.);curve=trial['pieces'][0]
                query.extend(curve.edges/curve.h);query.extend((curve.edges[1:]+curve.edges[:-1])/(2*curve.h))
                trials[tol]=trial
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
            base[tol]=row
        solver.close();query=np.unique(np.clip(np.round(query,14),0,1))
        ref,loose,optimized,budgets,audit_seconds=references(case,y,h,query)
        arrays.update(query=query,reference=ref,loose=loose,optimized=optimized)
        for tol,row0 in base.items():
            for method in ('inner','full'):
                key=f'{method}-{tol}';row=dict(row0,key=key,method=method,reference_budgets=budgets,audit_seconds=audit_seconds)
                if method=='inner' and tol not in trials:
                    rows.append(row);continue
                try:
                    trial=trials[tol] if method=='inner' else full_curve(case,y,h,tol,screen)
                    tick=time.perf_counter();curve=trial['pieces'][0];values=curve(query);guard=structural(case,values)
                    post_seconds=time.perf_counter()-tick
                    truth,_=assess_single(case,trial,query,ref)
                    row.update(status=truth['status'],truth=truth,seconds=trial['stats']['seconds']+post_seconds,
                        intervals=len(curve.pieces),guard=guard,endpoint=metrics(case,trial['fine'][:,None],ref[:,-1:],1e-6))
                    if method=='inner':
                        hermite=trial['hermite'];hv=hermite(query)
                        row['hermite']=dict(actual={str(t):metrics(case,hv,ref,t) for t in (1e-4,1e-6)},
                            ledger=physical_ledger(case,InnerCurve([hermite]),y,trial['fine']),
                            nonnegative=bool(np.all(hermite.minima()>=-1e-12*case['scale'])))
                    arrays[key+'-values']=values;arrays[key+'-endpoint']=trial['fine']
                    for k,v in curve.pack().items():arrays[key+'-'+k]=v
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                rows.append(row);save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                print(index,name,h,threshold,method,tol,row['status'],
                    {t:v['passes'] for t,v in row.get('truth',{}).get('targets',{}).items()},flush=True)
        tmp=OUT/f'{index}.tmp.npz';np.savez_compressed(tmp,**arrays)
        with np.load(tmp) as check:
            for key in check.files:check[key]
        tmp.replace(OUT/f'{index}.npz')

if __name__=='__main__':run()
