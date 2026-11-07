"""Fair local cost probe, not a performance admission experiment."""
import argparse
import json
import time
import numpy as np
from scipy.integrate import Radau
from scipy.sparse import csc_matrix
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import MRI,GAUSS
from research.ordered_radau import OrderedMRI,OrderedRadau
from research.defect_fast import estimate_fast
from research.m3_priority import full,metrics,assess
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-ordered-cost'
SPECS=[('teusink',1.,.01),('kecoli',.1,.001),('kecoli',.1,.01)]
METHODS=['mri-default','mri-ordered','full-default','full-ordered','full-cvode']
SOURCES=['research/m3_ordered_cost.py','research/ordered_radau.py','research/test_ordered_radau.py',
         'research/mri_kernel.py','research/mri_native.py','research/defect_fast.py','research/defect_shadow.py',
         'research/m3_priority.py','metabolism/m3_continuation.py']

def registration():return dict(date='2026-11-07',specs=SPECS,methods=METHODS,
    warmups=1,development_repeats=3,tolerances=[1e-8,1e-10],
    scope='Local development screen; full independent audits remain; no online controller or formal performance qualification.',
    fairness='Same ordering available to MRI and full Radau. Same RHS and analytic/complex-step Jacobian. CVODE remains an additional baseline. MRI structural checks retained.',
    timing='Setup, output, internal state diagnostics, pool/ledger postchecks and shadow estimation timed. Common independent reference audit separately recorded; included in audited MRI cost.',
    source_sha256={p:digest(ROOT/p) for p in SOURCES})

def register():
    OUT.mkdir(exist_ok=True,parents=True);reg=registration()
    # JSON round-trip normalizes tuple specs.
    reg=json.loads(json.dumps(reg));path=OUT/'registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)

def full_radau(case,y,h,tol,method,screen,query):
    sys=case['system'];scale=case['scale'];pools=case['options']['pools']
    rhs=lambda t,z:sys.S@sys.flux(0.,z)
    jac=lambda t,z:csc_matrix(sys.S@screen.gradient(z))
    sol=method(rhs,0.,y,h,rtol=tol,atol=.01*tol*scale,jac=jac)
    weights=case['options'].get('ledger_weights');ledger=quad=0.;pool_drift=0.
    values=np.empty((len(y),len(query)));values[:,0]=y;counts=dict(internal=0,minimum=float(y.min()))
    while sol.status=='running':
        left=sol.t;msg=sol.step()
        if sol.status=='failed':raise RuntimeError(msg)
        counts['internal']+=1;counts['minimum']=min(counts['minimum'],float(sol.y.min()))
        if not np.isfinite(sol.y).all() or np.any(sol.y < -1e-12*scale):raise ValueError('Negative/nonfinite accepted full state')
        pool_drift=max(pool_drift,float(np.max(abs(pools@(sol.y-y))))/max(1e-12,float(np.max(abs(pools@y)))))
        if weights is not None:
            integrals=[]
            for n in (4,8):
                nodes,w=GAUSS[n];ts=left+(sol.t-left)*(nodes+1)/2
                zz=sol.dense_output()(ts)
                if np.any(zz < -1e-12*scale[:,None]):raise ValueError('Negative ledger state')
                integrals.append(float((sol.t-left)/2*(w@np.array([weights@rhs(0.,z) for z in zz.T]))))
            ledger+=integrals[1];quad+=abs(integrals[1]-integrals[0])
        take=(h*query>left)&(h*query<=sol.t)
        values[:,take]=sol.dense_output()(h*query[take])
    counts.update(nfev=sol.nfev,njev=sol.njev,nlu=sol.nlu,pool_error=pool_drift)
    if weights is not None:
        denominator=max(1.,abs(float(weights@y)))
        counts.update(ledger_residual=abs(float(weights@(sol.y-y))-ledger)/denominator,ledger_quadrature=quad/denominator)
    return values,counts

def structural(case,values):
    scale=case['scale'];pool=case['options']['pools'];y=values[:,0]
    drift=float(np.max(abs(pool@(values-y[:,None]))))/max(1e-12,float(np.max(abs(pool@y))))
    return dict(nonnegative=bool(np.isfinite(values).all() and np.all(values>=-1e-12*scale[:,None])),pool=drift)

def run():
    reg=json.loads((OUT/'registration.json').read_text());assert reg==json.loads(json.dumps(registration()))
    rows=[];arrays={}
    for index,(name,start,h) in enumerate(SPECS):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale'])
        y=full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        query=np.unique(np.r_[np.linspace(0,1,65),(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h])
        tick=time.perf_counter();ref=full(case,y,h*query,1e-12);audit_seconds=time.perf_counter()-tick
        loose=full(case,y,h*query,1e-11);assert max(metrics(case,loose,ref,1e-6).values())<=.05
        arrays[f'{index}-reference']=ref;arrays[f'{index}-loose']=loose;arrays[f'{index}-query']=query
        methods=[m for m in METHODS if name=='kecoli' or m!='full-cvode']
        for rep in (-1,0,1,2):
            ordered=methods[rep%len(methods):]+methods[:rep%len(methods)]
            for method in ordered:
                for tol in reg['tolerances']:
                    key=f'{index}-{method}-{tol}-{rep}';row=dict(key=key,model=name,start=start,h=h,tolerance=tol,method=method,repeat=rep,audit_seconds=audit_seconds)
                    solver=None;tick=time.perf_counter()
                    try:
                        if method.startswith('mri'):
                            kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=.01)
                            cls=MRI if method=='mri-default' else OrderedMRI
                            solver=cls(case,groups,backend='radau')
                            trial=solver.trial(y,h,tol,timeout=20.)
                            truth,values=assess(case,trial,query,ref)
                            shadow,_=estimate_fast(case,trial['pieces'],values,query)
                            counts=trial['stats'];row['shadow']=shadow;row['truth']=truth
                        elif method=='full-cvode':values=full(case,y,h*query,tol);counts={}
                        else:
                            cls=Radau if method=='full-default' else OrderedRadau
                            values,counts=full_radau(case,y,h,tol,cls,screen,query)
                        guard=structural(case,values)
                        actual=({t:v['actual'] for t,v in truth.get('targets',{}).items()} if method.startswith('mri')
                                else {str(t):metrics(case,values,ref,t) for t in (1e-4,1e-6)})
                        elapsed=time.perf_counter()-tick
                        row.update(status='success',seconds=elapsed,actual=actual,guard=guard,counts=counts)
                        if method.startswith('mri'):row['audited_seconds']=elapsed+audit_seconds
                        arrays[key]=values
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                    finally:
                        if solver is not None:solver.close()
                    rows.append(row);save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                    print(key,row['status'],row['seconds'],flush=True)
        np.savez_compressed(OUT/'curves.npz',**arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args()
    {'register':register,'run':run}[a.stage]()
