"""History interpolation on actual successive MRI3 endpoints, with full audits."""
import json
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_dense import DenseMRI
from research.inner_dense import InnerCurve,physical_ledger
from research.history_output import cubic_history
from research.m3_priority import full,metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-inner-dense'
SPECS=[('teusink',1.,.01,.01),('teusink',1.,.03,.01),('teusink',1.,.001,.1),
       ('kecoli',.1,.001,.01),('kecoli',.1,.003,.01),('kecoli',.1,.001,.1)]


def run():
    source={p:digest(ROOT/p) for p in ['research/m3_history_study.py','research/history_output.py','research/test_history_output.py','research/mri_dense.py','research/inner_dense.py']}
    reg=dict(date='2026-11-08',specs=SPECS,steps=8,tolerance=1e-10,
        source='https://sundials.readthedocs.io/en/latest/arkode/Mathematics_link.html#lagrange-interpolation-module',
        local_cached_page_sha256=digest(ROOT/'results/scratch/arkode-math.html'),
        source_scope='Previously retrieved latest documentation; reread section2.2.2.2, not claiming this page is pinned to SUNDIALS6.4.1.',
        scope='Post-inner-curve failure development. Forced diagnostic paths; error rejection does not become accepted state. Negative curves stop path.',
        history='Last four computed macro endpoints only. No reference-state history. First two intervals cannot supply cubic history and are marked unavailable.',
        audit='65 uniform plus public observation points; independent local and whole-path references; local reference tightening at1e-11/1e-12.',
        source_sha256=source)
    reg=json.loads(json.dumps(reg));path=OUT/'history-registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    rows=[]
    for index,(name,start,h,threshold) in enumerate(SPECS):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale'])
        y=full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=threshold);solver=DenseMRI(case,groups)
        origin=y.copy();times=[0.];states=[y.copy()];arrays=dict(initial=y,groups=groups)
        grid=np.linspace(0,1,65);alltimes=np.unique(np.r_[np.arange(8*64+1)*h/64,
            case['times'][(case['times']>start)&(case['times']<start+8*h)]-start])
        global_ref=full(case,origin,alltimes,1e-12);global_loose=full(case,origin,alltimes,1e-11)
        global_budget=metrics(case,global_loose,global_ref,1e-6);assert max(global_budget.values())<=.05
        arrays.update(global_times=alltimes,global_reference=global_ref,global_loose=global_loose)
        for step in range(8):
            key=str(step);row=dict(shard=f'history-{index}',key=key,model=name,start=start,h=h,threshold=threshold,step=step,global_budget=global_budget)
            try:
                trial=solver.trial(y,h,1e-10,timeout=20.)
                if not trial['stats']['continuous_nonnegative']:raise ValueError('Negative inner candidate curve')
                times.append((step+1)*h);states.append(trial['fine'].copy())
                positions=np.flatnonzero((alltimes>=step*h-1e-13*h)&(alltimes<=(step+1)*h+1e-13*h))
                query=np.clip((alltimes[positions]-step*h)/h,0,1)
                atimes=np.unique(np.r_[0.,h*query]);indices=np.searchsorted(atimes,h*query)
                ref=full(case,y,atimes,1e-12)[:,indices];loose=full(case,y,atimes,1e-11)[:,indices]
                budget=metrics(case,loose,ref,1e-6);assert max(budget.values())<=.05
                curves=dict(inner=trial['pieces'][0],hermite=InnerCurve([trial['hermite']]))
                if len(states)>=4:curves['history']=InnerCurve([cubic_history(times[-4:],np.column_stack(states[-4:]))])
                output={}
                for label,curve in curves.items():
                    values=curve(query);guard=structural(case,values);positive=bool(np.all(curve.minima()>=-1e-12*case['scale']))
                    ledger=physical_ledger(case,curve,y,trial['fine']) if positive else {}
                    targets={}
                    for t in (1e-4,1e-6):
                        actual=metrics(case,values,ref,t) if positive else None
                        global_error=metrics(case,values,global_ref[:,positions],t) if positive else None
                        valid=positive and guard['nonnegative'] and max(guard['pool'],trial['stats']['pool_error'])<=1e-10
                        valid &= all(v<=1e-10 for v in ledger.values())
                        valid &= all(trial['stats'].get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature'))
                        targets[str(t)]=dict(actual=actual,global_error=global_error,passes=bool(valid and max(actual.values())<=1))
                    output[label]=dict(guard=guard,nonnegative=positive,ledger=ledger,targets=targets)
                    arrays[key+'-'+label+'-values']=values
                    for k,v in curve.pack().items():arrays[key+'-'+label+'-'+k]=v
                row.update(status='success',outputs=output,stats=trial['stats'],reference_budget=budget,history_available='history' in curves)
                arrays[key+'-query']=query;arrays[key+'-reference']=ref;arrays[key+'-loose']=loose;arrays[key+'-global_positions']=positions
                y=trial['fine'].copy()
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',reason=str(exc))
            rows.append(row);save(OUT/'history.json',dict(runs=rows,source_sha256=source))
            print(index,step,row['status'],{k:v['targets']['1e-06']['passes'] for k,v in row.get('outputs',{}).items()},flush=True)
            if row['status']=='failed':break
        solver.close();arrays['endpoint_times']=np.array(times);arrays['endpoints']=np.column_stack(states)
        tmp=OUT/f'history-{index}.tmp.npz';np.savez_compressed(tmp,**arrays)
        with np.load(tmp) as check:
            for k in check.files:check[k]
        tmp.replace(OUT/f'history-{index}.npz')

if __name__=='__main__':run()
