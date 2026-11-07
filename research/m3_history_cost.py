"""Cached endpoint plus quintic history versus guarded full-system cost."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.history_polynomial import HistoryPolynomial
from research.mri_endpoint import MRIEndpoint
from research.inner_dense import InnerCurve,physical_ledger,full_curve
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-history-degree';OLD=ROOT/'results/m3-inner-dense'


def run():
    reg=dict(date='2026-11-08',paths=list(range(6)),step=4,degree=5,tolerances=[1e-8,1e-10],repeats=3,warmups=1,
        scope='Development same-state cost with existing computed history supplied free; setup separately recorded. Not a full trajectory speedup.',
        timing='No unused Hermite output. Candidate: unchanged MRI3 endpoint/internal guards + quintic history, all polynomial/physical ledger guards. Full: equally ordered Radau with all polynomial/physical ledger guards. Reference error assessment outside both timings.',
        audits='Mandatory independent full reference retained. Compare candidate time even before adding audit; no claim that audits can be removed.',
        source_sha256={p:digest(ROOT/p) for p in ['research/m3_history_cost.py','research/mri_endpoint.py','research/test_mri_endpoint.py','research/history_polynomial.py','research/inner_dense.py','research/m3_ordered_cost.py']},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OLD/'history.json']+sorted(OLD.glob('history-[0-5].npz'))})
    path=OUT/'cost-registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[]
    for index in reg['paths']:
        oldrow=next(r for r in old if r['shard']==f'history-{index}' and r['step']==4)
        saved=np.load(OLD/f'history-{index}.npz');case=prepare(oldrow['model']);y=saved['endpoints'][:,4];h=oldrow['h']
        query=saved['4-query'];ref=saved['4-reference'];screen=GroupScreen(case['system'],case['scale']);solver=MRIEndpoint(case,saved['groups'])
        arrays={};times=saved['endpoint_times'][:6];previous=saved['endpoints'][:,:5]
        for repeat in (-1,0,1,2):
            for method in (('endpoint','full') if repeat%2==0 else ('full','endpoint')):
                for tol in reg['tolerances']:
                    key=f'{method}-{tol}-{repeat}';row=dict(key=key,source_shard=f'history-{index}',model=oldrow['model'],h=h,threshold=oldrow['threshold'],
                        method=method,tolerance=tol,repeat=repeat,setup_seconds=solver.setup_seconds)
                    tick=time.perf_counter()
                    try:
                        if method=='endpoint':
                            trial=solver.trial(y,h,tol,timeout=20.);stats=trial['stats']
                            poly=HistoryPolynomial(times,np.column_stack([previous,trial['fine']]))
                            curve=InnerCurve([poly]);ledger=physical_ledger(case,curve,y,trial['fine'])
                            stats.update(ledger);stats['continuous_nonnegative']=bool(np.all(poly.minima()>=-1e-12*case['scale']))
                        else:
                            trial=full_curve(case,y,h,tol,screen);curve=trial['pieces'][0];stats=trial['stats']
                        values=curve(query);guard=structural(case,values);seconds=time.perf_counter()-tick
                        targets={}
                        valid=stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                        valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
                        for t in (1e-4,1e-6):
                            actual=metrics(case,values,ref,t) if stats['continuous_nonnegative'] else None
                            targets[str(t)]=dict(actual=actual,passes=bool(valid and max(actual.values())<=1))
                        row.update(status='success',seconds=seconds,stats=stats,guard=guard,targets=targets)
                        arrays[key+'-values']=values;arrays[key+'-endpoint']=trial['fine']
                        if method=='endpoint':arrays[key+'-coefficients']=poly.coefficients
                        else:
                            for k,v in curve.pack().items():arrays[key+'-'+k]=v
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                    rows.append(row);save(OUT/'cost.json',dict(runs=rows,source_sha256=reg['source_sha256']))
            print(index,repeat,'done',flush=True)
        solver.close();tmp=OUT/f'cost-{index}.tmp.npz';np.savez_compressed(tmp,**arrays)
        with np.load(tmp) as check:
            for k in check.files:check[k]
        tmp.replace(OUT/f'cost-{index}.npz')

if __name__=='__main__':run()
