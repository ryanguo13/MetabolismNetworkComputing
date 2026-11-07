"""Registered reanalysis of degree 3/4/5 outputs on frozen computed paths."""
import json
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.history_polynomial import HistoryPolynomial
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-history-degree';OLD=ROOT/'results/m3-inner-dense'
SOURCES=['research/history_polynomial.py','research/test_history_polynomial.py','research/m3_history_degree.py',
         'research/inner_dense.py','research/m3_priority.py','research/m3_ordered_cost.py']


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    reg=dict(date='2026-11-08',degrees=[3,4,5],scope='Reanalysis of six already-known eight-step paths; no new independent trajectories or formal speed claims.',
        history='Computed MRI endpoints only; require degree+1 endpoints. Compare both all available and common last four intervals.',
        checks='Unchanged two error targets, net1e-4, positivity, pool1e-10, physical and modified ledger1e-10. No clipping or endpoint correction.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES},
        evidence_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OLD/'history.json']+sorted(OLD.glob('history-[0-5].npz'))})
    path=OUT/'registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[]
    for index in range(6):
        saved=np.load(OLD/f'history-{index}.npz');source=[r for r in old if r['shard']==f'history-{index}'];case=prepare(source[0]['model']);arrays={}
        for row in source:
            step=row['step'];q=saved[f'{step}-query'];reference=saved[f'{step}-reference']
            y=saved['endpoints'][:,step];end=saved['endpoints'][:,step+1]
            for degree in reg['degrees']:
                if step+2<degree+1:continue
                key=f'{step}-{degree}';tick=time.perf_counter()
                p=HistoryPolynomial(saved['endpoint_times'][step-degree+1:step+2],saved['endpoints'][:,step-degree+1:step+2])
                values=p(q);positive=bool(np.all(p.minima()>=-1e-12*case['scale']));guard=structural(case,values)
                ledger=physical_ledger(case,InnerCurve([p]),y,end) if positive else {}
                seconds=time.perf_counter()-tick
                valid=positive and guard['nonnegative'] and max(guard['pool'],row['stats']['pool_error'])<=1e-10
                valid &= all(v<=1e-10 for v in ledger.values())
                valid &= all(row['stats'].get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature'))
                targets={}
                for t in (1e-4,1e-6):
                    actual=metrics(case,values,reference,t) if positive else None
                    global_error=metrics(case,values,saved['global_reference'][:,saved[f'{step}-global_positions']],t) if positive else None
                    targets[str(t)]=dict(actual=actual,global_error=global_error,passes=bool(valid and max(actual.values())<=1))
                record=dict(key=key,source_shard=f'history-{index}',model=row['model'],h=row['h'],threshold=row['threshold'],step=step,degree=degree,
                    common_interval=step>=4,seconds=seconds,positive=positive,guard=guard,ledger=ledger,targets=targets)
                rows.append(record);arrays[key+'-values']=values;arrays[key+'-coefficients']=p.coefficients
        np.savez_compressed(OUT/f'{index}.npz',**arrays)
    save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
    for index in range(6):
        for degree in reg['degrees']:
            rr=[r for r in rows if r['source_shard']==f'history-{index}' and r['degree']==degree and r['common_interval']]
            print(index,degree,len(rr),sum(r['targets']['1e-06']['passes'] for r in rr),max(r['ledger'].get('physical_ledger_residual',0) for r in rr),flush=True)

if __name__=='__main__':run()
