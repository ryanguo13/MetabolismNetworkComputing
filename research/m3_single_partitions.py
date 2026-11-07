"""Development scan of fast-block size after the single-macro cost failure."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_single_study import OUT,assess_single
from research.m3_shadow_study import ROOT,digest,compare
from research.mri_single import SingleMRI
from research.defect_compact import estimate_compact
from research.m3_ordered_cost import structural

THRESHOLDS=[.001,.01,.1,1.,10.,100.]

def run():
    reg=dict(date='2026-11-08',thresholds=THRESHOLDS,multiplier=1e-3,tolerances=[1e-8,1e-10],grid=[4,8],
        cases=[['teusink',1.,.01],['kecoli',.1,.001],['kecoli',.1,.01]],
        scope='Post-cost-failure development scan, single measurement per configuration, not selection on fresh validation.',
        stop='Any false acceptance blocks variant. Only numerically and structurally passing candidates with timing prospect justify fresh-state testing.',
        source_sha256={p:digest(ROOT/p) for p in ['research/m3_single_partitions.py','research/m3_single_study.py','research/mri_single.py','research/defect_compact.py']},
        evidence_sha256={f'results/m3-single/{p}':digest(OUT/p) for p in ['cost.json','cost-0.npz','cost-1.npz','cost-2.npz']})
    path=OUT/'partition-registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    rows=[]
    for index,(name,start,h) in enumerate(reg['cases']):
        saved=np.load(OUT/f'cost-{index}.npz');y=saved['initial'];ref=saved['reference'];query=saved['query'];eq=saved['estimate_query']
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y);arrays={}
        for ti,threshold in enumerate(THRESHOLDS):
            groups=np.flatnonzero(1e-3*kappa>=threshold);solver=SingleMRI(case,groups)
            for tol in reg['tolerances']:
                key=f'{ti}-{tol}';row=dict(key=key,shard=f'partitions-{index}',reference_shard=f'cost-{index}',model=name,start=start,h=h,
                    threshold=threshold,tolerance=tol,groups=groups.tolist(),active=len(solver.active),setup_seconds=solver.setup_seconds)
                tick=time.perf_counter()
                try:
                    trial=solver.trial(y,h,tol,timeout=10.)
                    values=trial['pieces'][0](query);ev=trial['pieces'][0](eq);guard=structural(case,values)
                    base=time.perf_counter()-tick
                    shadow,evidence=estimate_compact(case,trial['pieces'],ev,eq,(4,8))
                    truth,_=assess_single(case,trial,query,ref)
                    row.update(status=truth['status'],base_seconds=base,shadow=shadow,truth=truth,comparison=compare(truth,shadow))
                    arrays[key+'-endpoints']=np.column_stack([y,trial['fine']]);arrays[key+'-values']=values;arrays[key+'-estimate_values']=ev
                    for k,v in evidence.items():arrays[key+'-'+k]=v
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',reason=str(exc),seconds=time.perf_counter()-tick,stats=getattr(exc,'stats',{}))
                rows.append(row);save(OUT/'partitions.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                print(name,h,threshold,tol,row['status'],row.get('base_seconds'),{t:c['actual_all_checks_pass'] for t,c in row.get('comparison',{}).items()},flush=True)
            solver.close()
        temp=OUT/f'partitions-{index}.tmp.npz';np.savez_compressed(temp,**arrays)
        # Validate compressed members before installing an evidence shard.
        with np.load(temp) as check:
            for key in check.files:check[key]
        temp.replace(OUT/f'partitions-{index}.npz')

if __name__=='__main__':run()
