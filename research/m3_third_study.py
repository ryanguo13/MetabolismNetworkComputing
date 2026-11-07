"""Literature-backed order-three local screen; no controller admission."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_single_study import references,queries,assess_single
from research.m3_ordered_cost import full_radau,structural
from research.m3_priority import full,metrics
from research.m3_shadow_study import ROOT,digest,compare
from research.defect_compact import estimate_compact
from research.mri_third import MRI3
from research.ordered_radau import OrderedRadau

OUT=ROOT/'results/m3-third'
SPECS={'teusink':dict(start=1.,steps=[.01,.03,.1,.3]),'kecoli':dict(start=.1,steps=[.001,.003,.01,.03,.1])}
SOURCES=['research/m3_third_study.py','research/mri_third.py','research/test_mri_third.py',
         'research/mri_single.py','research/defect_compact.py','research/m3_single_study.py','research/m3_ordered_cost.py',
         'research/ordered_radau.py','research/defect_fast.py','research/m3_priority.py']

def run():
    OUT.mkdir(exist_ok=True,parents=True)
    reg=dict(date='2026-11-08',specs=SPECS,tolerances=[1e-8,1e-10],grid=[4,8],threshold=.01,multiplier=1e-3,
        method='MRI-GARK-ERK33a, Sandu arXiv:1808.02759v3 section6.2 delta=-1/2, three H/3 physical subintervals, affine final forcing.',
        source='https://arxiv.org/pdf/1808.02759v3',source_pdf_sha256=digest(ROOT/'results/scratch/mri.pdf'),
        crosscheck='SUNDIALS 6.4.1 src/arkode/arkode_mri_tables.def ARKODE_MRI_GARK_ERK33a',
        table_sha256=digest(ROOT/'results/scratch/sundials-6.4.1/src/arkode/arkode_mri_tables.def'),
        scope='Known late states, 18 development candidates with equally optimized full baseline; one timing each, no full-window claim.',
        acceptance='Independent 65 point plus public grid reference and all original guards; shadow estimate sees 17 plus public points. No reduced audits.',
        stopping='No formal performance repeats absent a local same-accuracy cost prospect. Keep all failures.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES})
    path=OUT/'registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    rows=[]
    for name,spec in SPECS.items():
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);case['screen']=screen
        y=full(case,case['model'].y0,np.array([0.,spec['start']]),1e-12)[:,-1]
        kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=.01);solver=MRI3(case,groups)
        for hi,h in enumerate(spec['steps']):
            query,eq=queries(case,spec['start'],h);ref,loose,optimized,budgets,audit_seconds=references(case,y,h,query)
            arrays=dict(initial=y,groups=groups,query=query,estimate_query=eq,reference=ref,loose=loose,optimized=optimized)
            for tol in reg['tolerances']:
                for method in ('third','full'):
                    key=f'{method}-{tol}';row=dict(key=key,shard=f'{name}-{hi}',model=name,start=spec['start'],h=h,tolerance=tol,
                        method=method,reference_budgets=budgets,audit_seconds=audit_seconds,setup_seconds=solver.setup_seconds)
                    tick=time.perf_counter()
                    try:
                        if method=='third':
                            trial=solver.trial(y,h,tol,timeout=20.)
                            values=trial['pieces'][0](query);ev=trial['pieces'][0](eq);guard=structural(case,values)
                            base_seconds=time.perf_counter()-tick
                            shadow,evidence=estimate_compact(case,trial['pieces'],ev,eq,(4,8))
                            truth,_=assess_single(case,trial,query,ref)
                            row.update(status=truth['status'],base_seconds=base_seconds,shadow=shadow,truth=truth,comparison=compare(truth,shadow))
                            arrays[key+'-endpoints']=np.column_stack([y,trial['fine']]);arrays[key+'-estimate_values']=ev
                            for k,v in evidence.items():arrays[key+'-'+k]=v
                        else:
                            values,counts=full_radau(case,y,h,tol,OrderedRadau,screen,query);guard=structural(case,values)
                            seconds=time.perf_counter()-tick
                            row.update(status='success',seconds=seconds,counts=counts,actual={str(t):metrics(case,values,ref,t) for t in (1e-4,1e-6)})
                        arrays[key+'-values']=values;row['guard']=guard
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',reason=str(exc),seconds=time.perf_counter()-tick,stats=getattr(exc,'stats',{}))
                    rows.append(row);save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
                    print(name,h,tol,method,row['status'],row.get('base_seconds',row.get('seconds')),
                        {t:c['actual_all_checks_pass'] for t,c in row.get('comparison',{}).items()},flush=True)
            temp=OUT/f'{name}-{hi}.tmp.npz';np.savez_compressed(temp,**arrays)
            with np.load(temp) as check:
                for key in check.files:check[key]
            temp.replace(OUT/f'{name}-{hi}.npz')
        solver.close()

if __name__=='__main__':run()
