"""Reassess estimator cost against the equally optimized high-accuracy audit."""
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_priority import full,metrics
from research.ordered_radau import OrderedRadau
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-ordered-cost'
SOURCES=['research/m3_ordered_audit.py','research/ordered_radau.py','research/m3_ordered_cost.py']

def run():
    reg=dict(date='2026-11-07',scope='Post-discovery follow-up, same candidate/query as ordered-cost; optimized reference at 1e-12, warmup plus three development repeats.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES})
    path=OUT/'audit-registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    rows=[];arrays={};saved=np.load(OUT/'curves.npz');old=json.loads((OUT/'runs.json').read_text())['runs']
    for index,(name,start,h) in enumerate([('teusink',1.,.01),('kecoli',.1,.001),('kecoli',.1,.01)]):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);sys=case['system']
        y=full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        times=h*saved[f'{index}-query'];reference=saved[f'{index}-reference']
        for rep in (-1,0,1,2):
            tick=time.perf_counter()
            result=solve_ivp(lambda t,z:sys.S@sys.flux(0.,z),(0,h),y,t_eval=times,method=OrderedRadau,
                jac=lambda t,z:sys.S@screen.gradient(z),rtol=1e-12,atol=1e-14*case['scale'])
            seconds=time.perf_counter()-tick
            if not result.success:raise RuntimeError(result.message)
            delta=metrics(case,result.y,reference,1e-6)
            assert max(delta.values())<=.05
            estimates=[r['shadow']['seconds'] for r in old if r['model']==name and r['h']==h and r['repeat']>=0 and r['method']=='mri-ordered']
            row=dict(key=f'{index}-{rep}',model=name,h=h,repeat=rep,seconds=seconds,reference_difference=delta,
                median_estimator_seconds=float(np.median(estimates)))
            rows.append(row);arrays[row['key']]=result.y
    save(OUT/'optimized-audit.json',dict(runs=rows,source_sha256=reg['source_sha256']))
    np.savez_compressed(OUT/'optimized-audit.npz',**arrays)
    for name,h in [(r['model'],r['h']) for r in rows if r['repeat']==0]:
        rr=[r for r in rows if r['model']==name and r['h']==h and r['repeat']>=0]
        elapsed=float(np.median([r['seconds'] for r in rr]));print(name,h,elapsed,rr[0]['median_estimator_seconds']/elapsed)

if __name__=='__main__':run()
