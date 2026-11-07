"""Instrumented hotspot diagnosis; these timings are not speedup evidence."""
import cProfile
import json
import pstats
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from metabolism.adaptive import GroupScreen
from research.row_continuous import ROWContinuous
from research.fullstep_radau import fullstep_curve
from research.inner_dense import full_curve
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-row-continuous'


def run():
    rows=json.loads((OUT/'runs.json').read_text())['runs'];configs=[]
    for model in ('teusink','kecoli'):
        for method in ('row-spc','row','radau'):
            good=[r for r in rows if r['model']==model and r['h']==.003 and r['method']==method and r['status']=='success' and r['targets']['1e-06']['passes']]
            r=min(good,key=lambda r:r['seconds']);configs.append(dict(key=r['key'],model=model,method=method,iterations=20 if model=='teusink' else 5))
    reg=dict(date='2026-11-08',configs=configs,scope='Post-screen cProfile hotspots at H=.003 for best observed strict candidate,full ROW and mature Radau. Instrumentation changes costs; not formal repeats, not a speed comparison. Complete kernel guards included; setup outside.',
        source_sha256={p:digest(ROOT/p) for p in ('research/m3_row_profile.py','research/row_continuous.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-row-continuous/runs.json','results/m3-row-continuous/registration.json')})
    if (OUT/'profile-registration.json').exists():raise FileExistsError('Preserve profile')
    save(OUT/'profile-registration.json',reg);results=[]
    for spec in configs:
        row=next(r for r in rows if r['key']==spec['key']);case=prepare(row['model']);screen=GroupScreen(case['system'],case['scale']);kernel=None
        with np.load(OUT/f"reference-{row['model']}-{row['h']}.npz") as a:y=a['initial']
        with np.load(OUT/f"{row['key']}.npz") as a:expected=a['endpoint']
        if row['method']=='radau':call=lambda:(fullstep_curve if row['whole'] else full_curve)(case,y,row['h'],row['tolerance'],screen)
        else:
            kernel=ROWContinuous(case,row['groups']);call=lambda:kernel.trial(y,row['h'],row['tolerance'],base_only=row['method']=='row')
        profiler=cProfile.Profile();maxdiff=0.
        for _ in range(spec['iterations']):
            trial=profiler.runcall(call);maxdiff=max(maxdiff,float(np.max(abs(trial['fine']-expected)/case['scale'])))
        if kernel:kernel.close()
        assert maxdiff<1e-12
        stats=pstats.Stats(profiler);functions=[]
        for (file,line,function),(primitive,calls,selftime,total,callers) in stats.stats.items():
            name=file.removeprefix(str(ROOT)+'/') if file.startswith(str(ROOT)+'/') else file.split('/')[-1]
            functions.append(dict(file=name,line=line,function=function,primitive_calls=primitive,calls=calls,self_seconds=selftime,cumulative_seconds=total))
        results.append(dict(**spec,instrumented_seconds=stats.total_tt,endpoint_scaled_difference=maxdiff,
            by_self=sorted(functions,key=lambda r:r['self_seconds'],reverse=True)[:30],by_cumulative=sorted(functions,key=lambda r:r['cumulative_seconds'],reverse=True)[:30]))
    save(OUT/'profile.json',dict(results=results))


if __name__=='__main__':run()
