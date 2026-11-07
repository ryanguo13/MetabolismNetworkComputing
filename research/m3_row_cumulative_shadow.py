"""Post-path cumulative defect diagnostic, never an accepting controller."""
import json
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.defect_compact import estimate_compact
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-cached-row'


def evaluate(reg,write_arrays=False):
    original=json.loads((OUT/'runs.json').read_text());results=[]
    for index in reg['paths']:
        spec=json.loads((OUT/'registration.json').read_text())['paths'][index];case=prepare(spec['model'])
        summary=next(r for r in original['paths'] if r['key']==f'{index}-original');n=summary['accepted_steps'];pieces=[]
        with np.load(OUT/f'{index}-original.npz') as a:
            for step in range(n):pieces.extend(unpack(a[f'{step}-lengths'],a[f'{step}-coefficients']).pieces)
        curve=InnerCurve(pieces)
        with np.load(OUT/f'path-{index}.npz') as a:origin=a['origin'];grid=a['grid'];grid=grid[grid<=n*spec['h']+1e-14]
        query=np.clip(grid/(n*spec['h']),0,1);values=curve(query);ref,budget,reference_seconds=audit(case,origin,query,n*spec['h']);assert max(budget.values())<=.05
        actual={str(t):metrics(case,values,ref,t) for t in (1e-4,1e-6)}
        for grids in reg['grids']:
            estimate,arrays=estimate_compact(case,curve.pieces,values,query,tuple(grids));comparison={}
            for t,error in actual.items():
                accepted=estimate['status']=='success' and estimate['targets'][t]['would_accept'];true_ok=max(error.values())<=1
                comparison[t]=dict(actual_error=error,actual_error_pass=true_ok,estimator_would_accept=accepted,false_accept=bool(accepted and not true_ok))
            row=dict(path=index,model=spec['model'],steps=n,duration=n*spec['h'],full_path=summary['complete'],grids=grids,
                reference_budget=budget,reference_seconds=reference_seconds,estimate=estimate,comparison=comparison,
                scope='Error estimate only; previously failed ledger/full-path gates cannot be overridden. Linearized error carried across every piece, never reset to zero per macro.')
            results.append(row)
            if write_arrays:np.savez_compressed(OUT/f'shadow-{index}-{grids[0]}-{grids[1]}.npz',**arrays)
    return results


def run():
    sources=['research/'+p for p in ('m3_row_cumulative_shadow.py','defect_compact.py','defect_fast.py','defect_shadow.py','m3_priority.py')]
    reg=dict(date='2026-11-08',paths=[0,1,2,3],grids=[[4,8],[8,16]],
        motivation='All local strict steps can pass while global strict concentration error reaches1.465. Test existing frozen linearized cumulative defect estimator on own ROW output; no tuning of its factor4 score.',
        scope='Eight post-hoc diagnostics on previously accepted original-ROW-SPC paths/prefix. Not independent prospective validation, not replacement for full ODE audits or cumulative ledger checks.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ['results/m3-cached-row/runs.json']+[f'results/m3-cached-row/{i}-original.npz' for i in range(4)]})
    if (OUT/'shadow-registration.json').exists():raise FileExistsError('Preserve shadow evidence')
    save(OUT/'shadow-registration.json',reg);results=evaluate(reg,True);save(OUT/'shadow.json',dict(runs=results))
    for r in results:print(r['path'],r['grids'],r['estimate']['status'],r['comparison'],r['estimate']['seconds'],flush=True)


if __name__=='__main__':run()
