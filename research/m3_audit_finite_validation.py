"""Fail-closed finite-value audit before exposing posthoc cost rankings.

Additive validation: neither prior registration nor its source is rewritten.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np
from metabolism.m2_experiments import save
from research.m3_shadow_study import ROOT,digest
from research.m3_ledger_flow_report import registration,close
from research import m3_formal_audit_reanalysis as formal

OUT=ROOT/'results/m3-audit-finite-validation'
SOURCE=formal.SOURCE


def nonnegative_finite(value,path,strict=False):
    if isinstance(value,dict):
        for k,v in value.items():nonnegative_finite(v,path+'.'+k,strict)
    elif isinstance(value,list):
        for i,v in enumerate(value):nonnegative_finite(v,path+f'[{i}]',strict)
    elif isinstance(value,bool) or value is None or isinstance(value,str):return
    elif isinstance(value,Real):
        if not math.isfinite(value) or (value<=0 if strict else value<0):
            raise ValueError(f'Invalid nonnegative finite quantity: {path}')
    else:raise TypeError(path)


def validate_raw():
    data=formal.read(SOURCE/'runs.json');metrics=0;timings=0
    assert len(data['audits'])==336 and len(data['controls'])==128
    for row in data['audits']:
        nonnegative_finite(row['seconds'],row['key']+'.seconds',strict=True);timings+=1
        if row['status']=='success':
            for key in ('budgets','tightening','guards'):
                nonnegative_finite(row[key],row['key']+'.'+key)
            metrics+=sum(len(b) for b in row['budgets'])+len(row['tightening'])
            for member in row['solves']:
                nonnegative_finite(member['seconds'],row['key']+'.member.seconds',strict=True);timings+=1
                nonnegative_finite(member,row['key']+'.member')
            for key in ('discrepancy_seconds','practical_seconds','qualification_scoring_seconds'):
                nonnegative_finite(row[key],row['key']+'.'+key);timings+=1
        else:assert not row['qualified']
    for row in data['controls']:
        nonnegative_finite(row['seconds'],row['key']+'.seconds',strict=True);timings+=1
        if row['status']=='success':
            for key in ('stats','guard','targets'):nonnegative_finite(row[key],row['key']+'.'+key)
    for row in data['setups']:nonnegative_finite(row['seconds'],'setup.seconds');timings+=1
    arrays=0
    for path in SOURCE.glob('*.npz'):
        with np.load(path) as a:
            for name in a.files:
                if not np.isfinite(a[name]).all():raise ValueError(f'Nonfinite stored array: {path.name}:{name}')
                arrays+=1
    return dict(audit_rows_checked=len(data['audits']),mature_rows_checked=len(data['controls']),
        audit_budget_indicators_checked=metrics,timing_quantities_checked=timings,stored_arrays_checked=arrays,
        all_metric_indicators_finite_nonnegative=True,all_timing_quantities_finite_nonnegative=True,all_solve_seconds_strictly_positive=True)


def evaluate():
    reg=registration(OUT/'registration.json')
    # These checks precede every invocation of the posthoc ranking routine.
    checks=validate_raw()
    result=formal.evaluate();close(result,formal.read(formal.OUT/'summary.json'))
    comparisons=[]
    for row in result['comparisons']:
        for key in ('single_audits','pair_audits','mature_controls'):nonnegative_finite(row[key],row['reference_key']+'.'+key)
        comparisons.append({k:row[k] for k in ('reference_key','model','target','best_single','best_pair','best_baseline','single_solve_only_residual_1p5_budget','offline_pair_solve_residual_1p5_budget')})
    return dict(date=reg['date'],checks=checks,comparisons=comparisons,M4_admission=False,
        scope='Additive validation of locked calibration and posthoc reanalysis; all finite/positive checks run before ranking. Current stored data pass, so previous local numeric classifications remain unchanged.',
        limitations='Audit curves have only sampled nonnegative/pool guards, not internal RHS/continuous/physical-ledger certification; their times are optimistic cost bounds. Mature controls are optimized Radau only, without CVODE controls. Single solve budgets omit candidate scoring and numerical work. Local known-state posthoc results cannot grant M4.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);args=p.parse_args()
    if args.action=='run':
        OUT.mkdir(exist_ok=True)
        if (OUT/'registration.json').exists():raise FileExistsError('Preserve validation')
        save(OUT/'registration.json',dict(date='2026-11-15',posthoc=True,
            purpose='Explicit fail-closed validation of every saved budget/tightening component and timing quantity before formal posthoc ranking. No tolerance or prior evidence changes.',
            source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
            input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'manifest.json',SOURCE/'runs.json',formal.OUT/'manifest.json',formal.OUT/'summary.json',ROOT/'research/m3_formal_audit_reanalysis.py']}))
        result=evaluate();save(OUT/'summary.json',result)
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        result=evaluate();close(result,formal.read(OUT/'summary.json'))
        for path,sha in formal.read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
