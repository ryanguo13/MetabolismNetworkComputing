"""Exhaustive configuration/finiteness checks for the locked output-cost study."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from metabolism.m2_experiments import save
from research.m3_shadow_study import ROOT,digest
from research.m3_ledger_flow_report import close,registration
from research import m3_audit_output_cost as study

OUT=ROOT/'results/m3-audit-output-integrity'


def nonnegative(value,label,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<=0 if positive else value<0):
        raise ValueError('Invalid finite quantity: '+label)


def check():
    reg=registration(study.OUT/'registration.json');data=study.read(study.OUT/'runs.json')
    expected={'audits':{},'controls':{}}
    for cfg in reg['cases']:
        key=cfg['reference_key']
        for ai,ac in enumerate(cfg['audit_configs']):
            for mode,rep in itertools.product(reg['query_modes'],reg['repeats']):
                name=f'audit-{key}-{ai}-{mode}'
                expected['audits'][name,rep]=dict(reference_key=key,model=cfg['model'],target=ac['target'],method=ac['method'],tolerance=ac['tolerance'],query_mode=mode)
        for method,tol,mode,rep in itertools.product(reg['control_methods'][cfg['model']],reg['control_tolerances'],reg['query_modes'],reg['repeats']):
            name=f'control-{key}-{method}-{tol}-{mode}'
            expected['controls'][name,rep]=dict(reference_key=key,model=cfg['model'],method=method,tolerance=tol,query_mode=mode)
    assert len(expected['audits'])==128 and len(expected['controls'])==480
    scalarchecks=0
    for kind in ('audits','controls'):
        rows=data[kind];actual=[(r['key'],r['repeat']) for r in rows]
        if len(actual)!=len(set(actual)) or set(actual)!=set(expected[kind]):raise ValueError('Configuration/repeat set mismatch: '+kind)
        for row in rows:
            close({k:row[k] for k in expected[kind][row['key'],row['repeat']]},expected[kind][row['key'],row['repeat']])
            nonnegative(row['seconds'],row['key']+'.seconds',positive=True);scalarchecks+=1
            peers=[r for r in rows if r['key']==row['key']]
            first=next(r for r in peers if r['repeat']==-1)
            if row['status']!=first['status']:raise ValueError('Repeat status mismatch')
            if row['status']=='failed':
                if row['reason']!=first['reason']:raise ValueError('Repeat failure reason mismatch')
                if kind=='audits' and row['qualified']:raise ValueError('Failed audit qualified')
                continue
            if row['status']!='success':raise ValueError('Unexpected status')
            for key,value in row['stats'].items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    if not math.isfinite(value):raise ValueError('Nonfinite solver stat')
                    if key.endswith('seconds'):nonnegative(value,key);scalarchecks+=1
            nonnegative(row['guard']['pool'],'pool');scalarchecks+=1
            if kind=='audits':
                for key in ('runtime_score_seconds','audit_plus_scoring_seconds','offline_qualification_seconds'):
                    nonnegative(row[key],key);scalarchecks+=1
                study.finite_metrics(row['budget']);study.finite_metrics(row['runtime_candidate_error'])
                scalarchecks+=len(row['budget'])+len(row['runtime_candidate_error'])
            else:
                for target in row['targets'].values():study.finite_metrics(target['actual']);scalarchecks+=len(target['actual'])
    # Existing verifier independently reconstructs original candidate scores,
    # formal actual errors, positivity/pools/physical-ledger gates and decisions.
    result=study.verify();close(result,study.read(study.OUT/'summary.json'))
    for path,sha in study.read(study.OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    return dict(expected_audits=128,expected_mature_controls=480,unique_configuration_repeat_sets_exact=True,
        scalar_quantities_checked=scalarchecks,all_checked_metrics_times_finite_nonnegative=True,all_solve_seconds_positive=True,
        repeat_status_failure_reasons_consistent=True,original_curve_score_structural_and_pass_reconstruction=True,
        npz_crc_checked=result['npz_crc_checked'],M4_admission=False,
        limitation='Only warmup arrays retained per configuration; repeated qualification metrics and statuses are checked against reconstructed warmup curves. Individual repeat arrays were not saved, so bytewise per-repeat output replay cannot be certified.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);args=p.parse_args()
    if args.action=='run':
        OUT.mkdir(exist_ok=True)
        if (OUT/'registration.json').exists():raise FileExistsError('Preserve integrity evidence')
        save(OUT/'registration.json',dict(date='2026-11-15',posthoc=True,
            purpose='Add exhaustive specification-by-repeat identity sets, fail-closed finite quantities and failure consistency to locked output-cost evidence; retain all original registrations and data.',
            source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
            input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [study.OUT/'registration.json',study.OUT/'runs.json',study.OUT/'summary.json',study.OUT/'manifest.json']}))
        result=check();save(OUT/'summary.json',result)
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        registration(OUT/'registration.json');result=check();close(result,study.read(OUT/'summary.json'))
        for path,sha in study.read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(result)


if __name__=='__main__':main()
