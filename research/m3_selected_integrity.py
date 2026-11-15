"""Additional integrity audit, separate from pre-registered timing sources."""
import argparse
import itertools
import json
import math
import numbers
import numpy as np
from metabolism.m2_experiments import save
from research.m3_selected_cost import OUT,read,verify
from research.m3_shadow_study import ROOT,digest


def finite_nonnegative(value):
    if isinstance(value,dict):
        for v in value.values():finite_nonnegative(v)
    elif isinstance(value,list):
        for v in value:finite_nonnegative(v)
    elif isinstance(value,numbers.Number) and not isinstance(value,bool):
        assert math.isfinite(value) and value>=0,value


def audit():
    reg=read(OUT/'registration.json');data=read(OUT/'runs.json');rows=data['runs']
    expected=set()
    specs=[('candidate',v,t,True) for v,t in itertools.product(range(2),reg['candidate_tolerances'])]
    specs += [('full',v,t,w) for v,t,w in itertools.product(range(2),reg['control_tolerances'],reg['control_whole'])]
    for index,rep,spec in itertools.product(reg['paths'],reg['repeats'],specs):
        kind,v,t,w=spec
        expected.add((f'{index}-{rep}-{kind}-{v}-{t}-{w}',index,rep,kind,v,t,w))
    actual=[(r['key'],r['path'],r['repeat'],r['kind'],r['version'],r['tolerance'],r['whole']) for r in rows]
    assert len(actual)==len(set(actual))==len(expected)==192 and set(actual)==expected
    refs=data['references'];assert len(refs)==len(reg['paths'])
    assert {r['path'] for r in refs}==set(reg['paths'])
    for r in rows:
        assert r['status'] in ('success','failed')
        assert math.isfinite(r['seconds']) and r['seconds']>0
        if r['status']=='success':
            assert math.isfinite(r['scoring_seconds']) and r['scoring_seconds']>=0
            finite_nonnegative(r['targets']);finite_nonnegative(r['guard'])
            assert r['targets'].keys()=={'0.0001','1e-06'}
            for target in r['targets'].values():
                assert isinstance(target['passes'],bool)
                assert target['actual'] is not None
        else:
            assert isinstance(r['reason'],str) and r['reason']
    for r in refs:
        finite_nonnegative(r['budgets'])
        assert all(v<=.05 for b in r['budgets'].values() for v in b.values())
        assert math.isfinite(r['audit_seconds']) and r['audit_seconds']>0
    for setup in data['setups']:finite_nonnegative(setup)
    result=verify()
    finite_nonnegative([r['median_seconds'] for c in result['comparisons'] for r in c['costs']])
    return dict(unique_registered_records=192,configuration_product_complete=True,
                finite_positive_runtime_seconds=True,finite_nonnegative_budgets_and_scores=True,
                reference_budget_limit=.05,numerical_verification=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result=audit()
    if args.write:
        if (OUT/'integrity-manifest.json').exists():raise FileExistsError('Preserve integrity evidence')
        save(OUT/'integrity.json',result)
        files=[OUT/'integrity.json',ROOT/'research/m3_selected_integrity.py',ROOT/'docs/m3-selected-reactions.md',
               ROOT/'docs/m3-next-method-review.md',OUT/'manifest.json']
        save(OUT/'integrity-manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
