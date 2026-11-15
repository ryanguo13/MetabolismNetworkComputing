"""Explicit post-hoc scoring of saved audit curves at original formal targets.

No solve or timing is performed. This reanalysis cannot grant M4 admission.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from metabolism.m2_experiments import error,save
from metabolism.m3_continuation import prepare
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import close,registration
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-formal-audit-reanalysis'
SOURCE=ROOT/'results/m3-audit-calibration'
TRUTH=ROOT/'results/m3-remainder-curve'
TARGETS=(1e-3,1e-5)


def read(path):return json.loads(path.read_text())
def nominal(tol):return format(tol,'.10g')


def rates(case,values):
    return np.column_stack([case['system'].flux(0.,z) for z in values.T])


def formal_metrics(case,values,reference,target,value_rates,reference_rates):
    """Original concentration/one-way targets and fixed1e-3 net target."""
    result=dict(concentration=error(values,reference,case['scale'],target),
                flux=error(value_rates,reference_rates,case['options']['flux_scale'],target))
    if 'net_map' in case['options']:
        result['net_flux']=error(case['options']['net_map']@value_rates,
            case['options']['net_map']@reference_rates,case['options']['net_scale'],1e-3)
    return result


def evaluate():
    reg=registration(OUT/'registration.json');data=read(SOURCE/'runs.json');study=read(SOURCE/'registration.json')
    for path,sha in read(SOURCE/'manifest.json').items():assert digest(ROOT/path)==sha,path
    refs={r['key']:r for r in read(TRUTH/'runs.json')['references']}
    audits=[];controls=[];comparisons=[]
    for key in study['reference_keys']:
        rr=refs[key];case=prepare(rr['model'])
        with np.load(TRUTH/f'reference-{key}.npz') as a:y,q,truth=a['initial'],a['query'],a['reference']
        truth_rates=rates(case,truth)
        raw_audits=[r for r in data['audits'] if r['reference_key']==key]
        raw_controls=[r for r in data['controls'] if r['reference_key']==key]
        single_configs={str(target):{} for target in TARGETS};pair_configs={str(target):[] for target in TARGETS}
        for name in dict.fromkeys(r['key'] for r in raw_audits):
            rows=[r for r in raw_audits if r['key']==name];first=next(r for r in rows if r['repeat']==-1)
            record=dict(key=name,reference_key=key,method=first['method'],tolerance=first['tolerance'],status=first['status'])
            if first['status']=='failed':
                record['reason']=first['reason'];audits.append(record);continue
            with np.load(SOURCE/f'{name}.npz') as a:values=[a['loose'],a['tight']]
            rate_values=[rates(case,v) for v in values];guards=[structural(case,v) for v in values]
            record['guards']=guards;record['targets']={}
            timed=[r for r in rows if r['repeat']>=0]
            assert len(timed)==3 and all(r['status']=='success' for r in timed)
            for target in TARGETS:
                budget=[formal_metrics(case,v,truth,target,vf,truth_rates) for v,vf in zip(values,rate_values)]
                tightening=formal_metrics(case,values[0],values[1],target,*rate_values)
                member_good=[g['nonnegative'] and g['pool']<=1e-10 and max(b.values())<=.05 for b,g in zip(budget,guards)]
                good=all(member_good) and max(tightening.values())<=.05
                target_record=dict(budgets=budget,tightening=tightening,single_qualified=member_good,pair_qualified=good)
                record['targets'][str(target)]=target_record
                pair_configs[str(target)].append(dict(method=first['method'],loose_tolerance=first['tolerance'],qualified=good,
                    median_seconds=float(np.median([r['seconds'] for r in timed])),
                    median_solve_and_discrepancy_seconds=float(np.median([r['practical_seconds'] for r in timed]))))
                for member in (0,1):
                    tol=first['tolerance']*(study['tightening_factor'] if member else 1)
                    config=single_configs[str(target)].setdefault((first['method'],nominal(tol)),dict(method=first['method'],tolerance=float(nominal(tol)),qualified=True,observations=[],budgets=[],pair_discrepancy_seconds_context=[]))
                    config['qualified'] &= member_good[member]
                    config['budgets'].append(budget[member])
                    config['observations'].extend(r['solves'][member]['seconds'] for r in timed)
                    config['pair_discrepancy_seconds_context'].extend(r['discrepancy_seconds'] for r in timed)
            audits.append(record)
        mature_by_target={str(t):[] for t in TARGETS}
        for name in dict.fromkeys(r['key'] for r in raw_controls):
            rows=[r for r in raw_controls if r['key']==name];first=next(r for r in rows if r['repeat']==-1)
            record=dict(key=name,reference_key=key,tolerance=first['tolerance'],whole=first['whole'],status=first['status'])
            if first['status']=='failed':record['reason']=first['reason'];controls.append(record);continue
            with np.load(SOURCE/f'{name}.npz') as a:curve=InnerCurve.unpack(a['lengths'],a['coefficients']);end=a['endpoint']
            values=curve(q);guard=structural(case,values);stats=first['stats']
            assert curve.join_error(case['scale'])<=1e-10
            np.testing.assert_allclose(curve(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-12,atol=1e-12)
            nonnegative=bool(np.all(curve.minima()>=-1e-12*case['scale']))
            assert nonnegative==stats['continuous_nonnegative']
            if nonnegative:
                for name2,value in physical_ledger(case,curve,y,end).items():close(value,stats[name2])
            structural_pass=nonnegative and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
            structural_pass &= all(stats.get(k,0.)<=1e-10 for k in ('physical_ledger_residual','physical_ledger_quadrature'))
            record.update(guard=guard,structural_pass=bool(structural_pass),targets={})
            vf=rates(case,values) if nonnegative else None
            timed=[r for r in rows if r['repeat']>=0];assert len(timed)==3 and all(r['status']=='success' for r in timed)
            for target in TARGETS:
                actual=formal_metrics(case,values,truth,target,vf,truth_rates) if vf is not None else None
                good=bool(structural_pass and max(actual.values())<=1)
                record['targets'][str(target)]=dict(actual=actual,passes=good)
                mature_by_target[str(target)].append(dict(tolerance=first['tolerance'],whole=first['whole'],qualified=good,
                    median_seconds=float(np.median([r['seconds'] for r in timed])),min_seconds=min(r['seconds'] for r in timed),max_seconds=max(r['seconds'] for r in timed)))
            controls.append(record)
        for target in TARGETS:
            singles=[]
            for item in single_configs[str(target)].values():
                obs=item['observations'];contexts=item['pair_discrepancy_seconds_context']
                singles.append(dict(method=item['method'],tolerance=item['tolerance'],qualified=item['qualified'],measured_solves=len(obs),
                    median_seconds=float(np.median(obs)),min_seconds=min(obs),max_seconds=max(obs),
                    max_external_budget=max(v for b in item['budgets'] for v in b.values()),
                    median_pair_discrepancy_seconds_context=float(np.median(contexts))))
            best_single=min((r for r in singles if r['qualified']),key=lambda r:r['median_seconds'],default=None)
            pairs=pair_configs[str(target)];best_pair=min((r for r in pairs if r['qualified']),key=lambda r:r['median_seconds'],default=None)
            baselines=mature_by_target[str(target)];best_baseline=min((r for r in baselines if r['qualified']),key=lambda r:r['median_seconds'])
            comparisons.append(dict(reference_key=key,model=rr['model'],state_index=rr['state_index'],h=rr['h'],target=target,net_target=1e-3 if 'net_map' in case['options'] else None,
                single_audits=singles,pair_audits=pairs,mature_controls=baselines,best_single=best_single,best_pair=best_pair,best_baseline=best_baseline,
                single_solve_only_residual_1p5_budget=None if best_single is None else best_baseline['median_seconds']/1.5-best_single['median_seconds'],
                offline_pair_solve_residual_1p5_budget=None if best_pair is None else best_baseline['median_seconds']/1.5-best_pair['median_seconds']))
        print('posthoc formal rescored',key,flush=True)
    return dict(posthoc=True,date=reg['date'],timing_source='Original measured three development repeats; no new solves or timings',
        audit_configurations=len(audits),mature_configurations=len(controls),audits=audits,controls=controls,comparisons=comparisons,
        M4_admission=False,runtime_candidate_scoring_measured=False,
        limitation='Four known development states and .01 local windows only. Mature controls in this batch include optimized Radau only, not CVODE; positive residual is against this limited baseline set and cannot establish advantage over fastest mature solver. Single audit solve-only residual excludes candidate comparison, candidate numerical work, grouping, rollback and setup. Tolerance selection is posthoc; new prospective validation required. Original full windows/50% split/seven repeats/1.5x remain unmet.')


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve posthoc record')
    save(OUT/'registration.json',dict(date='2026-11-15',posthoc=True,targets=list(TARGETS),net_target=1e-3,reference_budget=.05,
        purpose='Explicit post-hoc rescore of already observed curves/timings at original formal concentration/one-way-flux1e-3 and1e-5 and net1e-3. Earlier development budget1e-6/net1e-4 was stricter; do not infer impossibility of formal architecture from it.',
        method='No new numerical solves, no retiming. Both audit members independently compared to immutable saved truth; pair discrepancy separately .05. Single runtime audit and offline tightening pair reported distinctly. Pool1e-10/nonnegative guards retained. Mature continuous positivity/pool/physical ledger checks unchanged. Mature baseline set is Radau-only; CVODE appears among audit backends but has no matched mature controls here, so positive residual cannot authorize promotion. Identical nominal single tolerances appearing in adjacent pairs pooled across all timed occurrences, avoiding selection of the faster duplicate.',
        sources_sha256_note='All input files and scripts frozen before reanalysis, but this registration follows observation of the experiment; it is explicitly not prospective.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'manifest.json',SOURCE/'runs.json',SOURCE/'registration.json',TRUTH/'manifest.json',ROOT/'metabolism/m3_continuation.py',ROOT/'research/m3_ordered_cost.py',ROOT/'metabolism/m2_experiments.py']}))
    result=evaluate();save(OUT/'summary.json',result)
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    compact(result)


def compact(result):
    print(json.dumps(dict(posthoc=result['posthoc'],comparisons=[{k:r[k] for k in ('reference_key','target','best_baseline','best_single','best_pair','single_solve_only_residual_1p5_budget','offline_pair_solve_residual_1p5_budget')} for r in result['comparisons']]),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);args=p.parse_args()
    if args.action=='run':run();return
    result=evaluate();close(result,read(OUT/'summary.json'))
    for path,sha in read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    compact(result)


if __name__=='__main__':main()
