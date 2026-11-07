"""Recompute saved evidence and apply the P1 stop gate without new integration."""
import hashlib
import json
from pathlib import Path

import numpy as np

from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import Hermite
from research.m3_priority import ROOT,OUT,STATES,STEPS,metrics


def main():
    datasets={name:json.loads((OUT/f'{name}.json').read_text()) for name in
              ('p0','startup','startup-early','growth','p1','p1-cached','p1-growth')}
    checks={}
    checks['all_declared_source_versions_resolve']=all(
        hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha or
        ((OUT/'source-snapshots'/f'{sha}.py').is_file() and
         hashlib.sha256((OUT/'source-snapshots'/f'{sha}.py').read_bytes()).hexdigest()==sha)
        for data in datasets.values() for path,sha in data['source_sha256'].items())
    checks['registered_48_P0_cases']=len(datasets['p0']['runs'])==48
    checks['four_startup_followups']=len(datasets['startup']['runs'])==4
    checks['four_early_layer_followups']=len(datasets['startup-early']['runs'])==4
    checks['twelve_step_growth_followups']=len(datasets['growth']['runs'])==12
    cases={name:prepare(name) for name in ('teusink','kecoli')}
    recomputed=True
    hermite_verified=True
    for name in ('p0','startup','startup-early','growth'):
        arrays=np.load(OUT/f'{name}.npz')
        for row in datasets[name]['runs']:
            if row['status']=='failed':
                continue
            case=cases[row['model']]
            if name=='p0':
                si=STATES[row['model']].index(row['start'])
                hi=STEPS[row['model']].index(row['h'])
                refkey=f"{row['model']}-{si}-{hi}"
            else:refkey=row['key']
            ref=arrays[refkey+'-reference']
            fractions=arrays[refkey+'-fractions']
            values=arrays[row['key']]
            nodes=arrays[row['key']+'-nodes']
            full=lambda y:case['system'].S@case['system'].flux(0.,y)
            deriv=[full(z) for z in nodes.T]
            pieces=[Hermite(nodes[:,i],nodes[:,i+1],deriv[i],deriv[i+1],row['h']/2) for i in (0,1)]
            mins=np.minimum(pieces[0].minima(),pieces[1].minima())
            hermite_verified &= bool(np.all(mins>=-1e-12*case['scale']))==row['stats']['continuous_nonnegative']
            for i,piece in enumerate(pieces):
                mask=(fractions<=.5) if i==0 else fractions>.5
                hermite_verified &= np.allclose(piece(2*fractions[mask]-i),values[:,mask],rtol=1e-12,atol=1e-14)
            for target,details in row.get('targets',{}).items():
                actual=metrics(case,values,ref,float(target))
                recomputed &= all(np.isclose(actual[k],v,rtol=1e-9,atol=1e-9) for k,v in details['actual'].items())
        if name=='p0':
            for reference in datasets[name]['references']:
                case=cases[reference['key'].split('-')[0]]
                actual=metrics(case,arrays[reference['key']+'-loose'],arrays[reference['key']+'-reference'],1e-6)
                recomputed &= all(np.isclose(actual[k],v,rtol=1e-9,atol=1e-9) for k,v in reference['tightening'].items())
    checks['saved_actual_errors_and_P0_reference_budgets_recompute']=bool(recomputed)
    checks['Hermite_curves_and_internal_extrema_recompute']=bool(hermite_verified)
    all_local=[r for name in ('p0','startup','startup-early','growth') for r in datasets[name]['runs']]
    checks['all_reference_budgets_pass']=all(r['reference_pass'] for r in all_local)
    checks['both_models_all_registered_starts_have_tight_local_candidate']=all(
        any(r['model']==model and r['start']==start and r.get('targets',{}).get('1e-06',{}).get('passes')
            for r in all_local) for model in STATES for start in STATES[model])
    checks['Hermite_negative_trials_rejected']=all(r['status']=='rejected'
        for r in all_local if r['stats'].get('continuous_nonnegative') is False)
    checks['accepted_open_ledger_checks_pass']=all(
        max(r['stats'][k] for k in ('modified_ledger_residual','modified_ledger_quadrature',
                                    'physical_ledger_residual','physical_ledger_quadrature'))<=1e-10
        for r in all_local if r['model']=='teusink' and any(v['passes'] for v in r.get('targets',{}).values()))
    costs=[]
    for dataset in ('p1-cached','p1-growth'):
        runs=datasets[dataset]['runs']
        for model in {r['model'] for r in runs}:
            baseline=[r for r in runs if r['model']==model and r['variant']=='full-local' and r['repeat']>=0]
            assert len(baseline)==3 and all(r['passes'] for r in baseline)
            for r in baseline:
                assert r['stats']['max_committed_pool_error'] <= 1e-10
                if model=='teusink':
                    denominator=max(1.,abs(float(cases[model]['options']['ledger_weights']@cases[model]['model'].y0)))
                    assert abs(r['stats']['ledger_residual'])/denominator<=1e-10
            base=float(np.median([r['seconds'] for r in baseline]))
            for variant in sorted({r['variant'] for r in runs if r['model']==model and r['variant']!='full-local'}):
                selected=[r for r in runs if r['model']==model and r['variant']==variant and r['repeat']>=0]
                assert len(selected)==3
                times=[r['seconds'] for r in selected]
                costs.append(dict(dataset=dataset,model=model,variant=variant,h=selected[0]['h'],
                    local_accuracy_pass=all(r.get('targets',{}).get('0.0001',{}).get('passes',False) for r in selected),
                    median=float(np.median(times)),iqr=float(np.ptp(np.quantile(times,[.25,.75]))),
                    kernel_median=float(np.median([r['stats']['seconds'] for r in selected])),
                    full_local_median=base,cost_ratio=float(np.median(times))/base))
    checks['three_interleaved_development_repeats_and_qualified_controls']=True
    checks['no_measured_P1_cost_prospect']=not any(r['local_accuracy_pass'] and r['cost_ratio']<1 for r in costs)
    tests=json.loads((OUT/'tests.json').read_text())
    checks['kernel_checks_pass']=tests['returncode']==0 and tests['sha256']==hashlib.sha256((ROOT/'research/test_mri_kernel.py').read_bytes()).hexdigest()
    assert all(checks.values()),checks
    summary=dict(checks=checks,P0='local_candidates_verified_with_failures_retained',P1='stop_no_measured_cost_prospect',
        P2='not_started_due_to_P1_gate',P3='not_started_due_to_P1_gate',M4='not_admitted',
        local_attempts=len(all_local),negative_curve_rejections=sum(r['status']=='rejected' for r in all_local),
        cases=[dict(model=model,stage=name,total=len([r for r in datasets[name]['runs'] if r['model']==model]),
            loose_pass=sum(r.get('targets',{}).get('0.0001',{}).get('passes',False) for r in datasets[name]['runs'] if r['model']==model),
            tight_pass=sum(r.get('targets',{}).get('1e-06',{}).get('passes',False) for r in datasets[name]['runs'] if r['model']==model))
            for name in ('p0','startup','startup-early','growth') for model in ('teusink','kecoli')],costs=costs,
        limits=['Local development data, no full-window or speedup qualification.',
                'No proof of global estimator reliability; no online acceptance controller.',
                'Polynomial extrema checked in float64, not a rigorous interval-arithmetic positivity proof.',
                'Startup/growth reference tightening was checked online; only P0 saves both reference tolerances.',
                'P1 timings are three development repeats; full-only native wrapper does not expose all internal accepted states.',
                'KLU allocation reuse does not preserve symbolic analysis across CVODE reinitialization.'])
    save(OUT/'summary.json',summary)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json')
    paths += sorted(ROOT.glob('research/mri_*.py'))+sorted(ROOT.glob('research/m3_*.py'))+[ROOT/'research/test_mri_kernel.py']
    save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
