"""Additional fail-closed checks without rewriting registered timing sources.

All previous candidate curves are reconstructed. Their ledgers are diagnostic:
an approximate prediction's physical ledger need not meet the final-candidate
threshold. Final-candidate gates and their stored decisions remain mandatory.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.inner_dense import InnerCurve, physical_ledger
from research.mri_kernel import GAUSS
from research.m3_ledger_flow_report import close, registration
from research.m3_shadow_study import ROOT, digest

SOURCE = ROOT/'results/m3-slow-tangent'
OUT = ROOT/'results/m3-slow-tangent-integrity'


def read(path):
    return json.loads(path.read_text())


def finite_nonnegative(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError('Invalid finite quantity: '+label)


def errors_finite(mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Missing error mapping: '+label)
    for key, value in mapping.items():
        finite_nonnegative(value, label+'.'+key)


def modified_ledger(case, curve, y, end, h, sf, sm, gradient, v0, forcing):
    weights = case['options'].get('ledger_weights')
    if weights is None:
        return {}
    js = sm@gradient; s0 = sm@v0
    totals = [0., 0.]; difference = 0.; left = 0.
    # sf columns are supplied separately so complete original rates can be used
    # independently, without calling the candidate's modified RHS.
    sf_matrix, columns = sf
    for piece in curve.pieces:
        integrals = []
        for degree in (8, 16):
            q = (GAUSS[degree][0]+1)/2
            tau = (left+piece.h*q)/h; states = piece(q)
            rates = np.column_stack([case['system'].flux(0., z)[columns] for z in states.T])
            if not np.isfinite(rates).all():
                raise ValueError('Nonfinite independently reconstructed fast rates')
            velocity = sf_matrix@rates+s0[:, None]+js@(states-y[:, None])
            velocity += forcing[:, :1]*tau**2+forcing[:, 1:]*tau**3
            integrals.append(piece.h/2*float(GAUSS[degree][1]@(weights@velocity)))
        totals[0] += integrals[0]; totals[1] += integrals[1]
        difference += abs(integrals[1]-integrals[0]); left += piece.h
    den = max(1., abs(float(weights@y)))
    return dict(modified_ledger_residual=abs(float(weights@(end-y))-totals[1])/den,
                modified_ledger_quadrature=difference/den)


def audit():
    reg = registration(SOURCE/'registration.json')
    data = read(SOURCE/'runs.json')
    refs = data['references']
    reference_by_key = {r['key']:r for r in refs}
    assert len(refs) == len(reference_by_key) == len(reg['reference_keys']) == 8
    assert set(reference_by_key) == set(reg['reference_keys'])
    expected_candidates = set(); expected_controls = set()
    for key in reg['reference_keys']:
        model = reference_by_key[key]['model']
        for threshold, sweeps, tol in itertools.product(reg['group_thresholds'][model], reg['sweeps'], reg['tolerances']):
            expected_candidates.add((f'candidate-{key}-{threshold}-{sweeps}-{tol}', key, model, threshold, sweeps, tol))
        for tol, whole in itertools.product(reg['control_tolerances'], reg['control_whole']):
            expected_controls.add((f'control-{key}-{tol}-{whole}', key, model, tol, whole))
    actual_candidates = [(r['key'],r['reference_key'],r['model'],r['threshold'],r['sweeps'],r['tolerance']) for r in data['candidates']]
    actual_controls = [(r['key'],r['reference_key'],r['model'],r['tolerance'],r['whole']) for r in data['controls']]
    assert len(actual_candidates) == len(set(actual_candidates)) == len(expected_candidates) == 96
    assert set(actual_candidates) == expected_candidates
    assert len(actual_controls) == len(set(actual_controls)) == len(expected_controls) == 48
    assert set(actual_controls) == expected_controls
    timing_count = 0
    for row in refs:
        finite_nonnegative(row['seconds'], 'reference.seconds', positive=True); timing_count += 1
        for key, values in row['budgets'].items():
            errors_finite(values, 'reference.budgets.'+key)
    assert len(data['setups']) == 8 and {r['reference_key'] for r in data['setups']} == set(reg['reference_keys'])
    for row in data['setups']:
        for key in ('classification_seconds', 'model_setup_seconds'):
            finite_nonnegative(row[key], 'setup.'+key); timing_count += 1
    failed = []; curves_checked = 0; rate_arrays_checked = 0; ledgers = []
    cases = {}
    for kind in ('candidates', 'controls'):
        for row in data[kind]:
            finite_nonnegative(row['seconds'], row['key']+'.seconds', positive=True); timing_count += 1
            if kind == 'candidates':
                finite_nonnegative(row['setup_seconds'], row['key']+'.setup_seconds'); timing_count += 1
            if row['status'] == 'failed':
                assert isinstance(row['reason'], str) and row['reason']
                failed.append(dict(kind=kind,key=row['key'],reason=row['reason']))
                continue
            assert row['status'] == 'success'
            if row['model'] not in cases:
                cases[row['model']] = prepare(row['model'])
            case = cases[row['model']]
            if 'ledger_weights' in case['options']:
                required = ['physical_ledger_residual', 'physical_ledger_quadrature']
                if kind == 'candidates':
                    required += ['modified_ledger_residual', 'modified_ledger_quadrature']
                for key in required:
                    if key not in row['stats']:
                        raise ValueError('Missing applicable ledger: '+row['key']+'.'+key)
                    finite_nonnegative(row['stats'][key], row['key']+'.'+key)
                ledger_pass = all(row['stats'][key] <= 1e-10 for key in required)
                if not ledger_pass:
                    assert all(not t['formal_pass'] and not t['local_pass'] for t in row['targets'].values())
            for name, target in row['targets'].items():
                errors_finite(target['actual'], row['key']+'.'+name+'.actual')
                errors_finite(target['local_actual'], row['key']+'.'+name+'.local_actual')
            if kind != 'candidates':
                continue
            assert row['stats']['reaction_split_coverage_eligible'] is False
            with np.load(SOURCE/f"{row['key']}.npz") as arrays:
                y = arrays['initial']; groups = arrays['groups']
                assert len(y) == row['implicit_dimension'] == row['stats']['implicit_dimension']
                columns = np.concatenate([case['system'].groups[g] for g in groups]) if len(groups) else np.array([],dtype=int)
                slow = np.setdiff1d(np.arange(case['system'].S.shape[1]), columns)
                sm = case['system'].S[:,slow]; sf = case['system'].S[:,columns]
                v0 = case['system'].flux(0.,y)
                gradient = GroupScreen(case['system'],case['scale']).gradient(y)[slow]
                assert np.isfinite(v0).all()
                close(v0, arrays['initial_rates'])
                sweeps = row['sweeps']
                for key in ('stages','rates','remainders','forcings'):
                    assert len(arrays[key]) == sweeps
                    assert np.isfinite(arrays[key]).all()
                h = reference_by_key[row['reference_key']]['h']
                for i in range(sweeps+1):
                    curve = InnerCurve.unpack(arrays[f'curve{i}_lengths'], arrays[f'curve{i}_coefficients'])
                    assert np.isfinite(curve.minima()).all() and np.all(curve.minima() >= -1e-12*case['scale'])
                    join = curve.join_error(case['scale'])
                    assert math.isfinite(join) and join <= 1e-10
                    np.testing.assert_allclose(curve(np.array([0.]))[:,0], y, rtol=2e-12, atol=1e-12)
                    end = curve(np.array([1.]))[:,0]
                    force = np.zeros((len(y),2)) if i == 0 else arrays['forcings'][i-1]
                    measured = modified_ledger(case,curve,y,end,h,(sf,columns),sm,gradient,v0[slow],force)
                    measured.update(physical_ledger(case,curve,y,end))
                    if measured:
                        errors_finite(measured, row['key']+f'.curve{i}.ledger')
                    ledgers.append(dict(key=row['key'],curve=i,final=i==sweeps,**measured))
                    if i == sweeps:
                        for key, value in measured.items():
                            close(value,row['stats'][key])
                    else:
                        stages = curve(np.array([.5,1.]))
                        rates = np.column_stack([case['system'].flux(0.,z) for z in stages.T])
                        assert np.isfinite(rates).all()
                        close(rates,arrays['rates'][i]); rate_arrays_checked += 1
                    curves_checked += 1
    return dict(date='2026-11-15',unique_candidate_records=96,unique_control_records=48,
        exact_configuration_product=True,timing_quantities_checked=timing_count,
        all_timing_quantities_finite_nonnegative=True,all_solve_seconds_positive=True,
        required_candidate_ledgers_present=True,full_stage_rate_arrays_checked=rate_arrays_checked,
        all_previous_curves_checked=curves_checked,failed_records=failed,
        reconstructed_curve_ledgers=ledgers,M4_admission=False,
        scope='Additive integrity and every-curve ledger reconstruction; previous physical ledger values are diagnostics, final original gates unchanged. No failed attempt removed, no timing rerun.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['write','verify']); args=parser.parse_args()
    if args.action == 'write':
        OUT.mkdir(exist_ok=True)
        if (OUT/'registration.json').exists():
            raise FileExistsError('Preserve independent integrity record')
        inputs = [SOURCE/'registration.json',SOURCE/'runs.json',SOURCE/'summary.json',SOURCE/'manifest.json']
        save(OUT/'registration.json',dict(date='2026-11-15',posthoc=True,
            purpose='Fail-closed ledger applicability, exact record identities, full-stage rates, previous curve positivity and ledger reconstruction; no source or timing changes.',
            source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
            input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}))
        result = audit(); save(OUT/'summary.json',result)
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        registration(OUT/'registration.json')
        result = audit(); close(result,read(OUT/'summary.json'))
        for p, sha in read(OUT/'manifest.json').items():
            assert digest(ROOT/p) == sha, p
    print(json.dumps({k:v for k,v in result.items() if k!='reconstructed_curve_ledgers'},indent=2))


if __name__ == '__main__':
    main()
