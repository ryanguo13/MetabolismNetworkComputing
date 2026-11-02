"""Audit stored M1 evidence and freeze the permitted M2 scope."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/m1-completion'
    def read(path):
        return json.loads((root / path).read_text(encoding='utf-8'))
    def digest(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    manifest = read('models/manifest.json')
    hash_ok = all((root / 'models' / entry['file']).exists() and
                  hashlib.sha256((root / 'models' / entry['file']).read_bytes()).hexdigest() == entry['sha256']
                  for entry in manifest)
    small = [read(f'results/m1/{mid}-audit.json') for mid in ('BIOMD0000000064', 'BIOMD0000000051')]
    large = read('results/m1-completion/kecoli-audit.json')
    net = read('results/m1-completion/kecoli-net-flux.json')
    curation = read('results/m1-completion/chassagnole-curation.json')
    balance = read('results/m1-completion/open-balance.json')
    structure = read('results/m1-completion/reaction-structure.json')
    large_structure = read('results/m1-completion/large-reaction-structure.json')
    summaries = read('results/m1-completion/kecoli-summary.json')
    large_runs = read('results/m1-completion/kecoli-runs.json')
    calibration = read('results/m1-completion/strict-calibration-runs.json')
    configs = []
    for method in ('BDF', 'Radau', 'CVODE-KLU'):
        candidates = []
        for row in summaries:
            flux = next(r for r in net['configs'] if r['method'] == row['method'] and r['rtol'] == row['rtol'])
            repeats = [r for r in large_runs if r['method'] == row['method'] and r['rtol'] == row['rtol']]
            if (row['method'] == method and row['successes'] == 7 and row['failures'] == 0
                    and row['concentration_error']['1e-05'] <= 1
                    and row['elementary_flux_error']['1e-05'] <= 1
                    and flux['net_flux_error']['0.001'] <= 1
                    and row['max_pool_drift'] / large['pool_normalization'] <= 1e-10
                    and all(r['significant_negative_output_count'] == 0 for r in repeats)):
                candidates.append(row)
        if candidates:
            best = min(candidates, key=lambda row: row['median_seconds'])
            configs.append({'method': method, 'rtol': best['rtol'], 'median_seconds': best['median_seconds'],
                            'concentration_target': 1e-5, 'elementary_flux_target': 1e-5, 'net_flux_target': 1e-3,
                            'reference': 'Frozen development configuration, not held-out validation'})
    strict_small = all(any(r['model_id'] == mid and r['method'] == method and r['status'] == 'success'
                           and r['concentration_error_at_1e7'] <= 1 and r['flux_error_at_1e7'] <= 1 for r in calibration)
                       for mid in ('BIOMD0000000064', 'BIOMD0000000051') for method in ('BDF', 'Radau', 'CVODE'))
    checks = {
        'pinned_public_artifacts_verified': hash_ok,
        'two_public_models_independent_rhs_semantics': all(a['semantic_checks']['rhs_max_scaled_difference'] < 1e-10 for a in small),
        'two_public_models_curated_reproduction': small[0]['curated_table4_steady_state']['within_0.02_mM'] and curation['pass'],
        'small_concentration_and_flux_1e7_calibrated': strict_small and all(max(v for row in read(f'results/m1-completion/{mid}-strict-reference.json')['checks'].values() for v in row.values()) < .05 for mid in ('BIOMD0000000064', 'BIOMD0000000051')),
        'medium_or_larger_source_and_original_executor': large['dimension'] >= 50 and large['source_initial_condition_max_difference'] == 0 and large['original_rhs_max_absolute_difference'] < 1e-9 and large['original_jacobian_max_scaled_difference'] < 1e-8,
        'large_reference_concentration_and_elementary_flux': large['status'] == 'pass',
        'large_original_net_flux_reference_and_source_steady_state': net['reference_pass'],
        'three_sparse_baselines_have_valid_configs': len(configs) == 3,
        'all_repeats_saved_with_memory_and_costs': len(large_runs) == 84 and all(r['status'] == 'success' and r['isolated_worker_peak_rss_bytes'] > 0 and r['counts_and_costs']['rhs_seconds'] > 0 for r in large_runs),
        'open_system_ledger': balance['pass'],
        'local_fast_modes_exist_on_multiple_states': all(any(c['h'] == 1e-5 and c['updated_state_count'] < 14 and c['spectral_radius_reduction'] > 50 for c in s['classifications']) for s in structure['snapshots']),
        'large_complete_reversible_groups_and_separable_modes': large_structure['groups'] == 2666 and all(any(c['h'] == 1e-5 and c['updated_states'] < 300 and c['largest_mode_reduction'] > 100 for c in s['classifications']) for s in large_structure['snapshots']),
        'nonautonomous_model_explicitly_excluded_from_autonomous_split': small[1]['time_dependent'] and 'blocked' in small[1]['split_scope'],
    }
    evidence_paths = [str(path.relative_to(root)).replace('\\', '/') for path in sorted(out.glob('*.json'))
                      if path.name not in {'acceptance.json', 'case-registration.json'}]
    evidence_paths += ['results/m1-completion/kecoli-reference.npz', 'results/m1-completion/kecoli-octave.mat', 'results/m1-completion/kecoli-source-inputs.npz']
    registration = {'date': '2026-11-02 Asia/Shanghai', 'seed': 20261102,
                    'models': [{'id': 'BIOMD0000000064', 'autonomous_split': True, 'window': [0, 10], 'time_unit': 'min',
                                'reference': 'results/m1/BIOMD0000000064-reference.npz', 'concentration_and_flux_targets': [1e-3, 1e-5, 1e-7]},
                               {'id': 'BIOMD0000000051', 'autonomous_split': False, 'window': [0, 20], 'curation_window': [0, 40], 'time_unit': 's',
                                'reason': 'Seven explicit time-dependent cofactor rules; retained unchanged for full-system validation'},
                               {'id': 'k-ecoli457', 'autonomous_split': True, 'window': [0, 1], 'time_unit': 'source coordinate',
                                'case': 'Original aerobic glucose wrapper, ACALD 2x overexpression',
                                'reference': 'results/m1-completion/kecoli-reference.npz', 'concentration_targets': [1e-3, 1e-5],
                                'elementary_flux_targets': [1e-3, 1e-5], 'net_flux_targets': [1e-2, 1e-3],
                                'event_policy': large['event_policy']}],
                    'grid_policy': 'Use saved reference times and species scales, plus the previously audited 2x dense small-model grid; no per-method output-grid or scaling adjustment',
                    'future_holdout': {'status': 'registered, not executed in M1', 'rng': 'NumPy PCG64 seed 20261102',
                                       'cases': 20, 'parameter_factors': 'log-uniform [0.9,1.1], one common multiplier per complete forward/reverse group; preserve equilibrium ratios',
                                       'initial_conditions': 'nonnegative, keep registered invariant pools; no observation noise in ODE'},
                    'frozen_large_baselines': configs,
                    'allowed_M2_work': ['B0 linear reversible chains and enzyme mechanism: Lie/Strang exact-flow checks',
                                       'Fixed complete-reaction splitting without PE on admitted autonomous models',
                                       'Measure restart and fast-block costs before any M3 efficiency claim'],
                    'excluded': ['Chassagnole autonomous splitting', 'PE/QSSA', 'Claimed performance gains', '93-metabolite core model not obtained']}
    save = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    save(out / 'case-registration.json', registration)
    status = {'M1': 'complete' if all(checks.values()) else 'incomplete',
              'M2_admission': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
              'scope': registration['allowed_M2_work'], 'limits': registration['excluded'],
              'evidence_sha256': {path: digest(path) for path in evidence_paths},
              'registration_sha256': digest('results/m1-completion/case-registration.json')}
    save(out / 'acceptance.json', status)
    print(json.dumps({'M1': status['M1'], 'M2_admission': status['M2_admission'], 'checks': checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
