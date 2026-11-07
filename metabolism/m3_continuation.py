"""Mac M3 continuation experiments, separate from frozen historical evidence.

bash scripts/run-macos.sh -m metabolism.m3_continuation --case probes
bash scripts/run-macos.sh -m metabolism.m3_continuation --case baselines
bash scripts/run-macos.sh -m metabolism.m3_continuation --case check
"""

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy

from .adaptive import AdaptiveConfig, AdaptiveSplit, GroupScreen
from .continuation import ContinuationConfig, MacroSplit
from .m2_experiments import digest, error, save
from .splitting import FixedSplit, ReactionSystem, SplitFailure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m3-continuation'
TARGETS = (1e-3, 1e-5)


def prepare(name):
    tick = time.perf_counter()
    large = name == 'kecoli'
    if large:
        from .kecoli import KEcoliModel
        model = KEcoliModel(ROOT / 'models/k-ecoli457.zip')
        system = ReactionSystem.from_kecoli(model)
        path = ROOT / 'results/m1-completion/kecoli-reference.npz'
        data = np.load(path)
        flux_scale = data['flux_scale']
        net_scale = np.array(json.loads((ROOT / 'results/m1-completion/kecoli-net-flux.json').read_text())['net_flux_scale'])
        options = dict(pools=model.pools, net_map=model.net_map, net_scale=net_scale, net_target=1e-3)
    else:
        from .sbml import SBMLModel
        model = SBMLModel(ROOT / 'models/BIOMD0000000064.xml')
        system = ReactionSystem.from_sbml(model)
        path = ROOT / 'results/m1/BIOMD0000000064-reference.npz'
        data = np.load(path)
        flux_reference = np.column_stack([model.flux(t, y) for t, y in zip(data['times'], data['concentrations'].T)])
        flux_scale = np.maximum(np.max(abs(flux_reference), axis=1), 1e-6)
        options = dict(pools=model.L.T,
                       ledger_weights=np.array([6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 2, 0, 21, 21], dtype=float))
    options['flux_scale'] = flux_scale
    reference_flux = np.column_stack([model.flux(t, y) for t, y in zip(data['times'], data['concentrations'].T)])
    return dict(model=model, system=system, times=data['times'], reference=data['concentrations'],
                scale=data['scale'], flux_reference=reference_flux, options=options,
                reference_path=path, setup_seconds=time.perf_counter()-tick)


def metrics(values, case, target, stats):
    model, options = case['model'], case['options']
    flux = np.column_stack([model.flux(t, y) for t, y in zip(case['times'], values.T)])
    initial = options['pools'] @ model.y0
    result = dict(concentration_error=error(values, case['reference'], case['scale'], target),
                  flux_error=error(flux, case['flux_reference'], options['flux_scale'], target),
                  invariant_error=float(np.max(abs(options['pools'] @ (values-model.y0[:, None])))) /
                      max(1e-12, float(np.max(abs(initial)))),
                  significant_negative_output=bool(np.any(values < -1e-12*case['scale'][:, None])),
                  finite=bool(np.isfinite(values).all()),
                  relative_ledger_error=abs(stats.get('ledger_residual', 0.)) /
                      max(1., abs(float(options['ledger_weights'] @ model.y0))) if 'ledger_weights' in options else 0.)
    if 'net_map' in options:
        result['net_flux_error'] = error(options['net_map'] @ flux, options['net_map'] @ case['flux_reference'], options['net_scale'], 1e-3)
    result['passes'] = (max(result['concentration_error'], result['flux_error'], result.get('net_flux_error', 0.)) <= 1
                        and result['invariant_error'] <= 1e-10 and result['relative_ledger_error'] <= 1e-10
                        and result['finite'] and not result['significant_negative_output'])
    return result


def measure(case, config, identity, repeat, fixed=None, original=False):
    tick = time.perf_counter()
    row = dict(config=identity, parameters=asdict(config), repeat=repeat)
    try:
        cls = AdaptiveSplit if original else MacroSplit
        controller = cls(case['system'], case['scale'], config, fixed_groups=fixed, **case['options'])
        values, stats = controller.integrate(case['times'])
        row.update(status='success', seconds=time.perf_counter()-tick, stats=stats)
        row['metrics'] = metrics(values, case, config.target, stats)
        return row, values
    except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
        row.update(status='failed', seconds=time.perf_counter()-tick, reason=str(exc), stats=getattr(exc, 'stats', None))
        return row, None


def registration(case):
    return dict(model_sha256=digest(case['model'].path),
                reference_path=case['reference_path'].relative_to(ROOT).as_posix(),
                reference_sha256=digest(case['reference_path']),
                window=[float(case['times'][0]), float(case['times'][-1])],
                output_count=len(case['times']), setup_seconds=case['setup_seconds'])


def source_fingerprints():
    paths = sorted((ROOT / 'tests').rglob('test_*.py')) + sorted((ROOT / 'checks').glob('test_*.py'))
    paths += [ROOT / 'metabolism' / name for name in (
        '__init__.py', 'adaptive.py', 'splitting.py', 'cvode.py', 'sbml.py', 'kecoli.py',
        'continuation.py', 'm3_continuation.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def probes():
    rows, saved, registrations = [], {}, {}
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        registrations[name] = registration(case)
        saved[name] = dict(times=case['times'], reference=case['reference'], scale=case['scale'])
        large = name == 'kecoli'
        for target in TARGETS:
            base = ContinuationConfig(target=target, initial_step=1e-3, max_step=.1 if large else 1.,
                                       max_attempts=512, max_rejections=16, timeout=60., group_every=1,
                                       audit_method='CVODE-KLU' if large else 'Radau',
                                       audit_rtol=1e-11 if large else 1e-12,
                                       fallback_method='CVODE-KLU' if large else 'BDF',
                                       fallback_rtol=1e-11 if large else 1e-12)
            # Frozen before running: separate grid, startup, grouping threshold,
            # and grouping-frequency changes; no per-result tolerance relaxation.
            configs = [
                ('macro-dynamic-1', base),
                ('startup-dynamic-1', replace(base, startup_end=1e-3)),
                ('startup-dynamic-5', replace(base, startup_end=1e-3, group_every=5)),
                ('startup-promoted-fast', replace(base, startup_end=1e-3, enter=.01, leave=.003)),
            ]
            for identity, config in configs:
                key = f'{identity}-{target}'
                row, values = measure(case, config, key, 0)
                row['model'] = name
                rows.append(row)
                if values is not None:
                    saved[name][key] = values
                save(OUT / 'probes.json', dict(registration=registrations, runs=rows,
                     scope='Single-run development probes; not repeated performance evidence. All attempts and failures retained.'))
                for model_name, arrays in saved.items():
                    np.savez_compressed(OUT / f'{model_name}-probes.npz', **arrays)
                print(name, key, row['status'], round(row['seconds'], 3),
                      row.get('metrics', {}).get('passes'),
                      (row.get('stats') or {}).get('split_coverage'), flush=True)


def baselines():
    """Calibrate then interleave seven repeats of each fastest valid method.

    All methods use the same diagnostics wrapper, model RHS, output grid and
    scale. The small model uses BDF/Radau; large also uses same-RHS CVODE-KLU.
    """
    calibration, runs, registrations, saved = [], [], {}, {}
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        registrations[name] = registration(case)
        saved[name] = dict(times=case['times'], reference=case['reference'], scale=case['scale'])
        configs = []
        for method in (('BDF', 'Radau', 'CVODE-KLU') if name == 'kecoli' else ('BDF', 'Radau')):
            candidates = []
            for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-11):
                config = AdaptiveConfig(target=1e-5, initial_step=1e-5, fallback_method=method,
                                        fallback_rtol=tol, timeout=60.)
                identity = f'{method}-{tol}'
                row, values = measure(case, config, identity, -2,
                                      fixed=np.arange(len(case['system'].groups)), original=True)
                row['model'] = name
                if values is not None:
                    row['target_metrics'] = {str(target): metrics(values, case, target, row['stats']) for target in TARGETS}
                calibration.append(row)
                candidates.append(row)
                save(OUT / 'baseline-calibration.json', calibration)
                print('calibrate', name, identity, row['status'], round(row['seconds'], 3), flush=True)
            for target in TARGETS:
                good = [r for r in candidates if r['status'] == 'success' and r['target_metrics'][str(target)]['passes']]
                if not good:
                    continue
                selected = min(good, key=lambda r: r['seconds'])
                config = AdaptiveConfig(**selected['parameters'])
                configs.append((f'{method}-{target}', replace(config, target=target)))
        for repeat in range(-1, 7):
            shift = max(0, repeat) % max(1, len(configs))
            for identity, config in configs[shift:]+configs[:shift]:
                row, values = measure(case, config, identity, repeat,
                                      fixed=np.arange(len(case['system'].groups)), original=True)
                row['model'] = name
                runs.append(row)
                if repeat == 0 and values is not None:
                    saved[name][identity] = values
                save(OUT / 'baselines.json', dict(registration=registrations, runs=runs,
                     repeat_policy='Separate calibration; one warmup, seven cyclically interleaved repeats per configuration.'))
                for model_name, arrays in saved.items():
                    np.savez_compressed(OUT / f'{model_name}-baselines.npz', **arrays)
                print('baseline', name, identity, repeat, row['status'], round(row['seconds'], 3), flush=True)


def record_tests():
    rows, logs = [], []
    for directory in ('tests', 'checks'):
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-v']
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rows.append(dict(command=command, returncode=result.returncode, source_sha256=source_fingerprints()))
        logs.extend(['$ '+' '.join(command), result.stdout])
    save(OUT / 'tests.json', rows)
    (OUT / 'tests.txt').write_text('\n'.join(logs), encoding='utf-8')
    if any(r['returncode'] for r in rows):
        raise SystemExit('Continuation tests failed')


def inner_sensitivity():
    """Isolate cancellation and inner tolerances with identical frozen maps."""
    from .cvode import solve
    case = prepare('kecoli')
    model, scale = case['model'], case['scale']
    prefix, prefix_counts = solve(model, np.array([0., .001]), 1e-11, 1e-13 * scale)
    state = prefix[:, -1]
    kappa, _ = GroupScreen(case['system'], scale).evaluate(state)
    groups = np.flatnonzero(1e-3*kappa >= .01)
    local_model = SimpleNamespace(y0=state, rhs=model.rhs, jacobian=model.jacobian, sparsity=model.sparsity)
    arrays = dict(state=state, scale=scale, groups=groups)
    rows, references = [], []

    def errors(values, reference):
        vf = np.column_stack([model.flux(0., y) for y in values.T])
        vr = np.column_stack([model.flux(0., y) for y in reference.T])
        return dict(concentration=error(values, reference, scale, 1e-4),
                    flux=error(vf, vr, case['options']['flux_scale'], 1e-4),
                    net_flux=error(model.net_map @ vf, model.net_map @ vr, case['options']['net_scale'], 1e-4))

    for hi, h in enumerate((1e-4, 1e-6, 3.0517578125e-8)):
        query = h * np.array([.25, .5, .75, 1.])
        reference, counts = solve(local_model, np.r_[0., query], 1e-11, 1e-13*scale)
        tighter, tight_counts = solve(local_model, np.r_[0., query], 1e-12, 1e-14*scale)
        ref = tighter[:, 1:]
        references.append(dict(h=h, reference_tightening=errors(reference[:, 1:], ref),
                               counts=counts, tighter_counts=tight_counts))
        arrays[f'reference-{hi}'] = ref
        arrays[f'reference-loose-{hi}'] = reference[:, 1:]
        for tol in (1e-8, 1e-10, 1e-12):
            tick = time.perf_counter()
            try:
                partition = FixedSplit(case['system'], groups, scale, rtol=tol, pools=model.pools, timeout=30.)
                _, fine, stats = partition.integrate(state, 0., h, 2)
                values = np.vstack([np.interp(query, [0., h/2, h], row) for row in fine])
                rows.append(dict(h=h, inner_rtol=tol, status='success', seconds=time.perf_counter()-tick,
                    interpolation_errors=errors(values, ref), endpoint_errors=errors(fine[:, -1:], ref[:, -1:]), stats=stats))
                arrays[f'fine-{hi}-{tol}'] = fine
            except SplitFailure as exc:
                rows.append(dict(h=h, inner_rtol=tol, status='failed', seconds=time.perf_counter()-tick,
                                 reason=str(exc), stats=exc.stats))
            save(OUT / 'inner-sensitivity.json', dict(registration=registration(case),
                scope='Single local sensitivity, identical frozen groups and h; prefix computed from original initial conditions, not loaded from a reference state. Errors in local 1e-4 concentration/flux/net-flux units; not a timing or full-window qualification.',
                prefix_counts=prefix_counts, fast_groups=groups.tolist(), references=references, runs=rows))
            np.savez_compressed(OUT / 'inner-sensitivity.npz', **arrays)
            print('inner-sensitivity', h, tol, rows[-1]['status'], rows[-1].get('endpoint_errors'), flush=True)


def repeat_candidates():
    """Spend formal repeats only on probes with a measured efficiency prospect."""
    probes_data = json.loads((OUT / 'probes.json').read_text())
    baseline_data = json.loads((OUT / 'baselines.json').read_text())
    selected, skipped, runs = [], [], []
    for name in ('teusink', 'kecoli'):
        for target in TARGETS:
            timings = {}
            for row in baseline_data['runs']:
                if (row['model'] == name and row['parameters']['target'] == target and row['repeat'] >= 0
                        and row['status'] == 'success' and row['metrics']['passes']):
                    timings.setdefault(row['config'], []).append(row['seconds'])
            qualified = [float(np.median(v)) for v in timings.values() if len(v) == 7]
            fastest = min(qualified) if qualified else None
            prospects = [r for r in probes_data['runs'] if r['model'] == name and r['parameters']['target'] == target
                         and r['status'] == 'success' and r['metrics']['passes']
                         and r['stats']['split_coverage'] >= .5 and fastest is not None and r['seconds'] < fastest]
            if prospects:
                selected.append(min(prospects, key=lambda r: r['seconds']))
            else:
                skipped.append(dict(model=name, target=target,
                    reason='No accurate probe with at least half-window split coverage is faster than a qualified repeated baseline.'))
    for chosen in selected:
        case = prepare(chosen['model'])
        config = ContinuationConfig(**chosen['parameters'])
        for repeat in range(-1, 7):
            row, values = measure(case, config, chosen['config'], repeat)
            row['model'] = chosen['model']
            runs.append(row)
            if repeat == 0 and values is not None:
                np.savez_compressed(OUT / f"{chosen['model']}-{config.target}-candidate.npz", values=values)
            print('candidate', chosen['model'], row['config'], repeat, row['status'], row['seconds'], flush=True)
    save(OUT / 'candidates.json', dict(selection_rule='Accurate single probe, split coverage >= 0.5, faster than qualified repeated baseline; select the fastest prospect per model/target, then warmup + 7 repeats.',
         selected=[dict(model=r['model'], config=r['config'], target=r['parameters']['target']) for r in selected],
         skipped=skipped, runs=runs))


def audit():
    probes_data = json.loads((OUT / 'probes.json').read_text())
    baselines_data = json.loads((OUT / 'baselines.json').read_text())
    calibration = json.loads((OUT / 'baseline-calibration.json').read_text())
    checks = dict(complete_16_development_probes=len(probes_data['runs']) == 16,
                  complete_25_baseline_calibrations=len(calibration) == 25)
    summaries = []
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        for label, records in (('probes', probes_data), ('baselines', baselines_data)):
            data = np.load(OUT / f'{name}-{label}.npz')
            reg = records['registration'][name]
            checks[f'{name}_{label}_original_model_grid_scale_reference'] = bool(
                digest(case['model'].path) == reg['model_sha256'] and digest(case['reference_path']) == reg['reference_sha256']
                and np.array_equal(data['times'], case['times']) and np.array_equal(data['reference'], case['reference'])
                and np.array_equal(data['scale'], case['scale']))
            relevant = [r for r in records['runs'] if r['model'] == name]
            checks[f'{name}_{label}_finite_failures_recorded'] = all(r['status'] in ('success', 'failed')
                and np.isfinite(r['seconds']) and (r['status'] == 'success' or bool(r.get('reason'))) for r in relevant)
            recomputed = True
            for row in (r for r in relevant if r['repeat'] == 0 and r['status'] == 'success'):
                value = metrics(data[row['config']], case, row['parameters']['target'], row['stats'])
                recomputed &= all(np.isclose(value[k], v, rtol=1e-9, atol=1e-10) for k, v in row['metrics'].items())
            checks[f'{name}_{label}_saved_metrics_recompute'] = bool(recomputed)
        for target in TARGETS:
            full = [r for r in baselines_data['runs'] if r['model'] == name and r['repeat'] >= 0 and r['parameters']['target'] == target]
            eligible = []
            for identity in sorted({r['config'] for r in full}):
                selected = [r for r in full if r['config'] == identity]
                if len(selected) == 7 and all(r['status'] == 'success' and r['metrics']['passes'] for r in selected):
                    values = [r['seconds'] for r in selected]
                    eligible.append(dict(config=identity, median=float(np.median(values)),
                                         iqr=float(np.ptp(np.quantile(values, [.25, .75])))))
            checks[f'{name}_{target}_repeated_qualified_strong_baseline'] = bool(eligible)
            fastest = min(eligible, key=lambda r: r['median']) if eligible else None
            for r in (r for r in probes_data['runs'] if r['model'] == name and r['parameters']['target'] == target):
                st = r.get('stats') or {}
                summaries.append(dict(model=name, target=target, config=r['config'], status=r['status'],
                    seconds=r['seconds'], passes=r.get('metrics', {}).get('passes', False),
                    split_coverage=st.get('split_coverage', 0.), accepted_steps=st.get('accepted_steps', 0),
                    rejected_steps=st.get('rejected_steps', 0), audit_seconds=st.get('audit_seconds', 0.),
                    split_seconds=st.get('split_seconds', 0.), full_intervals=st.get('full_intervals', []),
                    fastest_baseline=fastest,
                    exploratory_baseline_over_probe=fastest['median']/r['seconds'] if fastest else None))
    successful = [r for r in probes_data['runs'] if r['status'] == 'success']
    checks['accepted_splits_all_audited'] = bool(successful) and all(
        max(a['audit_max'], a['audit_flux_max'], a['audit_net_flux_max'], a['estimator_max']) <= 1
        for r in successful for a in r['stats']['attempts'] if a['status'] == 'accepted')
    checks['rejections_rollback_and_no_clipping'] = all(
        all(a['rollback_verified'] for a in r['stats']['attempts'] if a['status'] == 'rejected')
        and r['stats']['clipping_count'] == r['stats']['projection_count'] == 0
        for r in probes_data['runs'] if r.get('stats'))
    tests = json.loads((OUT / 'tests.json').read_text())
    checks['mac_tests_pass_on_current_sources'] = len(tests) == 2 and all(
        r['returncode'] == 0 and r['source_sha256'] == source_fingerprints() for r in tests)
    historical = json.loads((ROOT / 'results/m3/acceptance.json').read_text())
    checks['historical_M3_evidence_unchanged'] = all(digest(ROOT/path) == sha for path, sha in historical['evidence_sha256'].items())
    sensitivity = json.loads((OUT / 'inner-sensitivity.json').read_text())
    arrays = np.load(OUT / 'inner-sensitivity.npz')
    local_case = prepare('kecoli')
    local_model = local_case['model']

    def local_errors(values, reference):
        vf = np.column_stack([local_model.flux(0., y) for y in values.T])
        vr = np.column_stack([local_model.flux(0., y) for y in reference.T])
        return dict(concentration=error(values, reference, local_case['scale'], 1e-4),
                    flux=error(vf, vr, local_case['options']['flux_scale'], 1e-4),
                    net_flux=error(local_model.net_map @ vf, local_model.net_map @ vr, local_case['options']['net_scale'], 1e-4))

    same_maps, local_recomputed = True, True
    kappa, _ = GroupScreen(local_case['system'], local_case['scale']).evaluate(arrays['state'])
    same_maps &= np.array_equal(arrays['groups'], np.flatnonzero(1e-3*kappa >= .01))
    for hi, ref_record in enumerate(sensitivity['references']):
        ref = arrays[f'reference-{hi}']
        h = ref_record['h']
        diff = local_errors(arrays[f'reference-loose-{hi}'], ref)
        local_recomputed &= all(np.isclose(diff[k], v, rtol=1e-9, atol=1e-10) for k,v in ref_record['reference_tightening'].items())
        for row in (r for r in sensitivity['runs'] if r['h'] == h and r['status'] == 'success'):
            fine = arrays[f"fine-{hi}-{row['inner_rtol']}"]
            query = h * np.array([.25, .5, .75, 1.])
            values = np.vstack([np.interp(query, [0., h/2, h], a) for a in fine])
            same_maps &= np.array_equal(fine[:, 0], arrays['state'])
            for value, stored in ((local_errors(values, ref), row['interpolation_errors']),
                                  (local_errors(fine[:, -1:], ref[:, -1:]), row['endpoint_errors'])):
                local_recomputed &= all(np.isclose(value[k], v, rtol=1e-9, atol=1e-10) for k,v in stored.items())
    checks['nine_inner_sensitivities_with_identical_frozen_maps'] = bool(len(sensitivity['runs']) == 9 and same_maps)
    checks['inner_sensitivity_metrics_recompute'] = bool(local_recomputed)
    checks['local_reference_tightening_budget'] = len(sensitivity['references']) == 3 and all(
        max(r['reference_tightening'].values()) < .05 for r in sensitivity['references'])
    candidate_data = json.loads((OUT / 'candidates.json').read_text())
    checks['candidate_selection_accounts_for_all_model_targets'] = len(candidate_data['selected']) + len(candidate_data['skipped']) == 4
    checks['candidate_repeats_preserve_audits_and_rollback'] = all(
        r['status'] != 'success' or (all(max(a['audit_max'], a['audit_flux_max'], a['audit_net_flux_max'], a['estimator_max']) <= 1
            for a in r['stats']['attempts'] if a['status'] == 'accepted') and
            all(a['rollback_verified'] for a in r['stats']['attempts'] if a['status'] == 'rejected'))
        for r in candidate_data['runs'])
    efficiency = []
    for name in ('teusink', 'kecoli'):
        for target in TARGETS:
            candidate_runs = [r for r in candidate_data['runs'] if r['model'] == name and
                              r['parameters']['target'] == target and r['repeat'] >= 0]
            baseline = next(r['fastest_baseline'] for r in summaries if r['model'] == name and r['target'] == target)
            qualifying = (len(candidate_runs) == 7 and baseline is not None and
                          all(r['status'] == 'success' and r['metrics']['passes'] and
                              r['stats']['split_coverage'] >= .5 for r in candidate_runs))
            speedup = baseline['median'] / float(np.median([r['seconds'] for r in candidate_runs])) if qualifying else None
            efficiency.append(dict(model=name, target=target, qualified=qualifying,
                                   median_speedup=speedup, passes=bool(qualifying and speedup >= 1.5)))
    # A one-run probe cannot grant performance admission, regardless of speed.
    admitted = all(checks.values()) and all(r['passes'] for r in efficiency)
    result = dict(checks=checks, evidence_integrity='pass' if all(checks.values()) else 'fail',
                  M4_admission='pass' if admitted else 'not_admitted', summaries=summaries, efficiency=efficiency,
                  reason='Repeated same-accuracy efficiency gate passed.' if admitted else
                         'Independent audits remain mandatory; no repeated qualifying split advantage demonstrated.',
                  limits=['Probes are single development runs, not formal speedup claims.',
                          'Piecewise-linear split outputs are independently audited only on the registered grid and quarter points.',
                          'No PE, M5, M6 or held-out validation performed.'])
    return result


def write_report(result):
    probes_data = json.loads((OUT / 'probes.json').read_text())
    baseline_data = json.loads((OUT / 'baselines.json').read_text())
    sensitivity = json.loads((OUT / 'inner-sensitivity.json').read_text())
    probes_runs = probes_data['runs']
    formal = [r for r in baseline_data['runs'] if r['repeat'] >= 0]
    successes = sum(r['status'] == 'success' for r in probes_runs)
    lines = ['# M3 macOS 续作与 M4 再准入核验', '', '日期：2026-11-07（Asia/Shanghai）；分支：`test`。', '',
             f"**M4 准入：`{result['M4_admission']}`。** 16 个开发探针中 {successes} 个完成完整窗口，"
             f"{len(probes_runs)-successes} 个失败／超时，全部保留。积分成功仍需与分裂覆盖及效率区分。", '',
             '本轮落实用户继续 M3 的要求，重新检验旧实现的输出落步、8 次尝试上限及分区重建限制；'
             '没有改写历史 M1–M3 结果，也没有实施 PE、M5 或 M6。', '',
             '## macOS 环境', '',
             'Python 3.12 与 `uv.lock` 锁定依赖；CPU float64、BLAS／OpenMP 单线程。'
             'Homebrew 的 SUNDIALS 7 与原 ctypes 绑定不兼容，因此从 SHA-256 固定的 6.4.1 源码'
             '构建项目专属 double/int64 CVODE-KLU，链接 Homebrew SuiteSparse。'
             '`metabolism` 在 macOS 自动发现本项目运行时，不改变 Linux／Windows 的路径。', '',
             '本机默认 SDK 27 与 Apple 链接器不匹配；安装脚本使用已安装的 SDK 26.5，允许 `SDKROOT` 显式覆盖。'
             '运行时位于被忽略的 `.runtime/`，构建缓存位于 `results/scratch/`。', '',
             '## 算法续作', '',
             '- 宏步只受控制器和终点约束；公共 M1 观察网格、初值、物种／通量尺度及参考解均保持原样。',
             '- 接受两个细步，不外推；输出采用细步节点间凸线性插值，保持非负性及线性守恒池。'
             '所有请求输出、1/4、1/2、3/4 和终点都由同一起点的独立完整 ODE 审计；审计状态不用于冒充分裂输出。',
             '- 分组不变时复用分区矩阵；每步对缓存范数应用当前步长，避免步长增长时仍沿用旧快慢判定。'
             '频率 1／5 对比、初始层完整 ODE 前缀和进入阈值 1→0.01 的开发探针分别保留。',
             '- 尝试预算由 8 提升至 512；连续拒步上限 16、每条轨迹 60 秒。'
             '初始层前缀从原始初值实际积分，不取保存参考状态作初值；剩余回退区间明确登记。',
             '- 强刚性估计失准仍记录。因每个候选保留实际误差审计，控制器可依据实际误差继续缩步；'
             '这不构成廉价估计器，也不允许稀疏审计或未经审计的接受。', '',
             '## 同机强基线', '',
             '完整系统使用同一 RHS、输出网格、守恒／负值／开放收支诊断包装。Teusink 比较 BDF／Radau；'
             'k-ecoli457 再加入同 RHS、解析稀疏 Jacobian 的 CVODE-KLU。'
             '每种方法扫描 `rtol=10^-4,10^-6,10^-8,10^-10,10^-11`，按实际浓度、单向通量、'
             '大模型净通量及守恒／收支筛选配置，再独立预热并交错重复 7 次。', '',
             f"本轮保存 25 条单次基线校准、{len(formal)} 条正式基线运行、"
             f"{len(baseline_data['runs'])-len(formal)} 次预热。下面探针耗时为单次开发测量，"
             '基线为 7 次中位数（IQR）；不把它们包装成正式加速比。模型解析与参考／误差后处理单列，'
             '分组、粗细双算、逐步审计、拒步、重建、诊断和回退均计入求解耗时。', '',
             '| 模型 | 目标 | 达标最快基线 | 中位秒（IQR） |', '|---|---:|---|---:|']
    for name in ('teusink', 'kecoli'):
        for target in TARGETS:
            baseline = next(r['fastest_baseline'] for r in result['summaries'] if r['model'] == name and r['target'] == target)
            if baseline:
                lines.append(f"| {name} | {target:g} | {baseline['config']} | {baseline['median']:.4f} ({baseline['iqr']:.4f}) |")
    lines += ['', '| 模型 | 目标 | 开发配置 | 状态／精度 | 分裂覆盖 | 接受／拒步 | 秒 |',
              '|---|---:|---|---|---:|---:|---:|']
    for r in result['summaries']:
        status = '完成／达标' if r['status'] == 'success' and r['passes'] else ('完成／未达标' if r['status'] == 'success' else '失败／超时')
        lines.append(f"| {r['model']} | {r['target']:g} | {r['config']} | {status} | {100*r['split_coverage']:.6f}% | {r['accepted_steps']}/{r['rejected_steps']} | {r['seconds']:.3f} |")
    lines += ['', '## 子流容差与固定分组补查', '',
              f"另在实际积分到 `t=0.001` 的大模型状态冻结 {len(sensitivity['fast_groups'])} 个快组，"
              '比较 3 档宏步和 `rtol=10^-8,10^-10,10^-12` 共 9 个局部样例。'
              '初始状态、分组和宏步均在每个容差对照内保持一致；独立 CVODE 参考收紧检查均小于 0.05 个局部误差单位。', '',
              '| h | 子流 rtol | 端点净通量局部误差 | 含插值最大净通量误差 | 细步组合秒 |',
              '|---:|---:|---:|---:|---:|']
    for row in sensitivity['runs']:
        if row['status'] == 'success':
            lines.append(f"| {row['h']:.6g} | {row['inner_rtol']:g} | {row['endpoint_errors']['net_flux']:.6g} | {row['interpolation_errors']['net_flux']:.6g} | {row['seconds']:.4f} |")
    lines += ['', '误差单位为局部浓度／单向通量／净通量目标 `10^-4`。'
              '在 `h=10^-4` 时，收紧子流容差没有消除约 11.9 个单位的端点净通量误差。'
              '固定快组后，`h=10^-6` 的局部浓度、通量及净通量审计能通过；'
              '其中 `rtol=10^-8` 的一次细步组合约 0.22 秒，尚未含粗步或完整系统审计。'
              '局部固定分组可以达标，尚不构成全窗口动态求解或效率资格。', '',
              '## 决策与边界', '',
              '输出解耦、更多尝试、缓存及初始层前缀没有提供支持 M4 的效率证据。'
              '这些探针中，大模型没有接受分裂步；缩步后仍有净通量审计超标。'
              '小模型的完整窗口成功主要来自回退；扩大快组的一个探针在预算内未完成。'
              '因此本轮结果不能用于宣称纯分裂完整窗口资格、动态分组加速或 M4 准入。', '',
              '`candidates.json` 按“精度达标、分裂覆盖至少半窗口、单次耗时优于达标基线”的规则筛选'
              '值得正式重复的配置；不存在候选时明确记录原因。只有进一步完成重复验证、满足原规划'
              '两模型／两精度档 1.5 倍有用性目标并通过正确性检查，才自动建议准入；'
              '当前证据连较弱的效率前景条件也未满足。不得通过取消审计、放宽净通量目标或把回退标成分裂来取得准入。', '',
              '凸线性插值的资格仅覆盖请求网格与登记审计点，不是任意连续时刻误差定理；'
              '本轮未重新验证新控制器的事件／留出扰动／长期行为。原生 CVODE 未暴露全部内部接受状态或独立 LU 次数；'
              '失败时尚未返回的原生内部计数不可补造。未采集隔离 worker 峰值 RSS。', '',
              '实质剩余项是能在强刚性及净通量取消误差下可靠工作的廉价估计／连续输出策略，'
              '以及避免反复数值快流重启的办法。这属于算法研究；本次 macOS 适配本身已经完成。', '',
              '## 复现', '', '```bash', 'bash scripts/setup-macos.sh',
              'uv run python -m metabolism.m3_continuation --case probes',
              'uv run python -m metabolism.m3_continuation --case baselines',
              'uv run python -m metabolism.m3_continuation --case sensitivities',
              'uv run python -m metabolism.m3_continuation --case candidates',
              'uv run python -m metabolism.m3_continuation --case tests',
              'uv run python -m metabolism.m3_continuation --case report',
              'uv run python -m metabolism.m3_continuation --case check',
              'uv run python -m metabolism.m3_report --check', '```', '',
              '`check` 只读取保存轨迹并重算误差、登记条件和 SHA-256，不重跑积分。'
              '新增测试保存在 `checks/`，与历史测试源码指纹分开；其中显式运行 macOS 原生 CVODE 矩阵指数核验，'
              '补充旧测试按 Linux 平台筛选而跳过的一项。', '',
              '[机器验收](../results/m3-continuation/acceptance.json) · '
              '[原始证据](../results/m3-continuation/README.md)', '']
    (ROOT / 'docs/m3-macos-continuation.md').write_text('\n'.join(lines), encoding='utf-8')


def finalize(check_only=False):
    result = audit()
    if check_only:
        previous = json.loads((OUT / 'acceptance.json').read_text())
        result['checks']['saved_acceptance_recomputes'] = all(previous[k] == result[k] for k in ('checks', 'summaries', 'efficiency', 'M4_admission'))
        result['checks']['evidence_sha256'] = all(digest(ROOT/p) == sha for p, sha in previous['evidence_sha256'].items())
    else:
        write_report(result)
        save(OUT / 'summary.json', result['summaries'])
        paths = [p for p in OUT.iterdir() if p.name != 'acceptance.json']
        paths += [ROOT / p for p in ('metabolism/continuation.py', 'metabolism/m3_continuation.py',
                                     'metabolism/__init__.py', 'checks/test_continuation.py',
                                     'scripts/setup-macos.sh', 'scripts/run-macos.sh', 'docs/m3-macos-continuation.md')]
        result['evidence_sha256'] = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
        save(OUT / 'acceptance.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries', 'evidence_sha256')}, indent=2), flush=True)
    if not all(result['checks'].values()):
        raise SystemExit('Continuation evidence failed')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', choices=('probes', 'baselines', 'sensitivities', 'candidates', 'tests', 'report', 'check'), required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.case in ('probes', 'baselines'):
        save(OUT / 'environment.json', dict(date='2026-11-07 Asia/Shanghai', python=platform.python_version(),
             numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
             threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
             native='Project-local SUNDIALS 6.4.1, double/int64, SuiteSparse KLU; see scripts/setup-macos.sh'))
    {'probes': probes, 'baselines': baselines, 'sensitivities': inner_sensitivity,
     'candidates': repeat_candidates, 'tests': record_tests,
     'report': finalize, 'check': lambda: finalize(True)}[args.case]()


if __name__ == '__main__':
    main()
