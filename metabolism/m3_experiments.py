"""M3 calibration, audited adaptive ablations and full registered windows.

Run --case calibration|toy|teusink on Windows, --case kecoli in M1 WSL.
All records include rejected work, full-step audits and persistent fallback.
"""

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from dataclasses import asdict, replace
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from .adaptive import AdaptiveConfig, AdaptiveSplit, GroupScreen, integrate_segments, normalized_error
from .m2_experiments import digest, enzyme_system, error, linear_chain, save
from .splitting import FixedSplit, ReactionSystem, SplitFailure, compose_step, reversible_pair

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m3'


def calibration():
    rows = []
    for reversible in (False, True):
        for ratio in (1., 100., 1e4, 1e6):
            model, matrix, backward = linear_chain(ratio, reversible)
            for initial in ('far', 'fast_equilibrium'):
                y0 = model.y0 if initial == 'far' else np.array([.36, .54, .1])
                fast = lambda y, h: reversible_pair(y, h, 0, 1, 3 * ratio, 2 * ratio)
                slow = lambda y, h: reversible_pair(y, h, 1, 2, .4, backward)
                for snapshot in (0., 1e-4, .01):
                    y = expm(matrix * snapshot) @ y0
                    for h in (.001, .01, .1):
                        coarse = compose_step(fast, slow, y, h, 'Strang')
                        middle = compose_step(fast, slow, y, h / 2, 'Strang')
                        fine = compose_step(fast, slow, middle, h / 2, 'Strang')
                        reference = expm(matrix * h) @ y
                        midpoint_reference = expm(matrix * h / 2) @ y
                        estimator = (fine - coarse) / 3
                        predicted = normalized_error(fine - estimator, fine, np.ones(3), 1e-4)
                        actual = normalized_error(fine, reference, np.ones(3), 1e-4)
                        mid = normalized_error(middle, midpoint_reference, np.ones(3), 1e-4)
                        rows.append({'reversible': reversible, 'fast_multiplier': ratio, 'initial': initial,
                                     'snapshot': snapshot, 'h': h, 'predicted_wrms': predicted[0], 'predicted_max': predicted[1],
                                     'actual_wrms': actual[0], 'actual_max': actual[1], 'midpoint_max': mid[1],
                                     'underestimation_ratio': actual[1] / max(predicted[1], 1e-15),
                                     'naive_false_accept': predicted[0] <= 1 and max(actual[1], mid[1]) > 1,
                                     'independent_audit_rejects': max(actual[1], mid[1]) > 1})
    # RMS can conceal a single critical near-zero component in a wide system.
    rms_counterexample = normalized_error(np.r_[.0015, np.zeros(99)], np.zeros(100), np.ones(100), 1e-3)
    thresholds = []
    for limit in (1.5, 2., 4.):
        thresholds.append({'ratio_limit': limit,
                           'flagged': sum(r['actual_max'] > max(limit * r['predicted_max'], .1) for r in rows),
                           'unguarded_false_accepts': sum(r['naive_false_accept'] for r in rows),
                           'missed_false_accepts_with_max_audit': sum(r['naive_false_accept'] and not r['independent_audit_rejects'] for r in rows)})
    save(OUT / 'calibration.json', {'rows': rows, 'ratio_threshold_sensitivity': thresholds,
         'rms_counterexample': {'dimension': 100, 'wrms': rms_counterexample[0], 'maximum': rms_counterexample[1]},
         'scope': 'Exact complete flows and independent full matrix exponential; local target 1e-4. Development calibration, not a global theorem.',
         'selected_policy': 'Independent midpoint/endpoint maximum and observable audits every trial; ratio limit 2 with 0.1 noise floor; false accept guard remains active even if ratio criterion is not triggered.'})
    print('calibration', len(rows), 'exact local trials;', sum(r['naive_false_accept'] for r in rows), 'naive false acceptances', flush=True)


def case_configs(target, initial_step, max_step, budget=512):
    base = AdaptiveConfig(target=target, initial_step=initial_step, max_step=max_step,
                          max_attempts=budget, fallback_rtol=min(1e-10, target * 1e-4))
    configs = [('fixed', base, [0])]
    for period in (1, 5, 20):
        for hysteresis in (True, False):
            name = f'dynamic-{period}-' + ('hysteresis' if hysteresis else 'single-threshold')
            configs.append((name, replace(base, group_every=period, leave=.3 if hysteresis else 1.), None))
    return configs


def toy():
    model, matrix, _ = linear_chain(10.)
    times = np.array([0., .1, .2, .4, .6, .8, 1.])
    ref = np.column_stack([expm(matrix * t) @ model.y0 for t in times])
    configs = case_configs(1e-3, .01, .1)
    runs, trajectories = [], {}
    for repeat in range(-1, 7):
        shift = max(repeat, 0) % len(configs)
        for name, config, fixed in configs[shift:] + configs[:shift]:
            tick = time.perf_counter()
            try:
                controller = AdaptiveSplit(model, np.ones(3), config, fixed_groups=fixed,
                                            pools=np.ones((1, 3)), flux_scale=np.array([30., 20., .4, .2]))
                values, stats = controller.integrate(times)
                row = {'config': name, 'repeat': repeat, 'status': 'success',
                       'seconds': time.perf_counter() - tick, 'observed_error': error(values, ref, np.ones(3), config.target),
                       'stats': stats}
                if repeat == 0:
                    trajectories[name] = values
            except SplitFailure as exc:
                row = {'config': name, 'repeat': repeat, 'status': 'failed', 'reason': str(exc),
                       'seconds': time.perf_counter() - tick, 'stats': exc.stats}
            runs.append(row)
            print('toy', name, repeat, row['status'], round(row['seconds'], 3), flush=True)
    # Boundary layers, nonlinear enzyme pools, coupled slow guard and constant-input event.
    validation = []
    stiff, matrix, _ = linear_chain(1e6)
    values, stats = AdaptiveSplit(stiff, np.ones(3), AdaptiveConfig(initial_step=.1, max_step=.1),
                                  fixed_groups=[0], pools=np.ones((1, 3))).integrate([0., .2])
    validation.append({'case': 'stiff_estimator_rejection', 'target': 1e-3,
                       'error': error(values, np.column_stack([stiff.y0, expm(matrix * .2) @ stiff.y0]), np.ones(3), 1e-3),
                       'stats': stats})
    for ratio in (1., 1e2, 1e4, 1e6):
        for target in (1e-3, 1e-5):
            system, matrix, _ = linear_chain(ratio)
            grid = np.unique(np.r_[0., np.geomspace(1e-8, 1., 25), np.linspace(0., 1., 21)])
            cfg = AdaptiveConfig(target=target, initial_step=.05, max_step=.1, max_attempts=64)
            result, stats = AdaptiveSplit(system, np.ones(3), cfg, fixed_groups=[0], pools=np.ones((1, 3))).integrate(grid)
            reference = np.column_stack([expm(matrix * t) @ system.y0 for t in grid])
            validation.append({'case': 'initial_layer', 'fast_multiplier': ratio, 'target': target,
                               'error': error(result, reference, np.ones(3), target), 'stats': stats})
    system = enzyme_system()
    grid = np.linspace(0., 1., 21)
    reference = solve_ivp(lambda t, y: system.S @ system.flux(t, y), (0., 1.), system.y0,
                          method='DOP853', rtol=1e-12, atol=1e-14, t_eval=grid)
    pools = np.array([[1., 0., 1., 0.], [0., 1., 1., 1.]])
    for target in (1e-3, 1e-5):
        result, stats = AdaptiveSplit(system, np.ones(4), AdaptiveConfig(target=target),
                                      fixed_groups=[0], pools=pools).integrate(grid)
        validation.append({'case': 'nonlinear_enzyme', 'target': target,
                           'error': error(result, reference.y, np.ones(4), target), 'stats': stats})
    robertson = ReactionSystem(csc_matrix([[-1., 1., 0.], [1., -1., -1.], [0., 0., 1.]]),
                              lambda t, y: np.array([.04 * y[0], 1e4 * y[1] * y[2], 3e7 * y[1] ** 2]),
                              np.array([1., 0., 0.]), [np.array([i]) for i in range(3)],
                              csc_matrix([[1, 0, 0], [0, 1, 1], [0, 1, 0]]))
    grid = np.unique(np.r_[0., np.geomspace(1e-8, 1., 41)])
    scale = np.array([1., 1e-4, .1])
    ref = solve_ivp(lambda t, y: robertson.S @ robertson.flux(t, y), (0., 1.), robertson.y0,
                    method='Radau', rtol=1e-12, atol=1e-15, t_eval=grid)
    result, stats = AdaptiveSplit(robertson, scale, AdaptiveConfig(initial_step=.01, max_attempts=64),
                                  pools=np.ones((1, 3))).integrate(grid)
    validation.append({'case': 'Robertson', 'target': 1e-3, 'error': error(result, ref.y, scale, 1e-3), 'stats': stats})
    first, a, _ = linear_chain(1.)
    second, b, _ = linear_chain(100.)
    grid = np.array([0., .05, .1, .15, .2, .3])
    values, records = integrate_segments([first, second], [0., .1, .3], grid, np.ones(3),
                                         AdaptiveConfig(initial_step=.01), fixed_groups=[0], pools=np.ones((1, 3)))
    reference = np.column_stack([expm(a * t) @ first.y0 if t <= .1 else expm(b * (t - .1)) @ expm(a * .1) @ first.y0 for t in grid])
    validation.append({'case': 'known_input_event', 'target': 1e-3, 'error': error(values, reference, np.ones(3), 1e-3),
                       'event_time': .1, 'segment_stats': records})
    # A deterministic kappa trace makes the hysteresis chatter ablation explicit.
    kappa_trace = [.9, 1.1, .9, 1.1, .9, .2]
    from .adaptive import select_groups
    chatter = []
    for leave in (.3, 1.):
        previous, history = [], []
        for kappa in kappa_trace:
            previous = select_groups(np.array([kappa]), 1., previous, leave=leave)
            history.append(previous.tolist())
        chatter.append({'leave': leave, 'history': history, 'switches': sum(a != b for a, b in zip([[]] + history[:-1], history))})
    save(OUT / 'toy-registration.json', {'model': 'shared reversible chain', 'nominal_fast_multiplier': 10.,
         'times': times.tolist(), 'target': 1e-3, 'configs': [{'id': n, **asdict(c), 'fixed_groups': f} for n, c, f in configs],
         'warmups_per_config': 1, 'repeats': 7, 'scope': 'Development ablations; no held-out generalization claim'})
    save(OUT / 'toy-warmups.json', [r for r in runs if r['repeat'] < 0])
    save(OUT / 'toy-runs.json', [r for r in runs if r['repeat'] >= 0])
    save(OUT / 'validation.json', {'cases': validation, 'hysteresis_chatter': chatter})
    np.savez_compressed(OUT / 'toy-trajectories.npz', times=times,
                        reference=np.column_stack([expm(linear_chain(10.)[1] * t) @ model.y0 for t in times]), **trajectories)


def real_case(name, probe=False, full_only=False):
    large = name == 'kecoli'
    setup_start = time.perf_counter()
    if large:
        from .kecoli import KEcoliModel
        model = KEcoliModel(ROOT / 'models/k-ecoli457.zip')
        system = ReactionSystem.from_kecoli(model)
        model_path = ROOT / 'models/k-ecoli457.zip'
        prior_path = ROOT / 'results/m1-completion/kecoli-reference.npz'
        prior = np.load(prior_path)
        times, reference, scale, flux_scale = [prior[k] for k in ('times', 'concentrations', 'scale', 'flux_scale')]
        mapping, net_scale = model.net_map, np.array(json.loads((ROOT / 'results/m1-completion/kecoli-net-flux.json').read_text())['net_flux_scale'])
        pools, ledger = model.pools, None
        screening = GroupScreen(system, scale)
        kappa, _ = screening.evaluate(model.y0)
        fixed = np.flatnonzero(1e-5 * kappa >= 1).tolist()
        targets = (1e-3, 1e-5)
    else:
        from .sbml import SBMLModel
        model_path = ROOT / 'models/BIOMD0000000064.xml'
        model = SBMLModel(model_path)
        system = ReactionSystem.from_sbml(model)
        prior_path = ROOT / 'results/m1/BIOMD0000000064-reference.npz'
        prior = np.load(prior_path)
        times, reference, scale = [prior[k] for k in ('times', 'concentrations', 'scale')]
        flux = np.column_stack([model.flux(t, x) for t, x in zip(times, reference.T)])
        flux_scale = np.maximum(np.max(abs(flux), axis=1), 1e-6)
        mapping = net_scale = None
        pools = model.L.T
        ledger = np.array([6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 2, 0, 21, 21], dtype=float)
        fixed, targets = [6, 7], (1e-3, 1e-5)
    setup_seconds = time.perf_counter() - setup_start
    configs = []
    for target in targets:
        for identity, config, groups in case_configs(target, 1e-5, .1 if large else 1., budget=8):
            # Registered finite work budget; fallback is explicit, never labelled split speedup.
            config = replace(config, fallback_rtol=1e-11 if large else 1e-12,
                              audit_rtol=1e-11 if large else 1e-12,
                              fallback_method='CVODE-KLU' if large else 'BDF')
            configs.append((f'{identity}-{target}', config, fixed if groups is not None else None))
    if full_only:
        configs = [(f'full-only-{target}', replace(configs[7 * i][1], max_attempts=8),
                    list(range(len(system.groups)))) for i, target in enumerate(targets)]
    if probe:
        configs = [configs[0], configs[1]]
    flux_ref = np.column_stack([model.flux(t, y) for t, y in zip(times, reference.T)])
    options = {'flux_scale': flux_scale, 'net_map': mapping, 'net_scale': net_scale,
               'net_target': 1e-3 if large else None, 'pools': pools, 'ledger_weights': ledger}
    metadata = {'date': '2026-11-03 Asia/Shanghai', 'model': name, 'model_sha256': digest(model_path),
                'reference_path': prior_path.relative_to(ROOT).as_posix(), 'reference_sha256': digest(prior_path),
                'window': [float(times[0]), float(times[-1])], 'output_count': len(times),
                'output_policy': 'Unchanged M1 grid and scales; splitting lands exactly, fallback dense output uses same grid',
                'setup_seconds': setup_seconds, 'configs': [{'id': n, **asdict(c), 'fixed_groups': f} for n, c, f in configs],
                'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                            'platform': platform.platform(), 'threads': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
                'warmups_per_config': 1, 'repeats': 7,
                'limitations': ['Audited hybrid with persistent full ODE fallback', 'Output landing may force very short initial steps',
                                'All classification, audit and fallback work counted', 'No PE; no speedup claim']}
    if not probe:
        save(OUT / f'{name}{"-full" if full_only else ""}-registration.json', metadata)
    records, trajectories = [], {}
    for repeat in (range(1) if probe else range(-1, 7)):
        shift = max(repeat, 0) % len(configs)
        for identity, config, groups in configs[shift:] + configs[:shift]:
            tick = time.perf_counter()
            try:
                controller = AdaptiveSplit(system, scale, config, fixed_groups=groups, **options)
                result, stats = controller.integrate(times)
                row = {'config': identity, 'repeat': repeat, 'status': 'success',
                       'seconds': time.perf_counter() - tick,
                       'observed_error': error(result, reference, scale, config.target),
                       'concentration_errors': {str(r): error(result, reference, scale, r) for r in targets},
                       'stats': stats}
                obs = np.column_stack([model.flux(t, y) for t, y in zip(times, result.T)])
                row['flux_errors'] = {str(r): error(obs, flux_ref, flux_scale, r) for r in targets}
                row['invariant_error'] = float(np.max(abs(pools @ (result - model.y0[:, None]))) / max(1e-12, float(np.max(abs(pools @ model.y0)))))
                if large:
                    row['net_flux_errors'] = {str(r): error(mapping @ obs, mapping @ flux_ref, net_scale, r) for r in (1e-2, 1e-3)}
                else:
                    row['relative_open_ledger_error'] = abs(stats['ledger_residual']) / max(1., float(ledger @ model.y0))
                if repeat == 0:
                    # Store identical full-fallback trajectories once; each
                    # first-repeat record explicitly identifies its array.
                    key = next((key for key, old in trajectories.items() if np.array_equal(old, result)), identity)
                    trajectories[key] = result
                    row['trajectory_key'] = key
            except SplitFailure as exc:
                row = {'config': identity, 'repeat': repeat, 'status': 'failed', 'reason': str(exc),
                       'seconds': time.perf_counter() - tick, 'stats': exc.stats}
            records.append(row)
            print(name, identity, repeat, row['status'], round(row['seconds'], 3),
                  row.get('observed_error'), stats['fallbacks'][-1]['reason'] if row['status']=='success' and stats['fallbacks'] else '', flush=True)
    if probe:
        save(ROOT / f'results/scratch/m3-{name}-probe.json', {'registration': metadata, 'runs': records})
        return
    prefix = name + ('-full' if full_only else '')
    save(OUT / f'{prefix}-warmups.json', [r for r in records if r['repeat'] < 0])
    save(OUT / f'{prefix}-runs.json', [r for r in records if r['repeat'] >= 0])
    np.savez_compressed(OUT / f'{prefix}-trajectories.npz', times=times, scale=scale, reference=reference,
                        flux_scale=flux_scale, **trajectories)


def screen_benchmark(refresh=False):
    """One measured optimization of M1's per-group sparse construction."""
    from .kecoli import KEcoliModel
    model = KEcoliModel(ROOT / 'models/k-ecoli457.zip')
    system = ReactionSystem.from_kecoli(model)
    prior = np.load(ROOT / 'results/m1-completion/kecoli-reference.npz')
    tick = time.perf_counter()
    screen = GroupScreen(system, prior['scale'])
    setup = time.perf_counter() - tick
    print('screen setup', setup, flush=True)
    rows = []
    existing = OUT / 'screening.json'
    # Saved profiling samples may be reused while separately repairing or
    # extending the exploratory macro probe; no solver timing is recomputed.
    if existing.exists() and not refresh:
        saved = json.loads(existing.read_text())
        rows, setup = saved['snapshots'], saved['setup_seconds']
    else:
        for index in (0, np.searchsorted(prior['times'], 1e-4), np.searchsorted(prior['times'], .01), len(prior['times']) - 1):
            y = prior['concentrations'][:, index]
            optimized_times, original_times = [], []
            for repeat in range(7):
                tick = time.perf_counter()
                optimized, gradient = screen.evaluate(y)
                optimized_times.append(time.perf_counter() - tick)
                tick = time.perf_counter()
                original = np.array([np.max(np.asarray(abs(model.S[:, group] @ gradient[group]) @ prior['scale']).ravel() / prior['scale'])
                                     for group in system.groups])
                original_times.append(time.perf_counter() - tick)
            difference = float(np.max(abs(optimized - original) / np.maximum(1., abs(original))))
            rows.append({'time': float(prior['times'][index]), 'max_relative_difference': difference,
                         'optimized_including_gradient_seconds': optimized_times,
                         'original_group_loop_excluding_gradient_seconds': original_times})
            print('screen snapshot', float(prior['times'][index]), 'difference', difference, flush=True)
    # Initial macro-step calibration without the small mandatory output landings.
    fast, _ = screen.evaluate(model.y0)
    groups = np.flatnonzero(1e-5 * fast >= 1)
    config = AdaptiveConfig(initial_step=1e-5, max_step=1e-4, max_attempts=4, fallback_method='CVODE-KLU',
                            fallback_rtol=1e-11, timeout=20., slow_bound_limit=1e6)
    net_scale = np.array(json.loads((ROOT / 'results/m1-completion/kecoli-net-flux.json').read_text())['net_flux_scale'])
    # A sparse factorization can delay the next RHS timeout check. Bound this
    # exploratory worker externally as well; do not infer missing internal data.
    import multiprocessing as mp
    context = mp.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    def worker():
        try:
            _, stats = AdaptiveSplit(system, prior['scale'], config, fixed_groups=groups,
                                      flux_scale=prior['flux_scale'], pools=model.pools, net_map=model.net_map,
                                      net_scale=net_scale, net_target=1e-3).integrate([0., 1e-4])
            sender.send({'status': 'success', 'stats': stats})
        except SplitFailure as exc:
            sender.send({'status': 'failed', 'reason': str(exc), 'stats': exc.stats})
        except Exception as exc:
            sender.send({'status': 'failed', 'reason': repr(exc), 'stats': None})
        finally:
            sender.close()
    tick = time.perf_counter()
    process = context.Process(target=worker)
    process.start()
    sender.close()
    if receiver.poll(30.):
        probe = receiver.recv()
        process.join(5.)
    else:
        process.terminate()
        process.join(5.)
        probe = {'status': 'timeout', 'reason': 'Registered 30s external worker deadline', 'stats': None}
    receiver.close()
    probe.update(config=asdict(config), seconds=time.perf_counter() - tick, outer_timeout_seconds=30.,
                 scope='Single development audit without early output landings; the slow-norm guard is deliberately relaxed only here to exercise the estimator audit. Not a production setting or a formal timing comparison.')
    save(OUT / 'screening.json', {'setup_seconds': setup, 'snapshots': rows,
         'general_non_rank_one_groups': len(screen.general),
         'macro_probe': probe,
         'default_guard_probe': saved.get('default_guard_probe', saved.get('macro_probe')) if existing.exists() and not refresh else None,
         'scope': 'Same exact complete-pair Jacobian norms, no grouping heuristic change; seven screening samples/state'})
    print('screen optimization verified; macro probe', probe['status'], probe.get('reason'), flush=True)


def sensitivities():
    """Development accuracy checks, not repeated performance comparisons."""
    model, matrix, _ = linear_chain(10.)
    times = np.array([0., .1, .2, .4, .6, .8, 1.])
    ref = np.column_stack([expm(matrix * t) @ model.y0 for t in times])
    thresholds = []
    for enter in (.5, 1., 2.):
        config = AdaptiveConfig(initial_step=.01, max_step=.1, group_every=1, enter=enter, leave=.3 * enter)
        values, stats = AdaptiveSplit(model, np.ones(3), config, pools=np.ones((1, 3))).integrate(times)
        thresholds.append({'enter': enter, 'leave': .3 * enter, 'error': error(values, ref, np.ones(3), 1e-3), 'stats': stats})
    from .sbml import SBMLModel
    small = SBMLModel(ROOT / 'models/BIOMD0000000064.xml')
    prior = np.load(ROOT / 'results/m1/BIOMD0000000064-reference.npz')
    cases = [('shared_chain', model, model.y0, np.ones(3), [0], .02),
             ('enzyme', enzyme_system(), enzyme_system().y0, np.ones(4), [0], .01)]
    for index in (0, np.searchsorted(prior['times'], .01)):
        cases.append((f'teusink-{prior["times"][index]}', ReactionSystem.from_sbml(small),
                      prior['concentrations'][:, index], prior['scale'], [6, 7], 1e-5))
    inner = []
    for name, system, y0, scale, groups, h in cases:
        values = []
        for tol in (1e-8, 1e-9):
            _, states, stats = FixedSplit(system, groups, scale, rtol=tol).integrate(y0, 0., h, 2)
            values.append(states)
        inner.append({'case': name, 'h': h, 'rtol': [1e-8, 1e-9],
                      'difference_in_local_target_units': error(values[0], values[1], scale, 1e-4),
                      'scope': 'Same state, macro h, frozen groups and fine composition; tenfold inner tightening only'})
    save(OUT / 'sensitivities.json', {'group_thresholds': thresholds, 'inner_tolerances': inner,
                                     'scope': 'Single development accuracy checks; not formal repeated timings or held-out parameter tests'})
    print('threshold and inner-tolerance sensitivity complete', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', required=True, choices=['calibration', 'toy', 'teusink', 'kecoli', 'screening',
                                                         'teusink-full', 'kecoli-full', 'sensitivities'])
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--refresh-screening', action='store_true', help='Recompute all screening timing samples instead of reusing saved ones')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case == 'calibration':
        calibration()
    elif args.case == 'toy':
        toy()
    elif args.case == 'screening':
        screen_benchmark(refresh=args.refresh_screening)
    elif args.case == 'sensitivities':
        sensitivities()
    else:
        real_case(args.case.removesuffix('-full'), args.probe, full_only=args.case.endswith('-full'))


if __name__ == '__main__':
    main()
