"""M2 fixed splitting correctness experiments; no performance gain claims.

Windows: uv run python -m metabolism.m2_experiments --case b0|teusink
WSL:     python3 -m metabolism.m2_experiments --case kecoli
"""

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from .splitting import (FixedSplit, ReactionSystem, SplitFailure, compose_step,
                        enzyme_binding, enzyme_catalysis, reversible_pair)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m2'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error(y, ref, scale, target=1.):
    return float(np.max(abs(y - ref) / (target * (.01 * scale[:, None] + np.maximum(abs(ref), scale[:, None])))))


def orders(errors):
    return np.log2(np.asarray(errors[:-1]) / errors[1:]).tolist()


def exact_trajectory(fast, slow, y0, end, steps, method):
    ys = [y0.copy()]
    for _ in range(steps):
        ys.append(compose_step(fast, slow, ys[-1], end / steps, method))
    return np.asarray(ys).T


def linear_chain(ratio, reversible=True):
    S = csc_matrix([[-1., 1., 0., 0.], [1., -1., -1., 1.], [0., 0., 1., -1.]])
    reverse = .2 if reversible else 0.
    gradient = csc_matrix([[3 * ratio, 0., 0.], [0., 2 * ratio, 0.],
                           [0., .4, 0.], [0., 0., reverse]])
    system = ReactionSystem(S, lambda t, y: gradient @ y, np.array([1., 0., 0.]),
                            [np.array([0, 1]), np.array([2, 3])], gradient != 0,
                            lambda t, y: gradient)
    return system, (S @ gradient).toarray(), reverse


def enzyme_system():
    S = csc_matrix([[-1., 1., 1.], [-1., 1., 0.], [1., -1., -1.], [0., 0., 1.]])
    def flux(t, y):
        return np.array([4 * y[0] * y[1], 2 * y[2], .7 * y[2]])
    def gradient(t, y):
        return csc_matrix([[4 * y[1], 4 * y[0], 0., 0.], [0., 0., 2., 0.], [0., 0., .7, 0.]])
    return ReactionSystem(S, flux, np.array([.2, 1.3, 0., .1]),
                          [np.array([0, 1]), np.array([2])],
                          csc_matrix([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]), gradient)


def b0():
    rows = []
    counts = (8, 16, 32, 64, 128)
    for reversible in (False, True):
        for ratio in (1., 1e2, 1e4, 1e6):
            system, matrix, backward = linear_chain(ratio, reversible)
            for initial in ('far', 'fast_equilibrium'):
                y0 = system.y0 if initial == 'far' else np.array([.36, .54, .1])
                for method in ('Lie', 'Strang'):
                    errors, pool_errors, minima = [], [], []
                    for count in counts:
                        ts = np.linspace(0., 1., count + 1)
                        ref = np.column_stack([expm(matrix * t) @ y0 for t in ts])
                        y = exact_trajectory(lambda x, h: reversible_pair(x, h, 0, 1, 3 * ratio, 2 * ratio),
                                             lambda x, h: reversible_pair(x, h, 1, 2, .4, backward),
                                             y0, 1., count, method)
                        errors.append(error(y, ref, np.ones(3)))
                        pool_errors.append(float(np.max(abs(y.sum(axis=0) - y0.sum()))))
                        minima.append(float(y.min()))
                    rows.append({'case': 'reversible_chain' if reversible else 'terminal_chain',
                                 'fast_slow_ratio': ratio, 'initial': initial, 'method': method,
                                 'rate_constants': [3 * ratio, 2 * ratio, .4, backward],
                                 'isolated_group_relaxation_ratio': 5 * ratio / (.4 + backward),
                                 'steps': counts, 'trajectory_errors': errors, 'orders': orders(errors),
                                 'invariant_errors': pool_errors, 'min_raw_concentrations': minima,
                                 'reference': 'Independent full matrix exponential at every macro endpoint',
                                 'subflows': 'Exact reversible pair maps; no PE'})
    system = enzyme_system()
    references = [solve_ivp(lambda t, y: system.S @ system.flux(t, y), (0., 1.), system.y0,
                            method=method, rtol=1e-12, atol=1e-14, dense_output=True)
                  for method in ('Radau', 'DOP853')]
    grid = np.linspace(0., 1., 257)
    ref_difference = float(np.max(abs(references[0].sol(grid) - references[1].sol(grid))))
    enzyme_rows = []
    pools = np.array([[1., 0., 1., 0.], [0., 1., 1., 1.]])
    for method in ('Lie', 'Strang'):
        exact_errors, numerical_errors, discrepancies, invariants, minimum = [], [], [], [], []
        for count in counts:
            times = np.linspace(0., 1., count + 1)
            exact = exact_trajectory(lambda x, h: enzyme_binding(x, h, 4., 2.),
                                     lambda x, h: enzyme_catalysis(x, h, .7), system.y0, 1., count, method)
            _, numerical, _ = FixedSplit(system, [0], np.ones(4), rtol=1e-11).integrate(system.y0, 0., 1., count, method)
            exact_errors.append(error(exact, references[0].sol(times), np.ones(4)))
            numerical_errors.append(error(numerical, references[0].sol(times), np.ones(4)))
            discrepancies.append(float(np.max(abs(numerical - exact))))
            invariants.append(float(np.max(abs(pools @ (numerical - system.y0[:, None])))))
            minimum.append(float(numerical.min()))
        enzyme_rows.append({'method': method, 'steps': counts, 'exact_flow_errors': exact_errors,
                            'numerical_flow_errors': numerical_errors, 'orders': orders(exact_errors),
                            'numerical_vs_exact_split': discrepancies, 'two_pool_errors': invariants,
                            'min_raw_concentrations': minimum})
    # Robertson retains its encoded irreversible reactions; no invented PE pair.
    robertson = ReactionSystem(csc_matrix([[-1., 1., 0.], [1., -1., -1.], [0., 0., 1.]]),
                              lambda t, y: np.array([.04 * y[0], 1e4 * y[1] * y[2], 3e7 * y[1] ** 2]),
                              np.array([1., 0., 0.]), [np.array([i]) for i in range(3)],
                              csc_matrix([[1, 0, 0], [0, 1, 1], [0, 1, 0]]))
    rob_ref = solve_ivp(lambda t, y: robertson.S @ robertson.flux(t, y), (0., .01), robertson.y0,
                        method='Radau', rtol=1e-12, atol=1e-15, dense_output=True)
    rob_rows = []
    for method in ('Lie', 'Strang'):
        for count in (8, 16, 32, 64):
            try:
                times, y, stats = FixedSplit(robertson, [1, 2], np.array([1., 1e-4, 1e-3]), rtol=1e-11).integrate(
                    robertson.y0, 0., .01, count, method)
                rob_rows.append({'method': method, 'steps': count, 'status': 'success',
                                 'error': error(y, rob_ref.sol(times), np.array([1., 1e-4, 1e-3])),
                                 'invariant_error': float(np.max(abs(y.sum(axis=0) - 1))), 'stats': stats})
            except SplitFailure as exc:
                rob_rows.append({'method': method, 'steps': count, 'status': 'failed', 'reason': str(exc), 'stats': exc.stats})
    # Deliberately inadmissible non-quasipositive flow exercises the real failure path.
    bad = ReactionSystem(csc_matrix([[-1.]]), lambda t, y: np.ones(1), np.array([.01]),
                         [np.array([0])], csc_matrix([[0]]))
    try:
        FixedSplit(bad, [], np.ones(1)).integrate(bad.y0, 0., 1., 1)
        negative = {'detected': False}
    except SplitFailure as exc:
        negative = {'detected': True, 'reason': str(exc), 'stats': exc.stats}
    result = {'date': '2026-11-03 Asia/Shanghai', 'linear': rows, 'enzyme': enzyme_rows,
              'linear_ratio_definition': 'fast_slow_ratio is the nominal fast-rate multiplier. Fast rates=(3,2)*multiplier; slow rates=(0.4,0.2) or (0.4,0). The exact isolated nonzero group-mode ratio is separately saved.',
              'enzyme_independent_reference_max_difference': ref_difference,
              'robertson': rob_rows, 'negative_detection': negative,
              'interpretation': 'Expected orders apply only in the weak-stiffness asymptotic regime. High ratios expose order reduction; no adaptive estimator or fallback is implemented in M2.'}
    save(OUT / 'b0.json', result)
    print('B0 complete:', len(rows), 'linear scans;', len(enzyme_rows), 'enzyme scans;', len(rob_rows), 'Robertson runs', flush=True)


def real_case(name, repeats, probe=False):
    started = time.perf_counter()
    large = name == 'kecoli'
    if large:
        from .kecoli import KEcoliModel
        from .cvode import solve as cvode_solve
        model_path = ROOT / 'models/k-ecoli457.zip'
        model = KEcoliModel(model_path)
        system = ReactionSystem.from_kecoli(model)
        prior = np.load(ROOT / 'results/m1-completion/kecoli-reference.npz')
        scale, flux_scale = prior['scale'], prior['flux_scale']
        end, counts = 1e-4, (8, 16, 32)
        gradient = model.flux_jacobian(0., model.y0)
        kappas = np.array([np.max(np.asarray(abs(model.S[:, group] @ gradient[group]).dot(scale)).ravel() / scale)
                           for group in system.groups])
        fast = np.flatnonzero(1e-5 * kappas > 1)
        pools = model.pools
        weights = None
        net_scale = np.array(json.loads((ROOT / 'results/m1-completion/kecoli-net-flux.json').read_text())['net_flux_scale'])
        targets, net_targets = (1e-3, 1e-5), (1e-2, 1e-3)
    else:
        from .sbml import SBMLModel
        model_path = ROOT / 'models/BIOMD0000000064.xml'
        model = SBMLModel(model_path)
        system = ReactionSystem.from_sbml(model)
        prior = np.load(ROOT / 'results/m1/BIOMD0000000064-reference.npz')
        scale = prior['scale']
        all_flux = np.column_stack([model.flux(t, x) for t, x in zip(prior['times'], prior['concentrations'].T)])
        flux_scale = np.maximum(np.max(abs(all_flux), axis=1), 1e-6)
        end, counts = 1e-3, (16, 32, 64)
        fast = np.array([model.reactions.index(r) for r in ('vGAPDH', 'vPGK')])
        pools = model.L.T
        # Formal carbon-weighted concentration ledger of the encoded columns.
        # P is high-energy phosphate, not a carbon-bearing metabolite.
        weights = np.array([6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 2, 0, 21, 21], dtype=float)
        targets, net_targets = (1e-3, 1e-5, 1e-7), ()
    setup_seconds = time.perf_counter() - started
    times_ref = np.linspace(0., end, 2 * counts[-1] + 1)
    if large:
        ref, ref_stats = cvode_solve(model, times_ref, 1e-11, 1e-13 * scale)
        cross = solve_ivp(model.rhs, (0., end), model.y0, method='BDF', rtol=1e-10,
                          atol=1e-12 * scale, jac=model.jacobian, t_eval=times_ref)
    else:
        first = solve_ivp(model.rhs, (0., end), model.y0, method='Radau', rtol=1e-12,
                          atol=1e-14 * scale, jac_sparsity=model.sparsity, t_eval=times_ref)
        if not first.success:
            raise RuntimeError(first.message)
        ref, ref_stats = first.y, {'nfev': first.nfev, 'njev': first.njev, 'nlu': first.nlu}
        cross = solve_ivp(model.rhs, (0., end), model.y0, method='BDF', rtol=1e-12,
                          atol=1e-14 * scale, jac_sparsity=model.sparsity, t_eval=times_ref)
    if not cross.success:
        raise RuntimeError(cross.message)
    flux_ref = np.column_stack([model.flux(t, x) for t, x in zip(times_ref, ref.T)])
    flux_cross = np.column_stack([model.flux(t, x) for t, x in zip(times_ref, cross.y.T)])
    reference_checks = {'concentration': error(cross.y, ref, scale, min(targets)),
                        'flux': error(flux_cross, flux_ref, flux_scale, min(targets))}
    if large:
        net_ref = model.net_map @ flux_ref
        reference_checks['net_flux'] = error(model.net_map @ flux_cross, net_ref, net_scale, min(net_targets))
    if max(reference_checks.values()) > .05:
        raise RuntimeError(f'New short-window reference budget failed: {reference_checks}')
    configs = [(method, count, tol) for tol in (1e-9, 1e-10) for count in counts for method in ('Lie', 'Strang')]
    if probe:
        configs = [('Strang', counts[0], 1e-10)]
    config = {'date': '2026-11-03 Asia/Shanghai', 'model': name, 'model_sha256': digest(model_path),
              'window': [0., end], 'time_unit': 'source coordinate' if large else 'min',
              'initial_state': 'Same initial state as M1; start of transient, no equilibration',
              'scope': 'Bounded subwindow inside the M1 registered window; not a full-window trajectory qualification',
              'fast_groups': fast.tolist(), 'fast_directional_columns': np.concatenate([system.groups[i] for i in fast]).tolist(),
              'group_policy': 'Original elementary forward/reverse map' if large else 'Indivisible original net SBML reactions',
              'partition_policy': 'Frozen at initial h_screen=1e-5, kappa*h>1' if large else 'Frozen GAPDH+PGK from M1',
              'steps': counts, 'inner_rtol': [1e-9, 1e-10], 'inner_atol': '.01 * inner_rtol * frozen M1 species scale',
              'concentration_and_elementary_flux_targets': targets, 'net_flux_targets': net_targets,
              'reference_checks_at_strictest_targets': reference_checks, 'reference_stats': ref_stats,
              'reference_grid': 'Twice finest macro mesh; error reported at macro endpoints and common coarsest mesh, no composition interpolation',
              'repeats': 1 if probe else repeats, 'setup_seconds': setup_seconds,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'platform': platform.platform(), 'threads': {key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
              'ledger': 'Independent 4/8-point Gauss quadrature on each accepted subflow, included in timings' if weights is not None else '456 exact source enzyme pools; no physical whole-network mass claim'}
    if not probe:
        save(OUT / f'{name}-registration.json', config)
    records, trajectories = [], {}
    for repeat in range(-1 if not probe else 0, 1 if probe else repeats):
        # Identical deterministic rotation gives interleaved timings after one warmup/config.
        shift = max(repeat, 0) % len(configs)
        for method, count, tol in configs[shift:] + configs[:shift]:
            row = {'method': method, 'steps': count, 'inner_rtol': tol, 'repeat': repeat}
            partition_start = time.perf_counter()
            split = FixedSplit(system, fast, scale, rtol=tol, ledger_weights=weights, pools=pools)
            row['partition_setup_seconds'] = time.perf_counter() - partition_start
            try:
                times, y, stats = split.integrate(model.y0, 0., end, count, method)
                index = np.arange(count + 1) * (2 * counts[-1] // count)
                common = np.arange(counts[0] + 1) * (count // counts[0])
                reference = ref[:, index]
                flux = np.column_stack([model.flux(t, x) for t, x in zip(times, y.T)])
                drift = np.asarray(pools @ (y - model.y0[:, None]))
                invariant = float(np.max(abs(drift)) / max(1e-12, float(np.max(abs(pools @ model.y0)))))
                row.update(status='success', observed_error={str(r): error(y, reference, scale, r) for r in targets},
                           common_grid_error={str(r): error(y[:, common], reference[:, common], scale, r) for r in targets},
                           flux_error={str(r): error(flux, flux_ref[:, index], flux_scale, r) for r in targets},
                           invariant_error=invariant, min_raw_concentration=float(y.min()), stats=stats)
                if large:
                    row['net_flux_error'] = {str(r): error(model.net_map @ flux, net_ref[:, index], net_scale, r) for r in net_targets}
                if repeat == 0:
                    trajectories[f'{method}-{count}-{tol}'] = y
            except SplitFailure as exc:
                row.update(status='failed', failure_reason=str(exc), stats=exc.stats)
            row['partition_and_integration_seconds'] = row['partition_setup_seconds'] + row['stats']['integration_seconds']
            records.append(row)
            print(name, method, count, tol, repeat, row['status'], round(row['stats']['integration_seconds'], 3), flush=True)
    if probe:
        print(json.dumps({'registration': config, 'records': records}, indent=2), flush=True)
        return
    save(OUT / f'{name}-warmups.json', [r for r in records if r['repeat'] < 0])
    save(OUT / f'{name}-runs.json', [r for r in records if r['repeat'] >= 0])
    np.savez_compressed(OUT / f'{name}-trajectories.npz', reference_times=times_ref, reference=ref, cross_reference=cross.y,
                        scale=scale, flux_scale=flux_scale, **trajectories)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', choices=['b0', 'teusink', 'kecoli'], required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--probe', action='store_true', help='Exploratory single config, no saved formal evidence')
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error('At least seven formal timing repeats required')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case == 'b0':
        b0()
    else:
        real_case(args.case, args.repeats, args.probe)


if __name__ == '__main__':
    main()
