"""Local, offline M3 research; never used as an acceptance controller.

Run: bash scripts/run-macos.sh -m research.m3_coupling_probe
All methods use the same native subsolver to isolate the coupling formula.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.adaptive import GroupScreen
from metabolism.cvode import solve
from metabolism.m2_experiments import error, save
from metabolism.m3_continuation import prepare, registration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m3-research'


def step(method, y, h, fast, slow, flow):
    if method == 'mri2':
        # Sandu arXiv v3, (2.8), rescaled to physical half intervals.
        g0 = slow(y)
        mid = flow('fast', y, h/2, g0)
        return flow('fast', mid, h/2, -g0 + 2*slow(mid))
    if method not in ('strang', 'balanced'):
        raise ValueError(method)
    # Speth et al. (5.1)-(5.6), A=fast, B=slow; not rebalanced splitting.
    offset = (slow(y)-fast(y))/2 if method == 'balanced' else np.zeros_like(y)
    a = flow('fast', y, h/2, offset)
    b = flow('slow', a, h, -offset)
    return flow('fast', b, h/2, offset)


def self_checks():
    a = np.array([[-3., 1.], [3., -1.]])
    b = np.array([[-.2, 2.], [.2, -2.]])
    matrices = dict(fast=a, slow=b)
    def flow(name, y, h, offset):
        augmented = np.zeros((3, 3))
        augmented[:2, :2] = matrices[name]
        augmented[:2, 2] = offset
        return (expm(h*augmented) @ np.r_[y, 1.])[:2]
    y0 = np.array([.8, .2])
    reference = expm(a+b) @ y0
    results = {}
    for method in ('strang', 'balanced', 'mri2'):
        errors = []
        for n in (20, 40, 80):
            y = y0.copy()
            for _ in range(n):
                y = step(method, y, 1/n, lambda z:a@z, lambda z:b@z, flow)
            assert abs(y.sum()-1) < 1e-12
            errors.append(float(np.max(abs(y-reference))))
        ratios = np.array(errors[:-1])/errors[1:]
        assert np.all((ratios > 3.5) & (ratios < 4.5)), (method, errors)
        equilibrium = np.array([3/6.2, 3.2/6.2])
        deviation = np.max(abs(step(method, equilibrium, .1, lambda z:a@z,
                                   lambda z:b@z, flow)-equilibrium))
        if method != 'strang':
            assert deviation < 1e-14
        results[method] = dict(errors=errors, ratios=ratios.tolist(),
                               steady_state_deviation=float(deviation))
    # The exact affine map also checks the native forcing/Jacobian convention.
    offset = np.array([.1, -.1])
    model = SimpleNamespace(y0=y0, rhs=lambda t,y:a@y+offset,
                            jacobian=lambda t,y:csc_matrix(a), sparsity=csc_matrix(a != 0))
    actual, _ = solve(model, np.array([0., .1]), 1e-12, 1e-14)
    np.testing.assert_allclose(actual[:, -1], flow('fast', y0, .1, offset), atol=1e-10, rtol=1e-10)
    return results


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = self_checks()
    case = prepare('kecoli')
    model, system, scale = case['model'], case['system'], case['scale']
    screen = GroupScreen(system, scale)
    prefix, prefix_stats = solve(model, np.array([0., .001, .01]), 1e-11, 1e-13*scale)
    arrays = dict(prefix=prefix, scale=scale)
    rows, partitions, references = [], [], []

    def metrics(values, ref):
        vf = np.column_stack([model.flux(0., y) for y in values.T])
        vr = np.column_stack([model.flux(0., y) for y in ref.T])
        return dict(concentration=error(values, ref, scale, 1e-4),
                    flux=error(vf, vr, case['options']['flux_scale'], 1e-4),
                    net_flux=error(model.net_map@vf, model.net_map@vr, case['options']['net_scale'], 1e-4))

    for si, start in enumerate((.001, .01), 1):
        y = prefix[:, si]
        tick = time.perf_counter()
        kappa, gradient = screen.evaluate(y)
        groups = np.flatnonzero(1e-3*kappa >= .01)
        fast_cols = np.concatenate([system.groups[g] for g in groups])
        slow_cols = np.setdiff1d(np.arange(system.S.shape[1]), fast_cols)
        columns = dict(fast=fast_cols, slow=slow_cols)
        matrices = {k:system.S[:, cols] for k, cols in columns.items()}
        patterns = {k:(abs(matrices[k])@system.dependencies[cols]).astype(bool).tocsc()
                    for k, cols in columns.items()}
        def field(name, z):
            return matrices[name] @ model.flux(0., z)[columns[name]]
        fast, slow = lambda z:field('fast', z), lambda z:field('slow', z)
        f, g = fast(y), slow(y)
        jf, jg = (matrices[k]@gradient[columns[k]] for k in ('fast', 'slow'))
        bracket = jg@f-jf@g
        partitions.append(dict(start=start, fast_groups=groups.tolist(),
            fast_updated_states=int(np.count_nonzero(np.asarray(abs(matrices['fast']).sum(axis=1)))),
            slow_updated_states=int(np.count_nonzero(np.asarray(abs(matrices['slow']).sum(axis=1)))),
            classification_seconds=time.perf_counter()-tick,
            scaled_fast=float(np.max(abs(f)/scale)), scaled_slow=float(np.max(abs(g)/scale)),
            scaled_total=float(np.max(abs(f+g)/scale)),
            scaled_commutator=float(np.max(abs(bracket)/scale)),
            slow_jacobian_bound=screen.slow_bound(gradient, groups)))
        local = SimpleNamespace(y0=y, rhs=model.rhs, jacobian=model.jacobian, sparsity=model.sparsity)
        for hi, h in enumerate((1e-6, 1e-4, 1e-3)):
            query = h*np.array([0., .25, .5, .75, 1.])
            tick = time.perf_counter()
            ref, counts = solve(local, query, 1e-12, 1e-14*scale)
            ref_seconds = time.perf_counter()-tick
            loose, _ = solve(local, query, 1e-11, 1e-13*scale)
            tightening = metrics(loose, ref)
            references.append(dict(start=start, h=h, tightening=tightening,
                                   seconds=ref_seconds, counts=counts))
            arrays[f'ref-{si}-{hi}'] = ref
            arrays[f'ref-loose-{si}-{hi}'] = loose
            # A common 1e-8 sweep, plus identical-map tightening at h=1e-4.
            for tol in ((1e-8, 1e-10) if h == 1e-4 else (1e-8,)):
                for method in ('strang', 'balanced', 'mri2'):
                    tick = time.perf_counter()
                    stats = dict(subflow_calls=0, nfev=0, njev=0, linear_solver_setups=0,
                                 min_rhs=float(y.min()), min_subflow_output=float(y.min()),
                                 pool_drift=0.)
                    def validate(z, where):
                        if time.perf_counter()-tick > 25:
                            raise TimeoutError('25 second local attempt budget')
                        if not np.isfinite(z).all() or np.any(z < -1e-12*scale):
                            raise ValueError(f'nonfinite/significant negative {where}; no clipping')
                    def flow(name, z, duration, offset):
                        stats['subflow_calls'] += 1
                        def rhs(t, q):
                            stats['min_rhs'] = min(stats['min_rhs'], float(q.min()))
                            validate(q, 'RHS argument')
                            return field(name, q)+offset
                        def jac(t, q):
                            validate(q, 'Jacobian argument')
                            return (matrices[name]@model.flux_jacobian(0., q)[columns[name]]).tocsc()
                        sub = SimpleNamespace(y0=z, rhs=rhs, jacobian=jac, sparsity=patterns[name])
                        vals, counts = solve(sub, np.array([0., duration]), tol, .01*tol*scale)
                        answer = vals[:, -1]
                        validate(answer, 'subflow output')
                        stats['min_subflow_output'] = min(stats['min_subflow_output'], float(answer.min()))
                        stats['pool_drift'] = max(stats['pool_drift'], float(np.max(abs(model.pools@(answer-y)))))
                        for key in ('nfev', 'njev', 'linear_solver_setups'):
                            stats[key] += counts[key]
                        return answer
                    row = dict(start=start, h=h, inner_rtol=tol, method=method, stats=stats)
                    key = f'{si}-{hi}-{tol}-{method}'
                    try:
                        coarse = step(method, y, h, fast, slow, flow)
                        mid = step(method, y, h/2, fast, slow, flow)
                        fine = step(method, mid, h/2, fast, slow, flow)
                        elapsed = time.perf_counter()-tick
                        vals = np.column_stack([y, (y+mid)/2, mid, (mid+fine)/2, fine])
                        arrays[key] = vals
                        arrays[key+'-coarse'] = coarse
                        endpoint = metrics(fine[:, None], ref[:, -1:])
                        grid = metrics(vals, ref)
                        delta = metrics(fine[:, None], coarse[:, None])
                        row.update(status='success', seconds=elapsed, endpoint=endpoint, grid=grid,
                            step_doubling_over_three={k:v/3 for k,v in delta.items()},
                            local_reference_budget_pass=max(tightening.values()) <= .05,
                            endpoint_pass=max(endpoint.values()) <= 1,
                            grid_pass=max(grid.values()) <= 1)
                    except (RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                        row.update(status='failed', seconds=time.perf_counter()-tick, reason=str(exc))
                    rows.append(row)
                    save(OUT/'probe.json', dict(scope='Offline local mechanism probes, no controller, no full-window or speedup qualification. All subflows are full-coordinate CVODE-KLU, including the slow Strang subflow. Common linear output interpolation; timings include coarse+two fine maps, exclude offline references and error postprocessing.',
                        registration=registration(case), self_checks=checks, prefix_stats=prefix_stats,
                        partitions=partitions, references=references, runs=rows,
                        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
                    np.savez_compressed(OUT/'probe.npz', **arrays)
                    print(start, h, tol, method, row['status'], round(row['seconds'], 3),
                          row.get('endpoint'), row.get('reason', ''), flush=True)


if __name__ == '__main__':
    run()
