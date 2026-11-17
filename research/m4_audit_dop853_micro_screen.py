"""Audit-only DOP853 micro-screen on saved Kecoli candidate step sizes."""
import itertools
import json
import platform
import time
from types import SimpleNamespace

import numpy as np
from scipy.integrate import BDF, DOP853, solve_ivp

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare

OUT = ROOT / 'results/m4-audit-dop853-micro-screen'
SOURCE_RUNS = ROOT / 'results/m4-growth-factor-screen-v3/runs.json'
SOURCE_REGISTRATION = ROOT / 'results/m4-growth-factor-screen-v3/registration.json'
STATE_INDICES = (0, 61)
QUANTILES = (0., .5, .9, 1.)
TARGETS = (1e-3, 1e-5)
METHODS = ('DOP853', 'BDF', 'CVODE-KLU')
REPEATS = (0, 1, 2)
AUDIT_RTOL = 1e-9
NET_TARGET = 1e-3


def checked_rhs(system, model, scale):
    minimum = [np.inf]
    calls = [0]

    def rhs(t, y):
        if not np.isfinite(y).all() or np.any(y < -1e-12 * scale):
            raise FloatingPointError('nonfinite or negative full-ODE RHS state')
        minimum[0] = min(minimum[0], float(y.min()))
        calls[0] += 1
        return system.S @ model.flux(t, y)

    return rhs, minimum, calls


def stepped(method, rhs, jac, y0, end, query, rtol, atol, scale):
    solver_type = DOP853 if method == 'DOP853' else BDF
    solver = solver_type(rhs, 0., y0.copy(), end, rtol=rtol, atol=atol, jac=jac) if method == 'BDF' else solver_type(
        rhs, 0., y0.copy(), end, rtol=rtol, atol=atol)
    values = np.empty((len(y0), len(query)))
    index = 0
    minimum_internal = float(y0.min())
    while solver.status == 'running':
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(message or f'{method} failed')
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12 * scale):
            raise FloatingPointError('nonfinite or negative accepted audit state')
        minimum_internal = min(minimum_internal, float(solver.y.min()))
        dense = solver.dense_output()
        stop = np.searchsorted(query, solver.t, side='right')
        if stop > index:
            values[:, index:stop] = dense(query[index:stop])
            index = stop
    if index != len(query):
        raise RuntimeError(f'{method} did not cover every audit query')
    return values, dict(accepted_steps=solver.n, nfev=solver.nfev, njev=solver.njev,
                        nlu=getattr(solver, 'nlu', 0), min_internal_state=minimum_internal,
                        internal_states_checked=True)


def integrate(method, case, y0, end, query, rtol):
    system, model, scale = case['system'], case['model'], case['scale']
    rhs, minimum_rhs, calls = checked_rhs(system, model, scale)
    jac = lambda t, y: (system.S @ model.flux_jacobian(t, y)).tocsc()
    atol = np.maximum(scale * (.01 * rtol), 1e-18)
    started = time.perf_counter()
    if method == 'CVODE-KLU':
        from metabolism.cvode import solve as cvode_solve
        native = SimpleNamespace(y0=y0.copy(), rhs=rhs, jacobian=jac, sparsity=model.sparsity)
        values, counts = cvode_solve(native, np.r_[0., query], rtol, atol)
        values = values[:, 1:]
        counts.update(internal_states_checked=False,
                      min_internal_state=None, min_rhs_argument=counts['min_rhs_argument'])
    else:
        values, counts = stepped(method, rhs, jac, y0, end, query, rtol, atol, scale)
    seconds = time.perf_counter() - started
    counts['observed_rhs_calls'] = calls[0]
    counts['min_rhs_argument'] = float(minimum_rhs[0])
    return values, counts, seconds


def score(case, y0, values, truth, target):
    model = case['model']
    options = case['options']
    scale = case['scale']
    flux = np.column_stack([model.flux(0., y) for y in values.T])
    truth_flux = np.column_stack([model.flux(0., y) for y in truth.T])
    concentration = float(np.max(np.abs(values - truth) / (scale[:, None] * (.05 * target))))
    flux_error = float(np.max(np.abs(flux - truth_flux) /
                              (options['flux_scale'][:, None] * (.05 * target))))
    net_flux = 0.
    if 'net_map' in options:
        net_flux = float(np.max(np.abs(options['net_map'] @ (flux - truth_flux)) /
                                (options['net_scale'][:, None] * (.05 * NET_TARGET))))
    pools = options.get('pools')
    pool_error = 0. if pools is None else float(np.max(np.abs(pools @ (values - y0[:, None]))) /
        max(1e-12, float(np.max(np.abs(pools @ y0)))))
    nonnegative = bool(np.all(values >= -1e-12 * scale[:, None]))
    finite = bool(np.isfinite(values).all())
    return dict(concentration_error=concentration, flux_error=flux_error,
                net_flux_error=net_flux, pool_error=pool_error, finite=finite,
                nonnegative=nonnegative, passes=bool(finite and nonnegative and
                    max(concentration, flux_error, net_flux) <= 1. and pool_error <= 1e-10))


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing evidence: {OUT}')
    source_data = json.loads(SOURCE_RUNS.read_text())
    candidate = next(row for row in source_data['rows'] if row['kind'] == 'candidate'
        and row['target'] == 1e-3 and row['growth_safety'] == .9 and row['repeat'] == 0)
    accepted_h = np.asarray([a['h'] for a in candidate['stats']['attempts']
                             if a['status'] == 'accepted'], dtype=float)
    step_sizes = sorted(set(float(np.quantile(accepted_h, q)) for q in QUANTILES))
    OUT.mkdir(parents=True)
    case = prepare('kecoli')
    with np.load(case['reference_path']) as archived:
        state_times = archived['times']
        states = archived['concentrations']
        selected_states = [(index, float(state_times[index]), states[:, index].copy())
                           for index in STATE_INDICES]
    source_files = ('research/m4_audit_dop853_micro_screen.py', 'metabolism/cvode.py',
        'metabolism/kecoli.py', 'metabolism/m3_continuation.py',
        'results/m4-growth-factor-screen-v3/registration.json',
        'results/m4-growth-factor-screen-v3/runs.json')
    registration = dict(date='2026-11-15', model='k-ecoli457',
        scope='Audit-backend micro-screen on actual accepted Kecoli step-size quantiles; not a trajectory or M4 qualification.',
        source_candidate='results/m4-growth-factor-screen-v3, target=1e-3, growth_safety=0.9, repeat=0',
        state_indices=list(STATE_INDICES), state_times=[x[1] for x in selected_states],
        accepted_step_quantiles={str(q): float(np.quantile(accepted_h, q)) for q in QUANTILES},
        step_sizes=step_sizes, targets=list(TARGETS), net_flux_target=NET_TARGET,
        methods=list(METHODS), repeats=list(REPEATS), warmups=1, audit_rtol=AUDIT_RTOL,
        external_error_budget=.05, truth=dict(method='Radau', rtol=1e-13),
        checks=dict(full_nonlinear_rhs=True, every_requested_query=True,
            DOP853_and_BDF_internal_states_and_rhs_domain=True,
            CVODE_rhs_domain_and_outputs_checked=True, nonnegative_outputs=True,
            pools=True, concentration_flux_and_net_flux_against_tightened_truth=True),
        stop_rule='Do not advance unless every DOP853 state/step/target case qualifies and the measured audit plus existing non-audit candidate work has positive end-to-end margin against baseline/1.5.',
        candidate_non_audit_seconds=candidate['seconds'] - candidate['stats']['audit_seconds'],
        local_baselines={str(target): float(np.median([row['seconds'] for row in source_data['rows']
            if row['kind'] == 'baseline' and row['target'] == target and not row['warmup']]))
            for target in TARGETS},
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / 'registration.json', registration)
    rows = []
    cases = []
    for state_index, state_time, y0 in selected_states:
        for end in step_sizes:
            query = end * np.array([.5, 1.])
            rhs, _, _ = checked_rhs(case['system'], case['model'], case['scale'])
            jac = lambda t, y: (case['system'].S @ case['model'].flux_jacobian(t, y)).tocsc()
            ref = solve_ivp(rhs, (0., end), y0, method='Radau', t_eval=query,
                rtol=1e-13, atol=np.maximum(case['scale'] * 1e-15, 1e-18), jac=jac)
            if not ref.success:
                raise RuntimeError(f'Radau truth failed at state={state_index}, h={end}: {ref.message}')
            if np.any(ref.y < -1e-12 * case['scale'][:, None]):
                raise FloatingPointError('tight Radau truth violates the domain')
            truth_path = OUT / f'truth-state-{state_index}-h-{end:.17g}.npz'
            np.savez_compressed(truth_path, initial=y0, query=query, values=ref.y)
            cases.append((state_index, state_time, y0, end, query, ref.y, truth_path.name))
    schedule = list(itertools.product(cases, METHODS))
    for repeat in (-1, *REPEATS):
        offset = max(0, repeat) % len(schedule)
        rotated = schedule[offset:] + schedule[:offset]
        for (state_index, state_time, y0, end, query, truth, truth_name), method in rotated:
            row = dict(state_index=state_index, state_time=state_time, step=end,
                       method=method, repeat=repeat, warmup=repeat < 0)
            try:
                values, counts, seconds = integrate(method, case, y0, end, query, AUDIT_RTOL)
                row.update(status='success', seconds=seconds, counts=counts,
                    metrics={str(target): score(case, y0, values, truth, target) for target in TARGETS})
                artifact = OUT / f'{method}-state-{state_index}-h-{end:.17g}-rep-{repeat}.npz'
                np.savez_compressed(artifact, initial=y0, query=query, values=values)
                row['npz'] = artifact.name
            except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
                row.update(status='failed', reason=str(exc), seconds=None, counts={})
            rows.append(row)
            save(OUT / 'runs.json', dict(rows=rows))
            print(state_index, end, method, repeat, row['status'],
                  round(row.get('seconds') or 0., 6), flush=True)
    summary = []
    for state_index, _, _, end, _, _, _ in cases:
        for method in METHODS:
            samples = [r for r in rows if r['state_index'] == state_index and r['step'] == end
                       and r['method'] == method and not r['warmup']]
            summary.append(dict(state_index=state_index, step=end, method=method,
                repeats=len(samples), seconds=[r.get('seconds') for r in samples],
                median_seconds=float(np.median([r['seconds'] for r in samples if r['seconds'] is not None]))
                    if samples and all(r.get('seconds') is not None for r in samples) else None,
                all_targets_pass=len(samples) == len(REPEATS) and all(r.get('status') == 'success'
                    and all(v['passes'] for v in r['metrics'].values()) for r in samples),
                counts=[r.get('counts') for r in samples]))
    implicit = [r for r in summary if r['method'] in ('BDF', 'CVODE-KLU')
                and r['median_seconds'] is not None and r['all_targets_pass']]
    fastest_implicit = min((r['median_seconds'] for r in implicit), default=None)
    ratios = [fastest_implicit / r['median_seconds'] for r in summary
              if r['method'] == 'DOP853' and r['median_seconds'] and r['all_targets_pass']]
    result = dict(scope=registration['scope'], summary=summary,
        fastest_qualified_implicit_median=fastest_implicit,
        DOP853_speedup_vs_fastest_implicit=min(ratios) if len(ratios) == len(cases) else None,
        candidate_non_audit_seconds=registration['candidate_non_audit_seconds'],
        projected_candidate_seconds_lower_bound=registration['candidate_non_audit_seconds'],
        baseline_over_1_5={target: seconds / 1.5 for target, seconds in registration['local_baselines'].items()},
        local_baselines=registration['local_baselines'], M4_admission=False,
        interpretation='Audit backend calibration only. The projected lower bound excludes audit solves entirely and is compared with same-window baseline/1.5; no formal trajectory claim.')
    save(OUT / 'summary.json', result)
    files = [OUT / row['npz'] for row in rows if row.get('npz')]
    files += [OUT / name for name in ('registration.json', 'runs.json', 'summary.json')]
    save(OUT / 'manifest.json', {str(path.relative_to(ROOT)): digest(path) for path in files})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
