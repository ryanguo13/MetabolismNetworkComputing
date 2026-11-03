"""Audited adaptive reaction splitting, with transactional accepted-step state.

Every proposed Strang step uses one frozen partition for coarse and fine maps.
The first implementation audits midpoint and endpoint against an independent
full ODE step. That expensive safety mechanism is included in all timings.
No PE, extrapolation, clipping, projections or unaudited acceptance is used.
"""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import BDF, Radau
from scipy.sparse import csc_matrix, csr_matrix

from .splitting import FixedSplit, SplitFailure


def select_groups(kappa, h, previous, enter=1., leave=.3):
    previous = set(previous)
    return np.array([i for i, value in enumerate(h * kappa)
                     if value >= (leave if i in previous else enter)], dtype=int)


def normalized_error(value, reference, scale, target):
    weighted = np.abs(value - reference) / (target * (.01 * scale + np.maximum(abs(reference), scale)))
    return float(np.sqrt(np.mean(weighted ** 2))), float(np.max(weighted))


class GroupScreen:
    """Exact rank-one group norms when columns are equal/opposite; general fallback.

The publisher's reversible pairs have opposite columns. Collapsing their flux
gradients before taking absolute values preserves near-equilibrium cancellation
in the Jacobian, without mistaking small net flux itself for a slow reaction.
"""

    def __init__(self, system, scale):
        self.system, self.scale = system, scale
        self.S = csc_matrix(system.S)
        rows, cols, signs, first, general = [], [], [], [], []
        for i, group in enumerate(system.groups):
            first.append(group[0])
            base = self.S[:, group[0]]
            local = []
            for col in group:
                if (self.S[:, col] - base).nnz == 0:
                    local.append(1.)
                elif (self.S[:, col] + base).nnz == 0:
                    local.append(-1.)
                else:
                    general.append(i)
                    break
            if i not in general:
                rows.extend([i] * len(group)); cols.extend(group); signs.extend(local)
        self.mapping = csr_matrix((signs, (rows, cols)), shape=(len(first), self.S.shape[1]))
        self.factors = np.asarray(abs(self.S[:, first]).multiply((1 / scale)[:, None]).max(axis=0).toarray()).ravel()
        self.general = general

    def gradient(self, y):
        if self.system.flux_jacobian is not None:
            return csr_matrix(self.system.flux_jacobian(0., y))
        # SBML and synthetic admitted rates are holomorphic in the valid domain.
        # No finite-difference negative perturbation at a zero initial species.
        columns = []
        for i in range(len(y)):
            z = y.astype(complex)
            z[i] += 1j * 1e-24
            columns.append(self.system.flux(0., z).imag / 1e-24)
        return csr_matrix(np.column_stack(columns))

    def evaluate(self, y, need_kappa=True):
        gradient = self.gradient(y)
        if not need_kappa:
            return None, gradient
        kappa = self.factors * np.asarray(abs(self.mapping @ gradient) @ self.scale).ravel()
        for i in self.general:
            group = self.system.groups[i]
            kappa[i] = np.max(np.asarray(abs(self.S[:, group] @ gradient[group]) @ self.scale).ravel() / self.scale)
        if not np.isfinite(kappa).all():
            raise FloatingPointError('Nonfinite group time-scale proxy')
        return kappa, gradient

    def slow_bound(self, gradient, fast):
        selected = set(fast)
        columns = np.array([col for i, group in enumerate(self.system.groups) if i not in selected for col in group], dtype=int)
        if not len(columns):
            return 0.
        jac = self.S[:, columns] @ gradient[columns]
        return float(np.max(np.asarray(abs(jac) @ self.scale).ravel() / self.scale))


@dataclass(frozen=True)
class AdaptiveConfig:
    target: float = 1e-3
    initial_step: float = .01
    max_step: float = .1
    group_every: int = 5
    enter: float = 1.
    leave: float = .3
    inner_fraction: float = .05
    local_fraction: float = .1
    max_rejections: int = 4
    max_attempts: int = 512
    slow_bound_limit: float = 2.
    fallback_method: str = 'BDF'
    audit_rtol: float = 1e-11
    fallback_rtol: float = 1e-10
    estimator_ratio_limit: float = 2.
    timeout: float = 120.


class AdaptiveSplit:
    def __init__(self, system, scale, config=None, fixed_groups=None,
                 flux_scale=None, net_map=None, net_scale=None, net_target=None,
                 pools=None, ledger_weights=None):
        self.system, self.scale = system, np.asarray(scale, dtype=float)
        self.config = config or AdaptiveConfig()
        c = self.config
        if (not np.isfinite([c.target, c.initial_step, c.max_step, c.enter, c.leave, c.inner_fraction,
                             c.local_fraction, c.slow_bound_limit, c.audit_rtol, c.fallback_rtol,
                             c.estimator_ratio_limit, c.timeout]).all()
                or min(c.target, c.initial_step, c.max_step, c.inner_fraction, c.local_fraction,
                       c.audit_rtol, c.fallback_rtol, c.timeout) <= 0 or not 0 < c.leave <= c.enter
                or c.group_every not in (1, 5, 20) or c.max_rejections < 1 or c.max_attempts < 1
                or c.slow_bound_limit <= 0 or c.estimator_ratio_limit <= 1 or c.fallback_method not in ('BDF', 'Radau', 'CVODE-KLU')):
            raise ValueError('Invalid adaptive configuration')
        self.fixed = None if fixed_groups is None else np.asarray(fixed_groups, dtype=int)
        # Reuse M2 validation, preserving its saved implementation verbatim.
        FixedSplit(system, [] if self.fixed is None else self.fixed, self.scale)
        self.screen = GroupScreen(system, self.scale)
        self.flux_scale = None if flux_scale is None else np.asarray(flux_scale)
        self.net_map, self.net_scale = net_map, None if net_scale is None else np.asarray(net_scale)
        self.net_target = net_target
        self.pools = None if pools is None else csr_matrix(pools)
        self.weights = None if ledger_weights is None else np.asarray(ledger_weights)
        if self.flux_scale is not None and (self.flux_scale.shape != (self.screen.S.shape[1],)
                                           or np.any(self.flux_scale <= 0) or not np.isfinite(self.flux_scale).all()):
            raise ValueError('Invalid flux scale')
        if self.net_map is not None and (self.net_scale is None or net_target is None or net_target <= 0
                                         or np.any(self.net_scale <= 0) or not np.isfinite(self.net_scale).all()):
            raise ValueError('Net-flux map requires positive registered scales and target')

    def integrate(self, times, y0=None):
        """Land on requested times; fallback continues the remaining window once.

        A discarded trial owns no mutable state outside its local variables.
        Only counters and the attempt log survive rejection. Proposal history,
        time, state and the accepted open ledger are committed together.
        """
        c = self.config
        times = np.asarray(times, dtype=float)
        y = np.array(self.system.y0 if y0 is None else y0, dtype=float, copy=True)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or y.shape != self.scale.shape):
            raise ValueError('Invalid output times or state')
        start_clock = time.perf_counter()
        stats = {'attempts': [], 'partition_history': [], 'accepted_steps': 0, 'rejected_steps': 0,
                 'group_evaluations': 0, 'gradient_evaluations': 0, 'partition_switches': 0, 'controller_resets': 0,
                 'fallbacks': [], 'split_seconds': 0., 'audit_seconds': 0., 'fallback_seconds': 0.,
                 'classification_seconds': 0., 'partition_seconds': 0., 'diagnostic_seconds': 0.,
                 'nfev': 0, 'njev': 0, 'nlu': 0, 'observed_rhs_calls': 0,
                 'min_rhs_argument': float(y.min()), 'min_accepted_internal': float(y.min()),
                 'min_output': float(y.min()), 'negative_trial_failures': 0,
                 'max_committed_pool_error': 0., 'max_trial_pool_error': 0.,
                 'ledger_increment': 0., 'ledger_quadrature_difference': 0.,
                 'clipping_count': 0, 'projection_count': 0, 'false_acceptances_prevented': 0,
                 'rollback_checks': 0, 'full_ode_calls': 0}
        stats['native_linear_setups'] = 0
        stats['native_accepted_internal_states_exposed'] = None
        initial = y.copy()
        initial_pool = self.pools @ y if self.pools is not None else None
        pool_denominator = max(1e-12, float(np.max(abs(initial_pool)))) if initial_pool is not None else 1.
        def check(values, where):
            if not np.isfinite(values).all() or np.any(values < -1e-12 * self.scale[:, None] if values.ndim == 2 else values < -1e-12 * self.scale):
                raise RuntimeError(f'Nonfinite or significant negative {where}; no clipping')
        check(y, 'initial state')
        def pool_error(values):
            if self.pools is None:
                return 0.
            if values.ndim == 1:
                values = values[:, None]
            return float(np.max(abs(self.pools @ values - initial_pool[:, None]))) / pool_denominator
        def count_sub(stats_sub):
            for key in ('nfev', 'njev', 'nlu'):
                stats[key] += stats_sub[key]
            stats['observed_rhs_calls'] += stats_sub['rhs_calls_including_fd']
            stats['min_rhs_argument'] = min(stats['min_rhs_argument'], stats_sub['min_rhs_argument'])
            stats['min_accepted_internal'] = min(stats['min_accepted_internal'], stats_sub['min_accepted_state'])
            stats['max_trial_pool_error'] = max(stats['max_trial_pool_error'], stats_sub['max_substep_invariant_error'])
        def full_flow(state, begin, end, query, method, rtol):
            """Independent complete ODE; all accepted internal states observable."""
            stats['full_ode_calls'] += 1
            def rhs(t, x):
                if time.perf_counter() - start_clock > c.timeout:
                    raise TimeoutError('Registered trajectory timeout')
                stats['observed_rhs_calls'] += 1
                stats['min_rhs_argument'] = min(stats['min_rhs_argument'], float(x.min()))
                check(x, 'full ODE RHS argument')
                return self.screen.S @ self.system.flux(0., x)
            def jac(t, x):
                return (self.screen.S @ self.system.flux_jacobian(0., x)).tocsc()
            sparsity = (abs(self.screen.S) @ csc_matrix(self.system.dependencies)).astype(bool)
            if method == 'CVODE-KLU':
                if self.system.flux_jacobian is None or self.weights is not None:
                    raise ValueError('Native fallback requires analytic Jacobian and no quadrature ledger')
                from .cvode import solve
                local = SimpleNamespace(y0=state, rhs=rhs, jacobian=jac, sparsity=sparsity)
                values, native = solve(local, np.r_[0., query - begin], rtol, .01 * rtol * self.scale)
                values = values[:, 1:]
                check(values, 'native full ODE output')
                stats['nfev'] += native['nfev']
                stats['njev'] += native['njev']
                stats['native_linear_setups'] += native['linear_solver_setups']
                stats['native_accepted_internal_states_exposed'] = False
                stats['max_trial_pool_error'] = max(stats['max_trial_pool_error'], pool_error(values))
                # Native internal accepted states are unavailable; RHS arguments
                # and requested outputs are monitored and explicitly distinguished.
                return values, 0., 0.
            options = {'jac': jac} if self.system.flux_jacobian else {'jac_sparsity': sparsity}
            solver = {'BDF': BDF, 'Radau': Radau}[method](rhs, begin, state, end,
                          rtol=rtol, atol=.01 * rtol * self.scale, **options)
            values = np.empty((len(y), len(query)))
            i, ledger, quad_difference = 0, 0., 0.
            try:
                while solver.status == 'running':
                    left = solver.t
                    message = solver.step()
                    if solver.status == 'failed':
                        raise RuntimeError(message)
                    check(solver.y, 'accepted full ODE state')
                    stats['min_accepted_internal'] = min(stats['min_accepted_internal'], float(solver.y.min()))
                    stats['max_trial_pool_error'] = max(stats['max_trial_pool_error'], pool_error(solver.y))
                    right = solver.t
                    dense = solver.dense_output()
                    stop = np.searchsorted(query, right, side='right')
                    if stop > i:
                        values[:, i:stop] = dense(query[i:stop])
                        i = stop
                    if self.weights is not None:
                        tick = time.perf_counter()
                        integrals = []
                        weights_S = np.asarray(self.weights @ self.screen.S).ravel()
                        for order in (4, 8):
                            nodes, weights = np.polynomial.legendre.leggauss(order)
                            ts = left + (right - left) * (nodes + 1) / 2
                            xs = dense(ts)
                            vals = [weights_S @ self.system.flux(0., x) for x in xs.T]
                            integrals.append(float((right - left) / 2 * np.dot(weights, vals)))
                        ledger += integrals[1]
                        quad_difference += abs(integrals[1] - integrals[0])
                        stats['diagnostic_seconds'] += time.perf_counter() - tick
                if i != len(query):
                    raise RuntimeError('Incomplete full ODE output coverage')
                check(values, 'full ODE output')
                return values, ledger, quad_difference
            finally:
                stats['nfev'] += solver.nfev
                stats['njev'] += solver.njev
                stats['nlu'] += solver.nlu
        outputs = np.empty((len(y), len(times)))
        outputs[:, 0] = y
        t, output_index, h = float(times[0]), 1, min(c.initial_step, c.max_step)
        accepted_groups = np.array([], dtype=int)
        cached = None
        force_refresh, consecutive_rejections = True, 0
        # A proportional controller has no hidden PI state to leak on rollback.
        accepted_count_at_refresh = -1
        def fallback(reason):
            nonlocal y, t, output_index
            tick = time.perf_counter()
            query = times[output_index:]
            stats['fallbacks'].append({'time': t, 'reason': reason, 'from_accepted_state': True,
                                      'remaining_fraction': float((times[-1] - t) / (times[-1] - times[0]))})
            try:
                values, ledger, difference = full_flow(y, t, float(times[-1]), query,
                                                       c.fallback_method, c.fallback_rtol)
            finally:
                stats['fallback_seconds'] += time.perf_counter() - tick
            outputs[:, output_index:] = values
            stats['ledger_increment'] += ledger
            stats['ledger_quadrature_difference'] += difference
            stats['max_committed_pool_error'] = max(stats['max_committed_pool_error'], pool_error(values))
            y, t, output_index = values[:, -1].copy(), float(times[-1]), len(times)
        try:
            while output_index < len(times):
                if time.perf_counter() - start_clock > c.timeout:
                    raise TimeoutError('Registered trajectory timeout')
                if len(stats['attempts']) >= c.max_attempts:
                    fallback('registered_split_work_budget')
                    break
                h = min(h, c.max_step, float(times[output_index] - t))
                if h <= 32 * np.finfo(float).eps * max(1., abs(t)):
                    fallback('macro_step_underflow')
                    break
                baseline = (t, y.copy(), accepted_groups.copy(), cached, accepted_count_at_refresh,
                            stats['ledger_increment'], len(stats['partition_history']))
                refresh = self.fixed is None and (force_refresh or cached is None or
                    stats['accepted_steps'] - accepted_count_at_refresh >= c.group_every)
                tick = time.perf_counter()
                # The slow-bound check is done at every accepted/retry boundary;
                # a stale cached gradient never stands in for that check.
                kappa, gradient = self.screen.evaluate(y, need_kappa=refresh)
                stats['gradient_evaluations'] += 1
                stats['group_evaluations'] += int(refresh)
                if self.fixed is not None:
                    proposed = self.fixed.copy()
                elif refresh:
                    proposed = select_groups(kappa, h, accepted_groups, c.enter, c.leave)
                else:
                    proposed = accepted_groups.copy()
                bound = h * self.screen.slow_bound(gradient, proposed)
                stats['classification_seconds'] += time.perf_counter() - tick
                if bound > c.slow_bound_limit:
                    fallback('coupled_slow_norm_bound')
                    break
                if len(proposed) == len(self.system.groups):
                    fallback('all_groups_fast')
                    break
                trial = {'time': t, 'h': h, 'groups': proposed.tolist(), 'refresh': bool(refresh),
                         'slow_scaled_norm_times_h': bound, 'status': 'pending',
                         'accepted_state_index': stats['accepted_steps']}
                changed = not np.array_equal(proposed, accepted_groups)
                try:
                    tick = time.perf_counter()
                    inner = min(1e-8, c.target * c.local_fraction * c.inner_fraction)
                    split = FixedSplit(self.system, proposed, self.scale, rtol=inner,
                                       pools=self.pools, ledger_weights=self.weights,
                                       timeout=max(.01, c.timeout - (time.perf_counter() - start_clock)))
                    stats['partition_seconds'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    try:
                        try:
                            _, coarse, coarse_stats = split.integrate(y, t, t + h, 1)
                            count_sub(coarse_stats)
                            _, fine, fine_stats = split.integrate(y, t, t + h, 2)
                            count_sub(fine_stats)
                        except SplitFailure as exc:
                            count_sub(exc.stats)
                            raise
                    finally:
                        stats['split_seconds'] += time.perf_counter() - tick
                    predicted = (fine[:, -1] - coarse[:, -1]) / 3
                    wrms, maximum = normalized_error(fine[:, -1] - predicted, fine[:, -1], self.scale,
                                                    c.local_fraction * c.target)
                    trial.update(estimator_wrms=wrms, estimator_max=maximum)
                    tick = time.perf_counter()
                    try:
                        reference, _, _ = full_flow(y, t, t + h, np.array([t + h / 2, t + h]),
                                                    'Radau', c.audit_rtol)
                    finally:
                        stats['audit_seconds'] += time.perf_counter() - tick
                    actual = [normalized_error(fine[:, i + 1], reference[:, i], self.scale,
                                                c.local_fraction * c.target) for i in range(2)]
                    actual_max, actual_wrms = max(v[1] for v in actual), max(v[0] for v in actual)
                    flux_max = net_max = 0.
                    if self.flux_scale is not None or self.net_map is not None:
                        for i in range(2):
                            vf = self.system.flux(0., fine[:, i + 1])
                            vr = self.system.flux(0., reference[:, i])
                            if self.flux_scale is not None:
                                flux_max = max(flux_max, normalized_error(vf, vr, self.flux_scale, c.local_fraction * c.target)[1])
                            if self.net_map is not None:
                                net_max = max(net_max, normalized_error(self.net_map @ vf, self.net_map @ vr,
                                                                       self.net_scale, c.local_fraction * self.net_target)[1])
                    trial.update(audit_wrms=actual_wrms, audit_max=actual_max, audit_flux_max=flux_max,
                                 audit_net_flux_max=net_max, estimator_ratio=actual[-1][1] / max(maximum, 1e-15))
                    failed_audit = max(actual_max, flux_max, net_max) > 1
                    unreliable = actual[-1][1] > max(c.estimator_ratio_limit * maximum, .1)
                    if wrms <= 1 and failed_audit:
                        stats['false_acceptances_prevented'] += 1
                    if unreliable:
                        trial['reason'] = 'estimator_outside_calibrated_regime'
                    elif failed_audit or wrms > 1:
                        trial['reason'] = 'local_error_or_observable_budget'
                    else:
                        check(fine, 'candidate fine map')
                        committed_pool = pool_error(fine)
                        if committed_pool > 1e-10:
                            trial['reason'] = 'invariant_drift'
                        else:
                            trial['status'] = 'accepted'
                            y, t = fine[:, -1].copy(), t + h
                            accepted_groups = proposed.copy()
                            if refresh:
                                cached, accepted_count_at_refresh = kappa.copy(), stats['accepted_steps']
                            stats['accepted_steps'] += 1
                            if changed:
                                stats['partition_switches'] += 1
                                stats['controller_resets'] += 1
                            stats['partition_history'].append({'time': t - h, 'h': h, 'groups': proposed.tolist(),
                                                               'changed': changed, 'refresh': bool(refresh)})
                            stats['ledger_increment'] += fine_stats['ledger_increment']
                            stats['ledger_quadrature_difference'] += fine_stats['ledger_quadrature_difference']
                            stats['max_committed_pool_error'] = max(stats['max_committed_pool_error'], committed_pool)
                            if t >= times[output_index]:
                                outputs[:, output_index] = y
                                output_index += 1
                            consecutive_rejections, force_refresh = 0, False
                            control = max(wrms, actual_max, flux_max, net_max)
                            factor = 2. if control == 0 else float(np.clip(.9 * control ** (-1 / 3), .2, 2.))
                            # A new partition starts from proportional control, no inherited growth.
                            h *= min(factor, 1.) if changed else factor
                except (SplitFailure, FloatingPointError) as exc:
                    trial['reason'] = str(exc)
                    trial['domain_failure'] = True
                    if 'negative' in str(exc):
                        stats['negative_trial_failures'] += 1
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    trial.update(status='failed', reason=str(exc), rollback_verified=bool(
                        t == baseline[0] and np.array_equal(y, baseline[1]) and
                        np.array_equal(accepted_groups, baseline[2]) and stats['ledger_increment'] == baseline[5]))
                    stats['attempts'].append(trial)
                    raise
                if trial['status'] != 'accepted':
                    trial['status'] = 'rejected'
                    stats['rejected_steps'] += 1
                    consecutive_rejections += 1
                    same_cache = (cached is baseline[3])
                    rolled_back = (t == baseline[0] and np.array_equal(y, baseline[1]) and
                                   np.array_equal(accepted_groups, baseline[2]) and same_cache and
                                   accepted_count_at_refresh == baseline[4] and stats['ledger_increment'] == baseline[5]
                                   and len(stats['partition_history']) == baseline[6])
                    trial['rollback_verified'] = bool(rolled_back)
                    if not rolled_back:
                        raise RuntimeError('Rejected trial mutated accepted state')
                    stats['rollback_checks'] += 1
                    force_refresh = True
                    if trial['reason'] == 'estimator_outside_calibrated_regime':
                        stats['attempts'].append(trial)
                        fallback(trial['reason'])
                        break
                    if consecutive_rejections >= c.max_rejections:
                        stats['attempts'].append(trial)
                        fallback('consecutive_rejections')
                        break
                    h *= .5
                stats['attempts'].append(trial)
            check(outputs, 'returned trajectory')
            stats['min_output'] = float(outputs.min())
            if self.weights is not None:
                stats['ledger_residual'] = float(self.weights @ (outputs[:, -1] - initial) - stats['ledger_increment'])
            stats['status'] = 'success'
            return outputs, stats
        except (RuntimeError, ValueError, TimeoutError) as exc:
            stats['status'], stats['failure_reason'] = 'failed', str(exc)
            raise SplitFailure(str(exc), stats) from exc
        finally:
            stats['total_seconds'] = time.perf_counter() - start_clock


def integrate_segments(systems, boundaries, times, scale, config=None, **options):
    """Known piecewise-constant inputs: stop at each event and restart history.

    Systems are supplied separately; no in-place parameter mutation or autonomous
    Strang map is allowed to cross an event. Concentrations are continuous here.
    """
    boundaries, times = np.asarray(boundaries), np.asarray(times)
    if (len(boundaries) != len(systems) + 1 or np.any(np.diff(boundaries) <= 0)
            or times[0] != boundaries[0] or times[-1] != boundaries[-1] or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid constant-input segments')
    first = systems[0]
    for system in systems[1:]:
        if ((csc_matrix(system.S) != csc_matrix(first.S)).nnz or len(system.groups) != len(first.groups)
                or any(not np.array_equal(a, b) for a, b in zip(system.groups, first.groups))):
            raise ValueError('Event segments must retain state coordinates and reaction groups')
    state = first.y0.copy()
    output = np.empty((len(state), len(times)))
    output[:, 0] = state
    records = []
    for index, system in enumerate(systems):
        lo, hi = boundaries[index:index + 2]
        mask = (times > lo) & (times <= hi)
        segment_times = np.unique(np.r_[lo, times[mask], hi])
        values, stats = AdaptiveSplit(system, scale, config, **options).integrate(segment_times, state)
        output[:, mask] = values[:, np.searchsorted(segment_times, times[mask])]
        state = values[:, -1].copy()
        records.append(stats)
    return output, records
