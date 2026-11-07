"""M3 continuation: macro steps independent of observation times.

The continuous extension is piecewise linear through the two accepted fine
steps. Every requested output, quarter point and endpoint is independently
audited. It preserves linear pools and positivity by convexity, but is only
qualified on the registered output grid. No full-ODE audit state is committed
as a split result. Original M1–M3 implementations and evidence stay unchanged.
"""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from .adaptive import AdaptiveConfig, AdaptiveSplit, normalized_error, select_groups
from .splitting import FixedSplit, SplitFailure


@dataclass(frozen=True)
class ContinuationConfig(AdaptiveConfig):
    startup_end: float = 0.
    audit_method: str = 'Radau'


class MacroSplit(AdaptiveSplit):
    """Reuse validated grouping and subflows, cache partitions across steps.

    Independent actual-error audits control acceptance even when the doubling
    estimate loses order. Such mismatches remain explicit diagnostics; they
    never license unaudited acceptance or a claim of a cheap estimator.
    """

    def __init__(self, system, scale, config=None, **options):
        super().__init__(system, scale, config or ContinuationConfig(), **options)

    def integrate(self, times, y0=None):
        c = self.config
        grid = np.asarray(times, dtype=float)
        state = np.array(self.system.y0 if y0 is None else y0, dtype=float, copy=True)
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0) or state.shape != self.scale.shape
                or c.audit_method not in ('Radau', 'CVODE-KLU')
                or not np.isfinite(c.startup_end) or not grid[0] <= c.startup_end <= grid[-1]):
            raise ValueError('Invalid continuation grid, state or configuration')
        start = time.perf_counter()
        initial = state.copy()
        initial_pool = self.pools @ state if self.pools is not None else None
        denominator = max(1e-12, float(np.max(abs(initial_pool)))) if initial_pool is not None else 1.
        stats = dict(attempts=[], accepted_steps=0, rejected_steps=0, partition_builds=0,
                     group_evaluations=0, partition_switches=0, audit_seconds=0.,
                     split_seconds=0., classification_seconds=0., partition_seconds=0.,
                     full_seconds=0., split_coverage=0., full_intervals=[], full_calls=0,
                     estimator_mismatches=0, output_points_audited=0, rollback_checks=0,
                     nfev=0, njev=0, nlu=0, native_linear_setups=0,
                     ledger_increment=0., ledger_quadrature_difference=0.,
                     max_pool_error=0., min_rhs_argument=float(state.min()),
                     min_output=float(state.min()), clipping_count=0, projection_count=0)

        def check(values):
            scales = self.scale[:, None] if values.ndim == 2 else self.scale
            if not np.isfinite(values).all() or np.any(values < -1e-12 * scales):
                raise RuntimeError('Invalid concentration domain; no clipping')
            if self.pools is not None:
                values = values[:, None] if values.ndim == 1 else values
                drift = float(np.max(abs(self.pools @ values - initial_pool[:, None]))) / denominator
                stats['max_pool_error'] = max(stats['max_pool_error'], drift)
                if drift > 1e-10:
                    raise RuntimeError('Invariant drift')

        def rhs(t, y):
            if time.perf_counter() - start > c.timeout:
                raise TimeoutError('Registered continuation timeout')
            stats['min_rhs_argument'] = min(stats['min_rhs_argument'], float(y.min()))
            # Newton probes are checked for the kinetic domain, not conservation.
            if not np.isfinite(y).all() or np.any(y < -1e-12 * self.scale):
                raise RuntimeError('Invalid full ODE RHS domain; no clipping')
            return self.screen.S @ self.system.flux(0., y)

        def jac(t, y):
            return (self.screen.S @ self.system.flux_jacobian(0., y)).tocsc()

        def full(y, begin, end, query, method, rtol):
            stats['full_calls'] += 1
            if method == 'CVODE-KLU':
                if self.system.flux_jacobian is None:
                    raise ValueError('CVODE requires an analytic Jacobian')
                from .cvode import solve
                model = SimpleNamespace(y0=y, rhs=rhs, jacobian=jac,
                                        sparsity=(abs(self.screen.S) @ self.system.dependencies).astype(bool))
                values, counts = solve(model, np.r_[0., query - begin], rtol, .01 * rtol * self.scale)
                stats['nfev'] += counts['nfev']; stats['njev'] += counts['njev']
                stats['native_linear_setups'] += counts['linear_solver_setups']
                check(values)
                return values[:, 1:]
            options = {'jac': jac} if self.system.flux_jacobian else {
                'jac_sparsity': (abs(self.screen.S) @ self.system.dependencies).astype(bool)}
            result = solve_ivp(rhs, (begin, end), y, method=method, rtol=rtol,
                               atol=.01 * rtol * self.scale, dense_output=True, **options)
            stats['nfev'] += result.nfev; stats['njev'] += result.njev; stats['nlu'] += result.nlu
            if not result.success:
                raise RuntimeError(result.message)
            check(result.y)
            values = result.sol(query)
            check(values)
            return values

        def observable_error(values, reference):
            maximum = normalized_error(values, reference, self.scale[:, None], c.local_fraction * c.target)[1]
            flux_max = net_max = 0.
            if self.flux_scale is not None or self.net_map is not None:
                vf = np.column_stack([self.system.flux(0., y) for y in values.T])
                vr = np.column_stack([self.system.flux(0., y) for y in reference.T])
                if self.flux_scale is not None:
                    flux_max = normalized_error(vf, vr, self.flux_scale[:, None], c.local_fraction * c.target)[1]
                if self.net_map is not None:
                    net_max = normalized_error(self.net_map @ vf, self.net_map @ vr,
                                               self.net_scale[:, None], c.local_fraction * self.net_target)[1]
            return maximum, flux_max, net_max

        output = np.empty((len(state), len(grid)))
        output[:, 0] = state
        t, index, h = float(grid[0]), 1, min(c.initial_step, c.max_step)
        groups, partition, kappa = np.array([], dtype=int), None, None
        rejections = 0

        def commit_full(end, reason):
            nonlocal state, t, index
            stop = int(np.searchsorted(grid, end, side='right'))
            query = np.unique(np.r_[grid[index:stop], end])
            tick = time.perf_counter()
            # Reuse the original audited persistent full-flow path, including
            # its open ledger and accepted-state checks, for committed intervals.
            cfg = AdaptiveConfig(target=c.target, fallback_method=c.fallback_method,
                                 fallback_rtol=c.fallback_rtol,
                                 timeout=max(.001, c.timeout - (tick - start)))
            controller = AdaptiveSplit(self.system, self.scale, cfg,
                                       fixed_groups=np.arange(len(self.system.groups)),
                                       pools=self.pools, ledger_weights=self.weights)
            values, counts = controller.integrate(np.r_[t, query], state)
            stats['full_seconds'] += time.perf_counter() - tick
            stats['full_intervals'].append(dict(start=t, end=float(end), reason=reason))
            for key in ('nfev', 'njev', 'nlu', 'native_linear_setups', 'ledger_increment', 'ledger_quadrature_difference'):
                stats[key] += counts[key]
            output[:, index:stop] = values[:, 1:][:, np.searchsorted(query, grid[index:stop])]
            state, t, index = values[:, -1].copy(), float(end), stop

        try:
            check(state)
            if c.startup_end > t:
                commit_full(c.startup_end, 'registered_initial_layer')
            while t < grid[-1]:
                if time.perf_counter() - start > c.timeout:
                    raise TimeoutError('Registered continuation timeout')
                if len(stats['attempts']) >= c.max_attempts:
                    commit_full(grid[-1], 'registered_work_budget')
                    break
                h = min(h, c.max_step, float(grid[-1] - t))
                if h <= 32 * np.finfo(float).eps * max(1., abs(t)):
                    commit_full(grid[-1], 'macro_step_underflow')
                    break
                tick = time.perf_counter()
                refresh = kappa is None or rejections > 0 or stats['accepted_steps'] % c.group_every == 0
                new_kappa, gradient = self.screen.evaluate(state, need_kappa=refresh)
                stats['group_evaluations'] += int(refresh)
                if refresh:
                    kappa = new_kappa
                # Apply the current h to cached norms at every boundary, so a
                # growing step cannot silently retain a stale slow partition.
                proposed = self.fixed if self.fixed is not None else select_groups(kappa, h, groups, c.enter, c.leave)
                bound = h * self.screen.slow_bound(gradient, proposed)
                stats['classification_seconds'] += time.perf_counter() - tick
                if bound > c.slow_bound_limit:
                    h *= .5
                    rejections += 1
                    if rejections >= c.max_rejections:
                        commit_full(grid[-1], 'coupled_slow_norm_bound')
                        break
                    continue
                if len(proposed) == len(self.system.groups):
                    commit_full(grid[-1], 'all_groups_fast')
                    break
                tick = time.perf_counter()
                changed = not np.array_equal(proposed, groups)
                if partition is None or changed:
                    partition = FixedSplit(self.system, proposed, self.scale,
                                           rtol=min(1e-8, c.target * c.local_fraction * c.inner_fraction),
                                           pools=self.pools, ledger_weights=self.weights, timeout=c.timeout)
                    stats['partition_builds'] += 1
                stats['partition_seconds'] += time.perf_counter() - tick
                stop = int(np.searchsorted(grid, t + h, side='right'))
                query = np.unique(np.r_[t + h * np.array([.25, .5, .75, 1.]), grid[index:stop]])
                trial = dict(time=t, h=h, groups=proposed.tolist(), output_count=stop-index,
                             slow_scaled_norm_times_h=bound, status='pending')
                before = (state.copy(), t, index, stats['ledger_increment'], groups.copy())
                tick = time.perf_counter()
                try:
                    partition.timeout = max(.001, c.timeout - (tick - start))
                    _, coarse, cs = partition.integrate(state, t, t + h, 1)
                    for key in ('nfev', 'njev', 'nlu'):
                        stats[key] += cs[key]
                    _, fine, fs = partition.integrate(state, t, t + h, 2)
                    for key in ('nfev', 'njev', 'nlu'):
                        stats[key] += fs[key]
                    stats['split_seconds'] += time.perf_counter() - tick
                    for counts in (cs, fs):
                        stats['min_rhs_argument'] = min(stats['min_rhs_argument'], counts['min_rhs_argument'])
                    # Convex interpolation is checked, never used as a substitute
                    # for an independently calibrated splitting dense extension.
                    values = np.vstack([np.interp(query, [t, t+h/2, t+h], row) for row in fine])
                    check(values)
                    tick = time.perf_counter()
                    reference = full(state, t, t+h, query, c.audit_method, c.audit_rtol)
                    stats['audit_seconds'] += time.perf_counter() - tick
                    conc, flux, net = observable_error(values, reference)
                    estimate = normalized_error(fine[:, -1] - (fine[:, -1]-coarse[:, -1])/3, fine[:, -1],
                                                self.scale, c.local_fraction*c.target)[1]
                    endpoint_error = normalized_error(fine[:, -1], reference[:, -1], self.scale, c.local_fraction*c.target)[1]
                    mismatch = endpoint_error > max(c.estimator_ratio_limit * estimate, .1)
                    stats['estimator_mismatches'] += int(mismatch)
                    trial.update(audit_max=conc, audit_flux_max=flux, audit_net_flux_max=net,
                                 estimator_max=estimate, estimator_mismatch=bool(mismatch))
                    control = max(conc, flux, net, estimate)
                    if control <= 1:
                        trial['status'] = 'accepted'
                        output[:, index:stop] = values[:, np.searchsorted(query, grid[index:stop])]
                        state, t, index = fine[:, -1].copy(), t+h, stop
                        stats['accepted_steps'] += 1
                        stats['split_coverage'] += h / (grid[-1]-grid[0])
                        stats['partition_switches'] += int(changed)
                        stats['output_points_audited'] += stop - before[2]
                        stats['ledger_increment'] += fs['ledger_increment']
                        stats['ledger_quadrature_difference'] += fs['ledger_quadrature_difference']
                        groups = proposed.copy()
                        rejections = 0
                        factor = 2. if control == 0 else float(np.clip(.9*control**(-1/3), .2, 2.))
                        h *= min(factor, 1.) if changed else factor
                    else:
                        trial['reason'] = 'audited_error_or_estimator_budget'
                except SplitFailure as exc:
                    stats['split_seconds'] += time.perf_counter() - tick
                    for key in ('nfev', 'njev', 'nlu'):
                        stats[key] += exc.stats.get(key, 0)
                    trial['reason'] = str(exc)
                if trial['status'] != 'accepted':
                    trial['status'] = 'rejected'
                    trial['rollback_verified'] = bool(np.array_equal(state, before[0]) and t == before[1]
                        and index == before[2] and stats['ledger_increment'] == before[3]
                        and np.array_equal(groups, before[4]))
                    if not trial['rollback_verified']:
                        raise RuntimeError('Rejected trial mutated committed state')
                    stats['rollback_checks'] += 1
                    stats['rejected_steps'] += 1
                    rejections += 1
                    h *= .5
                    partition = None
                stats['attempts'].append(trial)
                if rejections >= c.max_rejections:
                    commit_full(grid[-1], 'consecutive_rejections')
                    break
            check(output)
            stats['min_output'] = float(output.min())
            if self.weights is not None:
                stats['ledger_residual'] = float(self.weights @ (state-initial) - stats['ledger_increment'])
            stats['status'] = 'success'
            return output, stats
        except (RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
            stats.update(status='failed', failure_reason=str(exc))
            raise SplitFailure(str(exc), stats) from exc
        finally:
            stats['total_seconds'] = time.perf_counter() - start
