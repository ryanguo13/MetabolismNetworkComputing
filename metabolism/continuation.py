"""M3 continuation: macro steps independent of observation times.

The continuous extension is piecewise linear through the two accepted fine
steps. Every requested output, quarter point and endpoint is independently
audited. It preserves linear pools and positivity by convexity, but is only
qualified on the registered output grid. No full-ODE audit state is committed
as a split result. Original M1–M3 implementations and evidence stay unchanged.
"""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
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
    audit_control_only: bool = False
    classification_step_floor: float = 0.
    growth_safety: float = .9
    rejection_audit_cache: bool = False
    split_substeps: int = 1
    # Fail-closed pre-audit filter.  A candidate can be rejected cheaply, but
    # every candidate that is accepted still passes the independent full ODE
    # audit below.
    shadow_first: bool = False
    shadow_limit: float = 1.0
    # Run candidate generation and its independent full-ODE audit concurrently
    # from the same committed state.  Disabled by default; incompatible with
    # cache/shadow modes because those modes change audit scheduling semantics.
    overlap_audit: bool = False


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
                or c.audit_method not in ('Radau', 'BDF', 'CVODE-KLU')
                or (c.rejection_audit_cache and c.audit_method == 'CVODE-KLU')
                or (c.rejection_audit_cache and self.system.time_dependent)
                or not np.isfinite(c.startup_end) or not 0 <= c.classification_step_floor
                or not np.isfinite(c.growth_safety) or not .5 <= c.growth_safety <= 2.
                or not isinstance(c.split_substeps, int) or isinstance(c.split_substeps, bool)
                or c.split_substeps < 1 or not grid[0] <= c.startup_end <= grid[-1]
                or not isinstance(c.shadow_first, bool) or not np.isfinite(c.shadow_limit)
                or c.shadow_limit <= 0 or not isinstance(c.overlap_audit, bool)
                or (c.overlap_audit and (c.rejection_audit_cache or c.shadow_first))):
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
                     coarse_split_calls=0, fine_split_calls=0,
                     audit_calls=0, audit_solver_calls=0, audit_cache_hits=0,
                     shadow_rejections=0, shadow_seconds=0.,
                     overlap_trials=0, overlap_seconds=0.,
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

        audit_cache = None

        def audited_reference(y, begin, end, query):
            nonlocal audit_cache
            stats['audit_calls'] += 1
            if not c.rejection_audit_cache:
                stats['audit_solver_calls'] += 1
                return full(y, begin, end, query, c.audit_method, c.audit_rtol)
            reusable = (audit_cache is not None and audit_cache['begin'] == begin
                        and audit_cache['method'] == c.audit_method
                        and audit_cache['rtol'] == c.audit_rtol
                        and np.array_equal(audit_cache['state'], y)
                        and end <= audit_cache['end']
                        and np.min(query) >= begin and np.max(query) <= audit_cache['end'])
            if reusable:
                stats['full_calls'] += 1
                stats['audit_cache_hits'] += 1
                values = audit_cache['solution'](query)
                check(values)
                return values
            stats['audit_solver_calls'] += 1
            stats['full_calls'] += 1
            def dense_rhs(q, z):
                return rhs(q, z)
            options = {'jac': jac} if self.system.flux_jacobian else {
                'jac_sparsity': (abs(self.screen.S) @ self.system.dependencies).astype(bool)}
            result = solve_ivp(dense_rhs, (begin, end), y, method=c.audit_method,
                               rtol=c.audit_rtol, atol=.01*c.audit_rtol*self.scale,
                               dense_output=True, **options)
            stats['nfev'] += result.nfev
            stats['njev'] += result.njev
            stats['nlu'] += result.nlu
            if not result.success:
                audit_cache = None
                raise RuntimeError(result.message)
            check(result.y)
            values = result.sol(query)
            check(values)
            audit_cache = dict(begin=begin, end=end, state=np.array(y, copy=True),
                               method=c.audit_method, rtol=c.audit_rtol,
                               solution=result.sol)
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
            nonlocal state, t, index, audit_cache
            audit_cache = None
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
                classification_h = max(h, c.classification_step_floor)
                proposed = self.fixed if self.fixed is not None else select_groups(
                    kappa, classification_h, groups, c.enter, c.leave)
                bound = h * self.screen.slow_bound(gradient, proposed)
                stats['classification_seconds'] += time.perf_counter() - tick
                if bound > c.slow_bound_limit:
                    # This is a pre-trial controller rejection: preserve it in
                    # the attempt ledger, but do not claim a full ODE audit.
                    stats['attempts'].append(dict(
                        time=t, h=h, classification_step=classification_h,
                        groups=proposed.tolist(), output_count=0,
                        slow_scaled_norm_times_h=bound, status='rejected',
                        reason='coupled_slow_norm_bound',
                        rollback_verified=True,
                        full_ode_audit=False,
                    ))
                    stats['rollback_checks'] += 1
                    stats['rejected_steps'] += 1
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
                trial = dict(time=t, h=h, classification_step=classification_h,
                             groups=proposed.tolist(), output_count=stop-index,
                             slow_scaled_norm_times_h=bound, status='pending')
                before = (state.copy(), t, index, stats['ledger_increment'], groups.copy())
                tick = time.perf_counter()
                overlap_timing_recorded = False
                try:
                    partition.timeout = max(.001, c.timeout - (tick - start))
                    def split_trial():
                        split_tick = time.perf_counter()
                        coarse_trial = None
                        coarse_stats = None
                        if not c.audit_control_only:
                            _, coarse_trial, coarse_stats = partition.integrate(
                                state, t, t + h, 1, substeps=c.split_substeps)
                        _, fine_trial, fine_stats = partition.integrate(
                            state, t, t + h, 2, substeps=c.split_substeps)
                        return (coarse_trial, fine_trial, coarse_stats, fine_stats,
                                time.perf_counter() - split_tick)

                    def audit_trial():
                        audit_tick = time.perf_counter()
                        reference_trial = audited_reference(state.copy(), t, t + h, query)
                        return reference_trial, time.perf_counter() - audit_tick

                    if c.overlap_audit:
                        def capture_trial(function):
                            trial_tick = time.perf_counter()
                            try:
                                return function(), time.perf_counter() - trial_tick, None
                            except BaseException as exc:
                                return None, time.perf_counter() - trial_tick, exc

                        overlap_tick = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='m4-audit') as pool:
                            split_future = pool.submit(capture_trial, split_trial)
                            audit_future = pool.submit(capture_trial, audit_trial)
                            split_result, split_elapsed, split_error = split_future.result()
                            audit_result, audit_elapsed, audit_error = audit_future.result()
                        stats['split_seconds'] += split_elapsed
                        stats['audit_seconds'] += audit_elapsed
                        stats['overlap_trials'] += 1
                        stats['overlap_seconds'] += max(
                            0., split_elapsed + audit_elapsed
                            - (time.perf_counter() - overlap_tick))
                        overlap_timing_recorded = True
                        if split_result is not None and (split_error is not None or audit_error is not None):
                            coarse, fine, cs, fs, _ = split_result
                            if cs is not None:
                                stats['coarse_split_calls'] += 1
                                for key in ('nfev', 'njev', 'nlu'):
                                    stats[key] += cs[key]
                            stats['fine_split_calls'] += 1
                            for key in ('nfev', 'njev', 'nlu'):
                                stats[key] += fs[key]
                        if audit_error is not None:
                            if isinstance(split_error, SplitFailure):
                                for key in ('nfev', 'njev', 'nlu'):
                                    stats[key] += split_error.stats.get(key, 0)
                            trial.update(status='failed', full_ode_audit=False,
                                         split_completed=split_error is None,
                                         split_failure=None if split_error is None else str(split_error),
                                         audit_failure=str(audit_error),
                                         audit_status='failed',
                                         audit_start_time=float(t), audit_end_time=float(t + h),
                                         audit_method=c.audit_method, audit_rtol=float(c.audit_rtol),
                                         audit_start_state_sha256=hashlib.sha256(
                                             np.ascontiguousarray(state, dtype=np.float64).tobytes()
                                         ).hexdigest(),
                                         rollback_verified=True)
                            stats['rollback_checks'] += 1
                            stats['attempts'].append(trial)
                            raise audit_error
                        if split_error is not None:
                            trial.update(full_ode_audit=True, audit_status='completed',
                                         split_failure=str(split_error))
                            trial.update(
                                audit_start_time=float(t), audit_end_time=float(t + h),
                                audit_method=c.audit_method, audit_rtol=float(c.audit_rtol),
                                audit_start_state_sha256=hashlib.sha256(
                                    np.ascontiguousarray(state, dtype=np.float64).tobytes()
                                ).hexdigest())
                            raise split_error
                        coarse, fine, cs, fs, split_elapsed = split_result
                        reference, _ = audit_result
                        trial.update(full_ode_audit=True, audit_status='completed')
                    else:
                        coarse, fine, cs, fs, split_elapsed = split_trial()
                        reference, audit_elapsed = None, 0.
                    if cs is not None:
                        stats['coarse_split_calls'] += 1
                        for key in ('nfev', 'njev', 'nlu'):
                            stats[key] += cs[key]
                    stats['fine_split_calls'] += 1
                    for key in ('nfev', 'njev', 'nlu'):
                        stats[key] += fs[key]
                    if not overlap_timing_recorded:
                        stats['split_seconds'] += split_elapsed
                        stats['audit_seconds'] += audit_elapsed
                    for counts in ((cs, fs) if cs is not None else (fs,)):
                        stats['min_rhs_argument'] = min(stats['min_rhs_argument'], counts['min_rhs_argument'])
                    # Convex interpolation is checked, never used as a substitute
                    # for an independently calibrated splitting dense extension.
                    values = np.vstack([np.interp(query, [t, t+h/2, t+h], row) for row in fine])
                    check(values)
                    # Use the already-computed coarse/fine pair as a fail-closed
                    # shadow filter.  A shadow rejection never commits state and
                    # never claims an audit; only shadow-pass trials reach the
                    # independent nonlinear ODE audit and can be accepted.
                    if c.shadow_first and not c.audit_control_only:
                        shadow_tick = time.perf_counter()
                        shadow_estimate = normalized_error(
                            fine[:, -1] - (fine[:, -1] - coarse[:, -1]) / 3,
                            fine[:, -1], self.scale, c.local_fraction * c.target)[1]
                        stats['shadow_seconds'] += time.perf_counter() - shadow_tick
                        trial['shadow_estimate_max'] = shadow_estimate
                        if shadow_estimate > c.shadow_limit:
                            trial['status'] = 'rejected'
                            trial['reason'] = 'shadow_estimate_budget'
                            trial['full_ode_audit'] = False
                            stats['shadow_rejections'] += 1
                            stats['rejected_steps'] += 1
                            rejections += 1
                            stats['rollback_checks'] += 1
                            # No mutable committed state has been touched yet.
                            trial['rollback_verified'] = True
                            h *= .5
                            partition = None
                            stats['attempts'].append(trial)
                            if rejections >= c.max_rejections:
                                commit_full(grid[-1], 'shadow_estimate_budget')
                                break
                            continue
                    if not c.overlap_audit:
                        reference, audit_elapsed = audit_trial()
                    stats['audit_seconds'] += audit_elapsed
                    trial.update(
                        full_ode_audit=True,
                        audit_start_time=float(t),
                        audit_end_time=float(t + h),
                        audit_method=c.audit_method,
                        audit_rtol=float(c.audit_rtol),
                        audit_start_state_sha256=hashlib.sha256(
                            np.ascontiguousarray(state, dtype=np.float64).tobytes()
                        ).hexdigest(),
                    )
                    conc, flux, net = observable_error(values, reference)
                    if c.audit_control_only:
                        estimate = 0.
                        mismatch = False
                    else:
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
                        audit_cache = None
                        stats['accepted_steps'] += 1
                        stats['split_coverage'] += h / (grid[-1]-grid[0])
                        stats['partition_switches'] += int(changed)
                        stats['output_points_audited'] += stop - before[2]
                        stats['ledger_increment'] += fs['ledger_increment']
                        stats['ledger_quadrature_difference'] += fs['ledger_quadrature_difference']
                        groups = proposed.copy()
                        rejections = 0
                        exponent = .5 if c.audit_control_only else 1/3
                        factor = 2. if control == 0 else float(np.clip(c.growth_safety*control**(-exponent), .2, 2.))
                        h *= min(factor, 1.) if changed else factor
                    else:
                        trial['reason'] = 'audited_error_or_estimator_budget'
                except SplitFailure as exc:
                    if not overlap_timing_recorded:
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
