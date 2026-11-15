"""Reusable allocations for independent same-state full nonlinear ODE audits.

BDF history is reset for EVERY call, including after rejection/failure. The
candidate endpoint must never be substituted into an existing Nordsieck
history. This module extends frozen NativeSession only with sampled queries;
it does not change that module or promise KLU numeric-factor reuse.
"""
import ctypes as c
from types import SimpleNamespace
import time

import numpy as np

from research.mri_native import NativeSession


class QueryNativeSession(NativeSession):
    def solve_queries(self, model, times, rtol, atol):
        """Return local-time queries, resetting the complete BDF history."""
        times = np.asarray(times, dtype=float)
        initial = np.asarray(model.y0, dtype=float)
        absolute = np.broadcast_to(atol, (self.n,)).astype(float, copy=True)
        if (not self.memory or times.ndim != 1 or len(times) < 2
                or not np.isfinite(times).all() or times[0] != 0.
                or np.any(np.diff(times) <= 0) or initial.shape != (self.n,)
                or not np.isfinite(initial).all() or not np.isfinite(rtol) or rtol <= 0
                or not np.isfinite(absolute).all() or np.any(absolute <= 0)):
            raise ValueError('Open session, finite ordered local times, initial state and positive tolerances required')
        start = time.perf_counter()
        self.model = model
        self.y[:] = initial
        self.av[:] = absolute
        self.errors = []
        self.counts = dict(rhs_seconds=0., jacobian_seconds=0.)
        tick = time.perf_counter()
        reset = self.initialized
        if reset:
            self.check(self.reinitialize(self.memory, 0., self.vector))
        else:
            self.check(self.initialize(self.memory, self.rhs_callback, 0., self.vector))
            self.check(self.set_linear(self.memory, self.linear, self.matrix))
            self.check(self.set_jac(self.memory, self.jac_callback))
            self.check(self.max_steps(self.memory, 1000000))
            self.initialized = True
        self.check(self.tolerances(self.memory, rtol, self.atolv))
        # Query calls may otherwise take internal steps beyond the audited
        # interval. Cold and reusable versions use the same stop boundary.
        self.check(self.stop_time(self.memory, float(times[-1])))
        self.counts['reinit_seconds'] = time.perf_counter()-tick
        result = np.empty((self.n, len(times)))
        result[:, 0] = initial
        returned = c.c_double()
        for i, t in enumerate(times[1:], 1):
            self.check(self.integrate(self.memory, float(t), self.vector, c.byref(returned), 1))
            result[:, i] = self.y
        for key, fn in self.getters.items():
            value = c.c_long()
            self.check(fn(self.memory, c.byref(value)))
            self.counts[key] = value.value
        self.counts.update(history_reset=True, reused_allocations=bool(reset),
            query_count=len(times)-1, seconds=time.perf_counter()-start,
            native_accepted_internal_states_exposed=False)
        return result, self.counts.copy()


class ReusableFullAudit:
    """Original autonomous full ODE, analytic Jacobian, original domain guard.

Only allocations and fixed sparsity resources survive calls. Returned values
are independent audit output, never committed split states. All observable
candidate scoring is deliberately the caller's additional runtime work.
"""
    def __init__(self, case):
        start = time.perf_counter()
        self.system = case['system']
        if self.system.time_dependent or self.system.flux_jacobian is None:
            raise ValueError('Autonomous original ODE with analytic Jacobian required')
        self.scale = np.asarray(case['scale'], dtype=float).copy()
        if (self.scale.shape != self.system.y0.shape or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError('Positive finite full-state scales required')
        self.pools = case['options'].get('pools')
        pattern = (abs(self.system.S) @ self.system.dependencies).astype(bool).tocsc()
        self.session = QueryNativeSession(pattern)
        self.closed = False
        self.setup_seconds = time.perf_counter()-start

    def solve(self, y, h, q, tol, timeout=20.):
        start = time.perf_counter()
        y = np.asarray(y, dtype=float).copy()
        q = np.asarray(q, dtype=float)
        if (self.closed or y.shape != self.scale.shape or not np.isfinite(y).all()
                or np.any(y < -1e-12*self.scale) or not np.isfinite(h) or h <= 0
                or q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all()
                or q[0] != 0. or q[-1] != 1. or np.any(np.diff(q) <= 0)
                or not np.isfinite(tol) or tol <= 0
                or not np.isfinite(timeout) or timeout <= 0):
            raise ValueError('Valid initial state, endpoint-inclusive ordered fractions and positive parameters required')
        minimum = float(y.min())
        def guard(z):
            if time.perf_counter()-start > timeout:
                raise TimeoutError('Reusable full audit timeout')
            scale = self.scale[:, None] if z.ndim == 2 else self.scale
            if not np.isfinite(z).all() or np.any(z < -1e-12*scale):
                raise RuntimeError('Invalid full ODE RHS domain; no clipping')
        def rhs(t, z):
            nonlocal minimum
            guard(z)
            minimum = min(minimum, float(z.min()))
            return self.system.S @ self.system.flux(0., z)
        def jac(t, z):
            guard(z)
            return (self.system.S @ self.system.flux_jacobian(0., z)).tocsc()
        values, stats = self.session.solve_queries(SimpleNamespace(y0=y, rhs=rhs, jacobian=jac),
            h*q, tol, .01*tol*self.scale)
        guard(values)
        pool_error = 0.
        if self.pools is not None:
            pool0 = self.pools@y
            pool_error = float(np.max(abs(self.pools@(values-y[:, None]))))/max(1e-12, float(np.max(abs(pool0))))
            if not np.isfinite(pool_error) or pool_error > 1e-10:
                raise RuntimeError('Invalid full audit output pool')
        stats.update(min_rhs_argument=minimum, pool_error=pool_error,
            seconds=time.perf_counter()-start)
        return values, stats

    def close(self):
        if not self.closed:
            self.session.close()
            self.closed = True
