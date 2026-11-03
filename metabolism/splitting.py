"""Fixed complete-reaction Lie/Strang flows. No PE, clipping or adaptivity.

Only coordinates with zero stoichiometry in a subproblem are held constant.
Every rate evaluation sees the full state, including all modifier dependencies.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix


@dataclass
class ReactionSystem:
    S: object
    flux: object
    y0: np.ndarray
    groups: list
    dependencies: object
    flux_jacobian: object = None
    time_dependent: bool = False

    @classmethod
    def from_sbml(cls, model):
        return cls(csc_matrix(model.S / model.volumes[:, None]), model.flux,
                   model.y0, [np.array([i]) for i in range(len(model.reactions))],
                   csc_matrix(model.dependencies), time_dependent=model.time_dependent)

    @classmethod
    def from_kecoli(cls, model):
        ids = model.data['network_data']['rxn_f_bInd_rxnInd'].astype(int)
        return cls(model.S, model.flux, model.y0,
                   [np.flatnonzero(ids == g) for g in np.unique(ids)],
                   model.dependency, model.flux_jacobian)


class SplitFailure(RuntimeError):
    """A failed fixed step is reported with its accumulated diagnostic counts."""

    def __init__(self, message, stats):
        super().__init__(message)
        self.stats = stats


def compose_step(fast, slow, y, h, method):
    """Chronological Lie F,S; Strang F/2,S,F/2 (rightmost map acts first)."""
    if method == 'Lie':
        return slow(fast(y, h), h)
    if method == 'Strang':
        return fast(slow(fast(y, h / 2), h), h / 2)
    raise ValueError('method must be Lie or Strang')


class FixedSplit:
    def __init__(self, system, fast_groups, scale, rtol=1e-10,
                 ledger_weights=None, pools=None, timeout=120):
        if system.time_dependent:
            raise ValueError('Autonomous flows only; time-dependent models are excluded')
        self.system = system
        self.S = csc_matrix(system.S)
        self.scale = np.asarray(scale, dtype=float)
        if (self.S.shape[0] != len(system.y0) or self.scale.shape != system.y0.shape
                or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)
                or not np.isfinite(rtol) or rtol <= 0):
            raise ValueError('Invalid dimensions, species scales or tolerance')
        groups = [np.asarray(g) for g in system.groups]
        if (not groups or any(g.ndim != 1 or not len(g) or g.dtype.kind not in 'iu' for g in groups)
                or not np.array_equal(np.sort(np.concatenate(groups)), np.arange(self.S.shape[1]))):
            raise ValueError('Complete reaction groups must partition every column exactly once')
        chosen = np.asarray(fast_groups)
        if chosen.size == 0:
            chosen = np.array([], dtype=int)
        if (chosen.ndim != 1 or chosen.dtype.kind not in 'iu' or len(np.unique(chosen)) != len(chosen)
                or np.any(chosen < 0) or np.any(chosen >= len(groups))):
            raise ValueError('Invalid fast group indices')
        self.fast_groups = chosen.copy()
        self.rtol, self.timeout = rtol, timeout
        self.weights = None if ledger_weights is None else np.asarray(ledger_weights, dtype=float)
        self.pools = None if pools is None else csc_matrix(pools)
        if self.pools is not None and self.pools.shape[1] != len(system.y0):
            raise ValueError('Invalid invariant pool dimensions')
        if self.weights is not None and (self.weights.shape != system.y0.shape or not np.isfinite(self.weights).all()):
            raise ValueError('Invalid open-ledger weights')
        self.parts = {}
        for name, selected in [('fast', chosen), ('slow', np.setdiff1d(np.arange(len(groups)), chosen))]:
            columns = np.concatenate([groups[i] for i in selected]) if len(selected) else np.array([], dtype=int)
            active = np.flatnonzero(np.asarray(abs(self.S[:, columns]).sum(axis=1)).ravel())
            submatrix = self.S[active][:, columns].tocsc()
            dependency = csc_matrix(system.dependencies)[columns][:, active]
            sparsity = (abs(submatrix) @ dependency).astype(bool).tocsc()
            self.parts[name] = (columns, active, submatrix, sparsity)

    def integrate(self, y0, start, end, steps, method='Strang'):
        if (method not in ('Lie', 'Strang') or not isinstance(steps, (int, np.integer)) or steps < 1
                or not np.isfinite([start, end]).all() or end <= start):
            raise ValueError('Invalid method, interval or fixed step count')
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != self.scale.shape:
            raise ValueError('Invalid initial state dimension')
        before = time.perf_counter()
        stats = {'method': method, 'macro_steps_completed': 0, 'subflow_calls': 0,
                 'nfev': 0, 'njev': 0, 'nlu': 0, 'accepted_substeps': 0,
                 'rhs_calls_including_fd': 0, 'fast_seconds': 0., 'slow_seconds': 0.,
                 'rhs_seconds': 0., 'jacobian_seconds': 0., 'ledger_seconds': 0.,
                 'significant_negative_rhs_components': 0,
                 'min_rhs_argument': float(y0.min()), 'min_accepted_state': float(y0.min()),
                 'min_subflow_endpoint': float(y0.min()), 'min_macro_state': float(y0.min()),
                 'clipping_count': 0, 'projection_count': 0, 'fallback_count': 0,
                 'fast_columns': self.parts['fast'][0].tolist(),
                 'fast_updated_states': self.parts['fast'][1].tolist(),
                 'slow_updated_state_count': len(self.parts['slow'][1]),
                 'max_substep_invariant_error': 0.,
                 'ledger_increment': 0., 'ledger_quadrature_difference': 0.}
        minima = y0.copy()

        def check(y, location):
            if not np.isfinite(y).all():
                raise SplitFailure(f'Nonfinite {location}', stats)
            if np.any(y < -1e-12 * self.scale):
                raise SplitFailure(f'Significant negative {location}; no clipping', stats)

        def flow(y, duration, name):
            columns, active, submatrix, sparsity = self.parts[name]
            if not len(active):
                return y.copy()
            tick = time.perf_counter()
            stats['subflow_calls'] += 1
            def expand(z):
                full = y.copy()
                full[active] = z
                return full
            def rhs(t, z):
                if time.perf_counter() - before > self.timeout:
                    raise SplitFailure(f'Trajectory exceeded {self.timeout}s limit', stats)
                full = expand(z)
                np.minimum(minima, full, out=minima)
                stats['rhs_calls_including_fd'] += 1
                stats['min_rhs_argument'] = float(minima.min())
                stats['significant_negative_rhs_components'] = int(np.count_nonzero(minima < -1e-12 * self.scale))
                check(full, f'{name} RHS argument (including Newton/FD probes)')
                stamp = time.perf_counter()
                result = submatrix @ self.system.flux(0., full)[columns]
                stats['rhs_seconds'] += time.perf_counter() - stamp
                return result
            def jac(t, z):
                stamp = time.perf_counter()
                value = submatrix @ self.system.flux_jacobian(0., expand(z))[columns][:, active]
                stats['jacobian_seconds'] += time.perf_counter() - stamp
                return value.tocsc()
            options = {}
            if name == 'fast':
                options = {'jac': jac} if self.system.flux_jacobian else {'jac_sparsity': sparsity}
            try:
                result = solve_ivp(rhs, (0., duration), y[active],
                                   method='Radau' if name == 'fast' else 'RK45',
                                   rtol=self.rtol, atol=.01 * self.rtol * self.scale[active],
                                   dense_output=self.weights is not None, **options)
                stats['nfev'] += result.nfev
                stats['njev'] += result.njev
                stats['nlu'] += result.nlu
                stats['accepted_substeps'] += len(result.t) - 1
                if not result.success:
                    raise SplitFailure(result.message, stats)
                stats['min_accepted_state'] = min(stats['min_accepted_state'], float(result.y.min()))
                if np.any(result.y < -1e-12 * self.scale[active, None]):
                    raise SplitFailure('Significant negative accepted substep', stats)
                if self.pools is not None:
                    local_drift = self.pools[:, active] @ (result.y - y[active, None])
                    base_drift = np.asarray(self.pools @ (y - y0)).ravel()[:, None]
                    denominator = max(1e-12, float(np.max(abs(self.pools @ y0))))
                    stats['max_substep_invariant_error'] = max(stats['max_substep_invariant_error'],
                                                               float(np.max(abs(local_drift + base_drift))) / denominator)
                new = expand(result.y[:, -1])
                check(new, f'{name} endpoint')
                stats['min_subflow_endpoint'] = min(stats['min_subflow_endpoint'], float(new.min()))
                if self.weights is not None:
                    # Independent Gauss quadrature of the encoded open contribution
                    # along every accepted subflow, not a projection onto initial mass.
                    stamp = time.perf_counter()
                    integrals = []
                    ledger_columns = np.asarray(self.weights @ self.S[:, columns]).ravel()
                    for order in (4, 8):
                        nodes, weights = np.polynomial.legendre.leggauss(order)
                        integral = 0.
                        for left, right in zip(result.t[:-1], result.t[1:]):
                            ts = left + (right - left) * (nodes + 1) / 2
                            zs = result.sol(ts)
                            values = [ledger_columns @ self.system.flux(0., expand(z))[columns] for z in zs.T]
                            integral += (right - left) / 2 * np.dot(weights, values)
                        integrals.append(float(integral))
                    stats['ledger_increment'] += integrals[1]
                    stats['ledger_quadrature_difference'] += abs(integrals[1] - integrals[0])
                    stats['ledger_seconds'] += time.perf_counter() - stamp
                return new
            finally:
                stats[name + '_seconds'] += time.perf_counter() - tick

        check(y0, 'initial state')
        times = np.linspace(start, end, steps + 1)
        states = np.empty((len(y0), steps + 1))
        states[:, 0] = y0
        try:
            for i, h in enumerate(np.diff(times)):
                states[:, i + 1] = compose_step(lambda y, d: flow(y, d, 'fast'),
                                               lambda y, d: flow(y, d, 'slow'), states[:, i], h, method)
                stats['macro_steps_completed'] += 1
                stats['min_macro_state'] = min(stats['min_macro_state'], float(states[:, i + 1].min()))
            if self.weights is not None:
                stats['ledger_residual'] = float(self.weights @ (states[:, -1] - y0) - stats['ledger_increment'])
            return times, states, stats
        except SplitFailure:
            raise
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            raise SplitFailure(f'Subflow failed: {exc}', stats) from exc
        finally:
            stats['integration_seconds'] = time.perf_counter() - before


def reversible_pair(y, duration, a, b, forward, reverse):
    """Exact A <-> B flow; the pair is one indivisible reaction group."""
    out = y.copy()
    total = y[a] + y[b]
    change = (reverse * total / (forward + reverse) - y[a]) * (-np.expm1(-(forward + reverse) * duration))
    out[a] += change
    out[b] -= change
    return out


def enzyme_binding(y, duration, kon, koff):
    """Exact Riccati binding flow for state (E,S,ES,P), no QSSA/PE."""
    e, s, c, _ = y
    et, st = e + c, s + c
    linear = kon * (et + st) + koff
    gap = np.sqrt((kon * (et - st)) ** 2 + 2 * kon * koff * (et + st) + koff ** 2)
    low, high = 2 * kon * et * st / (linear + gap), (linear + gap) / (2 * kon)
    ratio = (c - low) / (c - high) * np.exp(-gap * duration)
    new_c = (low - ratio * high) / (1 - ratio)
    out = y.copy()
    out[0] -= new_c - c
    out[1] -= new_c - c
    out[2] = new_c
    return out


def enzyme_catalysis(y, duration, kcat):
    out = y.copy()
    change = y[2] * (-np.expm1(-kcat * duration))
    out[[0, 3]] += change
    out[2] -= change
    return out
