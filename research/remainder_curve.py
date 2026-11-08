"""Full-J polynomial remainder iteration; a continuous predictor diagnostic.

With N(d)=f(y+d)-f(y)-Jd, interpolate N at H/2,H by a*q^2+b*q^3.
One affine prediction followed by one correction has classical local O(H^5)
continuous error for smooth fixed-J problems. This is NOT a stiff-order proof,
a reaction split, or an implementation of the published MERB4 tableau.
"""
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from research.ordered_radau import OrderedRadau
from research.shared_radau import ExactLUCache
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve, physical_ledger
from research.mri_kernel import GAUSS


def coefficients(remainders):
    half, end = np.asarray(remainders).T
    return np.column_stack([8*half-end, 2*end-8*half])


class RemainderCurve:
    def __init__(self, case):
        self.case = case
        self.system, self.scale = case['system'], case['scale']
        self.screen = GroupScreen(self.system, self.scale)
        if self.system.time_dependent:
            raise ValueError('Autonomous systems only')

    def trial(self, y, h, tol=1e-10, sweeps=1, timeout=20.):
        if sweeps not in (1, 2) or not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
            raise ValueError('One/two sweeps and positive finite step/tolerance required')
        tick = time.perf_counter()
        y = np.asarray(y, dtype=float).copy()
        stats = dict(nonlinear_nfev=0, nonlinear_njev=0, nfev=0, njev=0,
                     nlu=0, subflows=0, accepted_steps=0, pool_error=0., min_internal=float(y.min()))
        pools = self.case['options']['pools']
        p0 = pools@y
        pd = max(1e-12, float(np.max(abs(p0))))
        cache = ExactLUCache()

        def check(z):
            if time.perf_counter()-tick > timeout:
                raise TimeoutError('Remainder predictor budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):
                raise ValueError('Invalid remainder predictor state; no clipping')

        def accepted(z):
            check(z)
            stats['min_internal'] = min(stats['min_internal'], float(z.min()))
            stats['pool_error'] = max(stats['pool_error'], float(np.max(abs(pools@z-p0)))/pd)

        try:
            check(y)
            v0 = self.system.flux(0., y)
            stats['nonlinear_nfev'] += 1
            gradient = self.screen.gradient(y)
            j = (self.system.S@gradient).tocsc()
            f0 = self.system.S@v0
            stats['nonlinear_njev'] += 1

            def linear(forcing):
                stats['subflows'] += 1

                def rhs(t, d):
                    check(y+d)
                    q = t/h
                    return j@d+f0+q*q*(forcing[:, 0]+q*forcing[:, 1])

                solver = OrderedRadau(rhs, 0., np.zeros_like(y), h, rtol=tol,
                                      atol=.01*tol*self.scale, jac=j, first_step=h)
                factorize = solver.lu
                solver.lu = lambda a: cache.factor(a, factorize)
                pieces = []
                try:
                    while solver.status == 'running':
                        left = solver.t
                        message = solver.step()
                        if solver.status == 'failed':
                            raise RuntimeError(message)
                        accepted(y+solver.y)
                        dense = solver.dense_output()
                        pieces.append(DensePower(np.column_stack([y+dense.y_old, dense.Q]), solver.t-left))
                        stats['accepted_steps'] += 1
                finally:
                    for k in ('nfev', 'njev', 'nlu'):
                        stats[k] += getattr(solver, k)
                curve = InnerCurve(pieces)
                if np.any(curve.minima() < -1e-12*self.scale):
                    raise ValueError('Negative remainder predictor curve; no clipping')
                if curve.join_error(self.scale) > 1e-10:
                    raise ValueError('Discontinuous remainder predictor curve')
                return y+solver.y, curve

            end, curve = linear(np.zeros((len(y), 2)))
            curves = [curve]
            stages, rates, remainders, forcings = [], [], [], []
            for _ in range(sweeps):
                stage = curve(np.array([.5, 1.]))
                for z in stage.T:
                    accepted(z)
                rate = np.column_stack([self.system.flux(0., z) for z in stage.T])
                stats['nonlinear_nfev'] += 2
                remainder = rate-v0[:, None]-gradient@(stage-y[:, None])
                forcing = self.system.S@coefficients(remainder)
                end, curve = linear(forcing)
                stages.append(stage); rates.append(rate); remainders.append(remainder)
                forcings.append(forcing); curves.append(curve)
            stats['predictor_seconds'] = time.perf_counter()-tick
            stats.update(continuous_nonnegative=True, dense_join_error=curve.join_error(self.scale),
                         lu_cache_hits=cache.hits, lu_cache_misses=cache.misses)
            stats.update(physical_ledger(self.case, curve, y, end))
            stats['ledger_rate_evaluations'] = 24*len(curve.pieces) if self.case['options'].get('ledger_weights') is not None else 0
            weights = self.case['options'].get('ledger_weights')
            if weights is not None:
                totals = []
                for n in (8, 16):
                    nodes, w = GAUSS[n]; q = (nodes+1)/2; total = 0.; left = 0.
                    for piece in curve.pieces:
                        tau = (left+piece.h*q)/h
                        drive = f0[:, None]+forcing[:, :1]*tau**2+forcing[:, 1:]*tau**3
                        values = weights@(j@(piece(q)-y[:, None])+drive)
                        total += piece.h/2*float(w@values); left += piece.h
                    totals.append(total)
                den = max(1., abs(float(weights@y)))
                stats.update(modified_ledger_residual=abs(float(weights@(end-y))-totals[1])/den,
                             modified_ledger_quadrature=abs(totals[1]-totals[0])/den)
            stats['seconds'] = time.perf_counter()-tick
            return dict(fine=end, pieces=[curve], curves=curves, stages=np.array(stages),
                        rates=np.array(rates), remainders=np.array(remainders),
                        forcings=np.array(forcings), initial_rates=v0, stats=stats)
        except Exception as exc:
            stats.update(seconds=time.perf_counter()-tick, lu_cache_hits=cache.hits, lu_cache_misses=cache.misses)
            exc.stats = stats
            raise
