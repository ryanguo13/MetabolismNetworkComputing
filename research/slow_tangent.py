"""Original fast reactions with a frozen slow tangent and own-curve remainder.

This is a research coupling, not a published tableau or a stiff-order claim.
All state coordinates remain implicit because the slow tangent can couple
outside the fast block. A real reaction partition alone does not grant M4
split coverage. No saved reference is an input to the numerical method.
"""
import time
import numpy as np
from research.mri_kernel import MRI, GAUSS
from research.ordered_radau import OrderedRadau
from research.inner_dense import InnerCurve, Polynomial, physical_ledger
from research.remainder_curve import coefficients


class SlowTangent(MRI):
    def __init__(self, case, groups):
        super().__init__(case, groups, backend='radau')
        self.slow_columns = np.setdiff1d(np.arange(self.system.S.shape[1]), self.cols)
        self.sm = self.system.S[:, self.slow_columns].tocsc()

    def trial(self, y, h, tol=1e-10, sweeps=1, whole_step=False, timeout=20.):
        if (type(sweeps) is not int or sweeps not in (0, 1, 2)
                or not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0
                or not np.isfinite(timeout) or timeout <= 0):
            raise ValueError('Zero to two corrections and finite positive budgets required')
        y = np.asarray(y, dtype=float).copy()
        if y.shape != self.scale.shape:
            raise ValueError('Full state shape required')
        tick = time.perf_counter()
        stats = dict(nfev=0, njev=0, nlu=0, subflows=0, accepted_steps=0,
                     nonlinear_nfev=0, nonlinear_njev=0, ledger_rate_evaluations=0,
                     pool_error=0., min_rhs=float(y.min()), min_internal=float(y.min()),
                     rhs_seconds=0., jacobian_seconds=0., ledger_seconds=0.,
                     implicit_dimension=len(y), fast_reactions=len(self.cols),
                     slow_reactions=len(self.slow_columns),
                     reaction_split_coverage_eligible=False)
        pool0 = np.asarray(self.pools@y).ravel()
        poolden = max(1e-12, float(np.max(abs(pool0))))

        def check(z):
            if time.perf_counter()-tick > timeout:
                raise TimeoutError('Slow-tangent attempt budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):
                raise ValueError('Invalid slow-tangent state; no clipping')

        def accepted(z):
            check(z)
            stats['min_internal'] = min(stats['min_internal'], float(z.min()))
            stats['pool_error'] = max(stats['pool_error'],
                float(np.max(abs(self.pools@z-pool0)))/poolden)

        def rates(z, ledger=False):
            check(z)
            result = self.system.flux(0., z)
            if not np.isfinite(result).all():
                raise FloatingPointError('Nonfinite slow-tangent reaction rates')
            stats['ledger_rate_evaluations' if ledger else 'nonlinear_nfev'] += 1
            return result

        try:
            check(y)
            v0 = rates(y)
            gradient0 = self.screen.gradient(y)
            if not np.isfinite(gradient0.data).all():
                raise FloatingPointError('Nonfinite initial reaction gradient')
            stats['nonlinear_njev'] += 1
            slow0 = self.sm@v0[self.slow_columns]
            gs = gradient0[self.slow_columns]
            js = (self.sm@gs).tocsc()
            jf0 = (self.sf@gradient0[self.cols]).tocsc()
            all_coordinates = np.arange(len(y))

            def modified_ledger(curve, end, forcing):
                if self.weights is None:
                    return {}
                before = time.perf_counter(); totals = [0., 0.]; defect = 0.; left = 0.
                for piece in curve.pieces:
                    local = []
                    for n in (8, 16):
                        q = (GAUSS[n][0]+1)/2
                        tau = (left+piece.h*q)/h
                        vals = []
                        for t, z in zip(tau, piece(q).T):
                            f = self.sf@rates(z, ledger=True)[self.cols]+slow0+js@(z-y)
                            f += t*t*(forcing[:, 0]+t*forcing[:, 1])
                            vals.append(float(self.weights@f))
                        local.append(piece.h/2*float(GAUSS[n][1]@vals))
                    totals[0] += local[0]; totals[1] += local[1]
                    defect += abs(local[1]-local[0]); left += piece.h
                den = max(1., abs(float(self.weights@y)))
                stats['ledger_seconds'] += time.perf_counter()-before
                return dict(modified_ledger_residual=abs(float(self.weights@(end-y))-totals[1])/den,
                            modified_ledger_quadrature=defect/den)

            def integrate(forcing):
                stats['subflows'] += 1

                def rhs(t, z):
                    before = time.perf_counter(); check(z)
                    stats['min_rhs'] = min(stats['min_rhs'], float(z.min()))
                    q = t/h
                    value = self.sf@rates(z)[self.cols]+slow0+js@(z-y)
                    value += q*q*(forcing[:, 0]+q*forcing[:, 1])
                    stats['rhs_seconds'] += time.perf_counter()-before
                    return value

                def jac(t, z):
                    before = time.perf_counter(); check(z)
                    if np.array_equal(z, y):
                        jf = jf0
                    else:
                        gradient = self.screen.gradient(z)
                        if not np.isfinite(gradient.data).all():
                            raise FloatingPointError('Nonfinite current reaction gradient')
                        stats['nonlinear_njev'] += 1
                        jf = self.sf@gradient[self.cols]
                    stats['jacobian_seconds'] += time.perf_counter()-before
                    return (jf+js).tocsc()

                solver = OrderedRadau(rhs, 0., y, h, rtol=tol,
                    atol=.01*tol*self.scale, jac=jac, first_step=h if whole_step else None)
                pieces = []
                try:
                    while solver.status == 'running':
                        message = solver.step()
                        if solver.status == 'failed':
                            raise RuntimeError(message)
                        accepted(solver.y)
                        pieces.append(Polynomial.from_radau(solver.dense_output(), y,
                            np.zeros_like(y), np.zeros_like(y), h, all_coordinates))
                finally:
                    for name in ('nfev', 'njev', 'nlu'):
                        stats[name] += getattr(solver, name)
                stats['accepted_steps'] += len(pieces)
                curve = InnerCurve(pieces); end = solver.y.copy()
                if np.any(curve.minima() < -1e-12*self.scale):
                    raise ValueError('Negative slow-tangent continuous curve; no clipping')
                if curve.join_error(self.scale) > 1e-10:
                    raise ValueError('Discontinuous slow-tangent curve')
                ledger = modified_ledger(curve, end, forcing)
                # Original physical rates remain independent of the tangent RHS.
                before = time.perf_counter()
                ledger.update(physical_ledger(self.case, curve, y, end))
                if self.weights is not None:
                    stats['ledger_rate_evaluations'] += 24*len(pieces)
                stats['ledger_seconds'] += time.perf_counter()-before
                return end, curve, ledger

            forcing = np.zeros((len(y), 2))
            end, curve, ledger = integrate(forcing)
            curves = [curve]; ledgers = [ledger]
            stages, stage_rates, remainders, forcings = [], [], [], []
            for _ in range(sweeps):
                stage = curve(np.array([.5, 1.]))
                for z in stage.T:
                    accepted(z)
                rate = np.column_stack([rates(z) for z in stage.T])
                remainder = rate[self.slow_columns]-v0[self.slow_columns, None]-gs@(stage-y[:, None])
                forcing = self.sm@coefficients(remainder)
                end, curve, ledger = integrate(forcing)
                stages.append(stage); stage_rates.append(rate); remainders.append(remainder)
                forcings.append(forcing); curves.append(curve); ledgers.append(ledger)
            stats.update(ledger, continuous_nonnegative=True,
                         dense_join_error=curve.join_error(self.scale), seconds=time.perf_counter()-tick)
            return dict(fine=end, pieces=[curve], curves=curves, stages=np.asarray(stages),
                rates=np.asarray(stage_rates), remainders=np.asarray(remainders),
                forcings=np.asarray(forcings), initial_rates=v0, ledgers=ledgers,
                slow_columns=self.slow_columns.copy(), stats=stats)
        except Exception as exc:
            stats['seconds'] = time.perf_counter()-tick
            exc.stats = stats
            raise
