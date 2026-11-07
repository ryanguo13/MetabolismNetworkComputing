"""Trial-local Jacobian seeds and exact-matrix LU reuse for research Radau.

The Radau equations, Newton stopping test, error test and refresh logic are
unchanged. A borrowed Jacobian MUST start as stale, so Newton failure refreshes
it before reducing the step. No BDF history or numerical state is transferred.
SciPy internals: revalidate this adapter when the pinned SciPy version changes.
"""
from types import FunctionType
import numpy as np
from scipy.sparse import issparse
from research.ordered_radau import OrderedRadau
from research.mri_endpoint import MRIEndpoint


class ExactLUCache:
    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def factor(self, matrix, factorize):
        key = (issparse(matrix), matrix.shape, matrix.dtype.str)
        old = self.entries.get(key)
        same = False
        if old is not None:
            saved = old[0]
            same = (np.array_equal(matrix.indptr, saved.indptr)
                    and np.array_equal(matrix.indices, saved.indices)
                    and np.array_equal(matrix.data, saved.data)) if issparse(matrix) else np.array_equal(matrix, saved)
        if same:
            self.hits += 1
            return old[1]
        # Dense lu_factor may overwrite its input, so preserve the matrix first.
        saved = matrix.copy()
        factor = factorize(matrix)
        self.entries[key] = (saved, factor)
        self.misses += 1
        return factor


class SharedRadau(OrderedRadau):
    def __init__(self, *args, seed=None, cache=None, **kwargs):
        if not callable(kwargs.get('jac')):
            raise ValueError('Shared Radau requires a refreshable analytic Jacobian')
        jac = kwargs['jac']
        borrowed = seed is not None
        initial = True

        def refresh(t, y):
            nonlocal initial
            if initial and borrowed:
                initial = False
                return seed.copy()
            initial = False
            return jac(t, y)

        kwargs['jac'] = refresh
        super().__init__(*args, **kwargs)
        if borrowed:
            self.current_jac = False
            self.njev = 0  # The seed was copied, not evaluated at this stage.
        if cache is not None:
            factorize = self.lu
            self.lu = lambda matrix: cache.factor(matrix, factorize)


class SharedFactory:
    """One factory per macro trial and fixed fast-variable ordering."""
    def __init__(self, share_jacobian=True, share_lu=True):
        self.share_jacobian = share_jacobian
        self.cache = ExactLUCache() if share_lu else None
        self.solvers = []
        self.borrowed = 0

    def __call__(self, *args, **kwargs):
        seed = self.solvers[-1].J if self.share_jacobian and self.solvers else None
        solver = SharedRadau(*args, seed=seed, cache=self.cache, **kwargs)
        self.borrowed += seed is not None
        self.solvers.append(solver)
        return solver

    def stats(self):
        return dict(jacobian_seeds=self.borrowed,
                    lu_cache_hits=self.cache.hits if self.cache else 0,
                    lu_cache_misses=self.cache.misses if self.cache else 0,
                    actual_jacobians=sum(s.njev for s in self.solvers),
                    actual_factorizations=sum(s.nlu for s in self.solvers))


class SharedMRIEndpoint(MRIEndpoint):
    def trial(self, *args, share_jacobian=True, share_lu=True, **kwargs):
        if self.backend != 'radau':
            raise ValueError('Sharing experiment supports only Radau')
        factory = SharedFactory(share_jacobian, share_lu)
        # Freeze the old endpoint implementation without editing its evidence.
        original = MRIEndpoint.trial
        trial = FunctionType(original.__code__, dict(original.__globals__, Radau=factory),
                             name='trial', argdefs=original.__defaults__)
        result = trial(self, *args, **kwargs)
        result['stats'].update(factory.stats())
        return result


def shared_full_curve(case, y, h, tol, screen, share_lu=True):
    """Same full-curve guards and dense output as frozen inner_dense.full_curve."""
    import time
    from research.inner_dense import Polynomial, InnerCurve, physical_ledger
    start = time.perf_counter()
    sys, scale, pool = case['system'], case['scale'], case['options']['pools']
    initial_pool = pool @ y
    den = max(1e-12, float(np.max(abs(initial_pool))))
    drift = 0.

    def rhs(t, z):
        if time.perf_counter()-start > 20:
            raise TimeoutError('Full dense budget')
        if not np.isfinite(z).all() or np.any(z < -1e-12*scale):
            raise ValueError('Invalid full RHS state')
        return sys.S @ sys.flux(0., z)

    factory = SharedFactory(share_lu=share_lu)
    solver = factory(rhs, 0., y, h, rtol=tol, atol=.01*tol*scale,
                     jac=lambda t, z: sys.S @ screen.gradient(z))
    pieces = []
    zeros = np.zeros_like(y)
    active = np.arange(len(y))
    while solver.status == 'running':
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(message)
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12*scale):
            raise ValueError('Invalid full accepted state')
        drift = max(drift, float(np.max(abs(pool @ solver.y-initial_pool)))/den)
        pieces.append(Polynomial.from_radau(solver.dense_output(), y, zeros, zeros, h, active))
    curve = InnerCurve(pieces)
    stats = dict(nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                 accepted_steps=len(pieces), pool_error=drift,
                 continuous_nonnegative=bool(np.all(curve.minima() >= -1e-12*scale)),
                 dense_join_error=curve.join_error(scale))
    if stats['dense_join_error'] > 1e-10:
        raise ValueError('Discontinuous full curve')
    stats.update(physical_ledger(case, curve, y, solver.y))
    stats.update(factory.stats())
    stats['seconds'] = time.perf_counter()-start
    return dict(fine=solver.y.copy(), pieces=[curve], stats=stats)
