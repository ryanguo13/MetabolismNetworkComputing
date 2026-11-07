import unittest
import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from metabolism.adaptive import GroupScreen
from metabolism.splitting import ReactionSystem
from research.inner_dense import full_curve
from research.mri_endpoint import MRIEndpoint
from research.shared_radau import ExactLUCache, SharedRadau, SharedMRIEndpoint, shared_full_curve


def linear_case():
    s = csc_matrix([[-1., 1.], [1., -1.]])
    rates = csc_matrix([[1000., 0.], [0., 1.]])
    y = np.array([.5, .5])
    sys = ReactionSystem(s, lambda t,z: np.asarray(rates@z).ravel(), y,
                         [np.array([0]), np.array([1])], rates.astype(bool), lambda t,z: rates)
    return dict(system=sys, scale=np.ones(2), options=dict(pools=csc_matrix([[1., 1.]]))), y


class SharedChecks(unittest.TestCase):
    def test_cache_requires_exact_matrix_including_dtype_and_pattern(self):
        for sparse in (False, True):
            cache = ExactLUCache()
            a = np.array([[4., 1.], [1., 3.]])
            factorize = splu if sparse else lambda x: lu_factor(x, overwrite_a=True)
            convert = csc_matrix if sparse else np.asarray
            first = cache.factor(convert(a.copy()), factorize)
            second = cache.factor(convert(a.copy()), factorize)
            self.assertIs(first, second)
            changed = a.copy(); changed[0,0] = np.nextafter(changed[0,0], np.inf)
            self.assertIsNot(cache.factor(convert(changed), factorize), first)
            complex_a = a.astype(complex)
            complex_a[0,0] += 1j
            factor = cache.factor(convert(complex_a), factorize)
            b = np.ones(2)
            x = factor.solve(b.astype(complex)) if sparse else lu_solve(factor, b)
            np.testing.assert_allclose(complex_a@x, b, atol=1e-14)
            self.assertEqual((cache.hits, cache.misses), (1, 3))
        cache = ExactLUCache()
        a = csc_matrix([[2., 0.], [0., 3.]])
        b = csc_matrix([[0., 3.], [2., 0.]])
        cache.factor(a, splu); cache.factor(b, splu)
        self.assertEqual(cache.hits, 0)

    def test_bad_seed_refreshes_before_acceptance(self):
        calls = []
        def jac(t,y):
            calls.append(t)
            return np.array([[-1000.]])
        solver = SharedRadau(lambda t,y:-1000*y, 0., np.ones(1), .01,
                             first_step=.01, rtol=1e-9, atol=1e-12,
                             jac=jac, seed=np.zeros((1,1)), cache=ExactLUCache())
        self.assertFalse(solver.current_jac)
        self.assertEqual(solver.njev, 0)
        while solver.status == 'running':
            solver.step()
        self.assertEqual(solver.status, 'finished')
        self.assertGreaterEqual(solver.njev, 1)
        self.assertEqual(calls[0], 0.)
        np.testing.assert_allclose(solver.y, np.exp(-10.), rtol=2e-8)

    def test_endpoint_equivalence_ablation_and_fresh_trial(self):
        case, y = linear_case()
        ordinary = MRIEndpoint(case, [0,1]).trial(y, .001, 1e-10)
        kernel = SharedMRIEndpoint(case, [0,1])
        for share_jac, share_lu in ((False,False), (True,False), (True,True)):
            result = kernel.trial(y, .001, 1e-10, share_jacobian=share_jac, share_lu=share_lu)
            np.testing.assert_array_equal(result['fine'], ordinary['fine'])
            np.testing.assert_allclose(result['fine'], expm(.001*np.array([[-1000.,1.],[1000.,-1.]]))@y, rtol=1e-10)
            self.assertEqual(result['stats']['nfev'], ordinary['stats']['nfev'])
            self.assertEqual(result['stats']['jacobian_seeds'], 2 if share_jac else 0)
            self.assertEqual(result['stats']['njev'], ordinary['stats']['njev']-(2 if share_jac else 0))
            self.assertLess(result['stats']['pool_error'], 1e-12)
        again = kernel.trial(y, .001, 1e-10)
        self.assertEqual(again['stats']['njev'], result['stats']['njev'])
        self.assertEqual(again['stats']['nlu'], result['stats']['nlu'])

    def test_equal_small_substeps_share_factors(self):
        case,y = linear_case()
        result = SharedMRIEndpoint(case,[0,1]).trial(y, 1e-7, 1e-8)
        self.assertEqual(result['stats']['lu_cache_hits'], 4)
        self.assertEqual(result['stats']['nlu'], 2)
        self.assertEqual(result['stats']['njev'], 1)

    def test_full_curve_same_equations_guards_and_output(self):
        case,y = linear_case(); screen = GroupScreen(case['system'], case['scale'])
        ordinary = full_curve(case,y,.001,1e-10,screen)
        shared = shared_full_curve(case,y,.001,1e-10,screen)
        np.testing.assert_array_equal(shared['fine'], ordinary['fine'])
        np.testing.assert_array_equal(shared['pieces'][0](np.linspace(0,1,17)), ordinary['pieces'][0](np.linspace(0,1,17)))
        for key in ('nfev','njev','pool_error','continuous_nonnegative','dense_join_error'):
            self.assertEqual(shared['stats'][key], ordinary['stats'][key])


if __name__ == '__main__': unittest.main()
