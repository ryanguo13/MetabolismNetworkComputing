"""Independent matrix exponential/ODE checks for slow-tangent coupling."""
import unittest
from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.slow_tangent import SlowTangent
from research.test_rodas_continuous import nonlinear_case
from research.affine_partition import freeze
from research.remainder_curve import RemainderCurve


def reference(case, y, h):
    system = case['system']
    result = solve_ivp(lambda t,z:system.S@system.flux(0.,z), (0., h), y,
        method='DOP853', rtol=3e-14, atol=2e-14, dense_output=True)
    if not result.success:
        raise RuntimeError(result.message)
    return result.sol


def order_diagnostic():
    case, y = nonlinear_case(); q = np.linspace(0., 1., 65)
    truth = reference(case, y, .2); rows = []
    for sweeps in (0, 1, 2):
        for h in (.2, .1, .05, .025):
            trial = SlowTangent(case, [0]).trial(y, h, 1e-12, sweeps)
            values = trial['pieces'][0](q)
            rows.append(dict(sweeps=sweeps, h=h,
                continuous=float(np.max(abs(values-truth(h*q)))),
                endpoint=float(np.max(abs(values[:, -1]-truth(h))))))
    return rows


class SlowTangentChecks(unittest.TestCase):
    def test_affine_independent_exponential_and_full_implicit_dimension(self):
        case, y = nonlinear_case(); affine, context = freeze(case, y)
        h = .2; q = np.linspace(0., 1., 65); n = len(y)
        block = np.zeros((n+1, n+1))
        block[:n, :n] = context['jacobian'].toarray()
        block[:n, -1] = context['rhs']
        truth = np.column_stack([y+(expm(h*t*block)@np.r_[np.zeros(n), 1.])[:n] for t in q])
        for sweeps in (0, 1, 2):
            result = SlowTangent(affine, [0]).trial(y, h, 1e-12, sweeps)
            np.testing.assert_allclose(result['pieces'][0](q), truth, rtol=0., atol=8e-13)
            self.assertEqual(result['stats']['implicit_dimension'], n)
            self.assertFalse(result['stats']['reaction_split_coverage_eligible'])
            self.assertEqual(result['stats']['subflows'], sweeps+1)
            if sweeps:
                self.assertLess(np.max(abs(result['remainders'])), 4e-15)

    def test_independent_modified_ode_stages_and_reaction_forcing(self):
        case, y = nonlinear_case(); sys = case['system']; h = .1
        kernel = SlowTangent(case, [0]); result = kernel.trial(y, h, 1e-12, 2)
        slow = np.array([2, 3]); fast = np.array([0, 1])
        v0 = sys.flux(0., y); gs = sys.flux_jacobian(0., y)[slow]
        js = sys.S[:, slow]@gs; q = np.linspace(0., 1., 65)
        forcing = np.zeros((len(y), 2))
        for sweep, curve in enumerate(result['curves']):
            oracle = solve_ivp(lambda t,z:sys.S[:,fast]@sys.flux(0.,z)[fast]
                +sys.S[:,slow]@v0[slow]+js@(z-y)
                +(t/h)**2*(forcing[:,0]+t/h*forcing[:,1]),
                (0., h), y, method='DOP853', rtol=3e-14, atol=2e-14, dense_output=True)
            self.assertTrue(oracle.success)
            np.testing.assert_allclose(curve(q), oracle.sol(h*q), rtol=0., atol=8e-13)
            if sweep < 2:
                stages = oracle.sol(h*np.array([.5, 1.]))
                np.testing.assert_allclose(result['stages'][sweep], stages, rtol=0., atol=8e-13)
                nonlinear = np.column_stack([sys.flux(0.,z)[slow]-v0[slow]-gs@(z-y) for z in stages.T])
                ab = np.linalg.solve(np.array([[.25, .125], [1., 1.]]), nonlinear.T).T
                forcing = sys.S[:,slow]@ab
                np.testing.assert_allclose(result['forcings'][sweep], forcing, rtol=0., atol=6e-13)

    def test_full_fast_and_full_slow_limits(self):
        case, y = nonlinear_case(); q = np.linspace(0., 1., 65); h = .1
        truth = reference(case, y, h)(h*q)
        result = SlowTangent(case, [0, 1]).trial(y, h, 1e-12, 0)
        np.testing.assert_allclose(result['pieces'][0](q), truth, rtol=0., atol=8e-13)
        for sweeps in (1, 2):
            expected = RemainderCurve(case).trial(y, h, 1e-12, sweeps)
            result = SlowTangent(case, []).trial(y, h, 1e-12, sweeps)
            np.testing.assert_allclose(result['pieces'][0](q), expected['pieces'][0](q), rtol=0., atol=1e-12)

    def test_pool_and_open_physical_and_modified_ledgers(self):
        case, y = nonlinear_case(); case = dict(case, options=dict(case['options'], ledger_weights=np.array([1.,2.,3.])))
        result = SlowTangent(case, [0]).trial(y, .05, 1e-12, 2)
        for curve in result['curves']:
            for piece in curve.pieces:
                np.testing.assert_allclose(case['options']['pools']@np.column_stack([piece.a,piece.b,piece.c]), 0., atol=3e-16)
        self.assertLess(result['stats']['pool_error'], 2e-15)
        self.assertLess(result['stats']['modified_ledger_residual'], 1e-12)
        self.assertLess(result['stats']['modified_ledger_quadrature'], 1e-12)
        self.assertLess(result['stats']['physical_ledger_residual'], 1e-8)
        self.assertLess(result['stats']['physical_ledger_quadrature'], 1e-12)
        self.assertGreater(result['stats']['ledger_rate_evaluations'], 0)

    def test_classical_local_order_and_stiff_counterexample(self):
        rows = order_diagnostic()
        for sweeps, lower, upper in ((0, 2.7, 3.4), (1, 4.5, 5.5), (2, 4.5, 5.5)):
            a, b = [r for r in rows if r['sweeps']==sweeps][-2:]
            order = np.log2(a['continuous']/b['continuous'])
            self.assertGreater(order, lower); self.assertLess(order, upper)
        s = csc_matrix([[1., -1.]])
        rates = lambda t,z:np.array([.001, 1000*z[0]**2])
        gradient = lambda t,z:csc_matrix([[0.], [2000*z[0]]])
        y = np.array([1.]); system = ReactionSystem(s, rates, y,
            [np.array([0]), np.array([1])], csc_matrix([[False],[True]]), gradient)
        case = dict(system=system, scale=np.ones(1), options=dict(pools=csc_matrix([[0.]])))
        truth = reference(case, y, .1)
        trial = SlowTangent(case, [0]).trial(y, .1, 1e-12, 1)
        self.assertGreater(np.max(abs(trial['fine']-truth(.1))), .05)
        self.assertTrue(trial['stats']['continuous_nonnegative'])

    def test_invalid_origin_rejected_before_kinetic_evaluation(self):
        case, y = nonlinear_case(); y[0] = -.1
        with self.assertRaises(ValueError) as raised:
            SlowTangent(case, [0]).trial(y, .1)
        self.assertEqual(raised.exception.stats['nonlinear_nfev'], 0)


if __name__ == '__main__':
    unittest.main()
