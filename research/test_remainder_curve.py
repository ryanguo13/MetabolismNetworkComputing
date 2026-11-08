import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from research.test_rodas_continuous import nonlinear_case
from research.remainder_curve import RemainderCurve
from research.affine_partition import freeze


def exact_candidate(case, y, h, sweeps):
    """Independent augmented exponential, coefficients from Vandermonde solve."""
    j = (case['system'].S@case['system'].flux_jacobian(0, y)).toarray()
    f = lambda z: case['system'].S@case['system'].flux(0, z)
    f0 = f(y); n = len(y); stages = []
    def output(ab):
        block = np.zeros((n+4, n+4)); block[:n, :n] = h*j
        block[:n, n] = h*f0; block[:n, n+2:n+4] = h*ab
        block[n+1, n] = 1.; block[n+2, n+1] = 2.; block[n+3, n+2] = 3.
        return lambda q: y+(expm(q*block)@np.r_[np.zeros(n), 1., 0., 0., 0.])[:n]
    curve = output(np.zeros((n, 2)))
    for _ in range(sweeps):
        z = np.column_stack([curve(.5), curve(1.)]); stages.append(z)
        remainder = np.column_stack([f(v)-f0-j@(v-y) for v in z.T])
        ab = np.linalg.solve(np.array([[.5**2, .5**3], [1., 1.]]), remainder.T).T
        curve = output(ab)
    return stages, curve


def order_diagnostic():
    case, y = nonlinear_case(); q = np.linspace(0, 1, 65)
    rhs = lambda t, z: case['system'].S@case['system'].flux(0, z)
    truth = solve_ivp(rhs, (0, .2), y, method='DOP853', rtol=3e-14, atol=2e-14, dense_output=True)
    rows = []
    for sweeps in (1, 2):
        for h in (.2, .1, .05, .025):
            _, exact = exact_candidate(case, y, h, sweeps)
            values = np.column_stack([exact(t) for t in q])
            rows.append(dict(sweeps=sweeps, h=h, endpoint=float(np.max(abs(values[:, -1]-truth.sol(h)))),
                             continuous=float(np.max(abs(values-truth.sol(h*q))))))
    return rows


class RemainderChecks(unittest.TestCase):
    def test_independent_exponential_and_calls(self):
        case, y = nonlinear_case(); q = np.linspace(0, 1, 65)
        for sweeps in (1, 2):
            r = RemainderCurve(case).trial(y, .1, 1e-12, sweeps)
            stages, exact = exact_candidate(case, y, .1, sweeps)
            np.testing.assert_allclose(r['stages'], stages, rtol=0, atol=5e-13)
            np.testing.assert_allclose(r['pieces'][0](q), np.column_stack([exact(t) for t in q]), rtol=0, atol=5e-13)
            self.assertEqual(r['stats']['nonlinear_nfev'], 1+2*sweeps)
            self.assertEqual(r['stats']['nonlinear_njev'], 1)
            self.assertEqual(r['stats']['subflows'], 1+sweeps)
            self.assertEqual(r['stats']['lu_cache_misses'], r['stats']['nlu'])

    def test_affine_exactness_and_pools(self):
        case, y = nonlinear_case(); affine, _ = freeze(case, y)
        r = RemainderCurve(affine).trial(y, .2, 1e-12, 2)
        _, exact = exact_candidate(affine, y, .2, 2); q = np.linspace(0, 1, 65)
        np.testing.assert_allclose(r['pieces'][0](q), np.column_stack([exact(t) for t in q]), rtol=0, atol=5e-13)
        self.assertLess(np.max(abs(r['remainders'])), 3e-15)
        for c in r['curves']:
            for p in c.pieces:
                np.testing.assert_allclose(case['options']['pools']@p.coefficients[:, 1:], 0, atol=1e-15)

    def test_classical_continuous_local_order(self):
        rows = order_diagnostic()
        for sweeps in (1, 2):
            a, b = [r for r in rows if r['sweeps'] == sweeps][-2:]
            order = np.log2(a['continuous']/b['continuous'])
            self.assertGreater(order, 4.6); self.assertLess(order, 5.5)

    def test_negative_origin_rejected_before_rates(self):
        case, y = nonlinear_case(); y[0] = -.1
        with self.assertRaises(ValueError) as raised:
            RemainderCurve(case).trial(y, .1)
        self.assertEqual(raised.exception.stats['nonlinear_nfev'], 0)


if __name__ == '__main__':
    unittest.main()
