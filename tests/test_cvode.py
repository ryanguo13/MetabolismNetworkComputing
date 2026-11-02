"""Native SUNDIALS binding check, run with the registered Linux runtime."""

import sys
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix


@unittest.skipUnless(sys.platform == 'linux', 'Native SUNDIALS runtime is Linux/WSL')
class NativeChecks(unittest.TestCase):
    def test_stiff_reversible_chain_zero_initial_and_invariant(self):
        from metabolism.cvode import solve
        matrix = np.array([[-1000., 999., 0.], [1000., -1000., 0.], [0., 1., 0.]])
        model = SimpleNamespace(y0=np.array([1., 0., 0.]),
                                sparsity=csc_matrix(matrix != 0), rhs=lambda t, y: matrix @ y,
                                jacobian=lambda t, y: csc_matrix(matrix))
        times = np.array([0., 1e-5, .01, 1.])
        y, stats = solve(model, times, 1e-10, 1e-12)
        reference = np.column_stack([expm(matrix * t) @ model.y0 for t in times])
        np.testing.assert_allclose(y, reference, rtol=3e-9, atol=2e-10)
        np.testing.assert_allclose(y.sum(axis=0), 1, atol=1e-11)
        self.assertGreater(stats['njev'], 0)


if __name__ == '__main__':
    unittest.main()
