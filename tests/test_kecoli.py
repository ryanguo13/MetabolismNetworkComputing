"""k-ecoli source parity and sparse derivative checks; download model first."""

import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from metabolism.kecoli import KEcoliModel

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / 'models/k-ecoli457.zip').exists(), 'Run python -m metabolism.fetch_models first')
class KEcoliChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KEcoliModel(ROOT / 'models/k-ecoli457.zip')

    def test_sparse_matrix_is_not_mutated_by_dependency_construction(self):
        model = self.model
        np.testing.assert_array_equal(model.S.toarray(), model.data['network_data']['S_f_b'])
        self.assertEqual(model.S.nnz, 14727)

    def test_original_matlab_algebra_and_pools(self):
        model = self.model
        source = ROOT / 'results/m1-completion/kecoli-octave.mat'
        self.assertTrue(source.exists(), 'Original executor fixture missing')
        data = loadmat(source)
        np.testing.assert_allclose(model.rhs(0, model.y0), data['reference_rhs'][:, 0], rtol=1e-12, atol=1e-12)
        inputs = np.load(ROOT / 'results/m1-completion/kecoli-source-inputs.npz')
        for i, state in enumerate(inputs['check_states'].T):
            np.testing.assert_allclose(model.rhs(0, state), data['reference_rhs'][:, i], rtol=1e-12, atol=1e-12)
        self.assertEqual((model.pools @ model.S).nnz, 0)
        self.assertEqual(model.pools.shape, (456, 3003))
        self.assertEqual(float(data['initial_difference'][0, 0]), 0)

    def test_analytic_jacobian_directional_difference(self):
        model = self.model
        rng = np.random.default_rng(20261102)
        y = np.maximum(model.y0, .01)
        direction = rng.normal(size=len(y)) * np.maximum(y, .01)
        delta = 1e-5
        numerical = (model.rhs(0, y + delta * direction) - model.rhs(0, y - delta * direction)) / (2 * delta)
        analytic = model.jacobian(0, y) @ direction
        self.assertLess(np.max(abs(numerical - analytic) / np.maximum(1, abs(analytic))), 2e-6)

    def test_exact_subset_rate_and_jacobian_parity(self):
        model = self.model
        columns = np.array([0, 7, 31, 120, 456, 1024, 3000, 5238])
        y = np.maximum(model.y0, .01)
        np.testing.assert_array_equal(model.flux_subset(0., y, columns), model.flux(0., y)[columns])
        full = model.flux_jacobian(0., y).toarray()[columns]
        subset = model.flux_jacobian_subset(0., y, columns).toarray()
        np.testing.assert_array_equal(subset, full)

    def test_reaction_system_exposes_exact_subset_evaluators(self):
        from metabolism.splitting import ReactionSystem
        system = ReactionSystem.from_kecoli(self.model)
        self.assertIsNotNone(system.flux_subset)
        self.assertIsNotNone(system.flux_jacobian_subset)
        columns = np.array([2, 9, 77])
        y = np.maximum(self.model.y0, .01)
        np.testing.assert_array_equal(system.flux_subset(0., y, columns),
                                      self.model.flux_subset(0., y, columns))


if __name__ == '__main__':
    unittest.main()
