"""M2 correctness checks against matrix exponentials and unsplit ODE flows."""

import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.sbml import SBMLModel
from metabolism.splitting import (FixedSplit, ReactionSystem, SplitFailure, compose_step,
                                  enzyme_binding, enzyme_catalysis, reversible_pair)


def chain():
    S = csc_matrix([[-1., 1., 0., 0.], [1., -1., -1., 1.], [0., 0., 1., -1.]])
    gradient = csc_matrix([[3., 0., 0.], [0., 2., 0.], [0., .4, 0.], [0., 0., .2]])
    return ReactionSystem(S, lambda t, y: gradient @ y, np.array([1., 0., 0.]),
                          [np.array([0, 1]), np.array([2, 3])], gradient != 0,
                          lambda t, y: gradient)


class SplittingChecks(unittest.TestCase):
    def test_chronological_flow_order_and_durations(self):
        calls = []
        def fast(y, h):
            calls.append(('fast', h))
            return y
        def slow(y, h):
            calls.append(('slow', h))
            return y
        compose_step(fast, slow, np.ones(1), .2, 'Strang')
        self.assertEqual(calls, [('fast', .1), ('slow', .2), ('fast', .1)])
        calls.clear()
        compose_step(fast, slow, np.ones(1), .2, 'Lie')
        self.assertEqual(calls, [('fast', .2), ('slow', .2)])

    def test_exact_shared_chain_global_orders(self):
        model = chain()
        matrix = (model.S @ model.flux_jacobian(0, model.y0)).toarray()
        reference = expm(matrix) @ model.y0
        for method, expected in [('Lie', 1.), ('Strang', 2.)]:
            errors = []
            for steps in (16, 32, 64, 128):
                y = model.y0.copy()
                for _ in range(steps):
                    y = compose_step(lambda x, h: reversible_pair(x, h, 0, 1, 3., 2.),
                                     lambda x, h: reversible_pair(x, h, 1, 2, .4, .2), y, 1 / steps, method)
                errors.append(np.max(abs(y - reference)))
                self.assertGreaterEqual(y.min(), -1e-15)
                self.assertAlmostEqual(y.sum(), 1., places=13)
            orders = np.log2(np.array(errors[:-1]) / errors[1:])
            np.testing.assert_allclose(orders[-2:], expected, atol=.06)

    def test_numerical_flows_keep_shared_species_and_whole_pairs(self):
        system = chain()
        exact = system.y0.copy()
        for _ in range(16):
            exact = compose_step(lambda y, h: reversible_pair(y, h, 0, 1, 3., 2.),
                                 lambda y, h: reversible_pair(y, h, 1, 2, .4, .2), exact, 1 / 16, 'Strang')
        split = FixedSplit(system, [0], np.ones(3), rtol=1e-11, pools=np.ones((1, 3)))
        _, states, stats = split.integrate(system.y0, 0., 1., 16)
        np.testing.assert_allclose(states[:, -1], exact, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(states.sum(axis=0), 1., atol=2e-14)
        self.assertEqual(stats['fast_columns'], [0, 1])
        self.assertEqual(stats['fast_updated_states'], [0, 1])
        self.assertEqual(stats['subflow_calls'], 48)
        self.assertLess(stats['max_substep_invariant_error'], 2e-14)

    def test_full_modifier_dependency_survives_coordinate_reduction(self):
        # C regulates A->B although this subflow cannot update C.
        system = ReactionSystem(csc_matrix([[-1.], [1.], [0.]]),
                                lambda t, y: np.array([y[0] * y[2]]),
                                np.array([1., 0., 3.]), [np.array([0])],
                                csc_matrix([[True, False, True]]))
        split = FixedSplit(system, [0], np.ones(3))
        _, states, _ = split.integrate(system.y0, 0., .2, 1)
        np.testing.assert_allclose(states[:, -1], [np.exp(-.6), 1 - np.exp(-.6), 3], rtol=1e-10)

    def test_binding_exact_flow_two_pools_and_numeric_composition(self):
        y0 = np.array([.2, 1.3, 0., .1])
        kon, koff, kcat = 4., 2., .7
        S = csc_matrix([[-1., 1., 1.], [-1., 1., 0.], [1., -1., -1.], [0., 0., 1.]])
        def flux(t, y):
            return np.array([kon * y[0] * y[1], koff * y[2], kcat * y[2]])
        system = ReactionSystem(S, flux, y0, [np.array([0, 1]), np.array([2])],
                                csc_matrix([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]))
        reference = solve_ivp(lambda t, y: S[:, :2] @ flux(t, y)[:2], (0., .4), y0,
                              method='Radau', rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(enzyme_binding(y0, .4, kon, koff), reference.y[:, -1], atol=1e-13)
        exact = y0.copy()
        for _ in range(8):
            exact = compose_step(lambda y, h: enzyme_binding(y, h, kon, koff),
                                 lambda y, h: enzyme_catalysis(y, h, kcat), exact, .1, 'Strang')
        _, states, _ = FixedSplit(system, [0], np.ones(4), rtol=1e-11).integrate(y0, 0., .8, 8)
        np.testing.assert_allclose(states[:, -1], exact, atol=2e-12)
        pools = np.array([[1., 0., 1., 0.], [0., 1., 1., 1.]])
        np.testing.assert_allclose(pools @ (states - y0[:, None]), 0., atol=2e-14)

    def test_open_balance_is_integrated_not_projected(self):
        system = ReactionSystem(csc_matrix([[-1., 1.], [1., 0.]]),
                                lambda t, y: np.array([2 * y[0], .3]),
                                np.array([1., 0.]), [np.array([0]), np.array([1])],
                                csc_matrix([[1, 0], [0, 0]]))
        _, states, stats = FixedSplit(system, [0], np.ones(2), ledger_weights=np.ones(2)).integrate(
            system.y0, 0., 1., 4)
        self.assertAlmostEqual(states[:, -1].sum(), 1.3, places=12)
        self.assertAlmostEqual(stats['ledger_increment'], .3, places=13)
        self.assertLess(abs(stats['ledger_residual']), 2e-13)
        self.assertLess(stats['ledger_quadrature_difference'], 1e-13)

    def test_negative_domain_and_nonautonomous_inputs_fail_explicitly(self):
        system = ReactionSystem(csc_matrix([[-1.]]), lambda t, y: np.ones(1),
                                np.array([.01]), [np.array([0])], csc_matrix([[0]]))
        with self.assertRaises(SplitFailure) as caught:
            FixedSplit(system, [], np.ones(1)).integrate(system.y0, 0., 1., 1)
        self.assertIn('negative', str(caught.exception))
        self.assertEqual(caught.exception.stats['clipping_count'], 0)
        self.assertLess(caught.exception.stats['min_rhs_argument'], -1e-12)
        np.testing.assert_array_equal(system.y0, [.01])
        model = SBMLModel(Path(__file__).resolve().parents[1] / 'models/BIOMD0000000051.xml')
        with self.assertRaisesRegex(ValueError, 'Autonomous'):
            FixedSplit(ReactionSystem.from_sbml(model), [0], np.ones(len(model.y0)))
        system.groups = [np.array([0]), np.array([0])]
        with self.assertRaisesRegex(ValueError, 'partition'):
            FixedSplit(system, [0], np.ones(1))


if __name__ == '__main__':
    unittest.main()
