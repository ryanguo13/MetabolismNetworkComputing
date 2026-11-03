"""Independent M3 tests: modes, rollback, switching, observables and events."""

import unittest
import sys
from dataclasses import replace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit, GroupScreen, integrate_segments, select_groups
from metabolism.m2_experiments import linear_chain
from metabolism.splitting import FixedSplit, ReactionSystem, SplitFailure


class AdaptiveChecks(unittest.TestCase):
    def test_timeout_is_a_recorded_failure_not_partial_success(self):
        model, _, _ = linear_chain(1.)
        with self.assertRaises(SplitFailure) as caught:
            AdaptiveSplit(model, np.ones(3), AdaptiveConfig(timeout=1e-9)).integrate([0., 1.])
        self.assertEqual(caught.exception.stats['status'], 'failed')
        self.assertIn('timeout', caught.exception.stats['failure_reason'])
        self.assertEqual(caught.exception.stats['accepted_steps'], 0)

    def test_accepts_fine_map_without_richardson_extrapolation(self):
        model, _, _ = linear_chain(1.)
        config = AdaptiveConfig(target=1e-3, initial_step=.01, max_step=.01)
        actual, stats = AdaptiveSplit(model, np.ones(3), config, fixed_groups=[0]).integrate([0., .01])
        _, fine, _ = FixedSplit(model, [0], np.ones(3), rtol=1e-8).integrate(model.y0, 0., .01, 2)
        self.assertEqual(stats['accepted_steps'], 1)
        np.testing.assert_array_equal(actual[:, -1], fine[:, -1])

    def test_hysteresis_band_and_equal_threshold_ablation(self):
        previous = [0, 1]
        kappa = np.array([.2, .5, .5, 1.1])
        np.testing.assert_array_equal(select_groups(kappa, 1., previous), [1, 3])
        np.testing.assert_array_equal(select_groups(kappa, 1., previous, leave=1.), [3])

    def test_group_norm_matches_explicit_shared_column_jacobians(self):
        model, _, _ = linear_chain(100.)
        scale = np.array([.4, 2., .1])
        screen = GroupScreen(model, scale)
        kappa, gradient = screen.evaluate(model.y0)
        expected = [np.max(np.asarray(abs(model.S[:, g] @ gradient[g]) @ scale).ravel() / scale)
                    for g in model.groups]
        np.testing.assert_allclose(kappa, expected, rtol=1e-14)

    def test_adaptive_weak_chain_against_exponential(self):
        model, matrix, _ = linear_chain(1.)
        times = np.linspace(0., 1., 11)
        config = AdaptiveConfig(target=1e-3, initial_step=.05, max_step=.1)
        y, stats = AdaptiveSplit(model, np.ones(3), config, pools=np.ones((1, 3))).integrate(times)
        reference = np.column_stack([expm(matrix * t) @ model.y0 for t in times])
        self.assertLess(np.max(abs(y - reference)), 1e-3)
        self.assertGreater(stats['accepted_steps'], 0)
        self.assertTrue(all(max(r['audit_max'], r['audit_flux_max'], r['audit_net_flux_max']) <= 1
                            for r in stats['attempts'] if r['status'] == 'accepted'))
        np.testing.assert_allclose(y.sum(axis=0), 1., atol=2e-13)
        np.testing.assert_array_equal(model.y0, [1., 0., 0.])

    def test_rejections_leave_accepted_state_and_ledger_unchanged(self):
        model, matrix, _ = linear_chain(10.)
        config = AdaptiveConfig(target=1e-8, initial_step=.1, max_step=.1,
                                slow_bound_limit=1e6, max_rejections=2)
        y, stats = AdaptiveSplit(model, np.ones(3), config, fixed_groups=[0],
                                 ledger_weights=np.ones(3)).integrate([0., .2])
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertEqual(stats['rollback_checks'], stats['rejected_steps'])
        self.assertTrue(all(r['rollback_verified'] for r in stats['attempts'] if r['status'] == 'rejected'))
        self.assertGreater(len(stats['fallbacks']), 0)
        self.assertLess(np.max(abs(y[:, -1] - expm(matrix * .2) @ model.y0)), 1e-8)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)
        self.assertLess(abs(stats['ledger_increment']), 1e-12)

    def test_unreliable_stiff_estimator_triggers_audit_fallback(self):
        model, matrix, _ = linear_chain(1e6)
        config = AdaptiveConfig(target=1e-3, initial_step=.1, max_step=.1, slow_bound_limit=1e8)
        y, stats = AdaptiveSplit(model, np.ones(3), config, fixed_groups=[0]).integrate([0., .2])
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertIn(stats['fallbacks'][0]['reason'], ['estimator_outside_calibrated_regime', 'consecutive_rejections'])
        self.assertLess(np.max(abs(y[:, -1] - expm(matrix * .2) @ model.y0)), 1e-8)

    def test_piecewise_constant_event_lands_and_resets_history(self):
        first, a, _ = linear_chain(1.)
        second, b, _ = linear_chain(100.)
        config = AdaptiveConfig(initial_step=.01, max_step=.1)
        times = np.array([0., .1, .2, .3])
        y, records = integrate_segments([first, second], [0., .1, .3], times,
                                        np.ones(3), config, fixed_groups=[0])
        at_event = expm(a * .1) @ first.y0
        np.testing.assert_allclose(y[:, 1], at_event, atol=1e-4)
        np.testing.assert_allclose(y[:, -1], expm(b * .2) @ at_event, atol=1e-3)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]['attempts'][0]['time'], .1)
        self.assertEqual(records[1]['attempts'][0]['accepted_state_index'], 0)
        self.assertTrue(all(r['time'] + r['h'] <= .1 + 1e-15 for r in records[0]['attempts']))

    def test_coupled_slow_guard_and_no_clipping_on_invalid_model(self):
        # Individually small kappa*h, but the shared row sums exceed the bound.
        model = ReactionSystem(csc_matrix([[-1., -1.], [1., 1.]]),
                               lambda t, y: np.array([y[0], y[0]]), np.array([1., 0.]),
                               [np.array([0]), np.array([1])], csc_matrix([[1, 0], [1, 0]]))
        y, stats = AdaptiveSplit(model, np.ones(2), AdaptiveConfig(initial_step=.5, max_step=.5,
                                 slow_bound_limit=.75)).integrate([0., .5])
        self.assertEqual(stats['fallbacks'][0]['reason'], 'coupled_slow_norm_bound')
        np.testing.assert_allclose(y[:, -1], [np.exp(-1), 1 - np.exp(-1)], atol=1e-8)
        with self.assertRaises(ValueError):
            AdaptiveSplit(model, np.ones(2), replace(AdaptiveConfig(), leave=2.))
        bad = ReactionSystem(csc_matrix([[-1.]]), lambda t, y: np.ones(1), np.array([.01]),
                             [np.array([0])], csc_matrix([[0]]))
        with self.assertRaises(SplitFailure) as caught:
            AdaptiveSplit(bad, np.ones(1), AdaptiveConfig(initial_step=.1, max_step=1., max_rejections=2),
                          fixed_groups=[]).integrate([0., 1.])
        self.assertEqual(caught.exception.stats['clipping_count'], 0)
        np.testing.assert_array_equal(bad.y0, [.01])

    @unittest.skipIf(sys.platform == 'win32', 'Native SUNDIALS runtime is Linux/WSL')
    def test_native_fallback_from_accepted_state_with_shifted_times(self):
        model, matrix, _ = linear_chain(100.)
        config = AdaptiveConfig(fallback_method='CVODE-KLU', initial_step=.01, max_step=.1)
        y0 = expm(matrix * .2) @ model.y0
        values, stats = AdaptiveSplit(model, np.ones(3), config, fixed_groups=[0, 1],
                                      pools=np.ones((1, 3))).integrate([.2, .3, .5], y0)
        reference = np.column_stack([expm(matrix * (t - .2)) @ y0 for t in (.2, .3, .5)])
        np.testing.assert_allclose(values, reference, atol=2e-10, rtol=2e-9)
        self.assertFalse(stats['native_accepted_internal_states_exposed'])
        self.assertEqual(stats['fallbacks'][0]['time'], .2)


if __name__ == '__main__':
    unittest.main()
