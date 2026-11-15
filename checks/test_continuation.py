"""Continuation checks kept separate from the frozen M1–M3 test set."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m2_experiments import linear_chain
from metabolism.splitting import FixedSplit, SplitFailure


class ContinuationChecks(unittest.TestCase):
    def test_mac_native_runtime_against_matrix_exponential(self):
        from metabolism.cvode import solve
        system, matrix, _ = linear_chain(1e4)
        model = SimpleNamespace(y0=system.y0, rhs=lambda t, y: matrix @ y,
                                jacobian=lambda t, y: csc_matrix(matrix),
                                sparsity=csc_matrix(matrix != 0))
        times = np.array([0., 1e-8, 1e-4, .1, 1.])
        values, stats = solve(model, times, 1e-11, 1e-13)
        reference = np.column_stack([expm(matrix*t) @ model.y0 for t in times])
        np.testing.assert_allclose(values, reference, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(values.sum(axis=0), 1., atol=1e-11)
        self.assertGreater(stats['njev'], 0)

    def test_output_grid_does_not_force_macro_landings_and_keeps_fine_map(self):
        system, _, _ = linear_chain(1.)
        config = ContinuationConfig(initial_step=.001, max_step=.001)
        times = np.r_[0., np.geomspace(1e-10, .0009, 20), .001]
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0],
                                   pools=np.ones((1, 3))).integrate(times)
        _, fine, _ = FixedSplit(system, [0], np.ones(3), rtol=1e-8).integrate(system.y0, 0., .001, 2)
        self.assertEqual(stats['accepted_steps'], 1)
        self.assertEqual(stats['output_points_audited'], len(times)-1)
        self.assertEqual(stats['partition_builds'], 1)
        np.testing.assert_array_equal(values[:, -1], fine[:, -1])
        self.assertTrue(np.all(values >= 0))
        np.testing.assert_allclose(values.sum(axis=0), 1., atol=1e-13)

    def test_audited_interpolation_rejects_hidden_initial_layer(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        config = ContinuationConfig(target=1e-5, initial_step=.01, max_step=.01,
                                     slow_bound_limit=1e6, max_rejections=16, max_attempts=256)
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0],
                                   pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertEqual(stats['rejected_steps'], stats['rollback_checks'])
        self.assertTrue(all(r['rollback_verified'] for r in stats['attempts'] if r['status'] == 'rejected'))
        self.assertLess(np.max(abs(values-reference)), 1e-5)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)

    def test_audit_control_only_skips_coarse_split_but_keeps_full_audits(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        config = ContinuationConfig(target=1e-5, initial_step=.01, max_step=.01,
                                     slow_bound_limit=1e6, max_rejections=16, max_attempts=256,
                                     audit_control_only=True)
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0],
                                   pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertEqual(stats['coarse_split_calls'], 0)
        self.assertEqual(stats['fine_split_calls'], stats['accepted_steps'] + stats['rejected_steps'])
        self.assertGreater(stats['full_calls'], 0)
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertEqual(stats['rejected_steps'], stats['rollback_checks'])
        self.assertTrue(all(a['rollback_verified'] for a in stats['attempts'] if a['status'] == 'rejected'))
        self.assertTrue(all(max(a['audit_max'], a['audit_flux_max'], a['audit_net_flux_max']) <= 1
                            for a in stats['attempts'] if a['status'] == 'accepted'))
        self.assertLess(np.max(abs(values-reference)), 1e-5)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)

    def test_bdf_audit_backend_uses_full_audit_and_preserves_rollback(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        config = ContinuationConfig(target=1e-5, initial_step=.01, max_step=.01,
                                     slow_bound_limit=1e6, max_rejections=16, max_attempts=256,
                                     audit_control_only=True, audit_method='BDF', audit_rtol=1e-10)
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0],
                                   pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertEqual(stats['full_calls'], len(stats['attempts']))
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertEqual(stats['rejected_steps'], stats['rollback_checks'])
        self.assertTrue(all(a['rollback_verified'] for a in stats['attempts'] if a['status'] == 'rejected'))
        self.assertLess(np.max(abs(values-reference)), 1e-5)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)

    def test_rejection_audit_cache_reuses_only_same_origin_dense_solution(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        common = dict(target=1e-5, initial_step=.01, max_step=.01,
                      slow_bound_limit=1e6, max_rejections=16, max_attempts=256,
                      audit_control_only=True, audit_method='BDF', audit_rtol=1e-10)
        fresh, fresh_stats = MacroSplit(
            system, np.ones(3), ContinuationConfig(**common), fixed_groups=[0],
            pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        cached, cached_stats = MacroSplit(
            system, np.ones(3), ContinuationConfig(**common, rejection_audit_cache=True),
            fixed_groups=[0], pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertGreater(cached_stats['audit_cache_hits'], 0)
        self.assertLess(cached_stats['audit_solver_calls'], cached_stats['audit_calls'])
        self.assertEqual(cached_stats['audit_calls'], len(cached_stats['attempts']))
        self.assertEqual(cached_stats['full_calls'], cached_stats['audit_calls'])
        self.assertEqual(cached_stats['rejected_steps'], cached_stats['rollback_checks'])
        self.assertTrue(all(a['rollback_verified'] for a in cached_stats['attempts']
                            if a['status'] == 'rejected'))
        np.testing.assert_allclose(cached, fresh, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(cached, reference, rtol=1.1e-3, atol=1e-8)
        self.assertLess(abs(cached_stats['ledger_residual']), 1e-12)
        self.assertLess(abs(fresh_stats['ledger_residual']), 1e-12)

    def test_aggressive_growth_safety_keeps_audit_rejection_transactional(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        config = ContinuationConfig(target=1e-5, initial_step=.01, max_step=.01,
                                     slow_bound_limit=1e6, max_rejections=16, max_attempts=256,
                                     audit_control_only=True, growth_safety=1.8)
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0],
                                   pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertGreater(stats['rejected_steps'], 0)
        self.assertEqual(stats['rejected_steps'], stats['rollback_checks'])
        self.assertTrue(all(a['rollback_verified'] for a in stats['attempts'] if a['status'] == 'rejected'))
        self.assertEqual(stats['full_calls'], len(stats['attempts']))
        self.assertLess(np.max(abs(values-reference)), 1e-5)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)
        with self.assertRaises(ValueError):
            MacroSplit(system, np.ones(3), ContinuationConfig(growth_safety=2.1)).integrate(times)

    def test_classification_step_floor_is_only_a_grouping_probe(self):
        system, matrix, _ = linear_chain(100.)
        times = np.array([0., 1e-6, .001, .005, .01])
        config = ContinuationConfig(target=1e-5, initial_step=.01, max_step=.01,
                                     classification_step_floor=.01, slow_bound_limit=1e6,
                                     max_rejections=16, max_attempts=256,
                                     audit_control_only=True)
        values, stats = MacroSplit(system, np.ones(3), config,
                                   pools=np.ones((1, 3)), ledger_weights=np.ones(3)).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertTrue(all(a['classification_step'] >= a['h'] for a in stats['attempts']))
        self.assertTrue(all(a['h'] <= a['classification_step'] for a in stats['attempts']))
        self.assertLess(np.max(abs(values-reference)), 1e-5)
        self.assertLess(abs(stats['ledger_residual']), 1e-12)
        self.assertEqual(stats['rejected_steps'], stats['rollback_checks'])

    def test_startup_is_computed_and_fallback_is_explicit(self):
        system, matrix, _ = linear_chain(10.)
        config = ContinuationConfig(initial_step=.001, max_step=.01,
                                     startup_end=.01, max_attempts=1)
        times = np.array([0., .001, .01, .1])
        values, stats = MacroSplit(system, np.ones(3), config, fixed_groups=[0]).integrate(times)
        reference = np.column_stack([expm(matrix*t) @ system.y0 for t in times])
        self.assertEqual(stats['full_intervals'][0]['reason'], 'registered_initial_layer')
        self.assertEqual(stats['full_intervals'][-1]['reason'], 'registered_work_budget')
        self.assertLess(np.max(abs(values-reference)), 1e-5)

    def test_failure_never_returns_partial_success(self):
        system, _, _ = linear_chain(10.)
        with self.assertRaises(SplitFailure) as caught:
            MacroSplit(system, np.ones(3), ContinuationConfig(timeout=1e-9)).integrate([0., 1.])
        self.assertEqual(caught.exception.stats['status'], 'failed')
        self.assertEqual(caught.exception.stats['clipping_count'], 0)


if __name__ == '__main__':
    unittest.main()
