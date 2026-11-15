"""Independent oracle and history-reset/failure tests for sampled audits."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.splitting import ReactionSystem
from research.reusable_audit import ReusableFullAudit


def case():
    matrix = csc_matrix([[-80., 3.], [80., -3.]])
    system = ReactionSystem(csc_matrix([[-1., 1.], [1., -1.]]),
        lambda t, y:np.array([80.*y[0], 3.*y[1]]), np.array([.7, .3]),
        [np.array([0]),np.array([1])], csc_matrix(np.eye(2,dtype=bool)),
        lambda t,y:csc_matrix([[80.,0.],[0.,3.]]))
    return dict(system=system, scale=np.ones(2), options=dict(pools=csc_matrix([[1.,1.]]))), matrix


class ReusableAuditChecks(unittest.TestCase):
    def test_linear_oracle_and_cold_reusable_match(self):
        data, matrix = case(); q=np.array([0.,.25,.5,.75,1.]); audit=ReusableFullAudit(data)
        try:
            old = None
            for i in range(3):
                values, stats=audit.solve(data['system'].y0,.03,q,1e-11)
                truth=np.column_stack([expm(matrix.toarray()*.03*t)@data['system'].y0 for t in q])
                np.testing.assert_allclose(values,truth,rtol=2e-9,atol=2e-11)
                self.assertTrue(stats['history_reset'])
                self.assertEqual(stats['reused_allocations'],i>0)
                if old is not None:np.testing.assert_array_equal(old,values)
                old=values
        finally:audit.close()

    def test_changed_initial_state_tolerance_and_rejection_do_not_contaminate(self):
        data,_=case(); reused=ReusableFullAudit(data); q=np.array([0.,.1,.3,1.])
        try:
            for y,tol in ((np.array([.2,.8]),1e-6),(np.array([.9,.1]),1e-10),
                          (np.array([.7,.3]),1e-8)):
                values,stats=reused.solve(y,.01,q,tol)
                cold=ReusableFullAudit(data)
                try:expected,_=cold.solve(y,.01,q,tol)
                finally:cold.close()
                np.testing.assert_array_equal(values,expected)
                np.testing.assert_array_equal(values[:,0],y)
                self.assertTrue(stats['history_reset'])
                # The caller discards an audit on rejection; its next solve is
                # still independent. There is no accepted-state commit method.
        finally:reused.close()

    def test_callback_exception_propagates_and_session_recovers(self):
        data,_=case(); audit=ReusableFullAudit(data); q=np.array([0.,.5,1.])
        try:
            original=audit.system.flux
            with patch.object(audit.system,'flux',side_effect=RuntimeError('forced callback failure')):
                with self.assertRaisesRegex(RuntimeError,'forced callback failure'):
                    audit.solve(data['system'].y0,.01,q,1e-8)
            self.assertEqual(audit.system.flux,original)
            values,stats=audit.solve(data['system'].y0,.01,q,1e-8)
            self.assertTrue(np.isfinite(values).all());self.assertTrue(stats['history_reset'])
        finally:audit.close()

    def test_invalid_queries_state_and_unsupported_model_fail_closed(self):
        data,_=case(); audit=ReusableFullAudit(data)
        try:
            for q in ([0.,.5,.5,1.],[0.,1.,.5],[.25,.5,1.],[0.,.5,1.1],[0.,np.nan,1.]):
                with self.assertRaises(ValueError):audit.solve(data['system'].y0,.01,q,1e-8)
                self.assertFalse(audit.session.initialized)
            for y in (np.array([-1.,2.]),np.array([1.]),np.array([np.nan,1.])):
                with self.assertRaises(ValueError):audit.solve(y,.01,[0.,1.],1e-8)
            audit.close()
            with self.assertRaises(ValueError):audit.solve(data['system'].y0,.01,[0.,1.],1e-8)
        finally:audit.close()
        data,_=case();data['system'].flux_jacobian=None
        with self.assertRaisesRegex(ValueError,'analytic Jacobian'):ReusableFullAudit(data)


if __name__=='__main__':unittest.main()
