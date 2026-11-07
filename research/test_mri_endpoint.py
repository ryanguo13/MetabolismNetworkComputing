import unittest
import numpy as np
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.mri_third import MRI3
from research.mri_endpoint import MRIEndpoint


class EndpointChecks(unittest.TestCase):
    def test_endpoints_and_modified_ledgers_unchanged(self):
        s=csc_matrix(np.eye(2));j=csc_matrix([[0.,.2],[.3,0.]])
        y=np.ones(2);sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,
            [np.array([0]),np.array([1])],j.astype(bool),lambda t,z:j)
        case=dict(system=sys,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        original=MRI3(case,[0]).trial(y,.01,1e-12);new=MRIEndpoint(case,[0]).trial(y,.01,1e-12)
        np.testing.assert_array_equal(new['fine'],original['fine'])
        for key in ('subflows','nfev','njev','nlu','pool_error','modified_ledger_residual','modified_ledger_quadrature'):
            self.assertEqual(new['stats'][key],original['stats'][key])
        self.assertNotIn('physical_ledger_residual',new['stats'])
        self.assertNotIn('continuous_nonnegative',new['stats'])

if __name__=='__main__':unittest.main()
