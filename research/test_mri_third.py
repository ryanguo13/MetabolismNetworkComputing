import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.mri_third import MRI3


class ThirdChecks(unittest.TestCase):
    def test_third_order_conservation_and_full_rhs_output(self):
        s=csc_matrix([[-1,1,0,0],[1,-1,-1,1],[0,0,1,-1]])
        j=csc_matrix([[10,0,0],[0,3,0],[0,1,0],[0,0,2]])
        y=np.array([.8,.1,.1]);a=(s@j).toarray()
        sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,[np.array([0,1]),np.array([2,3])],j.astype(bool),lambda t,z:j)
        case=dict(system=sys,scale=np.ones(3),options=dict(pools=csc_matrix([[1,1,1]])))
        solver=MRI3(case,[0]);errors=[]
        for count in (8,16,32):
            z=y.copy()
            for _ in range(count):
                trial=solver.trial(z,.4/count,1e-12);z=trial['fine']
                self.assertEqual(trial['stats']['subflows'],3)
                self.assertLess(trial['stats']['pool_error'],1e-12)
            errors.append(np.max(abs(z-expm(.4*a)@y)))
        self.assertGreater(min(np.array(errors[:-1])/errors[1:]),7.)
        # All-fast limit must recover the full linear system, with no slow forcing.
        trial=MRI3(case,[0,1]).trial(y,.1,1e-12)
        np.testing.assert_allclose(trial['fine'],expm(.1*a)@y,atol=2e-11)

    def test_affine_forcing_inactive_coordinate_and_ledgers(self):
        # Slow inflow depends on X, so final-stage forcing is genuinely affine.
        s=csc_matrix(np.eye(2));j=csc_matrix([[0.,.2],[.3,0.]])
        y=np.ones(2);sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,
            [np.array([0]),np.array([1])],j.astype(bool),lambda t,z:j)
        case=dict(system=sys,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        reduced=MRI3(case,[0],reduced=True).trial(y,.01,1e-12)
        whole=MRI3(case,[0],reduced=False).trial(y,.01,1e-12)
        np.testing.assert_allclose(reduced['fine'],whole['fine'],atol=1e-12)
        np.testing.assert_allclose(reduced['fine'],expm(.01*j.toarray())@y,atol=2e-11)
        self.assertLess(reduced['stats']['modified_ledger_residual'],1e-12)
        self.assertLess(reduced['stats']['physical_ledger_residual'],1e-10)

if __name__=='__main__':unittest.main()
