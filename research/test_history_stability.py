import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.extrapolated_flow import ExtrapolatedFlow
from research.history_stability import amplification


class StabilityChecks(unittest.TestCase):
    def test_adams_and_all_fast_limits(self):
        a=amplification([[0.]],[[-1.]],.1,4)
        np.testing.assert_allclose(a[0],[1-.1*55/24,.1*59/24,-.1*37/24,.1*9/24],atol=1e-15)
        self.assertLess(max(abs(np.linalg.eigvals(a))),1.)
        self.assertGreater(max(abs(np.linalg.eigvals(amplification([[0.]],[[-1.]],1.,4)))),1.)
        b=amplification([[-1000.]],[[0.]],.01,4)
        np.testing.assert_allclose(b[0],[np.exp(-10),0,0,0],atol=1e-15)

    def test_companion_matches_actual_forced_solver(self):
        fast=np.array([[-3.,1.],[3.,-1.]]);slow=np.array([[-.2,.3],[.2,-.3]])
        s=csc_matrix(np.column_stack([np.eye(2),np.eye(2)]));rates=csc_matrix(np.vstack([fast,slow]));y=np.ones(2)
        system=ReactionSystem(s,lambda t,z:np.asarray(rates@z).ravel(),y,[np.array([0,1]),np.array([2,3])],rates.astype(bool),lambda t,z:rates)
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
        kernel=ExtrapolatedFlow(case,[0]);h=.02
        states=np.column_stack([expm(t*(fast+slow))@y for t in (-3*h,-2*h,-h,0.)])
        result=kernel.trial(y,h,np.arange(-3,1)*h,slow@states[:,:-1],1e-12)
        predicted=(amplification(fast,slow,h)@states[:,::-1].T.ravel())[:2]
        np.testing.assert_allclose(predicted,result['fine'],rtol=1e-11,atol=1e-12)
        self.assertLess(result['stats']['pool_error'],1e-12)


if __name__=='__main__':unittest.main()
