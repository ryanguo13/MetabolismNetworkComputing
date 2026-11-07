import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.implicit_history import amplification
from research.implicit_flow import EndpointImplicitFlow


class ImplicitFlowChecks(unittest.TestCase):
    def test_nonlinear_pure_slow_trapezoid(self):
        s=csc_matrix([[-1.],[1.]]);y=np.array([.5,.5]);h=.2
        system=ReactionSystem(s,lambda t,z:np.array([z[0]**2]),y,[np.array([0])],csc_matrix([[True,False]]),
                              lambda t,z:csc_matrix([[2*z[0],0.]]))
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
        result=EndpointImplicitFlow(case,[]).trial(y,h,[0.],np.empty((2,0)),residual_tol=1e-12)
        root=(-1+np.sqrt(1+2*h*y[0]-h*h*y[0]**2))/h
        np.testing.assert_allclose(result['fine'],[root,1-root],rtol=1e-11,atol=1e-12)
        self.assertEqual(result['stats']['nlu'],0)
        self.assertLess(result['stats']['pool_error'],1e-12)

    def test_converged_nonlinear_iteration_matches_linear_companion(self):
        fast=np.array([[-3.,1.],[3.,-1.]]);slow=np.array([[-.2,.3],[.2,-.3]])
        s=csc_matrix(np.column_stack([np.eye(2),np.eye(2)]));rates=csc_matrix(np.vstack([fast,slow]));y=np.array([1.,.5])
        system=ReactionSystem(s,lambda t,z:np.asarray(rates@z).ravel(),y,[np.array([0,1]),np.array([2,3])],rates.astype(bool),lambda t,z:rates)
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
        kernel=EndpointImplicitFlow(case,[0]);h=.02
        for q in (2,3,4):
            states=np.column_stack([expm(t*(fast+slow))@y for t in np.arange(-(q-2),1)*h])
            expected=(amplification(fast,slow,h,q)[0]@states[:,::-1].T.ravel())[:2]
            for method in ('fixed','newton'):
                result=kernel.trial(y,h,np.arange(-(q-2),1)*h,slow@states[:,:-1],1e-12,method,residual_tol=1e-12)
                np.testing.assert_allclose(expected,result['fine'],rtol=1e-11,atol=1e-12)
                self.assertLess(result['stats']['pool_error'],1e-12)
                self.assertLessEqual(result['stats']['endpoint_residual'],1e-12)
                self.assertTrue(result['stats']['continuous_nonnegative'])
        with self.assertRaisesRegex(RuntimeError,'did not converge'):
            kernel.trial(y,.1,[0.],np.empty((2,0)),1e-12,maxiter=1,residual_tol=1e-15)


if __name__=='__main__':unittest.main()
