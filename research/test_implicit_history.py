import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from research.implicit_history import EndpointPolynomial,amplification,response


class ImplicitHistoryChecks(unittest.TestCase):
    def test_adams_moulton_and_exact_fast_limits(self):
        for q,weights in ((2,[1/2,1/2]),(3,[5/12,8/12,-1/12]),(4,[9/24,19/24,-5/24,1/24])):
            e,k=response(np.zeros((1,1)),.1,q)
            np.testing.assert_allclose(np.array(k).ravel(),.1*np.array(weights),atol=1e-15)
            a,_,_=amplification([[-1000.]],[[0.]],.01,q)
            np.testing.assert_allclose(a[0],np.r_[np.exp(-10),np.zeros(q-2)],atol=1e-15)
        a,_,_=amplification([[0.]],[[-1000.]],.1,2)
        self.assertAlmostEqual(a[0,0],-49/51)

    def test_coupled_linear_endpoint_and_order(self):
        fast=np.array([[-3.,1.],[3.,-1.]]);slow=np.array([[-.2,.3],[.2,-.3]])
        y=np.array([1.,.5]);q=4;h=.03
        history=np.column_stack([expm(-i*h*(fast+slow))@y for i in range(q-1)])
        a,_,_=amplification(fast,slow,h,q);end=(a@history.T.ravel())[:2]
        times=np.arange(-(q-2),2)*h
        values=np.column_stack([slow@history[:,::-1],slow@end])
        forcing=EndpointPolynomial(times,values,0.,h)
        flow=solve_ivp(lambda t,z:fast@z+forcing.value(t),(0,h),y,method='DOP853',rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(flow.y[:,-1],end,rtol=1e-12,atol=1e-13)
        self.assertAlmostEqual(end.sum(),y.sum(),places=13)
        for q in (2,3,4):
            errors=[]
            for h in (.1,.05,.025):
                history=np.column_stack([expm(-i*h*(fast+slow))@y for i in range(q-1)])
                a,_,_=amplification(fast,slow,h,q)
                errors.append(np.linalg.norm((a@history.T.ravel())[:2]-expm(h*(fast+slow))@y))
            self.assertGreater(np.log2(errors[-2]/errors[-1]),q+.7)


if __name__=='__main__':unittest.main()
