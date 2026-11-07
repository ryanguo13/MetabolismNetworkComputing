import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.spc_mri import SPC,tableau


def linear_case():
    fast=np.array([[-3.,1.],[3.,-1.]]);slow=np.array([[-.2,.3],[.2,-.3]])
    s=csc_matrix(np.column_stack([np.eye(2),np.eye(2)]));rates=csc_matrix(np.vstack([fast,slow]));y=np.array([1.,.5])
    system=ReactionSystem(s,lambda t,z:np.asarray(rates@z).ravel(),y,[np.array([0,1]),np.array([2,3])],rates.astype(bool),lambda t,z:rates)
    return dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]]))),y,fast,slow


def exact_linear_step(fast,slow,y,h,order):
    a,gamma=tableau(order);n=len(y);stages=[];full=fast+slow
    for i in range(len(a)):
        rhs=y+h*sum((a[i,j]*full@stages[j] for j in range(i)),start=np.zeros_like(y))
        stages.append(np.linalg.solve(np.eye(n)-h*a[i,i]*full,rhs))
    aug=np.zeros((3*n,3*n));aug[:n,:n]=h*fast;aug[:n,n:2*n]=np.eye(n);aug[n:2*n,2*n:]=np.eye(n)
    e=expm(aug);coeff=slow@np.column_stack(stages)@gamma.T
    return e[:n,:n]@y+h*e[:n,n:2*n]@coeff[:,0]+h*e[:n,2*n:]@coeff[:,1],np.column_stack(stages)


class SPCChecks(unittest.TestCase):
    def test_primary_coefficients_and_base_order_conditions(self):
        for order in (2,3):
            a,g=tableau(order);c=a.sum(axis=1);b=a[-1]
            np.testing.assert_allclose(g.sum(axis=1),[1,0],atol=2e-15)
            np.testing.assert_allclose(g[0]+g[1]/2,b,atol=2e-15)
            self.assertAlmostEqual(b@c,.5,places=14)
            if order==3:
                self.assertAlmostEqual(b@c**2,1/3,places=14)
                self.assertAlmostEqual(b@a@c,1/6,places=14)

    def test_coupled_linear_and_partition_limits(self):
        case,y,fast,slow=linear_case();h=.1
        for order in (2,3):
            kernel=SPC(case,[0]);result=kernel.trial(y,h,order,1e-12,residual_tol=1e-13)
            end,stages=exact_linear_step(fast,slow,y,h,order)
            np.testing.assert_allclose(result['fine'],end,rtol=2e-12,atol=2e-13)
            np.testing.assert_allclose(result['predictor']['stages'],stages,rtol=2e-12,atol=2e-13)
            self.assertEqual(result['stats']['subflows'],1)
            self.assertLess(result['stats']['pool_error'],1e-12)
            allslow=SPC(case,[]);r=allslow.trial(y,h,order,1e-12,residual_tol=1e-13)
            np.testing.assert_allclose(r['fine'],r['predictor']['endpoint'],atol=1e-12)
            allfast=SPC(case,[0,1]);r=allfast.trial(y,h,order,1e-12,residual_tol=1e-13)
            np.testing.assert_allclose(r['fine'],expm(h*(fast+slow))@y,atol=1e-12)
            with self.assertRaisesRegex(RuntimeError,'did not converge'):
                kernel.predictor(y,h,order,residual_tol=1e-15,maxiter=1)

    def test_nonlinear_global_order(self):
        y=np.array([1.,.5]);s=csc_matrix([[-1,1,-1,1],[1,-1,1,-1]])
        rates=lambda t,z:np.array([3*z[0],z[1],.2*z[0]**2,.3*z[1]**2])
        grad=lambda t,z:csc_matrix([[3.,0.],[0.,1.],[.4*z[0],0.],[0.,.6*z[1]]])
        system=ReactionSystem(s,rates,y,[np.array([0,1]),np.array([2,3])],csc_matrix([[1,0],[0,1],[1,0],[0,1]]),grad)
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
        truth=solve_ivp(lambda t,z:s@rates(t,z),(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14).y[:,-1]
        for order in (2,3):
            errors=[]
            # This nonlinear example has pre-asymptotic cancellation at N=4/8;
            # use the observed asymptotic range rather than weakening the order test.
            for count in (16,32,64):
                kernel=SPC(case,[0]);z=y.copy()
                for _ in range(count):z=kernel.trial(z,.4/count,order,1e-12,residual_tol=1e-13)['fine']
                errors.append(np.linalg.norm(z-truth))
            self.assertGreater(np.log2(errors[-2]/errors[-1]),order-.2)


if __name__=='__main__':unittest.main()
