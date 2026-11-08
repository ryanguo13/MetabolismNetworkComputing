import unittest
import numpy as np
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from metabolism.splitting import ReactionSystem
from research.rodas_coupling import base,coupling,moments,targets
from research.rodas_continuous import RodasContinuous
from research.test_spc_mri import linear_case


def nonlinear_case():
    s=csc_matrix([[-1,0,1,-2],[1,-1,0,1],[0,1,-1,1.]])
    y=np.array([1.,.6,.9]);rates=lambda t,z:np.array([2*z[0]*z[1],.7*z[1]*z[2],3*z[2],.1*z[0]**2])
    grad=lambda t,z:csc_matrix([[2*z[1],2*z[0],0],[0,.7*z[2],.7*z[1]],[0,0,3],[.2*z[0],0,0]])
    system=ReactionSystem(s,rates,y,[np.array([0,1]),np.array([2,3])],grad(0,y).astype(bool),grad)
    return dict(system=system,scale=np.ones(3),options=dict(pools=csc_matrix([[1.,1.,1.]]),flux_scale=np.ones(4),ledger_weights=np.ones(3))),y


def order_diagnostic():
    case,y=nonlinear_case();rhs=lambda t,z:case['system'].S@case['system'].flux(t,z)
    truth=solve_ivp(rhs,(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True)
    q=np.linspace(0,1,33);rows=[]
    for augmented in (False,True):
        for whole in (False,True):
            for h in (.2,.1,.05,.025):
                kernel=RodasContinuous(case,[0]);r=kernel.trial(y,h,1e-12,base_only=whole,augmented=augmented);values=r['pieces'][0](q)
                rows.append(dict(augmented=augmented,whole=whole,h=h,endpoint=float(np.max(abs(r['fine']-truth.sol(h)))),continuous=float(np.max(abs(values-truth.sol(h*q)))),factors=r['stats']['predictor_nlu'],stages=r['stats']['predictor_nfev']));kernel.close()
    return rows


class RodasChecks(unittest.TestCase):
    def test_rank_obstruction_and_new_moments(self):
        a,g,b,*_=base();m=moments(a,g);t=targets()
        self.assertEqual(np.linalg.matrix_rank(m),5);self.assertEqual(np.linalg.matrix_rank(np.c_[m,t]),7)
        self.assertGreater(np.max(abs(m@np.linalg.lstsq(m,t,rcond=None)[0]-t)),.1)
        np.testing.assert_allclose(m@b,t.sum(1),atol=1e-13)
        a,g,b,mu=coupling();self.assertEqual(len(b),9)
        np.testing.assert_allclose(np.diag(g),.25,atol=0);np.testing.assert_allclose(moments(a,g)@mu,t,atol=1e-12)
        np.testing.assert_allclose(mu.sum(1),b,atol=1e-14)

    def test_original_transformed_rodas_matches_stages_and_endpoint(self):
        case,y=nonlinear_case();a,g,b,a0,c0=base();h=.1
        j=(case['system'].S@case['system'].flux_jacobian(0,y)).toarray();u=[]
        for i in range(6):
            state=y+sum((a0[i,k]*u[k] for k in range(i)),start=np.zeros_like(y))
            rhs=case['system'].S@case['system'].flux(0,state)+sum((c0[i,k]*u[k]/h for k in range(i)),start=np.zeros_like(y))
            u.append(np.linalg.solve(np.eye(3)/(h*.25)-j,rhs))
        u=np.column_stack(u);source_end=y+u@np.r_[a0[-1,:-1],1.]
        for augmented in (False,True):
            r=RodasContinuous(case,[0]).trial(y,h,1e-12,base_only=True,augmented=augmented)
            np.testing.assert_allclose(r['increments'][:,:6]@g.T,u,atol=3e-14)
            np.testing.assert_allclose(r['fine'],source_end,atol=4e-14)
            self.assertEqual(r['stats']['predictor_nlu'],1);self.assertEqual(r['stats']['predictor_solves'],9 if augmented else 6)
            ctx=r['rodas_linearization'];np.testing.assert_allclose(ctx['matrix']@ctx['solve'](y),y,atol=5e-16)
            self.assertNotIn('linearization',r)  # Old gamma=.435 streaming API is incompatible.

    def test_frozen_linear_full_curve(self):
        case,y,fast,slow=linear_case();h=.1;q=np.linspace(0,1,33)
        for augmented in (False,True):
            a,g,b,mu=coupling(augmented);n=len(y);j=fast+slow;inc=[];ks=[]
            for i in range(len(b)):
                stage=y+sum((a[i,k]*inc[k] for k in range(i)),start=np.zeros(n));prev=sum((g[i,k]*inc[k] for k in range(i)),start=np.zeros(n))
                ki=np.linalg.solve(np.eye(n)-h*g[i,i]*j,h*j@(stage+prev));inc.append(ki);ks.append(h*slow@(stage+prev+g[i,i]*ki))
            coeff=np.column_stack(ks)@mu*np.arange(1,mu.shape[1]+1)
            degree=mu.shape[1];block=np.zeros(((degree+1)*n,(degree+1)*n));block[:n,:n]=h*fast
            for k in range(degree):block[k*n:(k+1)*n,(k+1)*n:(k+2)*n]=np.eye(n)
            import math
            start=np.r_[y,*[math.factorial(k)*coeff[:,k] for k in range(degree)]]
            exact=np.column_stack([(expm(t*block)@start)[:n] for t in q])
            r=RodasContinuous(case,[0]).trial(y,h,1e-12,augmented=augmented)
            np.testing.assert_allclose(r['pieces'][0](q),exact,atol=2e-12)

    def test_nonlinear_continuous_order(self):
        rows=order_diagnostic()
        for whole in (False,True):
            for augmented,lower in ((False,3.5),(True,4.5)):
                a=[r for r in rows if r['whole']==whole and r['augmented']==augmented]
                self.assertGreater(np.log2(a[-2]['continuous']/a[-1]['continuous']),lower)
                self.assertTrue(all(r['factors']==1 for r in a))


if __name__=='__main__':unittest.main()
