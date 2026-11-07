import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.extrapolated_flow import SlowPolynomial,ExtrapolatedFlow,pack,unpack


class ExtrapolatedFlowChecks(unittest.TestCase):
    def test_nonuniform_history_and_inactive_dense_integral(self):
        times=np.array([-.4,-.1,0.]);h=.2
        values=np.array([1+2*times+3*times**2,2+times])
        p=SlowPolynomial(times,values,h)
        for t in (0.,.07,.2):
            np.testing.assert_allclose(p.value(t),[1+2*t+3*t*t,2+t],atol=1e-14)
            np.testing.assert_allclose(p.integral(t),[t+t*t+t**3,2*t+.5*t*t],atol=1e-14)
        curve=p.dense(np.ones(2),.05,.1,np.array([],dtype=int));q=np.linspace(0,1,9)
        np.testing.assert_allclose(curve(q),1+p.integral(.05+.1*q),atol=1e-14)
        with self.assertRaises(ValueError):SlowPolynomial([0.,0.],np.ones((2,2)),h)

    def test_global_orders_and_pool_with_computed_startup(self):
        s=csc_matrix([[-1,1,0,0],[1,-1,-1,1],[0,0,1,-1]])
        rates=csc_matrix([[10,0,0],[0,3,0],[0,1,0],[0,0,2]])
        y=np.array([.8,.1,.1]);a=(s@rates).toarray()
        sys=ReactionSystem(s,lambda t,z:np.asarray(rates@z).ravel(),y,[np.array([0,1]),np.array([2,3])],rates.astype(bool),lambda t,z:rates)
        case=dict(system=sys,scale=np.ones(3),options=dict(pools=csc_matrix([[1,1,1]])))
        for order in (2,3,4):
            errors=[]
            for count in (16,32,64):
                h=.5/count;ts=np.arange(order)*h
                startup=solve_ivp(lambda t,z:a@z,[0,ts[-1]],y,method='Radau',rtol=1e-12,atol=1e-14,t_eval=ts)
                states=list(startup.y.T);times=list(ts);kernel=ExtrapolatedFlow(case,[0])
                slow=[kernel.slow(z) for z in states]
                for step in range(order-1,count):
                    result=kernel.trial(states[-1],h,times[-order:],np.column_stack(slow[-order:-1]),1e-12)
                    states.append(result['fine']);times.append((step+1)*h);slow.append(kernel.slow(states[-1]))
                    self.assertTrue(result['stats']['continuous_nonnegative'])
                    self.assertLess(result['stats']['pool_error'],1e-12)
                errors.append(np.max(abs(states[-1]-expm(.5*a)@y)))
                packed=pack(result['pieces'][0]);restored=unpack(**packed)
                np.testing.assert_array_equal(restored(np.linspace(0,1,9)),result['pieces'][0](np.linspace(0,1,9)))
            self.assertGreater(min(np.array(errors[:-1])/errors[1:]),2**(order-.5),(order,errors))

    def test_exact_inactive_forcing_and_both_ledgers(self):
        sys=ReactionSystem(csc_matrix(np.eye(2)),lambda t,y:np.array([y[1],1.]),np.ones(2),
             [np.array([0]),np.array([1])],csc_matrix([[0,1],[0,0]]),lambda t,y:csc_matrix([[0.,1.],[0.,0.]]))
        case=dict(system=sys,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        result=ExtrapolatedFlow(case,[0]).trial(np.ones(2),.2,[-.2,0.],np.array([[0.],[1.]]),1e-12)
        q=np.linspace(0,1,17);t=.2*q
        np.testing.assert_allclose(result['pieces'][0](q),np.array([1+t+t*t/2,1+t]),atol=1e-12)
        for key in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'):
            self.assertLess(result['stats'][key],1e-12)

if __name__=='__main__':unittest.main()
