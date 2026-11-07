import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.mri_single import SingleMRI
from research.ordered_radau import OrderedMRI


class SingleChecks(unittest.TestCase):
    def test_inactive_forcing_exact_curve_and_both_ledgers(self):
        system=ReactionSystem(csc_matrix(np.eye(2)),lambda t,y:np.array([y[1],1.]),np.ones(2),
            [np.array([0]),np.array([1])],csc_matrix([[0,1],[0,0]]),
            lambda t,y:csc_matrix([[0.,1.],[0.,0.]]))
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        solver=SingleMRI(case,[0]);trial=solver.trial(system.y0,.2,1e-12)
        query=np.linspace(0,1,17);t=.2*query
        np.testing.assert_allclose(trial['pieces'][0](query),np.array([1+t+t*t/2,1+t]),atol=1e-12)
        self.assertEqual(trial['stats']['subflows'],2)
        for key in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'):
            self.assertLess(trial['stats'][key],1e-12)

    def test_same_single_macro_formula_and_second_order(self):
        s=csc_matrix([[-1,1,0,0],[1,-1,-1,1],[0,0,1,-1]])
        j=csc_matrix([[10,0,0],[0,3,0],[0,1,0],[0,0,2]])
        y=np.array([.8,.1,.1]);a=(s@j).toarray()
        sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,[np.array([0,1]),np.array([2,3])],j.astype(bool),lambda t,z:j)
        case=dict(system=sys,scale=np.ones(3),options=dict(pools=csc_matrix([[1,1,1]])))
        solver=SingleMRI(case,[0]);old=OrderedMRI(case,[0])
        one=solver.trial(y,.1,1e-12)
        np.testing.assert_allclose(one['fine'],old.trial(y,.1,1e-12)['coarse'],rtol=1e-10,atol=1e-12)
        curve=one['pieces'][0];np.testing.assert_allclose(curve.derivative(np.array([0.,1.])),a@np.column_stack([y,one['fine']]),atol=1e-12)
        self.assertLess(np.max(abs(curve(np.linspace(0,1,101)).sum(axis=0)-1)),1e-12)
        errors=[]
        for count in (8,16,32):
            z=y.copy()
            for _ in range(count):z=solver.trial(z,.4/count,1e-12)['fine']
            errors.append(np.max(abs(z-expm(.4*a)@y)))
        self.assertGreater(min(np.array(errors[:-1])/errors[1:]),3.5)

if __name__=='__main__':unittest.main()
