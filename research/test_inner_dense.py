import unittest
import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import expm
from metabolism.splitting import ReactionSystem
from research.inner_dense import Polynomial,InnerCurve
from research.mri_dense import DenseMRI
from research.mri_third import MRI3


class DenseChecks(unittest.TestCase):
    def test_constant_forcing_exact_curve_and_ledgers(self):
        sys=ReactionSystem(csc_matrix(np.eye(2)),lambda t,y:np.array([y[1],1.]),np.ones(2),
            [np.array([0]),np.array([1])],csc_matrix([[0,1],[0,0]]),lambda t,y:csc_matrix([[0.,1.],[0.,0.]]))
        case=dict(system=sys,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        trial=DenseMRI(case,[0]).trial(sys.y0,.2,1e-12);p=trial['pieces'][0];s=np.linspace(0,1,101);t=.2*s
        np.testing.assert_allclose(p(s),np.array([1+t+t*t/2,1+t]),atol=1e-12)
        np.testing.assert_allclose(p.derivative(s),np.array([1+t,np.ones_like(t)]),atol=1e-11)
        restored=InnerCurve.unpack(**p.pack());np.testing.assert_array_equal(restored(s),p(s))
        self.assertLess(p.join_error(np.ones(2)),1e-12)
        self.assertLess(trial['stats']['physical_ledger_residual'],1e-12)

    def test_variable_forcing_same_endpoint_and_dense_is_not_original_solution(self):
        s=csc_matrix([[-1,1,0,0],[1,-1,-1,1],[0,0,1,-1]])
        j=csc_matrix([[10,0,0],[0,3,0],[0,1,0],[0,0,2]])
        y=np.array([.8,.1,.1]);a=(s@j).toarray()
        sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,[np.array([0,1]),np.array([2,3])],j.astype(bool),lambda t,z:j)
        case=dict(system=sys,scale=np.ones(3),options=dict(pools=csc_matrix([[1,1,1]])))
        trial=DenseMRI(case,[0]).trial(y,.1,1e-12);old=MRI3(case,[0]).trial(y,.1,1e-12)
        np.testing.assert_array_equal(trial['fine'],old['fine'])
        p=trial['pieces'][0];grid=np.linspace(0,1,101);v=p(grid)
        self.assertLess(np.max(abs(v.sum(axis=0)-1)),1e-12)
        # Endpoint cancellation does not make intermediate modified-flow states exact.
        truth=np.column_stack([expm(.1*t*a)@y for t in grid])
        self.assertGreater(np.max(abs(v-truth)),np.max(abs(v[:,-1]-truth[:,-1])))

    def test_interior_negative_and_invalid_queries(self):
        p=Polynomial(np.zeros(1),np.ones(1),-np.ones(1),np.zeros(1),1.)
        curve=InnerCurve([p]);self.assertAlmostEqual(curve.minima()[0],-.25)
        with self.assertRaises(ValueError):curve(np.array([1.01]))

if __name__=='__main__':unittest.main()
