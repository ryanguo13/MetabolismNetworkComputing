"""Sparse block identity and stiff positivity counterexample."""
import unittest
import numpy as np
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import spsolve
from metabolism.splitting import ReactionSystem
from research.collocation_correction import StageSolve,tableau,correct
from research.inner_dense import InnerCurve
from research.extrapolated_flow import DensePower


class AdditionalChecks(unittest.TestCase):
    def test_sparse_transform_matches_independent_coupled_system(self):
        n=40;j=diags([np.full(n-1,2.),-np.arange(3.,n+3),np.full(n-1,.5)],[-1,0,1],format='csc')
        for stages in (3,5):
            a=tableau(stages)[1];rhs=np.sin(np.arange(n*stages)).reshape(n,stages);h=.13
            r=StageSolve(j,h,stages).solve(rhs,np.ones(n))
            direct=spsolve(eye(n*stages,format='csc')-h*kron(csc_matrix(a),j,format='csc'),rhs.ravel(order='F')).reshape(n,stages,order='F')
            np.testing.assert_allclose(r,direct,rtol=3e-12,atol=3e-13)

    def test_stiff_positive_exact_solution_does_not_excuse_negative_stages(self):
        j=csc_matrix([[-1e6,0],[1e6,0]]);y=np.array([1.,.1]);system=ReactionSystem(csc_matrix(np.eye(2)),lambda t,z:j@z,y,[np.array([0,1])],j.astype(bool),lambda t,z:j)
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
        predictor=InnerCurve([DensePower(np.column_stack([y,np.zeros(2)]),.001)])
        for stages in (3,5):
            with self.assertRaisesRegex(ValueError,'Invalid collocation stage|Negative corrected'):
                correct(case,y,predictor,j,.001,stages,1)


if __name__=='__main__':unittest.main()
