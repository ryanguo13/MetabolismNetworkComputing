import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from research.controlled_rational import RationalBasis,proposal


class ControlledChecks(unittest.TestCase):
    def test_full_basis_both_projections_match_exact_flow(self):
        j=csc_matrix([[-1e4,2.,0],[1e4,-5.,1.],[0,3.,-1.]])
        y=np.array([1.,2.,3.]);scale=np.array([.01,1.,100.]);b=np.array([-.3,.8,-.5]);q=np.linspace(0,1,65);h=.2
        aug=np.zeros((4,4));aug[:3,:3]=j.toarray();aug[:3,-1]=b
        truth=np.column_stack([y+(expm(h*t*aug)@np.r_[np.zeros(3),1.])[:3] for t in q])
        basis=RationalBasis(j,b,y,scale,h);basis.extend(3)
        for method in ('direct','inverse'):
            r=basis.evaluate(q,method);np.testing.assert_allclose(r['values'],truth,atol=2e-12,rtol=0)
            self.assertEqual(r['stats']['nlu'],1);self.assertEqual(r['stats']['solves'],r['stats']['dimension'])
            self.assertEqual(r['stats']['dimension'],2)  # Sum-zero forcing spans an invariant plane.
            self.assertTrue(r['stats']['breakdown'])

    def test_extension_reuses_factor_and_matches_fresh_basis(self):
        rng=np.random.default_rng(17);j=csc_matrix(rng.normal(size=(12,12))-10*np.eye(12));b=rng.normal(size=12);y=np.ones(12);q=np.linspace(0,1,33)
        basis=RationalBasis(j,b,y,y,.1);basis.extend(4);basis.evaluate(q);oldsolve=basis.solve;basis.extend(8);r=basis.evaluate(q)
        fresh=RationalBasis(j,b,y,y,.1);fresh.extend(8);other=fresh.evaluate(q)
        self.assertIs(basis.solve,oldsolve);self.assertEqual(r['stats']['solves'],8)
        for k in ('basis','values','projection','residual'):np.testing.assert_array_equal(r[k],other[k])
        d=r['displacement'];v=r['basis'];direct=v@(r['projection']@r['coefficients']+r['drive'][:,None])-(j@d+b[:,None])
        np.testing.assert_allclose(r['residual'],direct,atol=2e-15)

    def test_nonnormal_residual_and_spectral_sign_are_not_error_certificates(self):
        # Error from an omitted constant residual[0,.1] is amplified by K.
        # Both eigenvalues are negative, and H*||r|| is only .1, yet error>1.
        a=np.array([[-1.,100.],[0.,-1.]]);block=np.zeros((3,3));block[:2,:2]=a;block[:2,2]=[0.,.1]
        error=(expm(block)@np.array([0.,0.,1.]))[:2]
        self.assertLess(max(np.linalg.eigvals(a).real),0);self.assertGreater(np.max(abs(error)),1.)

    def test_proposal_requires_enrichment_and_is_explicitly_uncertified(self):
        y=np.ones(2);r=dict(residual=np.zeros((2,3)),displacement=np.zeros((2,3)))
        self.assertFalse(proposal(r,None,1.,y,y)['proposed']);p=proposal(r,r,1.,y,y)
        self.assertTrue(p['proposed']);self.assertFalse(p['certified'])


if __name__=='__main__':unittest.main()
