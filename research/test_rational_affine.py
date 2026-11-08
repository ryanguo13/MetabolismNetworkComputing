import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from research.rational_affine import action


class RationalChecks(unittest.TestCase):
    def test_full_basis_matches_augmented_exponential(self):
        j=np.array([[-1e4,2.,0],[1e4,-5.,1.],[0,3.,-1.]])
        y=np.array([1.,2.,3.]);b=np.array([-.3,.8,-.5]);scale=np.array([.01,1.,100.]);h=.2;q=np.unique(np.r_[np.linspace(0,1,65),.137,.913])
        aug=np.zeros((4,4));aug[:3,:3]=j;aug[:3,-1]=b
        truth=np.column_stack([y+(expm(h*t*aug)@np.r_[np.zeros(3),1.])[:3] for t in q])
        r=action(csc_matrix(j),b,y,scale,h,q,dimension=3)
        np.testing.assert_allclose(r['values'],truth,rtol=0,atol=2e-12)
        self.assertEqual(r['stats']['nlu'],1);self.assertEqual(r['stats']['solves'],2)
        np.testing.assert_allclose(r['values'].sum(0),6.,atol=3e-13)

    def test_eigenvector_breakdown_and_zero_rhs(self):
        j=csc_matrix(np.diag([-1.,-1e5]));y=np.ones(2);q=np.linspace(0,1,17)
        r=action(j,np.array([.3,0]),y,y,.5,q,dimension=2)
        self.assertTrue(r['stats']['breakdown']);self.assertEqual(r['stats']['actual_dimension'],1)
        np.testing.assert_allclose(r['values'][0],1+.3*(1-np.exp(-.5*q)),atol=4e-15)
        zero=action(j,np.zeros(2),y,y,.5,q);np.testing.assert_array_equal(zero['values'],np.ones((2,17)));self.assertEqual(zero['stats']['nlu'],0)

    def test_independent_projected_exponential_and_residual(self):
        rng=np.random.default_rng(83);j=rng.normal(size=(8,8));j-=10*np.eye(8);y=np.ones(8);b=rng.normal(size=8);q=np.linspace(0,1,33);h=.4
        r=action(csc_matrix(j),b,y,y,h,q,dimension=4);v=r['basis'];a=np.zeros((5,5));a[:4,:4]=v.T@j@v;a[:4,4]=v.T@b
        truth=np.column_stack([y+v@(expm(t*h*a)@np.r_[np.zeros(4),1.])[:4] for t in q])
        np.testing.assert_allclose(r['values'],truth,atol=2e-15)
        residual=r['derivatives']-j@(r['values']-y[:,None])-b[:,None]
        self.assertAlmostEqual(h*np.max(abs(residual)),r['stats']['residual_scaled'],places=14)


if __name__=='__main__':unittest.main()
