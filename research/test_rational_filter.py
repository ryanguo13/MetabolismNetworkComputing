import unittest
import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import expm
from research.controlled_rational import RationalBasis
from research.rational_filter import filtered_residual


class FilterChecks(unittest.TestCase):
    def test_filtered_formula_matches_explicit_shifted_solve(self):
        rng=np.random.default_rng(13);j=csc_matrix(rng.normal(size=(17,17))-np.diag(np.geomspace(1,1e5,17)));y=np.ones(17);b=rng.normal(size=17);h=.2;q=np.linspace(0,1,65)
        for shift in (.25,1.):
            basis=RationalBasis(j,b,y,y,h,shift);basis.extend(8);images=np.column_stack(basis.images)
            for method in ('direct','inverse'):
                r=basis.evaluate(q,method);filtered=filtered_residual(r['basis'],images,r['projection'],r['coefficients'],r['drive'],b,h,shift)
                direct=h*basis.solve(r['residual'])
                np.testing.assert_allclose(filtered,direct,rtol=2e-8,atol=2e-13)

    def test_filtered_residual_can_underestimate_nonnormal_error(self):
        # A larger Jordan chain defeats a fixed safety factor: stable spectrum
        # and a static resolvent do not certify its transient convolution.
        n=32;a=-32*np.eye(n)+96*np.diag(np.ones(n-1),1);r=np.zeros(n);r[-1]=1.
        block=np.zeros((n+1,n+1));block[:n,:n]=a;block[:n,-1]=r
        exact=(expm(block)@np.r_[np.zeros(n),1.])[:n]
        filtered=np.linalg.solve(np.eye(n)-.25*a,r)
        self.assertGreater(np.max(abs(exact)),4*np.max(abs(filtered)))


if __name__=='__main__':unittest.main()
