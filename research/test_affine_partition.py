import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from research.affine_partition import freeze,displacement_reference,affine_predictor,decomposition
from research.test_rodas_continuous import nonlinear_case
from research.rodas_centered import CenteredRodas
from research.rodas_coupling import RodasForcing


class AffineChecks(unittest.TestCase):
    def test_affine_reactions_and_tangent_are_exact_at_origin(self):
        case,y=nonlinear_case();aff,ctx=freeze(case,y);delta=np.array([.01,-.03,.02])
        np.testing.assert_array_equal(aff['system'].flux(0,y),case['system'].flux(0,y))
        np.testing.assert_allclose(aff['system'].flux(0,y+delta)-aff['system'].flux(0,y),ctx['gradient']@delta,atol=4e-16)
        self.assertEqual(aff['system'].groups,case['system'].groups)
        np.testing.assert_allclose(case['options']['pools']@ctx['jacobian'].toarray(),0,atol=1e-15)

    def test_displacement_oracle_against_augmented_matrix_exponential(self):
        j=np.array([[-1e4,0],[1e4,-2.]]);drive=np.array([.7,-.1]);origin=np.array([3.,4.]);h=.1;q=np.linspace(0,1,65)
        a=np.zeros((3,3));a[:2,:2]=j;a[:2,2]=drive
        truth=np.column_stack([origin+(expm(t*h*a)@np.array([0.,0.,1.]))[:2] for t in q])
        actual,_=displacement_reference(csc_matrix(j),lambda t:drive,origin,h,q,np.ones(2),1e-12)
        np.testing.assert_allclose(actual,truth,rtol=0,atol=2e-13)

    def test_independent_affine_predictor_and_exact_fast_oracle(self):
        case,y=nonlinear_case();aff,ctx=freeze(case,y);h=.1;q=np.linspace(0,1,33)
        for aug in (False,True):
            kernel=CenteredRodas(aff,[0]);r=kernel.trial(y,h,1e-12,augmented=aug);p=affine_predictor(ctx,kernel.sf,kernel.cols,h,aug)
            np.testing.assert_allclose(r['increments'],p['increments'],atol=1e-14)
            np.testing.assert_allclose(r['forcing_coefficients'],p['forcing'],atol=3e-13)
            forcing=RodasForcing(p['forcing'],h)
            expected,_=displacement_reference(p['fastj'],lambda t:p['fast']+forcing.value(t),y,h,q,case['scale'],1e-12)
            np.testing.assert_allclose(r['pieces'][0](q),expected,atol=2e-12)

    def test_signed_decomposition_detects_cancellation(self):
        truth=np.ones((2,3));nc=truth+2e-6;lc=truth+5e-6
        r=decomposition(nc,truth,lc,truth,np.ones(2),active=np.array([0]))
        self.assertLess(r['identity_roundoff'],1e-14);self.assertGreater(r['frozen'],r['nonlinear']);self.assertLess(r['worst']['difference'],0)
        self.assertAlmostEqual(r['cosine'],1.)


if __name__=='__main__':unittest.main()
