import unittest
import math
import numpy as np
from scipy.linalg import expm
from types import FunctionType
from research.test_rodas_subcycled import order_diagnostic as frozen_order
from research.rodas_subcycled import SubcycledRodas
from research.rodas_subcycled_refreshed import RefreshedSubcycledRodas
from research.affine_partition import freeze
from research.test_rodas_continuous import nonlinear_case
from research.test_spc_mri import linear_case


def order_diagnostic():
    # Same independent DOP853 truth and queries; only candidate type differs.
    return FunctionType(frozen_order.__code__,dict(frozen_order.__globals__,SubcycledRodas=RefreshedSubcycledRodas))()


class RefreshedChecks(unittest.TestCase):
    def test_linear_equivalence_and_factor_counts(self):
        case,y=nonlinear_case();aff,_=freeze(case,y);q=np.linspace(0,1,65)
        for whole in (False,True):
            a=SubcycledRodas(aff,[0]).trial(y,.1,1e-12,substeps=4,base_only=whole)
            b=RefreshedSubcycledRodas(aff,[0]).trial(y,.1,1e-12,substeps=4,base_only=whole)
            np.testing.assert_array_equal(a['increments'],b['increments'])
            np.testing.assert_array_equal(a['pieces'][0](q),b['pieces'][0](q))
            self.assertEqual(b['stats']['predictor_nlu'],4);self.assertEqual(b['stats']['predictor_njev'],4)

    def test_exact_piecewise_fast_oracle(self):
        case,y,fast,slow=linear_case();h=.2;m=4;dt=h/m
        for cls in (SubcycledRodas,RefreshedSubcycledRodas):
            r=cls(case,[0]).trial(y,h,1e-12,substeps=m);curve=r['pieces'][0];current=y.copy();n=len(y)
            for step,coeff in enumerate(r['forcing_coefficients']):
                # Augment polynomial forcing with its exact derivatives;
                # carry the corrected state, never the predictor endpoint.
                block=np.zeros((5*n,5*n));block[:n,:n]=fast
                for k in range(4):block[k*n:(k+1)*n,(k+1)*n:(k+2)*n]=np.eye(n)
                origin=np.r_[current,*[math.factorial(k)*coeff[:,k]/dt**k for k in range(4)]]
                q=np.linspace(0,1,17);truth=np.column_stack([(expm(dt*t*block)@origin)[:n] for t in q])
                np.testing.assert_allclose(curve((step+q)/m),truth,rtol=0,atol=3e-12)
                current=truth[:,-1]

    def test_refreshed_nonlinear_local_order(self):
        rows=order_diagnostic()
        for whole in (False,True):
            for m in (1,2,4):
                s=[r for r in rows if r['whole']==whole and r['substeps']==m]
                self.assertGreater(np.log2(s[-2]['continuous']/s[-1]['continuous']),4.4)


if __name__=='__main__':unittest.main()
