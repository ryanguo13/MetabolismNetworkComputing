import unittest

import numpy as np
from scipy.sparse import csc_matrix

from research.mri_kernel import Hermite
from research.defect_shadow import propagate


class DefectChecks(unittest.TestCase):
    def test_sign_and_stiff_damping_against_exact_scalar_error(self):
        piece=Hermite(np.ones(1),np.ones(1),np.zeros(1),np.zeros(1),1.)
        for rate in (1.,1e6):
            nodes,e,_=propagate([piece],lambda y:-rate*y,lambda y:csc_matrix([[-rate]]),32)
            expected=np.expm1(-rate*nodes)
            np.testing.assert_allclose(e[0],expected,atol=1e-4,rtol=1e-4)

    def test_hidden_output_hump_and_refinement(self):
        piece=Hermite(np.ones(1),np.ones(1),np.array([2.]),np.array([-2.]),1.)
        errors=[]
        for steps in (8,16,32):
            nodes,e,_=propagate([piece],lambda y:0*y,lambda y:csc_matrix((1,1)),steps)
            exact=1-piece(nodes)[0]
            errors.append(float(np.max(abs(e[0]-exact))))
            self.assertLess(e[0,steps//2],-.49)
        # Quadratic hump has linear defect and is integrated exactly by SDIRK2.
        self.assertLess(max(errors),1e-13)

    def test_conservative_error_dynamics(self):
        a=np.array([[-30.,2.],[30.,-2.]])
        y=np.array([.8,.2])
        piece=Hermite(y,y,np.zeros(2),np.zeros(2),.1)
        _,e,_=propagate([piece],lambda z:a@z,lambda z:csc_matrix(a),32)
        self.assertLess(float(np.max(abs(e.sum(axis=0)))),1e-13)


if __name__=='__main__':unittest.main()
