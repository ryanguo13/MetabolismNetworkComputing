import unittest
import numpy as np
from research.rodas_increment import IncrementRodas,IncrementForcing
from research.test_rodas_continuous import nonlinear_case
from research.rodas_continuous import RodasContinuous
from research.rodas_coupling import coupling
from types import SimpleNamespace


class IncrementChecks(unittest.TestCase):
    def test_shared_quartic_is_preserved_on_active_coordinates(self):
        forcing=IncrementForcing(np.array([[1.,2.,3.,4.],[-1.,-2.,-3.,-4.]]),.2)
        initial=np.array([2.,3.]);dense=SimpleNamespace(y_old=np.array([.1,-.1]),Q=np.array([[.01,.02,.03],[-.01,-.02,-.03]]))
        curve=forcing.dense(initial,.05,.1,np.array([0,1]),dense);q=np.linspace(0,1,31)
        expected=initial[:,None]+np.column_stack([forcing.integral(.05+.1*t) for t in q])+dense.y_old[:,None]+dense.Q@np.vander(q,4,increasing=True)[:,1:].T
        np.testing.assert_allclose(curve(q),expected,atol=8e-16)
        np.testing.assert_allclose(curve(q).sum(0),5.,atol=1e-15)
        self.assertGreater(np.max(abs(curve.coefficients[:,4])),0)

    def test_change_of_variables_preserves_solution_and_pools(self):
        case,y=nonlinear_case();q=np.linspace(0,1,65)
        original=RodasContinuous(case,[0]).trial(y,.1,1e-12)
        changed=IncrementRodas(case,[0]).trial(y,.1,1e-12)
        np.testing.assert_allclose(changed['fine'],original['fine'],atol=2e-12)
        np.testing.assert_allclose(changed['pieces'][0](q),original['pieces'][0](q),atol=2e-11)
        np.testing.assert_array_equal(changed['forcing_coefficients'],original['forcing_coefficients'])
        for p in changed['pieces'][0].pieces:
            np.testing.assert_allclose(np.asarray(case['options']['pools']@p.coefficients[:,1:]),0.,atol=1e-15)
        self.assertLess(changed['stats']['pool_error'],1e-14)
        self.assertEqual(changed['stats']['predictor_nlu'],1)


if __name__=='__main__':unittest.main()
