import unittest
import numpy as np
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from research.test_shared_radau import linear_case
from research.fullstep_radau import FullStepFactory, FullStepMRIEndpoint, fullstep_curve


class FullStepChecks(unittest.TestCase):
    def test_full_interval_attempt_is_rejected_when_inaccurate(self):
        factory = FullStepFactory()
        solver = factory(lambda t,y:-1000*y,0.,np.ones(1),.01,
                         jac=lambda t,y:np.array([[-1000.]]),rtol=1e-9,atol=1e-12)
        self.assertEqual(solver.h_abs,.01)
        solver.step()
        self.assertLess(solver.t,.01)
        while solver.status=='running': solver.step()
        self.assertEqual(solver.status,'finished')
        np.testing.assert_allclose(solver.y,np.exp(-10.),rtol=2e-8)

    def test_endpoint_and_full_curve_accuracy_and_guards(self):
        case,y = linear_case(); h=.001
        exact = expm(h*np.array([[-1000.,1.],[1000.,-1.]]))@y
        candidate = FullStepMRIEndpoint(case,[0,1]).trial(y,h,1e-10)
        baseline = fullstep_curve(case,y,h,1e-10,GroupScreen(case['system'],case['scale']))
        for result in (candidate,baseline):
            np.testing.assert_allclose(result['fine'],exact,rtol=1e-10)
            self.assertLess(result['stats']['pool_error'],1e-12)
        self.assertTrue(baseline['stats']['continuous_nonnegative'])
        self.assertLess(baseline['stats']['dense_join_error'],1e-12)


if __name__=='__main__': unittest.main()
