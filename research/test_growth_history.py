import unittest
import numpy as np
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from research.test_shared_radau import linear_case
from research.extrapolated_flow import ExtrapolatedFlow
from research.growth_history import schedule,bootstrap


class GrowthChecks(unittest.TestCase):
    def test_growth_covers_window_without_overshooting_ratio(self):
        for order in (3,4):
            for ratio in (1.,1.5,2.):
                times=schedule(.001,.016,order,ratio);steps=np.diff(times)[1:]
                self.assertEqual(times[-1],.016)
                self.assertTrue(np.all(np.diff(times)>0))
                self.assertLessEqual(steps[0],ratio*.001+1e-16)
                self.assertTrue(np.all(steps[1:]<=ratio*steps[:-1]+1e-16))
        for args in ((0.,1.,3,1.5),(.01,.02,3,1.5),(.01,1.,2,1.5),(.01,1.,3,2.1)):
            with self.assertRaises(ValueError):schedule(*args)

    def test_history_is_computed_dense_output_and_current_endpoint(self):
        case,y=linear_case();kernel=ExtrapolatedFlow(case,[0]);h=.001
        trial,times,states,previous=bootstrap(case,y,4,h,1e-12,GroupScreen(case['system'],case['scale']),kernel)
        matrix=np.array([[-1000.,1.],[1000.,-1.]])
        for t,z in zip(times,states.T):np.testing.assert_allclose(z,expm(t*matrix)@y,rtol=2e-10,atol=1e-12)
        np.testing.assert_array_equal(states[:,-1],trial['fine'])
        np.testing.assert_allclose(states.sum(axis=0),1.,atol=1e-12)
        np.testing.assert_array_equal(previous,np.column_stack([kernel.slow(z) for z in states[:,:-1].T]))
        self.assertTrue(trial['stats']['continuous_nonnegative'])


if __name__=='__main__':unittest.main()
