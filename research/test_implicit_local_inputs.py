import unittest
import numpy as np
from research.m3_implicit_local_corrected import history_values


class HistoryInputChecks(unittest.TestCase):
    def test_selected_samples_have_expected_ages(self):
        # Columns are the actual states at t-2h and t-h; current is evaluated inside trial.
        values=np.array([[2.,1.],[20.,10.]])
        self.assertEqual(history_values(values,2).shape,(2,0))
        np.testing.assert_array_equal(history_values(values,3),[[1.],[10.]])
        np.testing.assert_array_equal(history_values(values,4),values)


if __name__=='__main__':unittest.main()
