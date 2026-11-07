import unittest
import numpy as np
from research.history_output import cubic_history


class HistoryChecks(unittest.TestCase):
    def test_nonuniform_cubic_and_linear_invariant(self):
        t=np.array([.1,.3,.55,.9]);f=lambda x:1+x+2*x*x-3*x*x*x
        states=np.array([f(t),3-f(t)]);curve=cubic_history(t,states)
        s=np.linspace(0,1,101);q=t[-2]+s*(t[-1]-t[-2])
        np.testing.assert_allclose(curve(s)[0],f(q),atol=2e-14)
        np.testing.assert_allclose(curve.derivative(s)[0],1+4*q-9*q*q,atol=1e-13)
        np.testing.assert_allclose(curve(s).sum(axis=0),3,atol=1e-14)

    def test_insufficient_history_rejected(self):
        with self.assertRaises(ValueError):cubic_history([0,1,2],np.ones((2,3)))

if __name__=='__main__':unittest.main()
