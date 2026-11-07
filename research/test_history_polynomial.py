import unittest
import numpy as np
from research.history_polynomial import HistoryPolynomial


class HistoryPolynomialChecks(unittest.TestCase):
    def test_nonuniform_exactness_invariants_and_derivative(self):
        for degree in (3,4,5):
            times=np.linspace(.1,.9,degree+1)**1.2
            c=np.arange(1,degree+2)/10
            f=lambda t:np.polynomial.polynomial.polyval(t,c)
            values=np.array([f(times),3-f(times)])
            p=HistoryPolynomial(times,values);s=np.linspace(0,1,101);t=times[-2]+p.h*s
            np.testing.assert_allclose(p(s)[0],f(t),atol=1e-13)
            np.testing.assert_allclose(p.derivative(s)[0],np.polynomial.polynomial.polyval(t,c[1:]*np.arange(1,degree+1)),atol=1e-12)
            np.testing.assert_allclose(p(s).sum(axis=0),3,atol=1e-13)
            self.assertGreater(p.minima().min(),0)

    def test_negative_interior_and_bad_history(self):
        for degree in (3,4,5):
            t=np.arange(-degree+1,2,dtype=float)
            p=HistoryPolynomial(t,np.array([t*(t-1)]))
            self.assertAlmostEqual(p.minima()[0],-.25,places=12)
        with self.assertRaises(ValueError):HistoryPolynomial([0,1,1,2],np.ones((1,4)))

if __name__=='__main__':unittest.main()
