import unittest
import numpy as np
from types import SimpleNamespace
from research.rodas_centered import CenteredRodas,CenteredForcing
from research.rodas_continuous import RodasContinuous
from research.test_rodas_continuous import nonlinear_case


class CenteredChecks(unittest.TestCase):
    def test_reconstructed_polynomial(self):
        f=CenteredForcing(np.array([[1.,2.,3.,4.],[-1.,-2.,-3.,-4.]]),.2);y=np.array([2.,3.]);q=np.linspace(0,1,33);left=.05;dt=.1
        dense=SimpleNamespace(y_old=np.array([2.1,2.9]),Q=np.array([[.1,.02,.03],[-.1,-.02,-.03]]))
        p=f.dense(y,left,dt,np.array([0,1]),dense);t=left+dt*q
        expected=np.column_stack([f.integral(x) for x in t])+dense.y_old[:,None]+dense.Q@np.vander(q,4,increasing=True)[:,1:].T-f.coefficients[:,0,None]*t
        np.testing.assert_allclose(p(q),expected,atol=9e-16);np.testing.assert_allclose(p(q).sum(0),5.,atol=1e-15)

    def test_same_ode_tight_agreement_and_invariant_coefficients(self):
        case,y=nonlinear_case();q=np.linspace(0,1,65)
        old=RodasContinuous(case,[0]).trial(y,.1,1e-12);r=CenteredRodas(case,[0]).trial(y,.1,1e-12)
        np.testing.assert_array_equal(old['forcing_coefficients'],r['forcing_coefficients'])
        np.testing.assert_allclose(r['pieces'][0](q),old['pieces'][0](q),atol=3e-11)
        for p in r['pieces'][0].pieces:np.testing.assert_allclose(np.asarray(case['options']['pools']@p.coefficients[:,1:]),0.,atol=2e-15)


if __name__=='__main__':unittest.main()
