"""Apply the independent transaction/order/ledger checks to the new norm."""
import unittest
from types import FunctionType
import numpy as np
from research.test_cached_row import CacheChecks
from research.weighted_row import WeightedROW,weighted_full_curve
from research.weighted_lu import WeightedLU
from research.test_row_spc import nonlinear_case


class WeightedChecks(unittest.TestCase):
    def test_mature_weighted_control(self):
        from scipy.integrate import solve_ivp
        case,y=nonlinear_case();f=lambda t,z:case['system'].S@case['system'].flux(t,z)
        ref=solve_ivp(f,(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True)
        r=weighted_full_curve(case,y,.4,1e-10,reuse=True)
        np.testing.assert_allclose(r['pieces'][0](np.linspace(0,1,33)),ref.sol(np.linspace(0,.4,33)),atol=3e-10)


for name in dir(CacheChecks):
    if name.startswith('test_'):
        f=getattr(CacheChecks,name)
        setattr(WeightedChecks,name,FunctionType(f.__code__,dict(f.__globals__,CachedROW=WeightedROW,RefinedLU=WeightedLU),name=name,argdefs=f.__defaults__))


del CacheChecks
if __name__=='__main__':unittest.main()
