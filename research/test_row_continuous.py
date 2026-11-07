import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from research.row_continuous import ROWContinuous,coupling,linear_curve
from research.test_row_spc import nonlinear_case
from research.test_spc_mri import linear_case


def evaluate_order():
    case,y=nonlinear_case();f=lambda t,z:case['system'].S@case['system'].flux(t,z)
    ref=solve_ivp(f,(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True);rows=[];query=np.linspace(0,1,9)
    for base in (False,True):
        old=None
        for count in (4,8,16,32,64,128):
            kernel=ROWContinuous(case,[0]);z=y.copy();dense=0.
            for i in range(count):
                r=kernel.trial(z,.4/count,1e-12,base_only=base);z=r['fine']
                dense=max(dense,float(np.max(abs(r['pieces'][0](query)-ref.sol((i+query)*.4/count)))))
            error=float(np.linalg.norm(z-ref.y[:,-1]));kernel.close()
            rows.append(dict(base_only=base,steps=count,error=error,curve_error=dense,observed_order=None if old is None else float(np.log2(old/error))))
            old=error
    return rows


class ContinuousChecks(unittest.TestCase):
    def test_moments(self):
        a,g,b,m=coupling();beta=a+g;e=beta.sum(1);c=a.sum(1)
        np.testing.assert_allclose(np.vstack([np.ones(4),e,c*c,beta@e])@m,[[1,0,0],[0,.5,0],[0,0,1/3],[0,0,1/6]],atol=3e-14)
        np.testing.assert_allclose(m.sum(1),b,atol=1e-14)
        integral=m@np.array([.5,1/3,.25])
        np.testing.assert_allclose([integral@c,integral@g.sum(1)],[1/6,0],atol=3e-14)

    def test_linear_curve_and_limits(self):
        case,y,f,s=linear_case();q=np.linspace(0,1,17);h=.1
        r=ROWContinuous(case,[0]).trial(y,h,1e-12)
        np.testing.assert_allclose(r['pieces'][0](q),np.column_stack([m@y for m in linear_curve(f,s,h,q)]),atol=5e-12)
        np.testing.assert_array_equal(r['fine'],r['raw_endpoint'])
        allslow=ROWContinuous(case,[]).trial(y,h,1e-12)
        base=ROWContinuous(case,[]).trial(y,h,1e-12,base_only=True)
        np.testing.assert_allclose(allslow['pieces'][0](q),base['pieces'][0](q),atol=1e-13)
        allfast=ROWContinuous(case,[0,1]).trial(y,h,1e-12)
        np.testing.assert_allclose(allfast['pieces'][0](q),np.column_stack([expm(x*h*(f+s))@y for x in q]),atol=5e-12)
        self.assertLess(r['stats']['pool_error'],1e-12)

    def test_nonlinear_order(self):
        rows=evaluate_order()
        for base in (False,True):
            a=[r for r in rows if r['base_only']==base]
            self.assertGreater(a[-1]['observed_order'],2.8)
            self.assertGreater(np.log2(a[-2]['curve_error']/a[-1]['curve_error']),2.8)


if __name__=='__main__':unittest.main()
