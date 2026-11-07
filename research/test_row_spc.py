"""Independent analytic limits and nonlinear order tests for published ROW SPC."""
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from research.row_spc import ROWSPC,coefficients,linear_map
from research.test_spc_mri import linear_case
from research.m3_ledger_flow_report import modified


def nonlinear_case():
    y=np.array([1.,.5]);s=csc_matrix([[-1,1,-1,1],[1,-1,1,-1]])
    rates=lambda t,z:np.array([3*z[0],z[1],.2*z[0]**2,.3*z[1]**2])
    grad=lambda t,z:csc_matrix([[3.,0.],[0.,1.],[.4*z[0],0.],[0.,.6*z[1]]])
    system=ReactionSystem(s,rates,y,[np.array([0,1]),np.array([2,3])],csc_matrix([[1,0],[0,1],[1,0],[0,1]]),grad)
    return dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]]))),y


def order_diagnostic():
    case,y=nonlinear_case();f=lambda t,z:case['system'].S@case['system'].flux(t,z)
    refs=[solve_ivp(f,(0,.4),y,method='DOP853',rtol=t,atol=2e-14,dense_output=True) for t in (3e-14,1e-12)]
    rows=[];query=np.linspace(0,1,9)
    for base in (False,True):
        for theta in ((0.,.5,1.) if not base else (.5,)):
            previous=None
            for count in (4,8,16,32,64,128,256):
                kernel=ROWSPC(case,[0]);z=y.copy();curve_error=0.
                for i in range(count):
                    r=kernel.trial(z,.4/count,1e-12,theta,base_only=base);z=r['fine']
                    curve_error=max(curve_error,float(np.max(abs(r['pieces'][0](query)-refs[0].sol((i+query)*.4/count)))))
                error=float(np.linalg.norm(z-refs[0].y[:,-1]));kernel.close()
                rows.append(dict(base_only=base,theta=theta,steps=count,error=error,curve_error=curve_error,
                    observed_order=None if previous is None else float(np.log2(previous/error))))
                previous=error
    return dict(nonlinear=rows,reference_difference=float(np.linalg.norm(refs[0].y[:,-1]-refs[1].y[:,-1])))


class ROWChecks(unittest.TestCase):
    def test_coupling_and_base_conditions(self):
        for theta in (0.,.5,1.):
            a,g,b,mu=coefficients(theta);c=a.sum(axis=1);e=(a+g).sum(axis=1)
            np.testing.assert_allclose([mu.sum(),mu@c,mu@g.sum(axis=1)],[1,1/3,0],atol=2e-15)
            np.testing.assert_allclose([b.sum(),b@e,b@c**2,b@(a+g)@e],[1,.5,1/3,1/6],atol=3e-15)

    def test_linear_formula_and_partition_limits(self):
        case,y,fast,slow=linear_case();h=.1
        for theta in (0.,.5,1.):
            r=ROWSPC(case,[0]).trial(y,h,1e-12,theta)
            np.testing.assert_allclose(r['fine'],linear_map(fast,slow,h,theta)@y,rtol=2e-12,atol=2e-13)
            self.assertEqual(r['stats']['predictor_nlu'],1);self.assertEqual(r['stats']['predictor_solves'],4)
            self.assertLess(r['stats']['pool_error'],1e-12)
            r=ROWSPC(case,[]).trial(y,h,1e-12,theta)
            base=ROWSPC(case,[]).trial(y,h,1e-12,theta,base_only=True)
            np.testing.assert_allclose(r['fine'],base['fine'],atol=1e-13)
            r=ROWSPC(case,[0,1]).trial(y,h,1e-12,theta)
            np.testing.assert_allclose(r['fine'],expm(h*(fast+slow))@y,atol=1e-12)

    def test_endpoint_projection_and_ledger(self):
        from research.spc_mri import AffineForcing
        case,y,_,_=linear_case();case['options']['ledger_weights']=np.array([1.,2.]);kernel=ROWSPC(case,[0]);h=.1
        r=kernel.trial(y,h,1e-12);a,g,b,mu=coefficients()
        np.testing.assert_allclose(r['fine']-r['raw_endpoint'],r['slow_increments']@(b-mu),atol=2e-16)
        self.assertGreater(np.max(abs(r['fine']-r['raw_endpoint'])),1e-4)
        modified(case,kernel,r['raw'],y,r['raw_endpoint'],AffineForcing(r['forcing_coefficients'],h),r['stats'])
        self.assertLess(r['stats']['modified_ledger_residual'],1e-12)
        np.testing.assert_allclose(r['pieces'][0](np.array([0.,1.])),np.column_stack([y,r['fine']]),atol=2e-15)
        with self.assertRaisesRegex(ValueError,'Invalid ROW state'):kernel.trial(np.array([-1.,2.5]),h)

    def test_nonlinear_order(self):
        for r in order_diagnostic()['nonlinear']:
            if r['steps']==256:self.assertGreater(r['observed_order'],2.8)


if __name__=='__main__':unittest.main()
