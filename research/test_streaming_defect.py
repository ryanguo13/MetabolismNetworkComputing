import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from research.streaming_defect import tableau,advance,Rates,StreamingDefect
from research.linear_context_row import LinearContextROW
from research.row_continuous import ROWContinuous
from research.test_row_spc import nonlinear_case
from research.test_spc_mri import linear_case
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve


class StreamChecks(unittest.TestCase):
    def test_third_order_conditions_and_stiff_limit(self):
        a,b,c=tableau();np.testing.assert_allclose([b.sum(),b@c,b@c**2,b@a@c],[1,.5,1/3,1/6],atol=2e-15)
        for z in (-1.,-10.,-1e4):
            r=1+z*b@np.linalg.solve(np.eye(3)-z*a,np.ones(3));self.assertLess(abs(r),1)
        self.assertLess(abs(1-1e8*b@np.linalg.solve(np.eye(3)+1e8*a,np.ones(3))),1e-6)

    def test_analytic_error_equation_and_continuous_order(self):
        case,y,f,s=linear_case();j=csc_matrix(f+s);rates=Rates(case);h=.2
        # P(t)=constant y; exact error = exp(Jt)*(y+e0)-y.
        curve=InnerCurve([DensePower(np.column_stack([y,np.zeros_like(y)]),h)]);e0=np.array([.01,-.01]);q=np.linspace(0,1,101)
        truth=np.column_stack([expm(h*x*j.toarray())@(y+e0)-y for x in q]);errors=[]
        for n in (4,8,16,32):
            end,p,stats=advance(curve,j,e0,rates,n,scale=np.ones(2));errors.append(float(np.max(abs(p(q)-truth))))
            self.assertLess(stats['max_linear_residual'],1e-12)
        self.assertGreater(np.log2(errors[-2]/errors[-1]),2.8)

    def test_exposed_factor_matches_original_and_rolls_back(self):
        case,y=nonlinear_case();case['options']['flux_scale']=np.ones(4)
        original=ROWContinuous(case,[0]).trial(y,.01,1e-6);kernel=LinearContextROW(case,[0]);trial=kernel.trial(y,.01,1e-6)
        np.testing.assert_array_equal(trial['fine'],original['fine']);ctx=trial['linearization'];rhs=np.array([.4,-.7]);np.testing.assert_allclose(ctx['matrix']@ctx['solve'](rhs),rhs,atol=2e-16)
        estimator=StreamingDefect(case,y);query=np.linspace(0,1,17);first=estimator.trial(trial,query)
        self.assertEqual(first['stats'][0]['factorizations'],0);self.assertEqual(first['stats'][0]['predictor_factor_hits'],1)
        estimator.discard(first);self.assertEqual(estimator.generation,0)
        with self.assertRaises(ValueError):estimator.commit(first)
        r=estimator.trial(trial,query);estimator.commit(r)
        with self.assertRaises(ValueError):estimator.commit(r)
        next_trial=kernel.trial(trial['fine'],.013,1e-10);next_error=estimator.trial(next_trial,query)
        self.assertTrue(np.isfinite(next_error['errors'][1]).all());estimator.discard(next_error)
        np.testing.assert_array_equal(estimator.expected,trial['fine'])

    def test_propagation_does_not_reset_history(self):
        case,y,f,s=linear_case();j=csc_matrix(f+s);rates=Rates(case);e0=np.array([.02,-.02]);q=np.linspace(0,1,9)
        p=InnerCurve([DensePower(np.column_stack([y,np.zeros_like(y)]),.1)]);e1,_,_=advance(p,j,e0,rates,8);e2,_,_=advance(p,j,e1,rates,8)
        both=InnerCurve([DensePower(np.column_stack([y,np.zeros_like(y)]),.1),DensePower(np.column_stack([y,np.zeros_like(y)]),.1)]);end,_,_=advance(both,j,e0,rates,8)
        np.testing.assert_array_equal(e2,end)
        reset,_,_=advance(p,j,np.zeros(2),rates,8);self.assertGreater(np.linalg.norm(e2-reset),.01)


if __name__=='__main__':unittest.main()
