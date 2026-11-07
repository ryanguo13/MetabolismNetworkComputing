import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from research.streaming_defect import advance,Rates
from research.linear_error_output import advance_linear,LinearStreaming
from research.linear_context_row import LinearContextROW
from research.test_spc_mri import linear_case
from research.test_row_spc import nonlinear_case
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve


class LinearOutputChecks(unittest.TestCase):
    def test_nodes_unchanged_and_dense_second_order(self):
        case,y,f,s=linear_case();j=csc_matrix(f+s);rates=Rates(case);h=.2;e0=np.array([.01,-.01]);q=np.linspace(0,1,1001)
        curve=InnerCurve([DensePower(np.column_stack([y,np.zeros_like(y)]),h)])
        truth=np.column_stack([expm(h*x*j.toarray())@(y+e0)-y for x in q]);errors=[];ends=[]
        for n in (4,8,16,32):
            end,out,stats=advance_linear(curve,j,e0,rates,n,scale=np.ones(2));old,hermite,_=advance(curve,j,e0,rates,n,scale=np.ones(2))
            np.testing.assert_array_equal(end,old);np.testing.assert_allclose(out(out.edges/out.h),hermite(out.edges/out.h),atol=1e-15)
            errors.append(float(np.max(abs(out(q)-truth))));ends.append(float(np.max(abs(end-truth[:,-1]))))
        self.assertGreater(np.log2(errors[-2]/errors[-1]),1.8);self.assertGreater(np.log2(ends[-2]/ends[-1]),2.8)

    def test_no_derivative_overshoot_on_stiff_decay(self):
        from metabolism.splitting import ReactionSystem
        sys=ReactionSystem(csc_matrix([[-1.]]),lambda t,y:1e6*y,np.array([0.]),[np.array([0])],csc_matrix([[1]]))
        case=dict(system=sys);rates=Rates(case);curve=InnerCurve([DensePower(np.array([[0.,0.]]),1.)]);j=csc_matrix([[-1e6]])
        end,out,_=advance_linear(curve,j,np.array([1e-6]),rates,1)
        q=np.linspace(0,1,101);self.assertLessEqual(float(np.max(abs(out(q)))),1e-6)
        _,old,_=advance(curve,j,np.array([1e-6]),rates,1)
        self.assertGreater(float(np.max(abs(old(q)))),1e-3)

    def test_error_transaction(self):
        case,y=nonlinear_case();case['options']['flux_scale']=np.ones(4);kernel=LinearContextROW(case,[0]);est=LinearStreaming(case,y)
        trial=kernel.trial(y,.01,1e-6);r=est.trial(trial,np.linspace(0,1,17));est.discard(r)
        self.assertEqual(est.generation,0)
        with self.assertRaises(ValueError):est.commit(r)
        r=est.trial(trial,np.linspace(0,1,17));est.commit(r);np.testing.assert_array_equal(est.expected,trial['fine'])


if __name__=='__main__':unittest.main()
