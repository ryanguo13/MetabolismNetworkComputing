import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from research.merb_predictor import MERBPredictor
from research.test_rodas_continuous import nonlinear_case
from research.affine_partition import freeze


def exact_candidate(case,y,h):
    j=(case['system'].S@case['system'].flux_jacobian(0,y)).toarray();f0=case['system'].S@case['system'].flux(0,y);n=len(y)
    a=np.zeros((n+1,n+1));a[:n,:n]=j;a[:n,-1]=f0
    stage=y+(expm(.75*h*a)@np.r_[np.zeros(n),1.])[:n]
    D=case['system'].S@case['system'].flux(0,stage)-f0-j@(stage-y)
    block=np.zeros((n+3,n+3));block[:n,:n]=j;block[:n,n]=f0;block[:n,n+2]=(16/9)*D/h**2;block[n+1,n]=1.;block[n+2,n+1]=2.
    return stage,lambda t:y+(expm(t*block)@np.r_[np.zeros(n),1.,0.,0.])[:n]


def order_diagnostic():
    case,y=nonlinear_case();q=np.linspace(0,1,65);rhs=lambda t,z:case['system'].S@case['system'].flux(0,z)
    truth=solve_ivp(rhs,(0,.2),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True);rows=[]
    for h in (.2,.1,.05,.025):
        _,exact=exact_candidate(case,y,h);values=np.column_stack([exact(h*t) for t in q])
        rows.append(dict(h=h,endpoint=float(np.max(abs(values[:,-1]-truth.sol(h)))),continuous=float(np.max(abs(values-truth.sol(h*q))))))
    return rows


class MERBChecks(unittest.TestCase):
    def test_independent_matrix_exponential_stage_and_forcing(self):
        case,y=nonlinear_case();h=.1;r=MERBPredictor(case).trial(y,h,1e-12);stage,exact=exact_candidate(case,y,h);q=np.linspace(0,1,65)
        np.testing.assert_allclose(r['stage'],stage,rtol=0,atol=3e-13)
        np.testing.assert_allclose(r['pieces'][0](q),np.column_stack([exact(h*t) for t in q]),rtol=0,atol=4e-13)
        self.assertEqual(r['stats']['nonlinear_nfev'],2);self.assertEqual(r['stats']['nonlinear_njev'],1)
        self.assertEqual(r['stats']['lu_cache_misses'],r['stats']['nlu'])

    def test_affine_exactness_and_invariants(self):
        case,y=nonlinear_case();aff,ctx=freeze(case,y);r=MERBPredictor(aff).trial(y,.2,1e-12);q=np.linspace(0,1,65)
        _,exact=exact_candidate(aff,y,.2);np.testing.assert_allclose(r['pieces'][0](q),np.column_stack([exact(.2*t) for t in q]),rtol=0,atol=5e-13)
        self.assertLess(np.max(abs(r['remainder'])),2e-15)
        for p in r['pieces'][0].pieces:np.testing.assert_allclose(case['options']['pools']@p.coefficients[:,1:],0,atol=1e-15)

    def test_endpoint_order_does_not_imply_dense_order(self):
        rows=order_diagnostic();a,b=rows[-2:]
        self.assertGreater(np.log2(a['endpoint']/b['endpoint']),4.7)
        order=np.log2(a['continuous']/b['continuous']);self.assertGreater(order,3.7);self.assertLess(order,4.4)


if __name__=='__main__':unittest.main()
