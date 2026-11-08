import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from research.collocation_correction import tableau,StageSolve,correct,CorrectedROW
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve
from research.test_spc_mri import linear_case
from research.test_row_spc import nonlinear_case


class CollocationChecks(unittest.TestCase):
    def test_radau_moments_and_integrated_basis(self):
        for s in (3,5):
            c,a,l,b,values,vectors,inverse=tableau(s)
            np.testing.assert_allclose(a.sum(1),c,atol=2e-13)
            for k in range(2*s-1):self.assertAlmostEqual(a[-1]@c**k,1/(k+1),places=12)
            np.testing.assert_allclose(np.vander(c,s,increasing=True)@l,np.eye(s),atol=2e-13)
            np.testing.assert_allclose(vectors@np.diag(values)@inverse,a,atol=2e-13)

    def test_transformed_solver_matches_full_kronecker(self):
        j=np.array([[-100,7],[4,-3.]])
        for s in (3,5):
            a=tableau(s)[1];h=.2;r=np.arange(2*s).reshape(2,s)/7
            solver=StageSolve(csc_matrix(j),h,s)
            for rhs in (r,2*r-.3):
                actual=solver.solve(rhs,np.ones(2))
                direct=np.linalg.solve(np.eye(2*s)-h*np.kron(a,j),rhs.ravel(order='F')).reshape((2,s),order='F')
                np.testing.assert_allclose(actual,direct,atol=2e-13)
            self.assertEqual(len(solver.solvers),(s+1)//2)
            self.assertEqual(solver.solves,s+1)

    def test_linear_correction_is_seed_independent_collocation(self):
        case,y,f,slow=linear_case();j=f+slow;h=.1
        for s in (3,5):
            c,a,*_=tableau(s);exact=np.linalg.solve(np.eye(2*s)-h*np.kron(a,j),np.tile(y,s)).reshape((2,s),order='F')
            for slope in (np.zeros(2),j@y):
                seed=InnerCurve([DensePower(np.column_stack([y,h*slope]),h)])
                r=correct(case,y,seed,csc_matrix(j),h,s,1)
                np.testing.assert_allclose(r['stage_values'],exact,atol=2e-12)
                self.assertLess(r['stats']['pool_error'],1e-12)

    def test_nonlinear_continuous_and_endpoint_convergence(self):
        case,y=nonlinear_case();case['options']['flux_scale']=np.ones(4)
        ref=solve_ivp(lambda t,z:case['system'].S@case['system'].flux(t,z),(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True)
        q=np.linspace(0,1,17)
        for stages,sweeps in ((3,1),(5,1),(5,2)):
            errors=[]
            for steps in (2,4,8):
                z=y.copy();maximum=0.;kernel=CorrectedROW(case,[0]);h=.4/steps
                for k in range(steps):
                    r=kernel.trial(z,h,1e-12,stages,sweeps);z=r['fine'];maximum=max(maximum,float(np.max(abs(r['pieces'][0](q)-ref.sol((k+q)*h)))))
                errors.append((float(np.max(abs(z-ref.y[:,-1]))),maximum));kernel.close()
            # Broad lower bounds avoid claiming an unproved stiff order.
            self.assertGreater(np.log2(errors[0][0]/errors[1][0]),4.)
            self.assertGreater(np.log2(errors[0][1]/errors[1][1]),3.5)


if __name__=='__main__':unittest.main()
