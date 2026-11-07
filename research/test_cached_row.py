import unittest
import numpy as np
from scipy.sparse import csc_matrix
from research.refined_lu import RefinedLU
from research.cached_row import CachedROW
from research.row_continuous import ROWContinuous
from research.shared_ledger import optimized_case,ledgers
from research.test_row_spc import nonlinear_case
from research.test_spc_mri import linear_case
from research.m3_ledger_flow_report import modified
from research.row_continuous import PolynomialForcing
from research.inner_dense import physical_ledger


class CacheChecks(unittest.TestCase):
    def test_current_matrix_refinement_and_refresh(self):
        for complex_case in (False,True):
            a=csc_matrix([[3.,.2],[.1,2.]])*(1+1j if complex_case else 1);b=np.array([1.,2.]);scale=np.ones(2)
            cache=RefinedLU();cache.operator(a,scale).solve(b);before=cache.signatures();trial=cache.fork()
            current=a+csc_matrix([[.00001,0.],[0.,.00002]])
            x=trial.operator(current,scale).solve(b)
            np.testing.assert_allclose(x,np.linalg.solve(current.toarray(),b),atol=2e-13)
            self.assertEqual(trial.stats['factorizations'],0);self.assertGreater(trial.stats['corrections'],0);self.assertEqual(cache.signatures(),before)
            far=a*10;x=trial.operator(far,scale).solve(b)
            np.testing.assert_allclose(x,np.linalg.solve(far.toarray(),b),atol=2e-13)
            self.assertGreater(trial.stats['refreshes'],0);self.assertEqual(cache.signatures(),before)

    def test_transactions_and_fresh_equations(self):
        case,y=nonlinear_case();kernel=CachedROW(case,[0]);original=ROWContinuous(case,[0]);before=kernel.cache_state()
        rejected=kernel.trial(y,.01,1e-11);kernel.discard(rejected);self.assertEqual(kernel.cache_state(),before)
        with self.assertRaisesRegex(ValueError,'Stale'):kernel.commit(rejected)
        first=kernel.trial(y,.01,1e-11);kernel.commit(first);saved=kernel.cache_state()
        with self.assertRaises(ValueError):kernel.trial(np.array([-1.,2.5]),.01)
        self.assertEqual(kernel.cache_state(),saved)
        second=kernel.trial(first['fine'],.01,1e-11);ref=original.trial(first['fine'],.01,1e-11)
        np.testing.assert_allclose(second['fine'],ref['fine'],atol=2e-12)
        np.testing.assert_allclose(second['pieces'][0](np.linspace(0,1,17)),ref['pieces'][0](np.linspace(0,1,17)),atol=3e-11)
        self.assertEqual(second['stats']['predictor_nlu'],0);self.assertEqual(kernel.cache_state(),saved)
        kernel.commit(second)
        with self.assertRaisesRegex(ValueError,'Stale'):kernel.commit(first)
        # A changed step also solves the current system; no false exact hit.
        changed=kernel.trial(second['fine'],.013,1e-11);ref=original.trial(second['fine'],.013,1e-11)
        np.testing.assert_allclose(changed['fine'],ref['fine'],atol=2e-12)

    def test_ledgers_match_separate_audits(self):
        case,y,_,_=linear_case();case['options']['ledger_weights']=np.array([1.,2.]);kernel=CachedROW(case,[0]);r=kernel.trial(y,.01,1e-12)
        # Longer synthetic step need not satisfy the physical ledger gate; test
        # equality of the measured residuals, not an invented conservation law.
        separate=physical_ledger(case,r['pieces'][0],y,r['fine'])
        for k,v in separate.items():self.assertAlmostEqual(v,r['stats'][k],places=14)
        modified(case,kernel,r['raw'],y,r['fine'],PolynomialForcing(r['forcing_coefficients'],.01),r['stats'])
        self.assertEqual(r['stats']['ledger_rate_states'],28*len(r['raw'].pieces))

    def test_cached_global_order_and_mature_control(self):
        from scipy.integrate import solve_ivp
        from research.cached_full import full_curve
        case,y=nonlinear_case();fun=lambda t,z:case['system'].S@case['system'].flux(t,z)
        ref=solve_ivp(fun,(0,.4),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True)
        errors=[]
        for count in (16,32,64):
            kernel=CachedROW(case,[0]);z=y.copy();dense_error=0.
            for i in range(count):
                r=kernel.trial(z,.4/count,1e-12);kernel.commit(r);z=r['fine'];q=np.linspace(0,1,9)
                dense_error=max(dense_error,float(np.max(abs(r['pieces'][0](q)-ref.sol((i+q)*.4/count)))))
            errors.append((float(np.linalg.norm(z-ref.y[:,-1])),dense_error))
        self.assertTrue(np.all(np.log2(np.array(errors[-2])/errors[-1])>2.8))
        for reuse in (False,True):
            r=full_curve(case,y,.4,1e-10,reuse=reuse)
            np.testing.assert_allclose(r['pieces'][0](np.linspace(0,1,33)),ref.sol(np.linspace(0,.4,33)),atol=3e-10)

    def test_compiled_rates_and_complex_gradient(self):
        from metabolism.m3_continuation import prepare
        from metabolism.adaptive import GroupScreen
        case=prepare('teusink');opt=optimized_case(case);y=case['system'].y0;values=np.column_stack([y,y*1.001,y*.999])
        np.testing.assert_allclose(opt['batch_rates'](values),np.column_stack([case['system'].flux(0.,z) for z in values.T]),rtol=3e-15,atol=1e-13)
        np.testing.assert_allclose(GroupScreen(opt['system'],case['scale']).gradient(y).toarray(),GroupScreen(case['system'],case['scale']).gradient(y).toarray(),rtol=3e-14,atol=1e-12)


if __name__=='__main__':unittest.main()
