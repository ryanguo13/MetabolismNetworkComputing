import unittest
import numpy as np
from scipy.integrate import solve_ivp
from research.rodas_subcycled import SubcycledRodas
from research.rodas_centered import CenteredRodas
from research.affine_partition import freeze,affine_predictor
from research.rodas_coupling import coupling
from research.test_rodas_continuous import nonlinear_case


def order_diagnostic():
    case,y=nonlinear_case();rhs=lambda t,z:case['system'].S@case['system'].flux(t,z)
    truth=solve_ivp(rhs,(0,.2),y,method='DOP853',rtol=3e-14,atol=2e-14,dense_output=True)
    q=np.linspace(0,1,65);rows=[]
    for whole in (False,True):
        for m in (1,2,4):
            for h in (.2,.1,.05,.025):
                kernel=SubcycledRodas(case,[0]);r=kernel.trial(y,h,1e-12,substeps=m,base_only=whole);kernel.close()
                rows.append(dict(whole=whole,substeps=m,h=h,continuous=float(np.max(abs(r['pieces'][0](q)-truth.sol(h*q)))),endpoint=float(np.max(abs(r['fine']-truth.sol(h))))))
    return rows


class SubcycledChecks(unittest.TestCase):
    def test_one_substep_matches_frozen_kernel(self):
        case,y=nonlinear_case();q=np.linspace(0,1,65)
        for whole in (False,True):
            old=CenteredRodas(case,[0]).trial(y,.1,1e-12,base_only=whole)
            new=SubcycledRodas(case,[0]).trial(y,.1,1e-12,base_only=whole)
            np.testing.assert_array_equal(new['fine'],old['fine'])
            np.testing.assert_array_equal(new['pieces'][0](q),old['pieces'][0](q))
            np.testing.assert_array_equal(new['increments'][0],old['increments'])

    def test_affine_substeps_independent_algebra_and_one_factor(self):
        case,y=nonlinear_case();aff,ctx=freeze(case,y);h=.2;m=4;b=coupling()[2]
        kernel=SubcycledRodas(aff,[0]);r=kernel.trial(y,h,1e-12,substeps=m,base_only=True);current=y.copy()
        for step in range(m):
            local=dict(ctx,origin=current,rates=aff['system'].flux(0,current),rhs=aff['system'].S@aff['system'].flux(0,current))
            p=affine_predictor(local,kernel.sf,kernel.cols,h/m,True)
            np.testing.assert_allclose(r['increments'][step],p['increments'],atol=1e-14)
            np.testing.assert_allclose(r['forcing_coefficients'][step],p['forcing'],atol=2e-12)
            current+=p['increments']@b
        np.testing.assert_allclose(r['fine'],current,atol=2e-15)
        self.assertEqual(r['stats']['predictor_nlu'],1);self.assertEqual(r['stats']['predictor_solves'],36)
        linear=r['rodas_linearization'];np.testing.assert_allclose(linear['matrix']@linear['solve'](y),y,atol=5e-16)

    def test_continuity_pools_and_carried_fast_state(self):
        case,y=nonlinear_case();kernel=SubcycledRodas(case,[0]);r=kernel.trial(y,.2,1e-12,substeps=4)
        curve=r['pieces'][0];self.assertLess(curve.join_error(case['scale']),1e-13)
        np.testing.assert_allclose(curve(np.linspace(0,1,5)),r['correction_endpoints'],atol=3e-15)
        for p in curve.pieces:np.testing.assert_allclose(case['options']['pools']@p.coefficients[:,1:],0,atol=3e-15)
        self.assertEqual(r['stats']['subflows'],4)
        self.assertGreater(np.max(abs(r['correction_endpoints']-r['predictor_endpoints'])),1e-10)


if __name__=='__main__':unittest.main()
