"""Independent complete-rate slicing parity and preserved split-output checks."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from metabolism.adaptive import GroupScreen
from metabolism.kecoli import KEcoliModel
from metabolism.sbml import SBMLModel
from metabolism.splitting import ReactionSystem
from research.selected_reactions import SelectedReactions
from research.selected_extrapolated import SelectedExtrapolatedFlow
from research.extrapolated_flow import ExtrapolatedFlow

ROOT=Path(__file__).resolve().parents[1]


class SelectedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=SBMLModel(ROOT/'models/BIOMD0000000064.xml')
        cls.large=KEcoliModel(ROOT/'models/k-ecoli457.zip')

    def test_selected_rate_gradient_and_caller_order(self):
        rng=np.random.default_rng(20261115)
        for model,system in ((self.small,ReactionSystem.from_sbml(self.small)),
                             (self.large,ReactionSystem.from_kecoli(self.large))):
            scale=np.maximum(model.y0,.01)
            screen=GroupScreen(system,scale)
            columns=np.array([model.S.shape[1]-1,0,3,2])
            selected=SelectedReactions(model,columns)
            for y in (np.maximum(model.y0,.02),scale*rng.uniform(.5,1.5,len(scale))):
                np.testing.assert_array_equal(selected.flux(0.,y),model.flux(0.,y)[columns])
                np.testing.assert_allclose(selected.gradient(0.,y).toarray(),
                    screen.gradient(y)[columns].toarray(),rtol=2e-15,atol=1e-12)
                # Every selected dependency remains a full-state derivative.
                self.assertEqual(selected.gradient(0.,y).shape,(len(columns),len(y)))
            with patch.object(model,'flux',side_effect=AssertionError('Full kinetics called')):
                selected.flux(0.,scale)
                selected.gradient(0.,scale)

    def test_full_selection_and_zero_state_derivative(self):
        for model in (self.small,self.large):
            selected=SelectedReactions(model,np.arange(model.S.shape[1]))
            np.testing.assert_array_equal(selected.flux(0.,model.y0),model.flux(0.,model.y0))
        np.testing.assert_array_equal(SelectedReactions(self.large,np.arange(self.large.S.shape[1])).gradient(0.,self.large.y0).toarray(),
                                      self.large.flux_jacobian(0.,self.large.y0).toarray())

    def test_shared_modifier_dependency_not_frozen(self):
        model=SBMLModel(ROOT/'models/BIOMD0000000051.xml')
        system=ReactionSystem.from_sbml(model);y=np.maximum(model.y0,.03)
        # Find an actual kinetic dependency absent from the reaction's updated
        # coordinates: a modifier/boundary coupling cannot be frozen away.
        candidates=np.argwhere(model.dependencies.T & (model.S==0))
        self.assertTrue(len(candidates))
        species,reaction=next((int(i),int(j)) for i,j in candidates
                             if abs(GroupScreen(system,np.ones(len(y))).gradient(y)[j,i])>1e-10)
        selected=SelectedReactions(model,[reaction])
        z=y.copy();z[species]*=1.01
        self.assertNotEqual(float(selected.flux(0.,y)[0]),float(selected.flux(0.,z)[0]))
        self.assertNotEqual(float(selected.gradient(0.,y)[0,species]),0.)
        np.testing.assert_array_equal(selected.flux(0.,z),model.flux(0.,z)[[reaction]])

    def test_empty_and_invalid_inputs(self):
        model=self.small
        empty=SelectedReactions(model,[])
        self.assertEqual(empty.flux(0.,model.y0).shape,(0,))
        self.assertEqual(empty.gradient(0.,model.y0).shape,(0,len(model.y0)))
        for columns in ([0,0],[-1],[model.S.shape[1]],[.5],[True],[[0]]):
            with self.assertRaises(ValueError):SelectedReactions(model,columns)
        selected=SelectedReactions(model,[0])
        for y in (model.y0[:-1],np.full(len(model.y0),np.nan),-np.ones(len(model.y0))):
            with self.assertRaises(ValueError):selected.flux(0.,y)
        with self.assertRaises(ValueError):selected.flux(np.nan,model.y0)
        tolerant=SelectedReactions(model,[],np.ones(len(model.y0))*1e-12)
        self.assertEqual(tolerant.flux(0.,np.full(len(model.y0),-1e-13)).size,0)

    def test_selected_split_matches_original_output_and_guards(self):
        model=self.small;system=ReactionSystem.from_sbml(model)
        with np.load(ROOT/'results/m1/BIOMD0000000064-reference.npz') as a:
            y=a['concentrations'][:,-1];scale=a['scale']
        case=dict(model=model,system=system,scale=scale,options=dict(pools=csr_matrix(model.L.T),
            ledger_weights=np.array([6,6,6,6,3,3,3,3,3,3,2,0,21,21],dtype=float)))
        original=ExtrapolatedFlow(case,[0,1]);selected=SelectedExtrapolatedFlow(case,[0,1])
        np.testing.assert_allclose(selected.slow(y),original.slow(y),rtol=2e-13,atol=2e-13)
        h=1e-5;times=[-h,0.];past=original.slow(y)[:,None]
        old=original.trial(y,h,times,past,1e-10)
        new=selected.trial(y,h,times,past,1e-10)
        q=np.linspace(0.,1.,65)
        self.assertLess(np.max(abs(old['pieces'][0](q)-new['pieces'][0](q))/scale[:,None]),2e-12)
        for key in ('nfev','njev','nlu','subflows','accepted_steps','continuous_nonnegative'):
            self.assertEqual(old['stats'][key],new['stats'][key])
        for key in ('pool_error','physical_ledger_residual','physical_ledger_quadrature',
                    'modified_ledger_residual','modified_ledger_quadrature','dense_join_error'):
            self.assertLess(abs(old['stats'][key]-new['stats'][key]),1e-12)


if __name__=='__main__':unittest.main()
