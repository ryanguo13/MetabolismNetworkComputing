import unittest
import numpy as np
from research.ledger_diagnostic import split_ledger
from research.mri_kernel import Hermite,GAUSS


class LedgerDiagnosticChecks(unittest.TestCase):
    def test_endpoint_and_output_error_are_distinguishable(self):
        # y'=y, exact exp(t); independently perturb the computed endpoint.
        weights=np.ones(1);initial=np.ones(1);errors=[];oracle_errors=[]
        for h in (.1,.05,.025):
            exact=np.exp(h)*initial;end=exact+h*h
            curve=Hermite(initial,end,initial,end,h)
            oracle=Hermite(initial,exact,initial,exact,h)
            nodes,quadrature=GAUSS[16];q=(nodes+1)/2
            integral=lambda p:float(h/2*(quadrature@p(q)[0]))
            reference_integral=float(np.expm1(h))
            values=split_ledger(weights,initial,end,exact,integral(curve),reference_integral)
            self.assertAlmostEqual(values['endpoint_projection'],h*h,places=14)
            self.assertLess(abs(values['decomposition_roundoff']),1e-15)
            self.assertGreater(abs(values['signed_residual']),10*abs(values['reference_residual']))
            errors.append(abs(values['signed_residual']))
            oracle_errors.append(abs(split_ledger(weights,initial,exact,exact,integral(oracle),reference_integral)['signed_residual']))
        self.assertGreater(min(np.array(errors[:-1])/errors[1:]),3.8)
        self.assertGreater(min(np.array(oracle_errors[:-1])/oracle_errors[1:]),30.)


if __name__=='__main__':unittest.main()
