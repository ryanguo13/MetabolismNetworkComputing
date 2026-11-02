"""Independent numerical and semantic checks: uv run python -m unittest discover -s tests."""

import tempfile
import unittest
from pathlib import Path

import libsbml as sb
import numpy as np
import roadrunner
from scipy.integrate import solve_ivp

from metabolism.sbml import SBMLModel, UnsupportedSBML

ROOT = Path(__file__).resolve().parents[1]


class ModelChecks(unittest.TestCase):
    def test_public_models_against_independent_executor(self):
        rng = np.random.default_rng(20261102)
        for mid, expected_n, expected_time in (("0064", 14, False), ("0051", 18, True)):
            m = SBMLModel(ROOT / f"models/BIOMD000000{mid}.xml")
            rr = roadrunner.RoadRunner(str(m.path))
            self.assertEqual(len(m.species), expected_n)
            self.assertEqual(m.time_dependent, expected_time)
            for t in (0, .1, 1, 10):
                y = m.y0 * rng.uniform(.9, 1.1, len(m.species))
                rr.resetAll()
                rr.setValue("time", t)
                for s, x in zip(m.species, y):
                    rr[f"[{s}]"] = float(x)
                np.testing.assert_allclose(m.flux(t, y), rr.getReactionRates(), rtol=2e-12, atol=2e-11)
                np.testing.assert_allclose(m.rhs(t, y), [rr.getValue(f"{s}'") for s in m.species],
                                           rtol=2e-10, atol=2e-10)
                jac = m.jacobian(t, y, np.maximum(m.y0, 1e-3))
                self.assertLess(np.max(np.abs(jac[~m.sparsity])), 1e-8)
            if mid == "0064":
                self.assertEqual(m.L.shape[1], 1)
                np.testing.assert_allclose(m.L.T @ m.S, 0, atol=1e-13)

    def test_constant_volume_linear_flow(self):
        # Non-unit volume catches confusing substance/time with concentration/time.
        document = sb.SBMLDocument(2, 4)
        m = document.createModel()
        m.setId("linear")
        c = m.createCompartment(); c.setId("cell"); c.setSize(2); c.setConstant(True)
        for name, amount in (("A", 2), ("B", 0)):
            s = m.createSpecies(); s.setId(name); s.setCompartment("cell")
            s.setInitialAmount(amount); s.setBoundaryCondition(False); s.setConstant(False)
            s.setHasOnlySubstanceUnits(False)
        r = m.createReaction(); r.setId("decay"); r.setReversible(False)
        a = r.createReactant(); a.setSpecies("A"); a.setStoichiometry(1)
        b = r.createProduct(); b.setSpecies("B"); b.setStoichiometry(1)
        r.createKineticLaw().setMath(sb.parseL3Formula("cell * 3 * A"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.xml"
            sb.writeSBMLToFile(document, str(path))
            model = SBMLModel(path)
            np.testing.assert_allclose(model.y0, [1, 0])
            result = solve_ivp(model.rhs, (0, 1), model.y0, method="Radau", rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(result.y[:, -1], [np.exp(-3), 1 - np.exp(-3)], rtol=1e-10)
            np.testing.assert_allclose(model.L.T @ (result.y - model.y0[:, None]), 0, atol=1e-13)
            # Unsupported semantics must stop before any numerical integration.
            rule = m.createRateRule(); rule.setVariable("A"); rule.setMath(sb.parseL3Formula("1"))
            sb.writeSBMLToFile(document, str(path))
            with self.assertRaises(UnsupportedSBML):
                SBMLModel(path)

    def test_unknown_math_rejected(self):
        model = SBMLModel(ROOT / "models/BIOMD0000000064.xml")
        with self.assertRaises(UnsupportedSBML):
            model._expression(sb.parseL3Formula("sin(1)"), {}.__getitem__)


if __name__ == "__main__":
    unittest.main()
