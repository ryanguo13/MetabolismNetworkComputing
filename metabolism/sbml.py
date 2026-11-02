"""Small, fail-closed adapter for the two admitted constant-volume SBML models.

libSBML parses MathML. Only the explicitly supported AST operations below
generate Python; SBML formula strings are never executed as Python code.
States are concentrations; kinetic laws are substance/time, divided by volume.
"""

import hashlib
from pathlib import Path

import libsbml as sb
import numpy as np
from scipy.linalg import null_space


class UnsupportedSBML(ValueError):
    pass


class SBMLModel:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.document = sb.readSBML(str(path))
        errors = [self.document.getError(i).getMessage()
                  for i in range(self.document.getNumErrors())
                  if self.document.getError(i).getSeverity() >= sb.LIBSBML_SEV_ERROR]
        if errors:
            raise UnsupportedSBML("; ".join(errors))
        m = self.document.getModel()
        if m is None:
            raise UnsupportedSBML("Missing model")
        self.model = m
        if (any(m.getPlugin(i).getPackageName() not in {"layout", "render"}
                for i in range(m.getNumPlugins())) or m.getNumEvents() or m.getNumInitialAssignments()
                or m.isSetConversionFactor()):
            raise UnsupportedSBML("Packages, events, initial assignments or conversion factors")
        self.rules = {r.getVariable(): r.getMath() for r in m.getListOfRules()
                      if r.isAssignment()}
        if len(self.rules) != m.getNumRules():
            raise UnsupportedSBML("Rate/algebraic rules")
        for c in m.getListOfCompartments():
            if not c.getConstant() or c.id in self.rules or not c.getSize() > 0:
                raise UnsupportedSBML("Changing or nonpositive compartment")
        if any(s.getHasOnlySubstanceUnits() or s.isSetConversionFactor()
               for s in m.getListOfSpecies()):
            raise UnsupportedSBML("Amount-valued species or conversion factor")
        self.species = [s.id for s in m.getListOfSpecies()
                        if not s.getBoundaryCondition() and not s.getConstant()
                        and s.id not in self.rules]
        self.reactions = [r.id for r in m.getListOfReactions()]
        self.y0 = np.array([self._initial(m.getSpecies(s)) for s in self.species])
        self.volumes = np.array([m.getCompartment(m.getSpecies(s).getCompartment()).getSize()
                                 for s in self.species])
        constants = {p.id: p.getValue() for p in m.getListOfParameters()}
        constants.update({c.id: c.getSize() for c in m.getListOfCompartments()})
        constants.update({s.id: self._initial(s) for s in m.getListOfSpecies() if s.id not in self.rules})
        symbols = {k: repr(float(v)) for k, v in constants.items() if k not in self.rules}
        symbols.update({s: f"y[{i}]" for i, s in enumerate(self.species)})
        self.functions = {f.id: f.getMath() for f in m.getListOfFunctionDefinitions()}
        self.time_dependent = False
        # Recursive resolution respects assignment dependencies, regardless of file order.
        def resolve(name, stack=()):
            if name in symbols:
                return symbols[name]
            if name not in self.rules or name in stack:
                raise UnsupportedSBML(f"Unknown or cyclic symbol: {name}")
            return self._expression(self.rules[name], lambda k: resolve(k, (*stack, name)))

        expressions = []
        self.S = np.zeros((len(self.species), len(self.reactions)))
        for j, r in enumerate(m.getListOfReactions()):
            law = r.getKineticLaw()
            if law is None:
                raise UnsupportedSBML(f"Missing kinetic law: {r.id}")
            local = {p.id: repr(float(p.getValue())) for p in law.getListOfParameters()}
            local.update({p.id: repr(float(p.getValue())) for p in law.getListOfLocalParameters()})
            expressions.append(self._expression(law.getMath(),
                               lambda k: local[k] if k in local else resolve(k)))
            for sign, refs in [(-1, r.getListOfReactants()), (1, r.getListOfProducts())]:
                for ref in refs:
                    if ref.isSetStoichiometryMath() or ref.id in self.rules:
                        raise UnsupportedSBML("Variable stoichiometry")
                    if ref.getSpecies() in self.species:
                        self.S[self.species.index(ref.getSpecies()), j] += sign * ref.getStoichiometry()
        # Generated expressions contain only numeric literals, y[index], t and operators.
        code = compile("lambda t, y: array([" + ",".join(expressions) + "])",
                       str(path), "eval")
        self.flux = eval(code, {"__builtins__": {}, "array": np.array})
        # Exact structural dependencies, including rules, user functions and modifiers.
        dependencies = np.array([[f"y[{i}]" in expr for i in range(len(self.species))]
                                 for expr in expressions], dtype=int)
        self.dependencies = dependencies.astype(bool)
        self.sparsity = ((self.S != 0).astype(int) @ dependencies) != 0
        self.L = null_space((self.S / self.volumes[:, None]).T)

    def _initial(self, s):
        if s.isSetInitialConcentration():
            value = s.getInitialConcentration()
        elif s.isSetInitialAmount():
            value = s.getInitialAmount() / self.model.getCompartment(s.getCompartment()).getSize()
        else:
            raise UnsupportedSBML(f"Missing initial value: {s.id}")
        if not np.isfinite(value) or value < 0:
            raise UnsupportedSBML(f"Invalid initial value: {s.id}")
        return value

    def _expression(self, node, resolve, depth=0):
        if depth > 100:
            raise UnsupportedSBML("Recursive/overly deep math")
        kind = node.getType()
        if node.isNumber():
            value = node.getValue()
            if not np.isfinite(value):
                raise UnsupportedSBML("Nonfinite constant")
            return repr(float(value))
        if kind == sb.AST_NAME_TIME:
            self.time_dependent = True
            return "t"
        if kind == sb.AST_NAME:
            return resolve(node.getName())
        args = [self._expression(node.getChild(i), resolve, depth + 1)
                for i in range(node.getNumChildren())]
        ops = {sb.AST_PLUS: "+", sb.AST_TIMES: "*", sb.AST_MINUS: "-",
               sb.AST_DIVIDE: "/", sb.AST_POWER: "**", sb.AST_FUNCTION_POWER: "**"}
        if kind in ops:
            if len(args) == 1 and kind == sb.AST_MINUS:
                return f"(-{args[0]})"
            if len(args) < 2 or (kind in (sb.AST_DIVIDE, sb.AST_POWER, sb.AST_FUNCTION_POWER)
                                 and len(args) != 2):
                raise UnsupportedSBML("Invalid operator arity")
            return "(" + ops[kind].join(args) + ")"
        if kind == sb.AST_FUNCTION and node.getName() in self.functions:
            definition = self.functions[node.getName()]
            names = [definition.getChild(i).getName()
                     for i in range(definition.getNumChildren() - 1)]
            if len(names) != len(args):
                raise UnsupportedSBML("Function arity mismatch")
            bindings = dict(zip(names, args))
            return self._expression(definition.getChild(len(names)), bindings.__getitem__, depth + 1)
        raise UnsupportedSBML(f"Unsupported MathML: {sb.formulaToL3String(node)}")

    def rhs(self, t, y):
        return (self.S @ self.flux(t, y)) / self.volumes

    def jacobian(self, t, y, scale):
        """Centered differences for offline diagnostics, not a stiffness proof."""
        delta = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(y), scale)
        eye = np.eye(len(y))
        return np.column_stack([(self.rhs(t, y + eye[i] * d) - self.rhs(t, y - eye[i] * d)) / (2 * d)
                                for i, d in enumerate(delta)])
