"""Exact fixed-column kinetics for admitted models; full-state dependencies remain.

This changes evaluation work, not reaction laws. Empty selections and caller
ordering are supported; duplicate reaction columns are rejected to prevent
accidental double counting. No state clipping or dependency freezing is used.
"""
import numpy as np
from scipy.sparse import csr_matrix
from metabolism.kecoli import KEcoliModel
from metabolism.sbml import SBMLModel


class SelectedReactions:
    def __init__(self, model, columns, negative_tolerance=None):
        raw = np.asarray(columns)
        if raw.ndim != 1 or (raw.size and (raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b')):
            raise ValueError('One-dimensional integer reaction columns required')
        self.columns = raw.astype(int, copy=True)
        self.model = model
        self.n = len(model.y0)
        self.m = model.S.shape[1]
        self.negative_tolerance = (np.zeros(self.n) if negative_tolerance is None
                                   else np.asarray(negative_tolerance,dtype=float).copy())
        if (self.negative_tolerance.shape != (self.n,) or not np.isfinite(self.negative_tolerance).all()
                or np.any(self.negative_tolerance < 0)):
            raise ValueError('Finite nonnegative state-domain tolerance required')
        if (np.any(self.columns < 0) or np.any(self.columns >= self.m)
                or len(np.unique(self.columns)) != len(self.columns)):
            raise ValueError('Unique in-range reaction columns required')
        self.columns.flags.writeable = False
        if type(model) is KEcoliModel:
            self.kind = 'kecoli'
            self.k = model.k[self.columns].copy()
            self.indices = model.indices[:, self.columns].copy()
            self.orders = model.orders[:, self.columns].copy()
            self.dependencies = model.dependency[self.columns].copy()
        elif type(model) is SBMLModel:
            self.kind = 'sbml'
            self.dependencies = csr_matrix(model.dependencies[self.columns])
            expressions = self._sbml_expressions(model)
            code = compile('lambda t,y: array(['+', '.join(expressions[j] for j in self.columns)+'])',
                           '<selected admitted SBML expressions>', 'eval')
            self.compiled = eval(code, {'__builtins__': {}, 'array': np.array})
        else:
            raise ValueError('Only admitted KEcoliModel and SBMLModel adapters supported')
        self.derivative_columns = np.flatnonzero(np.asarray(self.dependencies.sum(axis=0)).ravel())

    @staticmethod
    def _sbml_expressions(model):
        # Reuse the adapter's bounded MathML translator. Formula text is never
        # executed, and all constants/rules/modifiers keep their original scope.
        m = model.model
        constants = {p.id:p.getValue() for p in m.getListOfParameters()}
        constants.update({c.id:c.getSize() for c in m.getListOfCompartments()})
        constants.update({s.id:model._initial(s) for s in m.getListOfSpecies() if s.id not in model.rules})
        symbols = {k:repr(float(v)) for k,v in constants.items() if k not in model.rules}
        symbols.update({s:f'y[{i}]' for i,s in enumerate(model.species)})
        def resolve(name, stack=()):
            if name in symbols: return symbols[name]
            if name not in model.rules or name in stack: raise ValueError('Unknown or cyclic SBML symbol')
            return model._expression(model.rules[name], lambda k:resolve(k, (*stack,name)))
        expressions = []
        for reaction in m.getListOfReactions():
            law = reaction.getKineticLaw()
            local = {p.id:repr(float(p.getValue())) for p in law.getListOfParameters()}
            local.update({p.id:repr(float(p.getValue())) for p in law.getListOfLocalParameters()})
            expressions.append(model._expression(law.getMath(),lambda k:local[k] if k in local else resolve(k)))
        return expressions

    def _state(self, t, y):
        y = np.asarray(y)
        if (not np.isscalar(t) or not np.isfinite(t) or y.shape != (self.n,)
                or not np.isfinite(y).all() or np.any(y.real < -self.negative_tolerance)):
            raise ValueError('Finite full nonnegative state and scalar time required')
        return y

    def flux(self, t, y):
        y = self._state(t,y)
        if not len(self.columns): return np.empty(0,dtype=np.result_type(y,float))
        with np.errstate(invalid='raise',over='raise',divide='raise'):
            result = (self.k*np.power(y[self.indices[0]],self.orders[0])
                      *np.power(y[self.indices[1]],self.orders[1]) if self.kind == 'kecoli'
                      else self.compiled(t,y))
        if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite selected rates')
        return result

    def gradient(self,t,y):
        y = self._state(t,y)
        if np.iscomplexobj(y): raise ValueError('Real state required for gradient')
        if self.kind == 'sbml':
            out = np.zeros((len(self.columns),self.n))
            for i in self.derivative_columns:
                z = y.astype(complex); z[i] += 1j*1e-24
                out[:,i] = self.flux(t,z).imag/1e-24
            return csr_matrix(out)
        rows,cols,values = [],[],[]
        for slot in range(2):
            active = self.orders[slot]>0
            first = self.indices[slot,active]; second = self.indices[1-slot,active]
            exponents = self.orders[slot,active]
            with np.errstate(invalid='raise',over='raise',divide='raise'):
                grad = self.k[active]*exponents*np.power(y[first],exponents-1)*np.power(y[second],self.orders[1-slot,active])
            rows.extend(np.flatnonzero(active));cols.extend(first);values.extend(grad)
        return csr_matrix((values,(rows,cols)),shape=(len(self.columns),self.n))
