"""Research-only Radau LU choice; same formula, tolerances and step control.

The scoped function clone avoids modifying the frozen historical MRI module.
SciPy's lu hook is internal: revalidate this adapter when SciPy changes.
"""
from types import FunctionType
import numpy as np
from scipy.integrate import Radau
from scipy.sparse import issparse
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI


class OrderedRadau(Radau):
    def __init__(self,fun,t0,y0,t_bound,**kwargs):
        jac=kwargs.get('jac')
        if len(y0)<=32 and jac is not None:
            def dense(value):return value.toarray() if issparse(value) else np.asarray(value)
            kwargs['jac']=(lambda t,y:dense(jac(t,y))) if callable(jac) else dense(jac)
        super().__init__(fun,t0,y0,t_bound,**kwargs)
        if issparse(self.J):
            def lu(matrix):
                self.nlu+=1
                return splu(matrix,permc_spec='MMD_AT_PLUS_A')
            self.lu=lu


class OrderedMRI(MRI):
    trial=FunctionType(MRI.trial.__code__,dict(MRI.trial.__globals__,Radau=OrderedRadau),
                       name='trial',argdefs=MRI.trial.__defaults__)
