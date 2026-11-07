"""Reusable research-only SUNDIALS 6 CVODE/KLU resources.

ReInit deliberately resets BDF history on every forcing change. The fixed CSC
pattern and allocations are reused; KLU numeric/symbolic reuse is not promised.
The historical metabolism.cvode binding and its evidence remain unchanged.
"""
import ctypes as c
from ctypes.util import find_library
import time

import numpy as np
from scipy.sparse import eye


class NativeSession:
    def __init__(self, pattern):
        tick = time.perf_counter()
        self.ctx, self.memory = c.c_void_p(), c.c_void_p()
        self.vector = self.atolv = self.matrix = self.linear = None
        self.n = pattern.shape[0]
        self.y = np.zeros(self.n)
        self.av = np.ones(self.n)
        self.pattern = (pattern+eye(self.n, format='csc', dtype=bool)).astype(bool).tocsc()
        self.pattern.sort_indices()
        self.columns = np.repeat(np.arange(self.n), np.diff(self.pattern.indptr))
        self.capacity = self.pattern.nnz
        self.initialized = False
        self.model = None
        self.errors = []
        self.counts = {}
        ptr, real, index = c.c_void_p, c.c_double, c.c_int64
        rp, ip = c.POINTER(real), c.POINTER(index)
        self.libs = {key:c.CDLL(find_library('sundials_'+name)) for key,name in
                     [('cv','cvode'),('nv','nvecserial'),('sm','sunmatrixsparse'),('klu','sunlinsolklu')]}
        def bind(lib, name, result, args):
            fn = getattr(self.libs[lib], name)
            fn.restype, fn.argtypes = result, args
            return fn
        self.context_free = bind('cv','SUNContext_Free',c.c_int,[c.POINTER(ptr)])
        self.vector_free = bind('nv','N_VDestroy_Serial',None,[ptr])
        self.matrix_free = bind('sm','SUNMatDestroy_Sparse',None,[ptr])
        self.linear_free = bind('klu','SUNLinSolFree_KLU',c.c_int,[ptr])
        self.destroy = bind('cv','CVodeFree',None,[c.POINTER(ptr)])
        data = bind('nv','N_VGetArrayPointer_Serial',rp,[ptr])
        md = bind('sm','SUNSparseMatrix_Data',rp,[ptr])
        mi = bind('sm','SUNSparseMatrix_IndexValues',ip,[ptr])
        mp = bind('sm','SUNSparseMatrix_IndexPointers',ip,[ptr])
        RHS = c.CFUNCTYPE(c.c_int,real,ptr,ptr,ptr)
        JAC = c.CFUNCTYPE(c.c_int,real,ptr,ptr,ptr,ptr,ptr,ptr,ptr)

        @RHS
        def rhs(t, yy, dy, user):
            stamp = time.perf_counter()
            try:
                state = np.ctypeslib.as_array(data(yy), shape=(self.n,))
                np.ctypeslib.as_array(data(dy), shape=(self.n,))[:] = self.model.rhs(t, state)
                return 0
            except Exception as exc:
                self.errors.append(repr(exc))
                return -1
            finally:
                self.counts['rhs_seconds'] += time.perf_counter()-stamp

        @JAC
        def jac(t, yy, fy, mat, user, t1, t2, t3):
            stamp = time.perf_counter()
            try:
                state = np.ctypeslib.as_array(data(yy), shape=(self.n,))
                j = self.model.jacobian(t, state).tocsc()
                np.ctypeslib.as_array(md(mat), shape=(self.capacity,))[:] = np.asarray(j[self.pattern.indices,self.columns]).ravel()
                np.ctypeslib.as_array(mi(mat), shape=(self.capacity,))[:] = self.pattern.indices
                np.ctypeslib.as_array(mp(mat), shape=(self.n+1,))[:] = self.pattern.indptr
                return 0
            except Exception as exc:
                self.errors.append(repr(exc))
                return -1
            finally:
                self.counts['jacobian_seconds'] += time.perf_counter()-stamp

        self.rhs_callback, self.jac_callback = rhs, jac
        self.initialize = bind('cv','CVodeInit',c.c_int,[ptr,RHS,real,ptr])
        self.reinitialize = bind('cv','CVodeReInit',c.c_int,[ptr,real,ptr])
        self.tolerances = bind('cv','CVodeSVtolerances',c.c_int,[ptr,real,ptr])
        self.set_linear = bind('cv','CVodeSetLinearSolver',c.c_int,[ptr,ptr,ptr])
        self.set_jac = bind('cv','CVodeSetJacFn',c.c_int,[ptr,JAC])
        self.max_steps = bind('cv','CVodeSetMaxNumSteps',c.c_int,[ptr,c.c_long])
        self.stop_time = bind('cv','CVodeSetStopTime',c.c_int,[ptr,real])
        self.integrate = bind('cv','CVode',c.c_int,[ptr,real,ptr,c.POINTER(real),c.c_int])
        self.getters = {k:bind('cv',name,c.c_int,[ptr,c.POINTER(c.c_long)]) for k,name in
            {'nfev':'CVodeGetNumRhsEvals','njev':'CVodeGetNumJacEvals',
             'linear_solver_setups':'CVodeGetNumLinSolvSetups','accepted_steps':'CVodeGetNumSteps'}.items()}
        try:
            self.check(bind('cv','SUNContext_Create',c.c_int,[ptr,c.POINTER(ptr)])(None,c.byref(self.ctx)))
            make = bind('nv','N_VMake_Serial',ptr,[index,rp,ptr])
            self.vector = make(self.n,self.y.ctypes.data_as(rp),self.ctx)
            self.atolv = make(self.n,self.av.ctypes.data_as(rp),self.ctx)
            self.matrix = bind('sm','SUNSparseMatrix',ptr,[index,index,index,c.c_int,ptr])(
                self.n,self.n,self.capacity,0,self.ctx)
            self.linear = bind('klu','SUNLinSol_KLU',ptr,[ptr,ptr,ptr])(self.vector,self.matrix,self.ctx)
            self.memory = ptr(bind('cv','CVodeCreate',ptr,[c.c_int,ptr])(2,self.ctx))
            if not all((self.vector,self.atolv,self.matrix,self.linear,self.memory)):
                raise MemoryError('Native resource allocation failed')
        except Exception:
            self.close()
            raise
        self.setup_seconds = time.perf_counter()-tick

    def check(self, code):
        if code < 0:
            raise RuntimeError(f'CVODE error {code}; callbacks={self.errors}')

    def solve(self, model, duration, rtol, atol):
        if not np.isfinite(duration) or duration <= 0 or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError('Invalid duration or tolerance')
        if np.shape(model.y0) != (self.n,) or not np.isfinite(model.y0).all():
            raise ValueError('Invalid initial state')
        self.model = model
        self.y[:] = model.y0
        self.av[:] = atol
        if not np.isfinite(self.av).all() or np.any(self.av <= 0):
            raise ValueError('Invalid absolute tolerance')
        self.errors = []
        self.counts = dict(rhs_seconds=0.,jacobian_seconds=0.)
        tick = time.perf_counter()
        if self.initialized:
            self.check(self.reinitialize(self.memory,0.,self.vector))
        else:
            self.check(self.initialize(self.memory,self.rhs_callback,0.,self.vector))
            self.check(self.set_linear(self.memory,self.linear,self.matrix))
            self.check(self.set_jac(self.memory,self.jac_callback))
            self.check(self.max_steps(self.memory,1000000))
            self.initialized = True
        self.check(self.tolerances(self.memory,rtol,self.atolv))
        self.check(self.stop_time(self.memory,duration))
        self.counts['reinit_seconds'] = time.perf_counter()-tick
        returned = c.c_double()
        while returned.value < duration:
            self.check(self.integrate(self.memory,duration,self.vector,c.byref(returned),2))
            model.accepted(returned.value,self.y)
        for key,fn in self.getters.items():
            value = c.c_long()
            self.check(fn(self.memory,c.byref(value)))
            self.counts[key] = value.value
        return self.y.copy(),self.counts.copy()

    def close(self):
        if self.memory:
            self.destroy(c.byref(self.memory))
        if self.linear:
            self.linear_free(self.linear)
            self.linear = None
        if self.matrix:
            self.matrix_free(self.matrix)
            self.matrix = None
        for name in ('vector','atolv'):
            if getattr(self,name,None):
                self.vector_free(getattr(self,name))
                setattr(self,name,None)
        if self.ctx:
            self.context_free(c.byref(self.ctx))
