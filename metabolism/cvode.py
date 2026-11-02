"""Minimal ctypes binding to Ubuntu SUNDIALS 6 double/int64 CVODE-BDF + KLU.

Uses the same Python RHS and analytic sparse Jacobian as SciPy. No clipping,
projections or positivity constraints. Native allocation is freed on failure.
"""

import ctypes as c
from ctypes.util import find_library
import time

import numpy as np
from scipy.sparse import eye


def solve(model, times, rtol, atol):
    ptr, real, index = c.c_void_p, c.c_double, c.c_int64
    realptr, indexptr = c.POINTER(real), c.POINTER(index)
    cv = c.CDLL(find_library("sundials_cvode"))
    nv = c.CDLL(find_library("sundials_nvecserial"))
    sm = c.CDLL(find_library("sundials_sunmatrixsparse"))
    klu = c.CDLL(find_library("sundials_sunlinsolklu"))

    def bind(library, name, result, args):
        fn = getattr(library, name)
        fn.restype, fn.argtypes = result, args
        return fn

    context_create = bind(cv, "SUNContext_Create", c.c_int, [ptr, c.POINTER(ptr)])
    context_free = bind(cv, "SUNContext_Free", c.c_int, [c.POINTER(ptr)])
    make = bind(nv, "N_VMake_Serial", ptr, [index, realptr, ptr])
    vector_data = bind(nv, "N_VGetArrayPointer_Serial", realptr, [ptr])
    vector_free = bind(nv, "N_VDestroy_Serial", None, [ptr])
    matrix_create = bind(sm, "SUNSparseMatrix", ptr, [index, index, index, c.c_int, ptr])
    matrix_free = bind(sm, "SUNMatDestroy_Sparse", None, [ptr])
    matrix_data = bind(sm, "SUNSparseMatrix_Data", realptr, [ptr])
    matrix_indices = bind(sm, "SUNSparseMatrix_IndexValues", indexptr, [ptr])
    matrix_indptr = bind(sm, "SUNSparseMatrix_IndexPointers", indexptr, [ptr])
    linear_create = bind(klu, "SUNLinSol_KLU", ptr, [ptr, ptr, ptr])
    linear_free = bind(klu, "SUNLinSolFree_KLU", c.c_int, [ptr])
    create = bind(cv, "CVodeCreate", ptr, [c.c_int, ptr])
    destroy = bind(cv, "CVodeFree", None, [c.POINTER(ptr)])
    RHS = c.CFUNCTYPE(c.c_int, real, ptr, ptr, ptr)
    JAC = c.CFUNCTYPE(c.c_int, real, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
    initialize = bind(cv, "CVodeInit", c.c_int, [ptr, RHS, real, ptr])
    tolerances = bind(cv, "CVodeSVtolerances", c.c_int, [ptr, real, ptr])
    set_linear = bind(cv, "CVodeSetLinearSolver", c.c_int, [ptr, ptr, ptr])
    set_jac = bind(cv, "CVodeSetJacFn", c.c_int, [ptr, JAC])
    max_steps = bind(cv, "CVodeSetMaxNumSteps", c.c_int, [ptr, c.c_long])
    integrate = bind(cv, "CVode", c.c_int, [ptr, real, ptr, c.POINTER(real), c.c_int])
    n = len(model.y0)
    pattern = (model.sparsity + eye(n, format="csc", dtype=bool)).astype(bool).tocsc()
    pattern.sort_indices()
    pattern_columns = np.repeat(np.arange(n), np.diff(pattern.indptr))
    y = np.array(model.y0, dtype=np.float64, copy=True)
    av = np.broadcast_to(atol, y.shape).astype(np.float64).copy()
    result = np.empty((n, len(times)))
    result[:, 0] = y
    elapsed_rhs, elapsed_jac = 0.0, 0.0
    raw_min, raw_calls = np.inf, 0
    callback_errors = []

    @RHS
    def rhs(t, yy, dy, user):
        nonlocal elapsed_rhs, raw_min, raw_calls
        start = time.perf_counter()
        try:
            state = np.ctypeslib.as_array(vector_data(yy), shape=(n,))
            raw_min = min(raw_min, float(state.min()))
            raw_calls += 1
            np.ctypeslib.as_array(vector_data(dy), shape=(n,))[:] = model.rhs(t, state)
            return 0
        except Exception as exc:
            callback_errors.append(repr(exc))
            return -1
        finally:
            elapsed_rhs += time.perf_counter() - start

    @JAC
    def jac(t, yy, fy, matrix, user, tmp1, tmp2, tmp3):
        nonlocal elapsed_jac
        start = time.perf_counter()
        try:
            state = np.ctypeslib.as_array(vector_data(yy), shape=(n,))
            j = model.jacobian(t, state).tocsc()
            # KLU reuses symbolic factorization: keep the structural pattern
            # fixed even when zero initial complexes make entries vanish.
            values = np.asarray(j[pattern.indices, pattern_columns]).ravel()
            np.ctypeslib.as_array(matrix_data(matrix), shape=(capacity,))[:] = values
            np.ctypeslib.as_array(matrix_indices(matrix), shape=(capacity,))[:] = pattern.indices
            np.ctypeslib.as_array(matrix_indptr(matrix), shape=(n + 1,))[:] = pattern.indptr
            return 0
        except Exception as exc:
            callback_errors.append(repr(exc))
            return -1
        finally:
            elapsed_jac += time.perf_counter() - start

    def check(code):
        if code < 0:
            raise RuntimeError(f"CVODE error {code}; callbacks={callback_errors}")

    ctx, memory = ptr(), ptr()
    vector = atolv = matrix = linear = None
    capacity = int(pattern.nnz)
    try:
        check(context_create(None, c.byref(ctx)))
        vector = make(n, y.ctypes.data_as(realptr), ctx)
        atolv = make(n, av.ctypes.data_as(realptr), ctx)
        matrix = matrix_create(n, n, capacity, 0, ctx)  # CSC_MAT=0
        linear = linear_create(vector, matrix, ctx)
        memory = ptr(create(2, ctx))  # CV_BDF=2
        if not all((vector, atolv, matrix, linear, memory)):
            raise MemoryError("SUNDIALS allocation failed")
        check(initialize(memory, rhs, 0, vector))
        check(tolerances(memory, rtol, atolv))
        check(set_linear(memory, linear, matrix))
        check(set_jac(memory, jac))
        check(max_steps(memory, 1000000))
        returned_time = real()
        for i in range(1, len(times)):
            check(integrate(memory, float(times[i]), vector, c.byref(returned_time), 1))
            result[:, i] = y
        stats = {}
        for key, name in {"accepted_steps": "CVodeGetNumSteps", "nfev": "CVodeGetNumRhsEvals",
                          "njev": "CVodeGetNumJacEvals", "linear_solver_setups": "CVodeGetNumLinSolvSetups",
                          "error_test_failures": "CVodeGetNumErrTestFails", "nonlinear_iterations": "CVodeGetNumNonlinSolvIters",
                          "nonlinear_convergence_failures": "CVodeGetNumNonlinSolvConvFails"}.items():
            value = c.c_long()
            check(bind(cv, name, c.c_int, [ptr, c.POINTER(c.c_long)])(memory, c.byref(value)))
            stats[key] = value.value
        stats.update({"rhs_seconds": elapsed_rhs, "jacobian_seconds": elapsed_jac,
                      "min_rhs_argument": raw_min, "observed_rhs_calls": raw_calls,
                      "linear_solver": "SUNDIALS 6.4.1 KLU sparse CSC, default COLAMD ordering"})
        return result, stats
    finally:
        if memory:
            destroy(c.byref(memory))
        if linear:
            linear_free(linear)
        if matrix:
            matrix_free(matrix)
        for v in (atolv, vector):
            if v:
                vector_free(v)
        if ctx:
            context_free(c.byref(ctx))
