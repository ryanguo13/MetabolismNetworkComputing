"""Micro-screen persistent CVODE/KLU structures with per-audit ReInit.

The session resets CVODE's nonlinear/history state for every candidate origin,
but reuses only immutable native handles, sparse structure, and KLU symbolic
data. Each run still integrates the complete nonlinear ODE independently from
the supplied state and checks every requested output. This is development
evidence only and never modifies M4 admission state.
"""

import ctypes as c
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from ctypes.util import find_library

import numpy as np
from scipy.sparse import eye

from metabolism.cvode import solve as cold_solve
from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare

OUT = ROOT / "results/m4-cvode-session-screen-v4"
H = 1e-6
STATE_INDEX = 61
REPEATS = (0, 1, 2)
RTOL = 1e-9


class CVODESession:
    """Reusable native CVODE/KLU shell with independent `CVodeReInit` solves."""

    def __init__(self, model, y0, atol):
        ptr, real, index = c.c_void_p, c.c_double, c.c_int64
        realptr, indexptr = c.POINTER(real), c.POINTER(index)
        self.ptr, self.real = ptr, real
        self.n = len(y0)
        self.model = model
        self.y = np.asarray(y0, dtype=np.float64).copy()
        self.atol = np.broadcast_to(atol, self.y.shape).astype(np.float64).copy()
        self.pattern = (model.sparsity + eye(self.n, format="csc", dtype=bool)).astype(bool).tocsc()
        self.pattern.sort_indices()
        self.pattern_columns = np.repeat(np.arange(self.n), np.diff(self.pattern.indptr))
        self.capacity = int(self.pattern.nnz)
        self.raw_calls = 0
        self.rhs_seconds = 0.
        self.jac_seconds = 0.
        self.raw_min = np.inf
        self.callback_errors = []

        def bind(library, name, result, args):
            fn = getattr(library, name)
            fn.restype, fn.argtypes = result, args
            return fn

        self.cv = c.CDLL(find_library("sundials_cvode"))
        self.nv = c.CDLL(find_library("sundials_nvecserial"))
        self.sm = c.CDLL(find_library("sundials_sunmatrixsparse"))
        self.klu = c.CDLL(find_library("sundials_sunlinsolklu"))
        self.context_create = bind(self.cv, "SUNContext_Create", c.c_int, [ptr, c.POINTER(ptr)])
        self.context_free = bind(self.cv, "SUNContext_Free", c.c_int, [c.POINTER(ptr)])
        self.make = bind(self.nv, "N_VMake_Serial", ptr, [index, realptr, ptr])
        self.vector_data = bind(self.nv, "N_VGetArrayPointer_Serial", realptr, [ptr])
        self.vector_free = bind(self.nv, "N_VDestroy_Serial", None, [ptr])
        self.matrix_create = bind(self.sm, "SUNSparseMatrix", ptr, [index, index, index, c.c_int, ptr])
        self.matrix_free = bind(self.sm, "SUNMatDestroy_Sparse", None, [ptr])
        self.matrix_data = bind(self.sm, "SUNSparseMatrix_Data", realptr, [ptr])
        self.matrix_indices = bind(self.sm, "SUNSparseMatrix_IndexValues", indexptr, [ptr])
        self.matrix_indptr = bind(self.sm, "SUNSparseMatrix_IndexPointers", indexptr, [ptr])
        self.linear_create = bind(self.klu, "SUNLinSol_KLU", ptr, [ptr, ptr, ptr])
        self.linear_free = bind(self.klu, "SUNLinSolFree_KLU", c.c_int, [ptr])
        self.create = bind(self.cv, "CVodeCreate", ptr, [c.c_int, ptr])
        self.destroy = bind(self.cv, "CVodeFree", None, [c.POINTER(ptr)])
        self.RHS = c.CFUNCTYPE(c.c_int, real, ptr, ptr, ptr)
        self.JAC = c.CFUNCTYPE(c.c_int, real, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
        self.initialize = bind(self.cv, "CVodeInit", c.c_int, [ptr, self.RHS, real, ptr])
        self.reinit = bind(self.cv, "CVodeReInit", c.c_int, [ptr, real, ptr])
        self.tolerances = bind(self.cv, "CVodeSVtolerances", c.c_int, [ptr, real, ptr])
        self.set_linear = bind(self.cv, "CVodeSetLinearSolver", c.c_int, [ptr, ptr, ptr])
        self.set_jac = bind(self.cv, "CVodeSetJacFn", c.c_int, [ptr, self.JAC])
        self.max_steps = bind(self.cv, "CVodeSetMaxNumSteps", c.c_int, [ptr, c.c_long])
        self.integrate = bind(self.cv, "CVode", c.c_int, [ptr, real, ptr, c.POINTER(real), c.c_int])
        self.getters = {key: bind(self.cv, name, c.c_int, [ptr, c.POINTER(c.c_long)]) for key, name in {
            "accepted_steps": "CVodeGetNumSteps",
            "nfev": "CVodeGetNumRhsEvals",
            "njev": "CVodeGetNumJacEvals",
            "linear_solver_setups": "CVodeGetNumLinSolvSetups",
            "error_test_failures": "CVodeGetNumErrTestFails",
            "nonlinear_iterations": "CVodeGetNumNonlinSolvIters",
            "nonlinear_convergence_failures": "CVodeGetNumNonlinSolvConvFails",
        }.items()}

        def check(code):
            if code < 0:
                raise RuntimeError(f"CVODE error {code}; callbacks={self.callback_errors}")

        @self.RHS
        def rhs(t, yy, dy, user):
            tick = time.perf_counter()
            try:
                state = np.ctypeslib.as_array(self.vector_data(yy), shape=(self.n,))
                self.raw_min = min(self.raw_min, float(state.min()))
                self.raw_calls += 1
                np.ctypeslib.as_array(self.vector_data(dy), shape=(self.n,))[:] = self.model.rhs(t, state)
                return 0
            except Exception as exc:
                self.callback_errors.append(repr(exc))
                return -1
            finally:
                self.rhs_seconds += time.perf_counter() - tick

        @self.JAC
        def jac(t, yy, fy, matrix, user, tmp1, tmp2, tmp3):
            tick = time.perf_counter()
            try:
                state = np.ctypeslib.as_array(self.vector_data(yy), shape=(self.n,))
                values = np.asarray(self.model.jacobian(t, state).tocsc()[self.pattern.indices, self.pattern_columns]).ravel()
                np.ctypeslib.as_array(self.matrix_data(matrix), shape=(self.capacity,))[:] = values
                np.ctypeslib.as_array(self.matrix_indices(matrix), shape=(self.capacity,))[:] = self.pattern.indices
                np.ctypeslib.as_array(self.matrix_indptr(matrix), shape=(self.n + 1,))[:] = self.pattern.indptr
                return 0
            except Exception as exc:
                self.callback_errors.append(repr(exc))
                return -1
            finally:
                self.jac_seconds += time.perf_counter() - tick

        self.rhs_callback, self.jac_callback = rhs, jac
        self.ctx = ptr()
        self.memory = ptr()
        self.vector = ptr()
        self.atolv = ptr()
        self.matrix = ptr()
        self.linear = ptr()
        try:
            check(self.context_create(None, c.byref(self.ctx)))
            self.vector = self.make(self.n, self.y.ctypes.data_as(realptr), self.ctx)
            self.atolv = self.make(self.n, self.atol.ctypes.data_as(realptr), self.ctx)
            self.matrix = self.matrix_create(self.n, self.n, self.capacity, 0, self.ctx)
            self.linear = self.linear_create(self.vector, self.matrix, self.ctx)
            self.memory = ptr(self.create(2, self.ctx))
            if not all((self.vector, self.atolv, self.matrix, self.linear, self.memory)):
                raise MemoryError("SUNDIALS allocation failed")
            check(self.initialize(self.memory, self.rhs_callback, 0., self.vector))
            check(self.tolerances(self.memory, RTOL, self.atolv))
            check(self.set_linear(self.memory, self.linear, self.matrix))
            check(self.set_jac(self.memory, self.jac_callback))
            check(self.max_steps(self.memory, 1000000))
        except Exception:
            self.close()
            raise

    def solve(self, y0, times, rtol, atol):
        times = np.asarray(times, dtype=float)
        self.y[:] = np.asarray(y0, dtype=float)
        self.atol[:] = np.broadcast_to(atol, self.y.shape)
        self.raw_calls = 0
        self.rhs_seconds = self.jac_seconds = 0.
        self.raw_min = float(self.y.min())
        self.callback_errors.clear()
        if self.reinit(self.memory, 0., self.vector) < 0:
            raise RuntimeError(f"CVodeReInit failed: {self.callback_errors}")
        if self.tolerances(self.memory, float(rtol), self.atolv) < 0:
            raise RuntimeError("CVodeSVtolerances failed")
        result = np.empty((self.n, len(times)))
        result[:, 0] = self.y
        returned = self.real()
        tick = time.perf_counter()
        for i in range(1, len(times)):
            code = self.integrate(self.memory, float(times[i]), self.vector, c.byref(returned), 1)
            if code < 0:
                raise RuntimeError(f"CVODE error {code}; callbacks={self.callback_errors}")
            result[:, i] = self.y
        stats = {}
        for key, getter in self.getters.items():
            value = c.c_long()
            if getter(self.memory, c.byref(value)) < 0:
                raise RuntimeError(f"CVODE getter failed: {key}")
            stats[key] = value.value
        stats.update(rhs_seconds=self.rhs_seconds, jacobian_seconds=self.jac_seconds,
                     min_rhs_argument=self.raw_min, observed_rhs_calls=self.raw_calls,
                     total_seconds=time.perf_counter() - tick, persistent=True)
        return result, stats

    def close(self):
        if getattr(self, "memory", None):
            self.destroy(c.byref(self.memory))
            self.memory = None
        if getattr(self, "linear", None):
            self.linear_free(self.linear); self.linear = None
        if getattr(self, "matrix", None):
            self.matrix_free(self.matrix); self.matrix = None
        for name in ("atolv", "vector"):
            value = getattr(self, name, None)
            if value:
                self.vector_free(value); setattr(self, name, None)
        if getattr(self, "ctx", None):
            self.context_free(c.byref(self.ctx)); self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(values, state, case):
    if not np.isfinite(values).all() or np.any(values < -1e-12 * case["scale"][:, None]):
        raise AssertionError("invalid audited output")
    if np.max(abs(case["options"]["pools"] @ (values - state[:, None]))) > 1e-10:
        raise AssertionError("pool drift")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    case = prepare("kecoli")
    state = case["reference"][:, STATE_INDEX].copy()
    times = H * np.array([0., .25, .5, .75, 1.])
    model = SimpleNamespace(y0=state, rhs=case["model"].rhs,
                            jacobian=case["model"].jacobian, sparsity=case["model"].sparsity)
    registration = dict(date="2026-11-16", model="k-ecoli457", state_index=STATE_INDEX,
                        horizon=H, repeats=list(REPEATS), rtol=RTOL,
                        semantics=dict(complete_nonlinear_rhs=True, complete_analytic_jacobian=True,
                                       independent_origin_reinit=True, requested_outputs_checked=True,
                                       no_clipping=True, no_projection=True),
                        source_sha256={"research/cvode_session_probe.py": digest(Path(__file__)),
                                       "metabolism/cvode.py": digest(ROOT / "metabolism/cvode.py"),
                                       "metabolism/kecoli.py": digest(ROOT / "metabolism/kecoli.py")},
                        environment=dict(python=platform.python_version(), numpy=np.__version__))
    OUT.mkdir(parents=True)
    save(OUT / "registration.json", registration)
    rows = []
    with CVODESession(model, state, .01 * RTOL * case["scale"]) as session:
        sequence = [("cold", -1), ("session", -1)]
        for rep in REPEATS:
            sequence.extend([("cold", rep), ("session", rep)])
        for kind, repeat in sequence:
            started = time.perf_counter()
            if kind == "cold":
                values, stats = cold_solve(model, times, RTOL, .01 * RTOL * case["scale"])
            else:
                values, stats = session.solve(state, times, RTOL, .01 * RTOL * case["scale"])
            elapsed = time.perf_counter() - started
            row = dict(kind=kind, repeat=repeat, seconds=elapsed, stats=stats,
                       output_sha256=checked(values, state, case),
                       values_shape=list(values.shape))
            rows.append(row)
            save(OUT / "runs.json", dict(rows=rows))
    formal = []
    for rep in REPEATS:
        cold = next(r for r in rows if r["kind"] == "cold" and r["repeat"] == rep)
        session = next(r for r in rows if r["kind"] == "session" and r["repeat"] == rep)
        formal.append(dict(repeat=rep, cold_seconds=cold["seconds"], session_seconds=session["seconds"],
                           speedup=cold["seconds"] / session["seconds"],
                           identical_output=cold["output_sha256"] == session["output_sha256"]))
    summary = dict(model="k-ecoli457", state_index=STATE_INDEX, horizon=H,
                   repeats=len(REPEATS), formal=formal,
                   median_cold=float(np.median([r["cold_seconds"] for r in formal])),
                   median_session=float(np.median([r["session_seconds"] for r in formal])),
                   median_speedup=float(np.median([r["speedup"] for r in formal])),
                   all_outputs_identical=all(r["identical_output"] for r in formal),
                   M4_admission=False, evidence_integrity="pass",
                   interpretation="Persistent CVODE/KLU structures are reused only with CVodeReInit; each row still starts from the candidate state and integrates the complete nonlinear ODE.")
    save(OUT / "summary.json", summary)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
