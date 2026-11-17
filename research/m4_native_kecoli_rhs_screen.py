"""Bound native C acceleration for the k-ecoli457 kinetic RHS.

This is a compiled-kernel micro-screen only. It compares the generated C
mass-action evaluator with the registered Python evaluator on archived valid
states. It does not change a solver, reaction partition, audit policy, or M4
admission state.
"""

import ctypes
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare

OUT = ROOT / "results/m4-native-kecoli-rhs-screen-v3"
REPEATS = 5
CALLS = 200


def c_array(values, ctype="double"):
    return "{" + ",".join(
        (f"{float(x):.17g}" if ctype == "double" else str(int(x))) for x in values
    ) + "}"


def source_for(model):
    n_state, n_rxn = model.S.shape
    indices = model.indices.T.ravel(order="C")
    orders = model.orders.T.ravel(order="C")
    stoich = model.S.tocoo()
    row = stoich.row.astype(int)
    col = stoich.col.astype(int)
    data = stoich.data.astype(float)
    return f"""#include <math.h>
#include <stddef.h>

static const int N = {n_state};
static const int R = {n_rxn};
static const int nnz = {len(data)};
static const int ri[{len(row)}] = {c_array(row, 'int')};
static const int rj[{len(col)}] = {c_array(col, 'int')};
static const double sv[{len(data)}] = {c_array(data)};
static const int idx[2*{n_rxn}] = {c_array(indices, 'int')};
static const double ord[2*{n_rxn}] = {c_array(orders)};
static const double kval[{n_rxn}] = {c_array(model.k)};

void kecoli_flux(const double *y, double *v) {{
    for (int j=0; j<R; ++j) {{
        double value = kval[j];
        const double x0 = y[idx[2*j]];
        const double o0 = ord[2*j];
        value *= (o0 == 0.0) ? 1.0 : pow(x0, o0);
        const double o1 = ord[2*j+1];
        if (o1 != 0.0) value *= pow(y[idx[2*j+1]], o1);
        v[j] = value;
    }}
}}

void kecoli_rhs(const double *y, double *dy) {{
    double v[{n_rxn}];
    kecoli_flux(y, v);
    for (int i=0; i<N; ++i) dy[i] = 0.0;
    for (int k=0; k<nnz; ++k) dy[ri[k]] += sv[k] * v[rj[k]];
}}
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    case = prepare("kecoli")
    model = case["model"]
    OUT.mkdir(parents=True)
    c_path = OUT / "kecoli_rhs.c"
    so_path = OUT / "libkecoli_rhs.dylib"
    c_path.write_text(source_for(model), encoding="utf-8")
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")
    command = ["clang", "-O3", "-fPIC", "-dynamiclib", "-isysroot", str(sdk),
               "-o", str(so_path), str(c_path)]
    compile_started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    compile_seconds = time.perf_counter() - compile_started
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    lib = ctypes.CDLL(str(so_path))
    flux = lib.kecoli_flux
    rhs = lib.kecoli_rhs
    ptr = ctypes.POINTER(ctypes.c_double)
    flux.argtypes = [ptr, ptr]
    rhs.argtypes = [ptr, ptr]
    states = [model.y0.copy()]
    with np.load(case["reference_path"]) as archived:
        ref = archived["concentrations"]
        for index in (0, 1, 61, len(archived["times"]) - 1):
            states.append(ref[:, index].copy())
    rows = []
    for index, state in enumerate(states):
        py_flux = model.flux(0.0, state)
        py_rhs = model.S @ py_flux
        c_flux = np.empty_like(py_flux)
        c_rhs = np.empty_like(py_rhs)
        rhs(state.ctypes.data_as(ptr), c_rhs.ctypes.data_as(ptr))
        flux(state.ctypes.data_as(ptr), c_flux.ctypes.data_as(ptr))
        np.testing.assert_allclose(c_flux, py_flux, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(c_rhs, py_rhs, rtol=2e-14, atol=2e-14)
        py_times, c_times = [], []
        for _ in range(REPEATS):
            tick = time.perf_counter()
            for _ in range(CALLS):
                model.flux(0.0, state)
            py_times.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            for _ in range(CALLS):
                rhs(state.ctypes.data_as(ptr), c_rhs.ctypes.data_as(ptr))
            c_times.append(time.perf_counter() - tick)
        rows.append(dict(state_index=index, python_median=float(np.median(py_times)),
                         native_median=float(np.median(c_times)),
                         speedup=float(np.median(py_times) / np.median(c_times)),
                         max_flux_abs=float(np.max(abs(c_flux - py_flux))),
                         max_rhs_abs=float(np.max(abs(c_rhs - py_rhs)))))
    result = dict(scope="Generated C RHS bound only; no solver or M4 qualification",
                  model="k-ecoli457", states=len(states), calls=CALLS, repeats=REPEATS,
                  compile_seconds=compile_seconds, rows=rows, M4_admission=False,
                  environment=dict(platform=platform.platform(), python=platform.python_version(),
                                   compiler=completed.args, clang_stderr=completed.stderr),
                  source_sha256=digest(c_path))
    save(OUT / "summary.json", result)
    save(OUT / "registration.json", dict(date="2026-11-15", source="research/m4_native_kecoli_rhs_screen.py",
        command=command, compiler="clang -O3 -fPIC -dynamiclib -isysroot MacOSX26.5.sdk", correctness="Python flux and S@y parity at five archived valid states",
        stopping="No solver integration or M4 route unless native kernel speed leaves a plausible positive audited budget"))
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
