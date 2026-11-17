"""Compare isolated SUNDIALS CVODE/KLU runtimes on two registered Kecoli states."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import ctypes as c
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare
from research.m4_audit_backend_screen import score, solve

OUT = ROOT / "results/m4-cvode-runtime-screen-v1"
RUNTIME_DIRS = {
    "6.4.1": ROOT / ".runtime/sundials-6.4.1/lib",
    "7.9.0": Path("/opt/homebrew/opt/sundials/lib"),
}
LIBRARIES = ("sundials_cvode", "sundials_nvecserial",
             "sundials_sunmatrixsparse", "sundials_sunlinsolklu")
MPI_ROOT = Path("/opt/homebrew/opt/open-mpi")
SUNDIALS_79_ROOT = Path("/opt/homebrew/opt/sundials")
SDK_ROOT = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk")
MPI_SHIM_SOURCE = ROOT / "research/m4_cvode_runtime_mpi.c"
MPI_SHIM = ROOT / ".runtime/m4_cvode_runtime_screen_mpi.dylib"
H = 1e-6
RTOL = 1e-11
REPEATS = tuple(range(7))
TARGETS = (1e-3, 1e-5)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def worker(version):
    libdir = RUNTIME_DIRS[version]
    from metabolism import cvode

    paths = {name: libdir / f"lib{name}.dylib" for name in LIBRARIES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {version} libraries: {missing}")
    cvode.find_library = lambda name: str(paths[name])
    shim = None
    mpi_started = False
    mpi_init_seconds = 0.0
    if version == "7.9.0":
        shim = c.CDLL(os.environ["M4_CVODE_MPI_SHIM"])
        shim.m4_cvode_mpi_initialize.restype = c.c_int
        shim.m4_cvode_mpi_context_create.restype = c.c_int
        shim.m4_cvode_mpi_context_create.argtypes = [c.c_void_p]
        shim.m4_cvode_mpi_finalize.restype = c.c_int
        mpi_started_at = time.perf_counter()
        if shim.m4_cvode_mpi_initialize() != 0:
            raise RuntimeError("Open MPI initialization failed")
        mpi_init_seconds = time.perf_counter() - mpi_started_at
        mpi_started = True
        original_cdll = c.CDLL

        class CvodeLibraryProxy:
            def __init__(self, library):
                self.library = library

            def __getattr__(self, name):
                if name == "SUNContext_Create":
                    def create_context(_comm, context):
                        return shim.m4_cvode_mpi_context_create(context)
                    return create_context
                return getattr(self.library, name)

        def load_library(path, *args, **kwargs):
            library = original_cdll(path, *args, **kwargs)
            if str(path) == str(paths["sundials_cvode"]):
                return CvodeLibraryProxy(library)
            return library

        cvode.c.CDLL = load_library

    try:
        prep_started = time.perf_counter()
        case = prepare("kecoli")
        with np.load(case["reference_path"]) as ref:
            times, concentrations = ref["times"], ref["concentrations"]
            state_indices = sorted(set((0, int(np.searchsorted(times, 1e-4)))))
            states = {int(i): (float(times[i]), concentrations[:, i].copy())
                      for i in state_indices}
            reference_hash = digest(case["reference_path"])
        query = H * np.array((0.25, 0.5, 0.75, 1.0))
        rhs = lambda t, y: case["system"].S @ case["model"].flux(t, y)
        jac = lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y)
        truths = {}
        for index, (_, y0) in states.items():
            atol = np.maximum(case["scale"] * 1e-15, 1e-18)
            result = solve_ivp(rhs, (0.0, H), y0, method="Radau", t_eval=query,
                               rtol=1e-13, atol=atol, jac=jac)
            if not result.success:
                raise RuntimeError(f"Radau truth failed at state {index}: {result.message}")
            truths[index] = result.y
        prep_seconds = time.perf_counter() - prep_started
        print(json.dumps(dict(kind="ready", version=version,
                              reference_path=str(case["reference_path"].relative_to(ROOT)),
                              reference_sha256=reference_hash,
                              state_indices=state_indices,
                              state_times={str(i): t for i, (t, _) in states.items()},
                              initial_state_sha256={
                                  str(i): sha256_bytes(np.ascontiguousarray(y).tobytes())
                                  for i, (_, y) in states.items()},
                              mpi_init_seconds=mpi_init_seconds,
                              model_and_truth_prep_seconds=prep_seconds),
                         allow_nan=False), flush=True)

        for line in sys.stdin:
            request = json.loads(line)
            index = int(request["state_index"])
            repeat = int(request["repeat"])
            state_time, y0 = states[index]
            started = time.perf_counter()
            values, counts, seconds = solve(case, y0, query, "CVODE-KLU", RTOL)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite output at state {index}, repeat {repeat}")
            metrics = {str(target): score(case, values, truths[index], y0, target)
                       for target in TARGETS}
            output = dict(version=version, state_index=index, state_time=state_time,
                          repeat=repeat, warmup=repeat < 0, seconds=seconds,
                          wall_seconds=time.perf_counter() - started, counts=counts,
                          metrics=metrics, values=values.tolist(),
                          value_sha256=sha256_bytes(np.ascontiguousarray(values).tobytes()),
                          truth_sha256=sha256_bytes(np.ascontiguousarray(truths[index]).tobytes()))
            if repeat < 0:
                output["truth_values"] = truths[index].tolist()
            print(json.dumps(output, allow_nan=False), flush=True)
    finally:
        if mpi_started and shim.m4_cvode_mpi_finalize() != 0:
            raise RuntimeError("Open MPI finalization failed")


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    for version, path in RUNTIME_DIRS.items():
        if not path.is_dir():
            raise FileNotFoundError(f"SUNDIALS {version} runtime not found: {path}")
    if MPI_SHIM.exists():
        raise FileExistsError(f"Preserve existing MPI shim: {MPI_SHIM}")
    for version, libdir in RUNTIME_DIRS.items():
        for name in LIBRARIES:
            library = libdir / f"lib{name}.dylib"
            if not library.is_file():
                raise FileNotFoundError(f"Missing SUNDIALS {version} library: {library}")
    if not MPI_ROOT.is_dir() or not SUNDIALS_79_ROOT.is_dir() or not SDK_ROOT.is_dir():
        raise FileNotFoundError("Required Homebrew Open MPI or SUNDIALS root is unavailable")
    subprocess.run([
        "cc", "-std=c11", "-dynamiclib", "-fPIC",
        f"-I{MPI_ROOT / 'include'}",
        f"-I{SUNDIALS_79_ROOT / 'include'}",
        str(MPI_SHIM_SOURCE),
        f"-L{SUNDIALS_79_ROOT / 'lib'}", "-lsundials_core",
        f"-L{MPI_ROOT / 'lib'}", "-lmpi",
        f"-Wl,-rpath,{SUNDIALS_79_ROOT / 'lib'}",
        f"-Wl,-rpath,{MPI_ROOT / 'lib'}",
        "-o", str(MPI_SHIM),
    ], check=True, env={**os.environ, "SDKROOT": str(SDK_ROOT)})
    OUT.mkdir(parents=True)

    source_paths = (
        "research/m4_cvode_runtime_screen.py",
        "research/m4_audit_backend_screen.py",
        "research/m4_cvode_runtime_mpi.c",
        "metabolism/cvode.py",
        "metabolism/kecoli.py",
        "metabolism/m3_continuation.py",
    )
    runtime_hashes = {version: {name: digest(RUNTIME_DIRS[version] / f"lib{name}.dylib")
                                for name in LIBRARIES}
                      for version in RUNTIME_DIRS}
    registration = dict(
        date="2026-11-18", model="kecoli457", state_indices=[0, 61],
        state_times=None, interval=H,
        query_fractions=[0.25, 0.5, 0.75, 1.0], method="CVODE-BDF/KLU",
        rtol=RTOL, repeats=list(REPEATS), warmups=1, targets=list(TARGETS),
        backends={"6.4.1": str(RUNTIME_DIRS["6.4.1"]),
                  "7.9.0": str(RUNTIME_DIRS["7.9.0"])},
        reference=dict(path="registered by worker ready handshake", sha256=None),
        runtime_library_sha256=runtime_hashes,
        mpi_shim=dict(source=str(MPI_SHIM_SOURCE.relative_to(ROOT)),
                      source_sha256=digest(MPI_SHIM_SOURCE),
                      binary_sha256=digest(MPI_SHIM),
                      init_finalize_outside_solve_timing=True, mpi_root=str(MPI_ROOT)),
        comparison="Same Python RHS/Jacobian and ctypes binding; each version runs in its own long-lived subprocess with an explicit runtime library map. Each solve creates a fresh CVODE/KLU audit instance. Requests are interleaved by state and repeat.",
        truth=dict(method="SciPy Radau", rtol=1e-13,
                   atol="max(concentration_scale*1e-15, 1e-18)"),
        gates=dict(complete_original_nonlinear_rhs=True, concentration_flux_and_net_flux=True,
                   external_error_budget=0.05, nonnegative_sampled_outputs=True,
                   pools=1e-10, no_clipping=True),
        limitation="Two k-ecoli457 states and one 1e-6 interval only. Backend timing is not a candidate trajectory, a full-window audit cost, or M4 qualification.",
        source_sha256={p: digest(ROOT / p) for p in source_paths},
        environment=dict(platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__))
    save(OUT / "registration.json", registration)

    env_base = os.environ.copy()
    env_base.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    workers = {}
    worker_started = {}
    try:
        for version, libdir in RUNTIME_DIRS.items():
            env = env_base.copy()
            env["DYLD_LIBRARY_PATH"] = str(libdir)
            if version == "7.9.0":
                env["M4_CVODE_MPI_SHIM"] = str(MPI_SHIM)
            worker_started[version] = time.perf_counter()
            workers[version] = subprocess.Popen(
                [sys.executable, "-m", "research.m4_cvode_runtime_screen",
                 "--worker", version],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1)
        ready = {}
        for version, process in workers.items():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"SUNDIALS {version} worker failed before ready: {process.poll()}")
            message = json.loads(line)
            if message.get("kind") != "ready" or message.get("version") != version:
                raise RuntimeError(f"Invalid SUNDIALS {version} worker handshake: {message}")
            message["parent_startup_to_ready_seconds"] = (
                time.perf_counter() - worker_started[version])
            ready[version] = message
        ready_values = list(ready.values())
        if len({item["reference_sha256"] for item in ready_values}) != 1:
            raise RuntimeError("Workers loaded different registered references")
        state_times = ready_values[0]["state_times"]
        if any(item["state_times"] != state_times for item in ready_values):
            raise RuntimeError("Workers selected different reference states")
        state_indices = ready_values[0]["state_indices"]
        if any(item["state_indices"] != state_indices for item in ready_values):
            raise RuntimeError("Workers selected different reference indices")
        registration["state_indices"] = state_indices
        registration["state_times"] = [state_times[str(i)] for i in state_indices]
        if len({item["reference_path"] for item in ready_values}) != 1:
            raise RuntimeError("Workers loaded references from different paths")
        if len({json.dumps(item["initial_state_sha256"], sort_keys=True)
                for item in ready_values}) != 1:
            raise RuntimeError("Workers loaded different initial state vectors")
        registration["reference"] = dict(path=ready_values[0]["reference_path"],
                                             sha256=ready_values[0]["reference_sha256"])
        registration["initial_state_sha256"] = ready_values[0]["initial_state_sha256"]
        registration["worker_startup"] = ready
        registration["build"] = dict(compiler="Apple clang via cc", sdk_root=str(SDK_ROOT))
        save(OUT / "registration.json", registration)

        rows = []
        output_values = {}

        def run_one(version, index, repeat):
            process = workers[version]
            process.stdin.write(json.dumps(dict(state_index=index, repeat=repeat)) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"SUNDIALS {version} worker exited: {process.poll()}")
            row = json.loads(line)
            values = np.asarray(row.pop("values"), dtype=np.float64)
            truth_values = row.pop("truth_values", None)
            np.savez_compressed(OUT / f"state-{index}-repeat-{repeat}-sundials-{version}.npz",
                                values=values)
            output_values[(version, index, repeat)] = values
            if truth_values is not None:
                truth_values = np.asarray(truth_values, dtype=np.float64)
                truth_path = OUT / f"truth-{index}.npz"
                if truth_path.exists():
                    with np.load(truth_path) as stored:
                        if not np.array_equal(stored["values"], truth_values):
                            raise RuntimeError(f"Backend truth differs at state {index}")
                else:
                    np.savez_compressed(truth_path, values=truth_values)
            row["artifact"] = f"state-{index}-repeat-{repeat}-sundials-{version}.npz"
            rows.append(row)
            save(OUT / "runs.json", dict(rows=rows))
            return row

        indices = tuple(int(i) for i in registration["state_indices"])
        for index in indices:
            for version in RUNTIME_DIRS:
                run_one(version, index, -1)
        for repeat in REPEATS:
            for index in indices:
                order = tuple(RUNTIME_DIRS) if repeat % 2 == 0 else tuple(reversed(RUNTIME_DIRS))
                for version in order:
                    run_one(version, index, repeat)

        summaries = []
        for index in indices:
            group = {version: [r for r in rows if r["version"] == version
                               and r["state_index"] == index and not r["warmup"]]
                     for version in RUNTIME_DIRS}
            medians = {version: float(np.median([r["seconds"] for r in samples]))
                       for version, samples in group.items()}
            max_errors = {}
            for version, samples in group.items():
                max_errors[version] = {
                    metric: max(r["metrics"][str(target)][metric]
                                for r in samples for target in TARGETS)
                    for metric in ("concentration", "flux", "net_flux", "pool_error")
                }
            summaries.append(dict(
                state_index=index,
                measured_repeats={version: len(samples) for version, samples in group.items()},
                all_accuracy_repeats_pass={
                    version: len(samples) == len(REPEATS) and all(
                        all(result["passes"] for result in row["metrics"].values())
                        for row in samples) for version, samples in group.items()},
                median_seconds=medians,
                speedup_7_9_over_6_4=medians["6.4.1"] / medians["7.9.0"],
                max_errors=max_errors,
                max_cross_backend_abs_difference=max(
                    float(np.max(np.abs(output_values[("6.4.1", index, repeat)] -
                                        output_values[("7.9.0", index, repeat)])))
                    for repeat in REPEATS)))

        save(OUT / "summary.json", dict(
            results=summaries, registration="registration.json",
            decision="Runtime-only evidence. A faster backend may justify a separate candidate-plus-audit budget screen, but cannot qualify a candidate or M4."))
        manifest = {p: digest(ROOT / p) for p in source_paths}
        manifest.update({f"{OUT.relative_to(ROOT)}/{p.name}": digest(p)
                         for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"})
        save(OUT / "manifest.json", manifest)
    finally:
        for process in workers.values():
            if process.poll() is None:
                process.stdin.close()
                process.wait(timeout=30)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
