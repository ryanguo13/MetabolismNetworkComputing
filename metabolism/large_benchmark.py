"""Linux/WSL M1 audit with a shared Python RHS and analytic sparse Jacobian.

python3 -m metabolism.large_benchmark --reference
python3 -m metabolism.large_benchmark --benchmark
"""

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import BDF, Radau
from scipy.io import loadmat

from .cvode import solve as cvode_solve
from .kecoli import KEcoliModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m1-completion"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def weighted_error(value, ref, scale, target=1e-5):
    return float(np.max(abs(value - ref) / (target * (.01 * scale[:, None] + np.maximum(abs(ref), scale[:, None])))))


def integrate(model, times, method, tol, scale, timeout=120):
    if method == "CVODE-KLU":
        return cvode_solve(model, times, tol, .01 * tol * scale)
    start = time.perf_counter()
    counts = {"rhs_seconds": 0.0, "jacobian_seconds": 0.0,
              "lu_factor_seconds": 0.0, "linear_solve_seconds": 0.0, "rhs_calls": 0}
    minimum = np.inf
    def rhs(t, y):
        nonlocal minimum
        before = time.perf_counter()
        if before - start > timeout:
            raise TimeoutError(f"Registered {timeout}s trajectory limit")
        minimum = min(minimum, float(y.min()))
        value = model.rhs(t, y)
        counts['rhs_seconds'] += time.perf_counter() - before
        counts['rhs_calls'] += 1
        return value
    def jac(t, y):
        before = time.perf_counter()
        value = model.jacobian(t, y)
        counts['jacobian_seconds'] += time.perf_counter() - before
        return value
    solver = {"BDF": BDF, "Radau": Radau}[method](rhs, 0, model.y0, float(times[-1]),
                                                   jac=jac, rtol=tol, atol=.01 * tol * scale)
    factor, triangular = solver.lu, solver.solve_lu
    def lu(a):
        before = time.perf_counter()
        value = factor(a)
        counts['lu_factor_seconds'] += time.perf_counter() - before
        return value
    def solve_lu(a, b):
        before = time.perf_counter()
        value = triangular(a, b)
        counts['linear_solve_seconds'] += time.perf_counter() - before
        return value
    solver.lu, solver.solve_lu = lu, solve_lu
    result = np.empty((len(model.y0), len(times)))
    result[:, 0], index, steps = model.y0, 1, 0
    accepted_minimum = float(model.y0.min())
    while solver.status == 'running':
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(message)
        steps += 1
        accepted_minimum = min(accepted_minimum, float(solver.y.min()))
        next_index = np.searchsorted(times, solver.t, side='right')
        if next_index > index:
            result[:, index:next_index] = solver.dense_output()(times[index:next_index])
            index = next_index
    counts.update({"nfev": solver.nfev, "njev": solver.njev, "nlu": solver.nlu,
                   "accepted_steps": steps, "min_rhs_argument": minimum,
                   "min_accepted_state": accepted_minimum, "error_test_failures": None})
    return result, counts


def reference():
    model = KEcoliModel(ROOT / "models/k-ecoli457.zip")
    source = loadmat(OUT / "kecoli-octave.mat")
    inputs = np.load(OUT / 'kecoli-source-inputs.npz')
    times = source['times'].ravel()
    scale = np.maximum(abs(source['concentrations']).max(axis=0), 1e-3)
    # Tightening below this range hits severe cancellation costs in Radau.
    references = {}
    reference_stats = {}
    for method, tol in (("CVODE-KLU", 1e-10), ("CVODE-KLU", 1e-11), ("BDF", 1e-10), ("Radau", 1e-9)):
        start = time.perf_counter()
        y, stats = integrate(model, times, method, tol, scale)
        name = f"{method}-{tol}"
        references[name] = y
        reference_stats[name] = {"seconds": time.perf_counter() - start, "counts": stats}
        print("reference", name, reference_stats[name]['seconds'], flush=True)
    ref = references['CVODE-KLU-1e-11']
    source_y = source['concentrations'].T
    checks = {name: weighted_error(y, ref, scale) for name, y in references.items()}
    checks['original_Octave_ode15s'] = weighted_error(source_y, ref, scale)
    flux_ref = np.column_stack([model.flux(t, y) for t, y in zip(times, ref.T)])
    flux_scale = np.maximum(np.max(abs(flux_ref), axis=1), 1e-6)
    flux_checks = {name: weighted_error(np.column_stack([model.flux(t, y) for t, y in zip(times, states.T)]), flux_ref, flux_scale)
                   for name, states in references.items()}
    flux_checks['original_Octave_ode15s'] = weighted_error(np.column_stack([model.flux(t, y) for t, y in zip(times, source_y.T)]), flux_ref, flux_scale)
    source_rhs = source['reference_rhs']
    python_rhs = np.column_stack([model.rhs(0, y) for y in inputs['check_states'].T])
    j = model.jacobian(0, inputs['check_states'][:, -1]).toarray()
    original_j = source['jacobian'].toarray()
    invariants = float(np.max(abs(model.pools @ (ref - model.y0[:, None]))))
    pool_scale = max(1, float(np.max(abs(model.pools @ model.y0))))
    audit = {"model_hash": hashlib.sha256(model.path.read_bytes()).hexdigest(), "dimension": len(model.y0),
             "directional_reactions": len(model.k), "case": "aerobic_glucose, reaction 1 ACALD 2x overexpression, original deterministic perturbModel semantics",
             "window": [0, 1], "time_unit": "original MATLAB integration coordinate; not independently assigned physical seconds",
             "event_policy": "Fixed registered window; original steady-state and CPU-time stop conditions replaced for every solver, not chemical forcing events",
             "target_levels": [1e-3, 1e-5], "strictest_target_reason": "At 1e-7, cross-reference differences exceed the 5% reference budget; 1e-5 has verified margin. No 1e-7 accuracy claim on this model.",
             "reference_checks_at_1e-5": checks, "elementary_flux_reference_checks_at_1e-5": flux_checks,
             "reference_stats": reference_stats, "source_octave_version": str(source['octave_version'][0]),
             "source_initial_condition_max_difference": float(source['initial_difference'][0, 0]),
             "original_rhs_max_absolute_difference": float(np.max(abs(python_rhs - source_rhs))),
             "optimized_Octave_rhs_max_absolute_difference": float(np.max(abs(source['optimized_rhs'] - source_rhs))),
             "original_jacobian_max_scaled_difference": float(np.max(abs(j - original_j) / np.maximum(1, abs(original_j)))),
             "closed_enzyme_pools": model.pools.shape[0], "excluded_pseudo_pool": "Biomass_Bulk (reaction 457); algebraically not conserved",
             "reference_max_enzyme_pool_drift_absolute": invariants,
             "reference_relative_enzyme_pool_drift": invariants / pool_scale,
             "pool_normalization": pool_scale, "reference_min_raw_output": float(ref.min()),
             "jacobian_structure_nonzeros": model.sparsity.nnz,
             "status": "pass" if max(checks.values()) < .05 and max(flux_checks.values()) < .05 and invariants / pool_scale < 1e-10 else "fail",
             "accuracy_limit_probes": [{"method": "Radau", "rtol": 1e-10,
                                        "atol": "1e-12*scale", "status": "manually stopped after >120s; exploratory, not included as a successful reference"}]}
    np.savez_compressed(OUT / "kecoli-reference.npz", times=times, concentrations=ref, scale=scale,
                        flux_scale=flux_scale, species=np.array(model.species),
                        BDF=references['BDF-1e-10'], Radau=references['Radau-1e-09'], CVODE_looser=references['CVODE-KLU-1e-10'])
    save_json(OUT / "kecoli-audit.json", audit)
    if audit['status'] != 'pass':
        raise RuntimeError(f"Large model reference gate failed: {checks}, flux={flux_checks}, invariants={invariants}")


def worker(model, times, ref, scale, flux_ref, flux_scale, method, tol, pipe):
    record = {"method": method, "rtol": tol, "model_id": "k-ecoli457", "case_id": "aerobic_ACALD_2x"}
    started = time.perf_counter()
    try:
        y, stats = integrate(model, times, method, tol, scale)
        elapsed = time.perf_counter() - started
        flux = np.column_stack([model.flux(t, state) for t, state in zip(times, y.T)])
        if not np.isfinite(y).all():
            raise RuntimeError("Nonfinite output")
        record.update({"status": "success", "failure_reason": None, "integration_output_seconds": elapsed,
                       "concentration_error": {str(target): weighted_error(y, ref, scale, target) for target in (1e-3, 1e-5)},
                       "elementary_flux_error": {str(target): weighted_error(flux, flux_ref, flux_scale, target) for target in (1e-3, 1e-5)},
                       "pool_drift": float(np.max(abs(model.pools @ (y - model.y0[:, None])))),
                       "min_raw_output": float(y.min()), "significant_negative_output_count": int(np.count_nonzero(y < -1e-12 * scale[:, None])),
                       "counts_and_costs": stats})
    except Exception as exc:
        record.update({"status": "failed", "failure_reason": repr(exc), "integration_output_seconds": time.perf_counter() - started})
    record['isolated_worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    pipe.send(record)
    pipe.close()


def benchmark():
    setup_start = time.perf_counter()
    model = KEcoliModel(ROOT / "models/k-ecoli457.zip")
    setup = time.perf_counter() - setup_start
    data = np.load(OUT / "kecoli-reference.npz")
    times, ref, scale, fs = data['times'], data['concentrations'], data['scale'], data['flux_scale']
    flux_ref = np.column_stack([model.flux(t, y) for t, y in zip(times, ref.T)])
    configs = [(method, tol) for tol in (1e-4, 1e-6, 1e-8, 1e-9) for method in ("BDF", "Radau", "CVODE-KLU")]
    rng = np.random.default_rng(20261102)
    ctx = mp.get_context("fork")
    runs, warmups = [], []
    for repetition in range(-1, 7):
        for i in rng.permutation(len(configs)):
            method, tol = configs[i]
            receive, send = ctx.Pipe(duplex=False)
            process = ctx.Process(target=worker, args=(model, times, ref, scale, flux_ref, fs, method, tol, send))
            process.start()
            send.close()
            if receive.poll(120):
                try:
                    record = receive.recv()
                except EOFError:
                    record = {"method": method, "rtol": tol, "status": "failed", "failure_reason": "worker exited without result", "exitcode": process.exitcode}
            else:
                process.terminate()
                record = {"method": method, "rtol": tol, "status": "failed", "failure_reason": "120s isolated worker timeout"}
            process.join()
            receive.close()
            record['repeat'] = repetition
            (warmups if repetition < 0 else runs).append(record)
            save_json(OUT / "kecoli-runs.json", runs)
            save_json(OUT / "kecoli-warmups.json", warmups)
            print("run", repetition, method, tol, record['status'], record.get('integration_output_seconds'), flush=True)
    summaries = []
    for method, tol in configs:
        group = [r for r in runs if r['method'] == method and r['rtol'] == tol]
        success = [r for r in group if r['status'] == 'success']
        row = {"method": method, "rtol": tol, "successes": len(success), "failures": len(group) - len(success)}
        if success:
            seconds = [r['integration_output_seconds'] for r in success]
            row.update({"all_seconds": seconds, "median_seconds": float(np.median(seconds)),
                        "iqr_seconds": float(np.ptp(np.quantile(seconds, [.25, .75]))),
                        "peak_rss_bytes": max(r['isolated_worker_peak_rss_bytes'] for r in success),
                        "setup_plus_median_seconds": setup + float(np.median(seconds)),
                        "concentration_error": success[0]['concentration_error'], "elementary_flux_error": success[0]['elementary_flux_error'],
                        "max_pool_drift": max(r['pool_drift'] for r in success),
                        "min_raw_output": min(r['min_raw_output'] for r in success)})
        summaries.append(row)
    save_json(OUT / "kecoli-summary.json", summaries)
    save_json(OUT / "linux-environment.json", {"python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "sundials": "6.4.1+dfsg1-3build4, double/int64", "octave": "8.4.0",
              "platform": platform.platform(), "processor": platform.processor(), "setup_seconds": setup,
              "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "timing": "Identical Python RHS and sparse analytic Jacobian; isolated fork worker for each run; 1 warmup + 7 interleaved runs; minimum trajectory duration >20ms",
              "memory": "Isolated worker ru_maxrss includes inherited model/reference arrays, integration and postprocessing, not incremental solver-only memory",
              "native_linear_cost": "SciPy LU/solve instrumented; CVODE linear algebra is included in total minus callbacks, not separately exposed as nlu"})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", action='store_true')
    parser.add_argument("--benchmark", action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reference:
        reference()
    if args.benchmark:
        benchmark()


if __name__ == '__main__':
    main()
