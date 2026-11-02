"""Run M1 pilot: uv run python -m metabolism.benchmark."""

import os

# Apply before NumPy/SciPy/RoadRunner import; do not inherit hidden BLAS threading.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner
import libsbml
from scipy.integrate import solve_ivp
from scipy.io import loadmat

from .sbml import SBMLModel

ROOT = Path(__file__).resolve().parents[1]
CASES = [("BIOMD0000000064", 10.0, "minute"), ("BIOMD0000000051", 20.0, "second")]
TARGETS = (1e-3, 1e-5, 1e-7)
TOLERANCES = (1e-3, 1e-5, 1e-7, 1e-9)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def grid(end, density=1):
    # Development grids; formal peak/event grid registration remains an M1 task.
    return np.unique(np.r_[0, np.geomspace(1e-8, end, 240 * density),
                           np.linspace(0, end, 401 * density)])


def error(y, reference, scale, target):
    return float(np.max(np.abs(y - reference) /
                        (0.01 * target * scale[:, None] + target * np.maximum(np.abs(reference), scale[:, None]))))


def scipy_solve(model, times, method, rtol, scale):
    raw_min = np.full(len(model.species), np.inf)
    calls = 0
    def rhs(t, y):
        nonlocal calls
        calls += 1
        np.minimum(raw_min, y, out=raw_min)
        return model.rhs(t, y)
    result = solve_ivp(rhs, (0, times[-1]), model.y0, method=method,
                       rtol=rtol, atol=rtol * 0.01 * scale,
                       jac_sparsity=model.sparsity, dense_output=True)
    if not result.success:
        raise RuntimeError(result.message)
    y = result.sol(times)
    stats = {"nfev": result.nfev, "njev": result.njev, "nlu": result.nlu,
             "observed_rhs_calls_including_fd": calls,
             "accepted_steps": len(result.t) - 1,
             "min_accepted_state": float(result.y.min()),
             "min_rhs_argument_including_newton_and_fd": float(raw_min.min()),
             "significant_negative_rhs_components": int(np.count_nonzero(raw_min < -1e-12 * scale))}
    return y, stats


def cvode_solve(rr, model, times, rtol, scale):
    rr.resetAll()
    rr.integrator.stiff = True
    rr.integrator.relative_tolerance = rtol
    rr.integrator.maximum_num_steps = 100000
    # RoadRunner also includes rule-defined species in its tolerance vector.
    rr.integrator.absolute_tolerance = float(rtol * 0.01 * scale.min())
    values = rr.simulate(times=times.tolist(), selections=["time", *[f"[{s}]" for s in model.species]])
    return np.asarray(values[:, 1:]).T, {
        "nfev": None, "njev": None, "nlu": None, "accepted_steps": None,
        "min_accepted_state": None, "min_rhs_argument_including_newton_and_fd": None,
        "note": "RoadRunner public API here does not expose internal steps/counters; dense default CVODE, compiled RHS. Scalar atol uses smallest common species scale."}


def parity(model, rr, times, states):
    max_rhs, max_flux = 0.0, 0.0
    for i in np.linspace(0, len(times) - 1, 12, dtype=int):
        rr.resetAll()
        rr.setValue("time", float(times[i]))
        for s, v in zip(model.species, states[:, i]):
            rr[f"[{s}]"] = float(v)
        reference_rhs = np.array([rr.getValue(f"{s}'") for s in model.species]) / model.volumes
        reference_flux = np.asarray(rr.getReactionRates())
        max_rhs = max(max_rhs, float(np.max(np.abs(model.rhs(times[i], states[:, i]) - reference_rhs)
                                           / np.maximum(1, np.abs(reference_rhs)))))
        max_flux = max(max_flux, float(np.max(np.abs(model.flux(times[i], states[:, i]) - reference_flux)
                                             / np.maximum(1, np.abs(reference_flux)))))
    if max(max_rhs, max_flux) > 1e-10:
        raise RuntimeError(f"Independent executor disagreement: {max_rhs}, {max_flux}")
    return {"sampled_states": 12, "rhs_max_scaled_difference": max_rhs,
            "flux_max_scaled_difference": max_flux}


def model_audit(model):
    m = model.model
    return {"model_id": m.id, "sha256": model.sha256,
            "species_total": m.getNumSpecies(), "dynamic_concentration_count": len(model.species),
            "dynamic_species": model.species, "initial_concentrations": model.y0.tolist(),
            "boundary_species": [s.id for s in m.species if s.getBoundaryCondition()],
            "constant_species": [s.id for s in m.species if s.getConstant()],
            "assignment_rules": {r.getVariable(): libsbml.formulaToL3String(r.getMath())
                                 for r in m.getListOfRules()},
            "compartments": {c.id: {"volume": c.getSize(), "constant": c.getConstant()}
                             for c in m.getListOfCompartments()},
            "reactions": len(model.reactions), "reaction_ids": model.reactions,
            "events": m.getNumEvents(), "time_dependent": model.time_dependent,
            "jacobian_structural_nonzeros": int(model.sparsity.sum()),
            "closed_linear_invariants": model.L.shape[1],
            "linear_invariant_basis": model.L.tolist(),
            "split_scope": "blocked: explicitly time-dependent rules" if model.time_dependent else "autonomous candidate"}


def large_model_audit():
    path = ROOT / "models/k-ecoli457.zip"
    with zipfile.ZipFile(path) as outer:
        inner_bytes = outer.read("Supplementary Software 1/k-ecoli457.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        data = loadmat(io.BytesIO(inner.read("k-ecoli457/Data.mat")), simplify_cells=True)
        report = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "inner_zip_sha256": hashlib.sha256(inner_bytes).hexdigest(),
                  "files": [n for n in inner.namelist() if n.startswith('k-ecoli457/')],
                  "mat_variables": {k: {"type": type(v).__name__, "shape": list(v.shape) if hasattr(v, "shape") else None,
                                         "fields": list(v) if isinstance(v, dict) else None}
                                    for k, v in data.items() if not k.startswith('__')},
                  "status": "source available; original MATLAB execution/port parity not yet verified"}
        # Save short original source for audit, not the 669 KB generated Jacobian.
        audit_dir = ROOT / "results/m1/large-model-source"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for name in ("Main_Module.m", "mass_balance_ode.m", "solve_ode.m"):
            (audit_dir / name).write_bytes(inner.read("k-ecoli457/" + name))
        expanded = data.get("network_data")
        if isinstance(expanded, dict):
            report["expanded_fields"] = {k: {"type": type(v).__name__, "shape": list(v.shape) if hasattr(v, "shape") else None}
                                         for k, v in expanded.items()}
            report["expanded_ode_dimension"] = int(expanded["S_f_b"].shape[0])
            report["elementary_directional_reactions"] = int(expanded["S_f_b"].shape[1])
            report["unexpanded_metabolites"] = int(len(expanded["unexpanded_model"]["metab"]))
            report["unexpanded_reactions"] = int(len(expanded["unexpanded_model"]["rxn"]))
    return report


def run_case(mid, end, unit, out, repeats):
    start = time.perf_counter()
    model = SBMLModel(ROOT / f"models/{mid}.xml")
    python_setup = time.perf_counter() - start
    start = time.perf_counter()
    rr = roadrunner.RoadRunner(str(model.path))
    rr_setup = time.perf_counter() - start
    times = grid(end)
    initial_scale = np.maximum(model.y0, 1e-3)
    ref, _ = scipy_solve(model, times, "Radau", 1e-10, initial_scale)
    # Freeze scales after a development reference; every method uses this same vector.
    scale = np.maximum(np.max(np.abs(ref), axis=1), 1e-6)
    ref, _ = scipy_solve(model, times, "Radau", 1e-11, scale)
    bdf, _ = scipy_solve(model, times, "BDF", 1e-11, scale)
    tighter, _ = scipy_solve(model, times, "Radau", 1e-12, scale)
    cvode, _ = cvode_solve(rr, model, times, 1e-11, scale)
    reference_checks = {"BDF_vs_Radau": error(bdf, ref, scale, 1e-7),
                        "Radau_tightening": error(tighter, ref, scale, 1e-7),
                        "CVODE_vs_Radau": error(cvode, ref, scale, 1e-7)}
    if max(reference_checks.values()) > 0.05:
        raise RuntimeError(f"Reference gate failed: {reference_checks}")
    # Dense grid audit of the same references, computed independently of timed runs.
    dense_times = grid(end, 2)
    dense_ref, _ = scipy_solve(model, dense_times, "Radau", 1e-12, scale)
    dense_bdf, _ = scipy_solve(model, dense_times, "BDF", 1e-11, scale)
    reference_checks["dense_grid_BDF_vs_Radau"] = error(dense_bdf, dense_ref, scale, 1e-7)
    if reference_checks["dense_grid_BDF_vs_Radau"] > 0.05:
        raise RuntimeError("Dense reference gate failed")
    semantic_checks = parity(model, rr, times, ref)
    np.savez_compressed(out / f"{mid}-reference.npz", times=times, concentrations=ref,
                        scale=scale, species=np.array(model.species), BDF=bdf,
                        Radau_tighter=tighter, CVODE=cvode)
    np.savetxt(out / f"{mid}-reference.csv", np.column_stack([times, ref.T]), delimiter=",",
               header=",".join([f"time_{unit}", *model.species]), comments="")
    snapshots = []
    for index in (0, len(times) // 2, len(times) - 1):
        jac = model.jacobian(times[index], ref[:, index], scale)
        ev = np.linalg.eigvals(jac)
        nonzero = np.abs(ev)[np.abs(ev) > max(1e-8, np.max(np.abs(ev)) * 1e-10)]
        snapshots.append({"time": float(times[index]), "eigenvalues_real": ev.real.tolist(),
                          "eigenvalues_imag": ev.imag.tolist(),
                          "effective_magnitude_ratio": float(nonzero.max() / nonzero.min()) if len(nonzero) else None,
                          "scaled_jacobian_inf_norm": float(np.linalg.norm(jac * scale[None, :] / scale[:, None], np.inf))})
    audit = model_audit(model)
    audit.update({"case_id": "original_initial_values", "time_end": end, "time_unit": unit,
                  "time_window_basis": "development numerical window, not claimed to match all publication figures",
                  "scale": scale.tolist(), "reference_checks_at_target_1e-7": reference_checks,
                  "semantic_checks": semantic_checks, "jacobian_snapshots": snapshots,
                  "setup_seconds": {"python_adapter": python_setup, "roadrunner_parse_compile": rr_setup}})
    if mid.endswith("0064"):
        # BioModels curation notes reproduce the branched model's Table 4 steady state.
        expected = {"G6P": 1.03, "F6P": .11, "F16P": .60, "P3G": .36,
                    "P2G": .04, "PEP": .07, "PYR": 8.52, "ACE": .17, "NAD": 1.55, "NADH": .04}
        rr.resetAll()
        rr.simulate(0, 100, 2)
        observed = {s: float(rr[f"[{s}]"]) for s in expected}
        audit["curated_table4_steady_state"] = {"expected_rounded": expected, "cvode_at_100_minutes": observed,
                                               "within_0.02_mM": all(abs(observed[s] - v) < .02 for s, v in expected.items())}
        if not audit["curated_table4_steady_state"]["within_0.02_mM"]:
            raise RuntimeError("Published curated steady-state check failed")
    configs = [(method, tol) for tol in TOLERANCES for method in ("BDF", "Radau", "CVODE")]
    records = []
    def integrate(method, tol, target_times=times):
        return cvode_solve(rr, model, target_times, tol, scale) if method == "CVODE" else scipy_solve(model, target_times, method, tol, scale)
    # Warm every configuration, then deterministic interleaved timing order.
    warmup_seconds = []
    for method, tol in configs:
        start = time.perf_counter()
        integrate(method, tol)
        warmup_seconds.append(time.perf_counter() - start)
    # All configurations use the same batch size, including the slow methods.
    batch_size = max(1, int(np.ceil(0.02 / min(warmup_seconds))))
    audit["timing_batch_size"] = batch_size
    audit["timing_batch_target_seconds"] = 0.02
    rng = np.random.default_rng(20261102)
    flux_ref = np.column_stack([model.flux(t, y) for t, y in zip(times, ref.T)])
    flux_scale = np.maximum(np.max(np.abs(flux_ref), axis=1), 1e-6)
    for repetition in range(repeats):
        for index in rng.permutation(len(configs)):
            method, tol = configs[index]
            record = {"model_id": mid, "model_hash": model.sha256, "case_id": "original_initial_values",
                      "seed": 20261102, "method": method, "rtol": tol, "repeat": repetition,
                      "implementation": "compiled SBML/RoadRunner" if method == "CVODE" else "Python shared RHS/SciPy"}
            start = time.perf_counter()
            try:
                for _ in range(batch_size):
                    y, stats = integrate(method, tol)
                batch_elapsed = time.perf_counter() - start
                elapsed = batch_elapsed / batch_size
                if not np.isfinite(y).all():
                    raise RuntimeError("Nonfinite trajectory")
                f = np.column_stack([model.flux(t, x) for t, x in zip(times, y.T)])
                invariant_error = float(np.max(np.abs(model.L.T @ (y - model.y0[:, None]))) /
                                        max(1e-12, float(np.max(np.abs(model.L.T @ model.y0))))) if model.L.shape[1] else None
                record.update({"status": "success", "failure_reason": None, "integration_and_output_seconds": elapsed,
                               "batch_seconds": batch_elapsed, "trajectories_per_batch": batch_size,
                               "observed_error": {str(r): error(y, ref, scale, r) for r in TARGETS},
                               "flux_error": {str(r): error(f, flux_ref, flux_scale, r) for r in TARGETS},
                               "invariant_error": invariant_error, "min_raw_output_concentration": float(y.min()),
                               "significant_negative_output_count": int(np.count_nonzero(y < -1e-12 * scale[:, None])),
                               "solver_counts": stats})
            except Exception as exc:
                record.update({"status": "failed", "failure_reason": repr(exc),
                               "integration_and_output_seconds": time.perf_counter() - start})
            records.append(record)
    write_json(out / f"{mid}-audit.json", audit)
    write_json(out / f"{mid}-runs.json", records)
    summaries = []
    for method, tol in configs:
        runs = [r for r in records if r['method'] == method and r['rtol'] == tol]
        successful = [r for r in runs if r['status'] == 'success']
        values = [r['integration_and_output_seconds'] for r in successful]
        setup = rr_setup if method == "CVODE" else python_setup
        summary = {"model_id": mid, "method": method, "rtol": tol, "successes": len(successful),
                   "failures": len(runs) - len(successful), "all_seconds": values,
                   "median_seconds": float(np.median(values)) if values else None,
                   "iqr_seconds": float(np.ptp(np.quantile(values, [.25, .75]))) if values else None,
                   "single_task_setup_plus_median_seconds": setup + float(np.median(values)) if values else None}
        if successful:
            summary.update({k: successful[0][k] for k in ("observed_error", "flux_error", "invariant_error", "min_raw_output_concentration")})
            dense_y, _ = integrate(method, tol, dense_times)
            summary["dense_grid_observed_error"] = {str(r): error(dense_y, dense_ref, scale, r) for r in TARGETS}
        summaries.append(summary)
    write_json(out / f"{mid}-summary.json", summaries)
    print(mid, "reference checks", reference_checks, "runs", len(records), flush=True)
    return audit, summaries


def plot_results(out, audits, summaries):
    fig, axes = plt.subplots(1, len(audits), figsize=(11, 4), constrained_layout=True)
    for ax, audit, summary in zip(np.atleast_1d(axes), audits, summaries):
        for method, marker in (("BDF", "o"), ("Radau", "s"), ("CVODE", "^")):
            points = [r for r in summary if r['method'] == method and r['successes']]
            ax.loglog([r['observed_error']['1e-07'] for r in points],
                      [r['median_seconds'] for r in points], marker=marker,
                      label=method + (" (compiled RHS)" if method == "CVODE" else " (Python RHS)"))
        ax.axvline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(title=audit['model_id'], xlabel="Observed concentration error / target 1e-7", ylabel="Median integration + output (s)")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "work-precision.png", dpi=180)
    plt.close(fig)


def write_report(out, audits, summaries):
    lines = ["# M1 baseline pilot", "", "Generated from the saved audit and summary JSON files.", "",
             "Times include integration and common output sampling, after warm-up; median of 7 or more interleaved batches.",
             "CVODE uses compiled RoadRunner RHS and dense default linear algebra; SciPy uses shared Python RHS plus structural Jacobian sparsity.",
             "This is a software baseline pilot, not evidence of splitting speedup.", ""]
    for audit, summary in zip(audits, summaries):
        lines += [f"## {audit['model_id']}", "",
                  f"Dynamic states: {audit['dynamic_concentration_count']}; reactions: {audit['reactions']}; "
                  f"window: 0–{audit['time_end']} {audit['time_unit']}; batch size: {audit['timing_batch_size']}.", "",
                  "| Method | rtol | Median ms / trajectory | IQR ms | Concentration E (1e-7) | Dense grid E (1e-7) | Flux E (1e-7) | Failures |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary:
            if row['successes']:
                lines += [f"| {row['method']} | {row['rtol']:.0e} | {row['median_seconds'] * 1000:.3f} | "
                          f"{row['iqr_seconds'] * 1000:.3f} | {row['observed_error']['1e-07']:.4g} | "
                          f"{row['dense_grid_observed_error']['1e-07']:.4g} | {row['flux_error']['1e-07']:.4g} | {row['failures']} |"]
            else:
                lines += [f"| {row['method']} | {row['rtol']:.0e} | — | — | — | — | — | {row['failures']} |"]
        lines += ["", "E ≤ 1 passes that error metric on the specified grid only. Concentration and flux pass/fail are separate.", ""]
    lines += ["![Work–precision curves](work-precision.png)", "",
              "See ../../docs/m1-baseline-report.md for reference checks, scope and remaining acceptance gates."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("Protocol requires at least 7 repeats")
    out = ROOT / "results/m1"
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "python-libsbml", "libroadrunner", "matplotlib")},
                "repeats": args.repeats, "peak_memory": None,
                "limitations": ["Pilot, not a speedup claim", "CVODE uses compiled RHS and default dense linear algebra",
                                "No exposed CVODE internal negativity or solver counters", "No measured peak memory or detailed factorization timing",
                                "Chassagnole publication-figure digitization still pending"]}
    write_json(out / "environment.json", metadata)
    write_json(out / "large-model-audit.json", large_model_audit())
    audits, summaries = [], []
    for case in CASES:
        audit, summary = run_case(*case, out, args.repeats)
        audits.append(audit)
        summaries.append(summary)
    plot_results(out, audits, summaries)
    write_report(out, audits, summaries)


if __name__ == "__main__":
    main()
