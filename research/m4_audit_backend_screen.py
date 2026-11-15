"""Prospective comparison of complete ODE audit backends on early k-ecoli states."""
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m3_continuation import ROOT, digest, prepare
from metabolism.m2_experiments import save

OUT = ROOT / "results/m4-audit-backend-screen"
METHODS = ("CVODE-KLU", "BDF")
RTOLS = (1e-9, 1e-10, 1e-11)
REPEATS = (-1, 0, 1, 2)
H = 1e-6
TARGETS = (1e-3, 1e-5)


def rates(model, values):
    return np.column_stack([model.flux(0.0, y) for y in values.T])


def solve(case, y0, query, method, rtol):
    system, model = case["system"], case["model"]
    atol = np.maximum(case["scale"] * (0.01 * rtol), 1e-18)
    rhs = lambda t, y: system.S @ model.flux(t, y)
    started = time.perf_counter()
    if method == "CVODE-KLU":
        from metabolism.cvode import solve as cvode_solve
        native = SimpleNamespace(y0=y0, rhs=rhs,
            jacobian=lambda t, y: (system.S @ model.flux_jacobian(t, y)).tocsc(),
            sparsity=model.sparsity)
        values, stats = cvode_solve(native, np.r_[0.0, query], rtol, atol)
        values = values[:, 1:]
    else:
        result = solve_ivp(rhs, (0.0, H), y0, method="BDF", t_eval=query,
                           rtol=rtol, atol=atol,
                           jac=lambda t, y: system.S @ model.flux_jacobian(t, y))
        if not result.success:
            raise RuntimeError(result.message)
        values = result.y
        stats = {"nfev": result.nfev, "njev": result.njev, "nlu": result.nlu}
    return values, stats, time.perf_counter() - started


def score(case, values, truth, initial, target):
    model, options = case["model"], case["options"]
    flux, truth_flux = rates(model, values), rates(model, truth)
    conc_error = float(np.max(np.abs(values - truth) / (case["scale"][:, None] * target)))
    flux_error = float(np.max(np.abs(flux - truth_flux) / (options["flux_scale"][:, None] * target)))
    net_error = 0.0
    if "net_map" in options:
        net_error = float(np.max(np.abs(options["net_map"] @ (flux - truth_flux)) /
                                 (options["net_scale"][:, None] * 1e-3)))
    pool = options["pools"]
    pool_error = float(np.max(np.abs(pool @ (values - initial[:, None]))) /
                       max(1e-12, float(np.max(np.abs(pool @ initial)))))
    return dict(concentration=conc_error, flux=flux_error, net_flux=net_error,
                pool_error=pool_error, finite=bool(np.isfinite(values).all()),
                nonnegative=bool(np.all(values >= -1e-12 * case["scale"][:, None])),
                passes=bool(max(conc_error, flux_error, net_error) <= 0.05
                            and pool_error <= 1e-10 and np.isfinite(values).all()
                            and np.all(values >= -1e-12 * case["scale"][:, None])))


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as ref:
        times, concentrations = ref["times"], ref["concentrations"]
        state_indices = sorted(set([0, int(np.searchsorted(times, 1e-4))]))
        states = [(int(i), float(times[i]), concentrations[:, i].copy()) for i in state_indices]
    query = H * np.array([0.25, 0.5, 0.75, 1.0])
    sources = ["research/m4_audit_backend_screen.py", "metabolism/cvode.py",
               "metabolism/kecoli.py", "metabolism/m3_continuation.py"]
    registration = dict(date="2026-11-15", model="kecoli", state_indices=state_indices,
        state_times=[x[1] for x in states], interval=H, query_fractions=[0.25, 0.5, 0.75, 1.0],
        methods=list(METHODS), rtols=list(RTOLS), repeats=list(REPEATS), targets=list(TARGETS),
        truth_method="Radau", truth_rtol=1e-13,
        hypothesis="SciPy BDF may reduce complete per-attempt audit cost relative to CVODE-KLU while matching independent tighter Radau truth.",
        gates=dict(complete_original_nonlinear_rhs=True, concentration_flux_and_net_flux=True,
                   external_error_budget=0.05, nonnegative_sampled_outputs=True, pools=1e-10,
                   no_clipping=True),
        limitation="Two early states and one 1e-6 interval only; exploratory audit-backend calibration, not controller or M4 qualification. BDF internal states are not exposed, so positivity is checked at requested queries; CVODE internal RHS minima are recorded by native wrapper.",
        source_sha256={p: digest(ROOT / p) for p in sources},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / "registration.json", registration)
    rows, truth_costs = [], {}
    for state_index, state_time, y0 in states:
        truth_atol = np.maximum(case["scale"] * 1e-15, 1e-18)
        start = time.perf_counter()
        truth_result = solve_ivp(lambda t, y: case["system"].S @ case["model"].flux(t, y),
            (0.0, H), y0, method="Radau", t_eval=query, rtol=1e-13, atol=truth_atol,
            jac=lambda t, y: case["system"].S @ case["model"].flux_jacobian(t, y))
        if not truth_result.success:
            raise RuntimeError(f"Radau truth failed: {truth_result.message}")
        truth, truth_seconds = truth_result.y, time.perf_counter() - start
        np.savez_compressed(OUT / f"truth-{state_index}.npz", initial=y0, values=truth)
        truth_costs[str(state_index)] = dict(seconds=truth_seconds, nfev=truth_result.nfev,
            njev=truth_result.njev, nlu=truth_result.nlu)
        schedule = list(itertools.product(METHODS, RTOLS))
        for repeat in REPEATS:
            offset = max(0, repeat) % len(schedule)
            for method, rtol in schedule[offset:] + schedule[:offset]:
                row = dict(state_index=state_index, state_time=state_time, repeat=repeat,
                           warmup=repeat == -1, method=method, rtol=rtol)
                try:
                    values, counts, seconds = solve(case, y0, query, method, rtol)
                    target_scores = {str(target): score(case, values, truth, y0, target)
                                     for target in TARGETS}
                    row.update(status="success", seconds=seconds, counts=counts,
                               metrics=target_scores, finite=bool(np.isfinite(values).all()),
                               nonnegative=bool(np.all(values >= -1e-12*case["scale"][:, None])))
                    row["value_sha256"] = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                    np.savez_compressed(OUT / f"state-{state_index}-{repeat}-{method}-{rtol}.npz", values=values)
                except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exc:
                    row.update(status="failed", reason=str(exc))
                rows.append(row)
                save(OUT / "runs.json", dict(rows=rows, truth_costs=truth_costs))
                print(state_index, repeat, method, rtol, row["status"],
                      round(row.get("seconds", 0.0), 5), flush=True)
    summaries = []
    for state_index, method, rtol in itertools.product(state_indices, METHODS, RTOLS):
        samples = [r for r in rows if r["state_index"] == state_index and r["method"] == method
                   and r["rtol"] == rtol and r["repeat"] >= 0]
        summaries.append(dict(state_index=state_index, method=method, rtol=rtol,
            all_repeats_pass=len(samples) == 3 and all(r["status"] == "success"
                and all(v["passes"] for v in r["metrics"].values()) for r in samples),
            seconds=[r["seconds"] for r in samples] if samples else [],
            median_seconds=float(np.median([r["seconds"] for r in samples])) if samples else None,
            max_errors={metric: max((r["metrics"][str(target)][metric] for r in samples
                for target in TARGETS), default=float("inf"))
                for metric in ("concentration", "flux", "net_flux", "pool_error")}))
    save(OUT / "summary.json", dict(results=summaries, truth_costs=truth_costs,
        conclusion="Exploratory local calibration only; no audit backend substitution is authorized by this screen."))
    save(OUT / "manifest.json", {p: digest(ROOT / p) for p in sources})


if __name__ == "__main__":
    main()
