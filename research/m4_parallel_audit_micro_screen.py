"""Short-window CPU parallel audit screen.

Candidate states are generated sequentially from a real k-ecoli state.  All
candidate intervals are held pending until every independent full nonlinear
ODE audit finishes.  Sequential, thread-pool and reusable fork-process audit
wall times are compared; parallel results are only a timing experiment and
never bypass the audit or transaction semantics.
"""

import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from metabolism.m2_experiments import save
from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.m3_continuation import ROOT, digest, prepare
from research.m4_wmri_residual_micro_screen import _groups_from_source, _maps, _score, _trial

OUT = ROOT / "results/m4-parallel-audit-micro-screen-v2"
STATE_INDEX = 100
STEPS = 8
H = 1e-3
AUDIT_RTOL = 1e-11
WORKERS = (2, 4)

_WORKER_CASE = None


def _audit(case, y0, h):
    model, system, scale = case["model"], case["system"], case["scale"]
    stoich = csc_matrix(system.S)
    query = h * np.array((0.5, 1.0))
    minimum = float(y0.min())

    def rhs(t, z):
        del t
        if not np.isfinite(z).all() or np.any(z < -1e-12 * scale):
            raise FloatingPointError("invalid full-ODE audit RHS state")
        return stoich @ model.flux(0.0, z)

    def jac(t, z):
        del t
        return (stoich @ model.flux_jacobian(0.0, z)).tocsc()

    started = time.perf_counter()
    result = solve_ivp(rhs, (0.0, h), y0.copy(), method="Radau", t_eval=query,
                       rtol=AUDIT_RTOL, atol=np.maximum(scale * (0.01 * AUDIT_RTOL), 1e-18),
                       jac=jac)
    if not result.success:
        raise RuntimeError(result.message)
    minimum = min(minimum, float(result.y.min()))
    if not np.isfinite(result.y).all() or np.any(result.y < -1e-12 * scale[:, None]):
        raise FloatingPointError("invalid full-ODE audit output")
    return dict(values=result.y, seconds=time.perf_counter() - started,
                nfev=int(result.nfev), njev=int(result.njev), nlu=int(result.nlu),
                min_state=minimum, internal_states_checked=True)


def _worker_init():
    global _WORKER_CASE
    _WORKER_CASE = prepare("kecoli")


def _worker_task(payload):
    index, y0, h = payload
    out = _audit(_WORKER_CASE, y0, h)
    return index, out


def _thread_task(payload, case):
    index, y0, h = payload
    return index, _audit(case, y0, h)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, source_row = _groups_from_source()
    archive = np.load(case["reference_path"])
    initial = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    states = [initial]
    candidate_curves = []
    candidate_rows = []
    candidate_started = time.perf_counter()
    for index in range(STEPS):
        y0 = states[-1]
        maps = _maps(case, y0, groups)
        started = time.perf_counter()
        candidate, candidate_stats = _trial(case, maps, y0, H)
        candidate_curves.append(candidate.copy())
        endpoint = candidate[:, -1].copy()
        states.append(endpoint)
        candidate_rows.append(dict(index=index, start_state_sha256=_canonical_sha256(y0.tolist()),
                                   end_state_sha256=_canonical_sha256(endpoint.tolist()),
                                   seconds=time.perf_counter() - started, candidate_stats=candidate_stats))
    candidate_seconds = time.perf_counter() - candidate_started
    payloads = [(i, states[i], H) for i in range(STEPS)]

    registration = dict(
        date="2026-11-15", model="k-ecoli457", state_index=STATE_INDEX,
        start_time=float(archive["times"][STATE_INDEX]), interval=float(STEPS * H),
        steps=STEPS, h=H, groups=groups.tolist(),
        candidate_method="quadratic slow-remainder W-MRI",
        candidate_seconds=candidate_seconds, audit_method="Radau", audit_rtol=AUDIT_RTOL,
        workers=list(WORKERS),
        semantics=dict(candidate_generation_sequential=True, audits_independent_full_nonlinear=True,
                       all_audits_complete_before_commit=True, no_audit_cache=True,
                       no_clipping=True, internal_states_checked=True,
                       rollback_on_audit_failure=True),
        scope="Short-window audit scheduling screen; no full-window or M4 qualification.",
        source_sha256={p: digest(ROOT / p) for p in (
            "research/m4_parallel_audit_micro_screen.py",
            "research/m4_wmri_residual_micro_screen.py",
            "metabolism/kecoli.py", "metabolism/splitting.py",
            "metabolism/m3_continuation.py",
            "results/m4-growth-factor-screen-v3/runs.json")},
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         start_method="fork"),
    )
    save(OUT / "registration.json", registration)
    save(OUT / "candidate-runs.json", dict(rows=candidate_rows, candidate_seconds=candidate_seconds))
    np.savez_compressed(OUT / "candidate-states.npz", states=np.column_stack(states), h=H, groups=groups)

    # Warm a sequential audit, then measure the same independent tasks.
    warm = _audit(case, states[0], H)
    sequential = []
    started = time.perf_counter()
    for payload in payloads:
        index, result = _worker_task.__name__, None
        del index, result
        audit_result = _audit(case, payload[1], payload[2])
        sequential.append(audit_result)
    sequential_seconds = time.perf_counter() - started

    runs = [dict(kind="sequential", workers=1, seconds=sequential_seconds,
                 per_task_seconds=[float(r["seconds"]) for r in sequential],
                 outputs_sha256=[_canonical_sha256(r["values"].tolist()) for r in sequential],
                 internal_states_checked=all(r["internal_states_checked"] for r in sequential))]
    reference_arrays = [r["values"] for r in sequential]
    candidate_metrics = []
    for index, audit_result in enumerate(sequential):
        candidate_metrics.append(dict(index=index, metrics={str(target): _score(
            case, states[index], candidate_curves[index], audit_result["values"], target)
            for target in (1e-3, 1e-5)}))
    save(OUT / "candidate-metrics.json", dict(rows=candidate_metrics))
    for workers in WORKERS:
        for kind in ("threads", "processes"):
            started = time.perf_counter()
            if kind == "threads":
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(lambda p: _thread_task(p, case), payloads))
            else:
                context = mp.get_context("fork")
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                                                             mp_context=context,
                                                             initializer=_worker_init) as pool:
                    # Ensure model loading is outside the timed audit batch.
                    pool.submit(_worker_task, ("warmup", states[0], H)).result()
                    started = time.perf_counter()
                    results = list(pool.map(_worker_task, payloads))
            elapsed = time.perf_counter() - started
            results.sort(key=lambda x: x[0])
            outputs = [r[1]["values"] for r in results]
            max_difference = max(float(np.max(np.abs(a - b))) for a, b in zip(reference_arrays, outputs))
            runs.append(dict(kind=kind, workers=workers, seconds=elapsed,
                             per_task_seconds=[float(r[1]["seconds"]) for r in results],
                             outputs_sha256=[_canonical_sha256(r.tolist()) for r in outputs],
                             max_abs_difference_vs_sequential=max_difference,
                             internal_states_checked=all(r[1]["internal_states_checked"] for r in results),
                             independent_task_count=len(results)))

    # Save each sequential audit curve; candidate commits remain pending in this screen.
    for index, result in enumerate(sequential):
        artifact = OUT / f"audit-{index:02d}.npz"
        np.savez_compressed(artifact, initial=states[index], values=result["values"],
                            query=H * np.array((0.5, 1.0)), h=H, groups=groups)
    save(OUT / "runs.json", dict(rows=runs, pending_commit=dict(
        all_audits_completed=True, all_audits_independent=True,
        rollback_on_failure="No committed candidate state is exposed before audit completion")))
    baseline_times = []
    baseline_grid = np.linspace(0.0, STEPS * H, STEPS + 1)
    for _ in range(3):
        config = AdaptiveConfig(target=1e-5, initial_step=1e-5, max_step=STEPS * H,
                                max_attempts=512, max_rejections=16, timeout=240.,
                                fallback_method="CVODE-KLU", fallback_rtol=1e-11)
        started = time.perf_counter()
        baseline = AdaptiveSplit(case["system"], case["scale"], config,
                                 fixed_groups=np.arange(len(case["system"].groups)),
                                 **case["options"])
        baseline.integrate(baseline_grid, y0=initial)
        baseline_times.append(time.perf_counter() - started)
    baseline_median = float(np.median(baseline_times))
    save(OUT / "baseline.json", dict(method="AdaptiveSplit-fixed-all", target=1e-5,
                                         repeats=3, seconds=baseline_times, median_seconds=baseline_median))
    fastest_parallel = min((r["seconds"] for r in runs if r["kind"] != "sequential"),
                           default=sequential_seconds)
    summary = dict(scope=registration["scope"], candidate_seconds=candidate_seconds,
                   sequential_audit_seconds=sequential_seconds, runs=runs,
                   candidate_metrics=candidate_metrics, baseline_seconds=baseline_times,
                   baseline_median_seconds=baseline_median,
                   sequential_end_to_end_seconds=candidate_seconds + sequential_seconds,
                   fastest_parallel_end_to_end_seconds=candidate_seconds + fastest_parallel,
                   fastest_parallel_speedup=sequential_seconds / fastest_parallel,
                   end_to_end_ratio_vs_baseline=(candidate_seconds + fastest_parallel) / baseline_median,
                   max_parallel_difference=max(r.get("max_abs_difference_vs_sequential", 0.0) for r in runs),
                   M4_admission=False,
                   interpretation="Parallel audits preserve independent full nonlinear solves and can reduce"
                       " audit wall time only after candidate states are generated; this does not reduce"
                       " candidate work, accepted-step count, or the frozen M4 evidence requirements.")
    save(OUT / "summary.json", summary)
    files = [OUT / name for name in ("registration.json", "candidate-runs.json",
                                     "candidate-metrics.json", "candidate-states.npz",
                                     "baseline.json", "runs.json", "summary.json")]
    files.extend(OUT / f"audit-{i:02d}.npz" for i in range(STEPS))
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    (OUT / "README.md").write_text(
        "# Parallel audit micro-screen\n\n"
        "Candidate states are generated sequentially. Every pending interval is independently"
        " audited before a commit can be considered; thread and fork-process timings are"
        " scheduling diagnostics only.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
