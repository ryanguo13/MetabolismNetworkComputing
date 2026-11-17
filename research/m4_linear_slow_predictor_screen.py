"""Registered local screen for predictor-only linear slow coupling.

Every candidate interval receives a separately restarted complete nonlinear
BDF audit. The study is development evidence and never changes M4 admission.
"""

import concurrent.futures
import hashlib
import json
import platform
import time

import numpy as np
from scipy.integrate import BDF, Radau
from scipy.sparse import csc_matrix

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare
from research.m4_wmri_residual_micro_screen import _groups_from_source, _score

OUT = ROOT / "results/m4-linear-slow-predictor-screen-v6"
STATE_INDEX, STEPS, H = 100, 8, 1e-3
TARGETS = (1e-3, 1e-5)
PREDICTOR_RTOL = CORRECTOR_RTOL = 1e-6
AUDIT_RTOL, WORKERS, ATOL_FLOOR = 1e-9, 4, 1e-18
GX, GW = np.polynomial.legendre.leggauss(16)


def sha_json(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def check(y, scale, label):
    y = np.asarray(y)
    threshold = scale if y.ndim <= 1 else np.asarray(scale).reshape((-1,) + (1,) * (y.ndim - 1))
    if not np.isfinite(y).all() or np.any(y < -1e-12 * threshold):
        raise FloatingPointError(f"invalid {label}; no clipping")


def make_maps(case, origin, groups):
    system, model = case["system"], case["model"]
    S = csc_matrix(system.S)
    fast_cols = np.concatenate([system.groups[g] for g in groups])
    slow_cols = np.setdiff1d(np.arange(S.shape[1]), fast_cols)
    F, Q = S[:, fast_cols], S[:, slow_cols]
    q0 = Q @ model.flux_subset(0., origin, slow_cols)
    Jq = (Q @ model.flux_jacobian_subset(0., origin, slow_cols)).tocsc()
    return dict(S=S, fast_cols=fast_cols, slow_cols=slow_cols, F=F, Q=Q,
                q0=np.asarray(q0).ravel(), Jq=Jq)


def rhs_value(case, maps, origin, z):
    model = case["model"]
    ff = maps["F"] @ model.flux_subset(0., z, maps["fast_cols"])
    return ff + maps["q0"] + maps["Jq"] @ (z - origin)


def jac_value(case, maps, z):
    gradient = case["model"].flux_jacobian_subset(0., z, maps["fast_cols"])
    return (maps["F"] @ gradient + maps["Jq"]).tocsc()


def equivalence(case, maps, origin):
    scale = case["scale"]
    # Keep the finite-difference equivalence probe inside the nonnegative domain.
    z = origin + np.linspace(0., 1., len(origin)) * scale * 1e-4
    check(z, scale, "equivalence probe")
    model = case["model"]
    direct = (maps["F"] @ model.flux_subset(0., z, maps["fast_cols"])
              + maps["Q"] @ model.flux_subset(0., origin, maps["slow_cols"])
              + maps["Jq"] @ (z - origin))
    rhs_err = float(np.max(np.abs(rhs_value(case, maps, origin, z) - direct)))
    analytic = jac_value(case, maps, z)
    cols = np.unique(np.linspace(0, len(z) - 1, 12).astype(int))
    fd = np.empty((len(z), len(cols)))
    for k, col in enumerate(cols):
        step = 1e-7 * max(scale[col], abs(z[col]), 1e-8)
        d = np.zeros_like(z); d[col] = step
        fd[:, k] = (rhs_value(case, maps, origin, z + d)
                    - rhs_value(case, maps, origin, z - d)) / (2 * step)
    jac_err = float(np.max(np.abs(fd - analytic[:, cols].toarray())))
    jac_scale = max(1., float(np.max(np.abs(analytic.data))))
    result = dict(rhs_max_abs_error=rhs_err, jacobian_fd_max_abs_error=jac_err,
                  jacobian_fd_scaled_error=jac_err / jac_scale, columns=cols.tolist(),
                  passes=rhs_err <= 1e-10 * max(1., float(np.max(np.abs(direct))) )
                         and jac_err <= 2e-5 * jac_scale)
    if not result["passes"]:
        raise AssertionError(f"candidate RHS/Jacobian equivalence failed: {result}")
    return result


def integrate_dense(rhs, jac, y0, h, scale, rtol, sample_points):
    """One Radau solve with checked accepted states and dense samples."""
    started = time.perf_counter()
    solver = Radau(rhs, 0., y0.copy(), h, rtol=rtol,
                   atol=np.maximum(scale * (0.01 * rtol), ATOL_FLOOR), jac=jac)
    pieces, minimum = [], float(y0.min())
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        check(solver.y, scale, "candidate accepted state")
        dense = solver.dense_output()
        local = np.unique(np.r_[np.linspace(solver.t_old, solver.t, 9),
                                sample_points[(sample_points >= solver.t_old)
                                              & (sample_points <= solver.t)]])
        sampled = dense(local)
        check(sampled, scale, "candidate internal dense state")
        minimum = min(minimum, float(solver.y.min()), float(sampled.min()))
        pieces.append((float(solver.t_old), float(solver.t), dense))

    def evaluate(query):
        query = np.atleast_1d(np.asarray(query, dtype=float))
        scalar = query.size == 1
        value = np.empty((len(y0), len(query)))
        for left, right, dense in pieces:
            mask = (query >= left) & (query <= right)
            if np.any(mask):
                value[:, mask] = dense(query[mask])
        check(value, scale, "candidate dense output")
        return value[:, 0] if scalar else value

    stats = dict(seconds=time.perf_counter() - started, nfev=int(solver.nfev),
                 njev=int(solver.njev), nlu=int(solver.nlu),
                 accepted_internal_steps=len(pieces), min_internal_state=minimum)
    return evaluate, stats


def ledger_stats(case, origin, endpoint, dense, h, candidate_rhs):
    pools = csc_matrix(case["options"]["pools"])
    before = np.asarray(pools @ origin).ravel()
    delta = np.asarray(pools @ (endpoint - origin)).ravel()
    denom = np.maximum(1., np.abs(before))
    physical, modified = [], []
    for tau in (GX + 1.) * 0.5:
        z = dense(tau * h)
        check(z, case["scale"], "ledger quadrature state")
        physical.append(np.asarray(pools @ (case["system"].S @ case["system"].flux(0., z))).ravel())
        modified.append(np.asarray(pools @ candidate_rhs(0., z)).ravel())
    p_int = h * (np.asarray(physical).T @ (GW * 0.5))
    m_int = h * (np.asarray(modified).T @ (GW * 0.5))
    return dict(pool_endpoint_drift=float(np.max(np.abs(delta) / denom)),
                physical_reaction_ledger_residual=float(np.max(np.abs(delta-p_int) / denom)),
                modified_reaction_ledger_residual=float(np.max(np.abs(delta-m_int) / denom)),
                physical_ledger_rate_max=float(np.max(np.abs(physical))),
                modified_ledger_rate_max=float(np.max(np.abs(modified))))


def candidate_step(case, origin, groups, corrected):
    started = time.perf_counter(); scale = case["scale"]
    maps = make_maps(case, origin, groups)
    eq = equivalence(case, maps, origin)
    public = H * np.array((0., .25, .5, .75, 1.))
    dense_nodes = np.array((0., .25, .5, .75, 1.)) * H
    def base_rhs(t, z):
        check(z, scale, "candidate RHS argument")
        return rhs_value(case, maps, origin, z)
    jac = lambda t, z: jac_value(case, maps, z)
    predictor_dense, pred_stats = integrate_dense(base_rhs, jac, origin, H, scale,
                                                   PREDICTOR_RTOL, dense_nodes)
    if corrected:
        pred_samples = predictor_dense(H * np.array((0., .25, .5, .75, 1.)))
        remainders = []
        for z in pred_samples.T:
            slow = maps["Q"] @ case["model"].flux_subset(0., z, maps["slow_cols"])
            remainders.append(np.asarray(slow - maps["q0"] - maps["Jq"] @ (z-origin)).ravel())
        rem = np.column_stack(remainders)
        def forcing(t):
            x = float(t / H)
            return (rem[:, 0] * ((x-.5)*(x-1.)/.5)
                    + rem[:, 2] * (-4.*x*(x-1.))
                    + rem[:, 4] * (x*(x-.5)/.5))
        def rhs(t, z):
            check(z, scale, "corrected candidate RHS argument")
            return rhs_value(case, maps, origin, z) + forcing(t)
        candidate_dense, solve_stats = integrate_dense(rhs, jac, origin, H, scale,
                                                        CORRECTOR_RTOL, dense_nodes)
        stats = dict(predictor=pred_stats, corrector=solve_stats,
                     remainder_max_abs=float(np.max(np.abs(rem))))
        final_rhs = rhs
    else:
        candidate_dense = predictor_dense
        stats = dict(predictor=pred_stats)
        final_rhs = base_rhs
    values = candidate_dense(public)
    check(values, scale, "candidate public output")
    stats.update(ledger_stats(case, origin, values[:, -1], candidate_dense, H, final_rhs))
    stats.update(equivalence=eq, seconds=time.perf_counter()-started,
                 split_coverage=float(len(maps["fast_cols"]) / case["system"].S.shape[1]))
    return dict(values=values, dense=candidate_dense, rhs=final_rhs, stats=stats)


def audit_one(case, task):
    method, index, origin = task
    scale, system, model = case["scale"], case["system"], case["model"]
    S = csc_matrix(system.S); query = H * np.array((0., .25, .5, .75, 1.))
    started = time.perf_counter()
    def rhs(t, z):
        check(z, scale, "independent complete audit RHS argument")
        return S @ model.flux(0., z)
    def jac(t, z):
        return (S @ model.flux_jacobian(0., z)).tocsc()
    solver = BDF(rhs, 0., origin.copy(), H, rtol=AUDIT_RTOL,
                 atol=np.maximum(scale * (0.01 * AUDIT_RTOL), ATOL_FLOOR), jac=jac)
    values = np.empty((len(origin), len(query))); cursor = 0; minimum = float(origin.min())
    internal_steps = 0; pieces = []
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        check(solver.y, scale, "independent complete audit accepted state")
        minimum = min(minimum, float(solver.y.min()))
        dense = solver.dense_output()
        pieces.append((float(solver.t_old), float(solver.t), dense))
        stop = int(np.searchsorted(query, solver.t, side="right"))
        if stop > cursor:
            values[:, cursor:stop] = dense(query[cursor:stop])
            check(values[:, cursor:stop], scale, "audit public output")
            cursor = stop
        internal_steps += 1
    if cursor != len(query):
        raise RuntimeError("full audit did not cover the registered public nodes")
    def evaluate(times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        scalar = times.size == 1
        result = np.empty((len(origin), len(times)))
        for left, right, dense in pieces:
            mask = (times >= left) & (times <= right)
            if np.any(mask):
                result[:, mask] = dense(times[mask])
        check(result, scale, "audit dense ledger state")
        return result[:, 0] if scalar else result
    audit_ledger = ledger_stats(case, origin, values[:, -1], evaluate, H, rhs)
    return dict(method=method, index=index, values=values, seconds=time.perf_counter()-started,
                nfev=int(solver.nfev), njev=int(solver.njev), nlu=int(solver.nlu),
                min_internal_state=minimum, accepted_internal_steps=internal_steps,
                internal_states_checked=True, ledger=audit_ledger)


def run_candidate(case, initial, groups, name, corrected, audit_pool):
    states, curves, rows, audit_futures = [initial.copy()], [], [], []
    started = time.perf_counter()
    for index in range(STEPS):
        origin = states[-1]
        row = dict(method=name, index=index, status="failed",
                   start_state_sha256=sha_json(origin.tolist()))
        try:
            trial = candidate_step(case, origin, groups, corrected)
            states.append(trial["values"][:, -1].copy()); curves.append(trial["values"].copy())
            audit_futures.append(audit_pool.submit(audit_one, case, (name, index, origin.copy())))
            row.update(status="success", candidate_seconds=trial["stats"]["seconds"],
                       stats=trial["stats"], end_state_sha256=sha_json(states[-1].tolist()))
        except (FloatingPointError, RuntimeError, ValueError, AssertionError) as exc:
            row.update(reason=str(exc), candidate_seconds=None)
            rows.append(row)
            rows.extend(dict(method=name, index=j, status="not_run_after_failure")
                        for j in range(index+1, STEPS))
            break
        rows.append(row)
    candidate_elapsed = time.perf_counter() - started
    audit_started = time.perf_counter()
    audits, failures = [], []
    for index, future in enumerate(audit_futures):
        try:
            audits.append(future.result())
        except BaseException as exc:
            failures.append(dict(method=name, index=index, failure=str(exc)))
    pipeline_wall = time.perf_counter() - started
    return dict(name=name, states=states, curves=curves, rows=rows, audits=audits,
                audit_failures=failures, pipeline_wall_seconds=pipeline_wall,
                post_generation_wait_seconds=time.perf_counter()-audit_started,
                candidate_wall_seconds=candidate_elapsed,
                complete=len(curves) == STEPS)


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, _ = _groups_from_source()
    archive = np.load(case["reference_path"])
    initial = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    source_paths = ("research/m4_linear_slow_predictor_screen.py",
                    "research/m4_wmri_residual_micro_screen.py", "metabolism/kecoli.py",
                    "metabolism/splitting.py", "results/m4-growth-factor-screen-v3/runs.json",
                    "results/m4-same-machine-baselines-v2/runs.json",
                    "results/m4-admission/state.json")
    baseline = dict(target_1e3=1.4079387921374291, target_1e5=1.3920109998434782)
    registration = dict(date="2026-11-16", model="k-ecoli457", state_index=STATE_INDEX,
        state_time=float(archive["times"][STATE_INDEX]), steps=STEPS, h=H,
        public_output_fractions=[0., .25, .5, .75, 1.],
        methods=["predictor_only_linear_slow", "quadratic_slow_remainder_corrected"],
        predictor_rtol=PREDICTOR_RTOL, corrector_rtol=CORRECTOR_RTOL,
        audit=dict(method="BDF", rtol=AUDIT_RTOL, workers=WORKERS,
                   independent_complete_nonlinear_solve_per_candidate_step=True,
                   all_internal_accepted_states_checked=True,
                   overlap_policy="submit each interval audit after its candidate curve exists; generate later intervals while audits run; trajectory usable only after every audit and gate passes",
                   adaptive_rejection_or_commit_qualified=False),
        ledger_definitions=dict(
            pool_endpoint_drift="max(abs(P @ (y1-y0)) / max(1, abs(P @ y0)))",
            physical_reaction_ledger_residual="max(abs(P @ (y1-y0) - integral(P @ S @ v(y(t)))) / max(1, abs(P @ y0)))",
            modified_reaction_ledger_residual="max(abs(P @ (y1-y0) - integral(P @ candidate_rhs(t,y(t)))) / max(1, abs(P @ y0)))",
            quadrature="16-point Gauss-Legendre over the full interval using dense output",
            pool_source="case['options']['pools'] from metabolism.m3_continuation.prepare('kecoli')"),
        groups=groups.tolist(), fast_reaction_columns=int(sum(len(case["system"].groups[g]) for g in groups)),
        split_coverage=float(sum(len(case["system"].groups[g]) for g in groups) / case["system"].S.shape[1]),
        gates=dict(no_clipping=True, manual_rhs_jacobian_equivalence=True,
                   physical_and_modified_reaction_ledgers=True, enzyme_pool_drift=True,
                   concentration_and_unidirectional_flux_targets=list(TARGETS), net_flux_target=1e-3,
                   every_candidate_has_independent_audit=True, no_full_window_or_M4_claim=True),
        cost_hypothesis="One Radau solve per predictor-only step may leave positive short-window k-ecoli budget after four-worker independent BDF audits.",
        baseline_medians_seconds=baseline,
        baseline_source="Frozen gate values from results/m4-admission/state.json: CVODE-KLU-0.001 and CVODE-KLU-1e-05, formal repeats 0-6. The newer same-machine baseline-v2 results are separately fingerprinted context only; the admission gatekeeper excludes development timings from historical formal baselines.",
        candidate_budgets_seconds={k: v/1.5 for k, v in baseline.items()},
        source_sha256={p: digest(ROOT / p) for p in source_paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / "registration.json", registration)

    methods = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS,
                                               thread_name_prefix="m4-speculative-independent-bdf") as pool:
        methods.append(run_candidate(case, initial, groups, "predictor_only_linear_slow", False, pool))
        methods.append(run_candidate(case, initial, groups, "quadratic_slow_remainder_corrected", True, pool))
    audit_results = [audit for method in methods for audit in method["audits"]]
    audit_failures = [failure for method in methods for failure in method["audit_failures"]]
    pipeline_wall_by_method = {m["name"]: m["pipeline_wall_seconds"] for m in methods}
    audit_map = {(a["method"], a["index"]): a for a in audit_results}
    steps, candidate_arrays, audit_arrays = [], {}, {}
    expected_audits = sum(len(m["curves"]) for m in methods)
    all_pass = (all(m["complete"] for m in methods) and not audit_failures
                and len(audit_map) == expected_audits)
    for method in methods:
        for i, curve in enumerate(method["curves"]):
            key = method["name"], i
            audit = audit_map.get(key); origin = method["states"][i]
            row = dict(method=method["name"], index=i, status="failed")
            if audit is None:
                row["reason"] = next((x["failure"] for x in audit_failures
                                      if (x["method"], x["index"]) == key), "missing audit")
                all_pass = False
            else:
                targets = {str(t): _score(case, origin, curve[:, 1:], audit["values"][:, 1:], t)
                           for t in TARGETS}
                cstats = method["rows"][i]["stats"]
                ledger_pass = max(cstats["pool_endpoint_drift"],
                                  cstats["physical_reaction_ledger_residual"],
                                  cstats["modified_reaction_ledger_residual"]) <= 1e-10
                audit_ledger_pass = max(audit["ledger"]["pool_endpoint_drift"],
                                        audit["ledger"]["physical_reaction_ledger_residual"],
                                        audit["ledger"]["modified_reaction_ledger_residual"]) <= 1e-10
                accuracy_pass = all(v["passes"] for v in targets.values())
                row.update(status="success" if ledger_pass and audit_ledger_pass and accuracy_pass else "failed",
                           candidate_seconds=method["rows"][i]["candidate_seconds"],
                           audit_seconds=audit["seconds"], audit_internal_states_checked=True,
                           audit_accepted_internal_steps=audit["accepted_internal_steps"],
                           audit_nfev=audit["nfev"], audit_njev=audit["njev"], audit_nlu=audit["nlu"],
                           audit_ledger=audit["ledger"],
                           metrics=targets, ledger_pass=ledger_pass, audit_ledger_pass=audit_ledger_pass,
                           accuracy_pass=accuracy_pass,
                           candidate_plus_audit_seconds=method["rows"][i]["candidate_seconds"] + audit["seconds"])
                if row["status"] != "success":
                    all_pass = False
                candidate_arrays[f"{method['name']}_step_{i}"] = curve
                audit_arrays[f"{method['name']}_step_{i}"] = audit["values"]
            steps.append(row)

    predictor = next(m for m in methods if m["name"] == "predictor_only_linear_slow")
    predictor_audit_sum = sum(r["seconds"] for r in audit_results
                              if r["method"] == predictor["name"])
    predictor_cost = pipeline_wall_by_method[predictor["name"]]
    budgets = registration["candidate_budgets_seconds"]
    summary = dict(scope="One real state, eight fixed intervals; short-window screen only.",
        candidate_wall_seconds={m["name"]: m["candidate_wall_seconds"] for m in methods},
        predictor_individual_audit_sum_seconds=predictor_audit_sum,
        all_candidate_generation_wall_seconds=sum(m["candidate_wall_seconds"] for m in methods),
        candidate_and_audit_wall_seconds_predictor_only=predictor_cost,
        speculative_pipeline_wall_seconds_by_method=pipeline_wall_by_method,
        post_generation_wait_seconds_by_method={m["name"]: m["post_generation_wait_seconds"] for m in methods},
        candidate_and_audit_wall_seconds_by_method=pipeline_wall_by_method,
        all_speculative_pipelines_wall_seconds=sum(pipeline_wall_by_method.values()),
        budget_seconds=budgets,
        predictor_budget_ratio={k: predictor_cost/v for k, v in budgets.items()},
        audit_failures=audit_failures, step_results=steps,
        predictor_candidate_complete=predictor["complete"],
        all_correctness_gates_pass=bool(all_pass), M4_admission=False,
        interpretation="Short-window cost evidence only; no full-window, seven-repeat, or admission claim.")
    save(OUT / "runs.json", dict(methods=[dict(name=m["name"], complete=m["complete"],
         candidate_wall_seconds=m["candidate_wall_seconds"],
         pipeline_wall_seconds=m["pipeline_wall_seconds"],
         post_generation_wait_seconds=m["post_generation_wait_seconds"],
         attempts=m["rows"]) for m in methods],
         audits=[{k: v for k, v in a.items() if k != "values"} for a in audit_results],
         audit_failures=audit_failures, step_results=steps))
    save(OUT / "summary.json", summary)
    np.savez_compressed(OUT / "curves.npz", initial=initial, h=H, groups=groups,
                        fractions=np.array((0., .25, .5, .75, 1.)),
                        candidate_curves=np.stack([candidate_arrays[k] for k in sorted(candidate_arrays)])
                            if candidate_arrays else np.empty((0, len(initial), 5)),
                        audit_curves=np.stack([audit_arrays[k] for k in sorted(audit_arrays)])
                            if audit_arrays else np.empty((0, len(initial), 5)),
                        artifact_keys=np.array(sorted(candidate_arrays)))
    (OUT / "README.md").write_text(
        "# Linear slow-predictor local screen\n\n"
        "Predictor-only linear slow coupling is compared with the separately integrated quadratic slow-remainder candidate. Each interval audit is submitted as soon as its candidate curve exists, while subsequent fixed-path steps are generated; trajectories are usable only after all audits and gates pass. This fixed-path speculation screen does not qualify adaptive rejection or commit behavior and is not a full-window or M4 qualification.\n",
        encoding="utf-8")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    print(json.dumps({k: summary[k] for k in ("candidate_wall_seconds",
          "speculative_pipeline_wall_seconds_by_method", "post_generation_wait_seconds_by_method",
          "candidate_and_audit_wall_seconds_predictor_only",
          "predictor_budget_ratio", "all_correctness_gates_pass", "M4_admission")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
