"""Fixed-state predictor step-size cost falsification; not an adaptive solver."""
import concurrent.futures
import json
import platform
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare
from research import m4_linear_slow_predictor_screen as core

OUT = ROOT / "results/m4-linear-slow-step-size-screen-v1"
STATE_INDEX = 100
H_VALUES = (1e-3, 3e-3, 1e-2, 3e-2)
TARGETS = (1e-3, 1e-5)
BASELINES = {"target_1e3": 1.4079387921374291, "target_1e5": 1.3920109998434782}

def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, _ = core._groups_from_source()
    archive = np.load(case["reference_path"])
    origin = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    budgets = {k: v / 1.5 for k, v in BASELINES.items()}
    source_paths = ("research/m4_linear_slow_step_size_screen.py",
        "research/m4_linear_slow_predictor_screen.py", "research/m4_wmri_residual_micro_screen.py",
        "metabolism/kecoli.py", "metabolism/splitting.py", "metabolism/m3_continuation.py",
        "results/m4-growth-factor-screen-v3/runs.json", "results/m4-admission/state.json")
    registration = dict(date="2026-11-16", model="k-ecoli457", state_index=STATE_INDEX,
        state_time=float(archive["times"][STATE_INDEX]), h_values=list(H_VALUES),
        scope="Single-state fixed predictor step-size screen; not full-window/adaptive qualification",
        public_output_fractions=[0., .25, .5, .75, 1.], predictor_rtol=core.PREDICTOR_RTOL,
        audit=dict(method="BDF", rtol=core.AUDIT_RTOL, concurrent=True,
                   independent_complete_nonlinear_solve_from_identical_start=True,
                   all_internal_accepted_states_checked=True),
        groups=groups.tolist(),
        split_coverage=float(sum(len(case["system"].groups[g]) for g in groups) / case["system"].S.shape[1]),
        targets=dict(concentration_flux=TARGETS, net_flux=1e-3),
        ledger_definitions=dict(pool_endpoint_drift="max(abs(P @ (y1-y0)) / max(1, abs(P @ y0)))",
          physical="P @ (y1-y0) minus integral(P @ S @ v(y(t)))",
          modified="P @ (y1-y0) minus integral(P @ candidate_rhs(t,y(t)))",
          quadrature="16-point Gauss-Legendre over dense output"),
        baseline_medians_seconds=BASELINES, candidate_window_budgets_seconds=budgets,
        baseline_source="Frozen results/m4-admission/state.json, CVODE-KLU formal repeats 0-6",
        source_sha256={p: digest(ROOT / p) for p in source_paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / "registration.json", registration)
    rows, candidate_curves, audit_curves = [], {}, {}
    for h in H_VALUES:
        core.H = h
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="m4-step-audit") as pool:
            audit_future = pool.submit(core.audit_one, case, ("predictor_only_linear_slow", 0, origin.copy()))
            candidate_start = time.perf_counter()
            try:
                candidate = core.candidate_step(case, origin, groups, corrected=False)
                candidate_error = None
            except BaseException as exc:
                candidate, candidate_error = None, f"{type(exc).__name__}: {exc}"
            candidate_seconds = time.perf_counter() - candidate_start
            try:
                audit = audit_future.result()
                audit_error = None
            except BaseException as exc:
                audit, audit_error = None, f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - start
        row = dict(h=h, status="failed", candidate_seconds=candidate_seconds,
                   pipeline_wall_seconds=wall, candidate_failure=candidate_error,
                   audit_failure=audit_error)
        if candidate is not None and audit is not None:
            cvals, avals = candidate["values"], audit["values"]
            targets = {str(t): core._score(case, origin, cvals[:, 1:], avals[:, 1:], t) for t in TARGETS}
            cs, al = candidate["stats"], audit["ledger"]
            c_ledger = max(cs[k] for k in ("pool_endpoint_drift", "physical_reaction_ledger_residual", "modified_reaction_ledger_residual")) <= 1e-10
            a_ledger = max(al[k] for k in ("pool_endpoint_drift", "physical_reaction_ledger_residual", "modified_reaction_ledger_residual")) <= 1e-10
            accuracy = all(v["passes"] for v in targets.values())
            row.update(status="success" if c_ledger and a_ledger and accuracy else "failed",
                candidate_ledger=cs, audit_ledger=al, targets=targets,
                candidate_ledger_pass=c_ledger, audit_ledger_pass=a_ledger, accuracy_pass=accuracy,
                audit_seconds=audit["seconds"], audit_internal_states_checked=audit["internal_states_checked"],
                audit_accepted_internal_steps=audit["accepted_internal_steps"],
                audit_nfev=audit["nfev"], audit_njev=audit["njev"], audit_nlu=audit["nlu"],
                amortized_full_window_seconds={k: wall / h for k in budgets},
                amortized_budget_ratio={k: wall / h / b for k, b in budgets.items()})
            candidate_curves[f"H_{h:g}"] = cvals
            audit_curves[f"H_{h:g}"] = avals
        rows.append(row)
    summary = dict(scope=registration["scope"], rows=rows,
        all_points_correctness_pass=all(r["status"] == "success" for r in rows),
        M4_admission=False,
        interpretation="H-normalized local cost is a falsification estimate only; no full-window or dynamic-adaptivity claim.")
    save(OUT / "runs.json", dict(rows=rows))
    save(OUT / "summary.json", summary)
    keys = sorted(candidate_curves)
    np.savez_compressed(OUT / "curves.npz", initial=origin, state_index=STATE_INDEX,
        h_values=np.asarray(H_VALUES), fractions=np.array((0., .25, .5, .75, 1.)),
        candidate_curves=np.stack([candidate_curves[k] for k in keys]) if keys else np.empty((0, len(origin), 5)),
        audit_curves=np.stack([audit_curves[k] for k in keys]) if keys else np.empty((0, len(origin), 5)),
        artifact_keys=np.array(keys))
    (OUT / "README.md").write_text("# Predictor step-size cost screen\n\nFour fixed macrosteps from one real k-ecoli state are independently audited with concurrent complete nonlinear BDF solves. Per-window extrapolation divides local wall by H and is a stopping screen only; this artifact does not qualify full-window execution or dynamic adaptivity.\n", encoding="utf-8")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    print(json.dumps([{k: r.get(k) for k in ("h", "status", "candidate_seconds", "audit_seconds", "pipeline_wall_seconds", "amortized_budget_ratio", "candidate_failure", "audit_failure")} for r in rows], indent=2), flush=True)

if __name__ == "__main__":
    main()
