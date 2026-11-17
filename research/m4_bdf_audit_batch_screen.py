"""Registered batch screen for BDF versus Radau full-ODE audits.

Each candidate interval is audited independently from its own split state.
This screen changes only the audit backend and never changes admission state.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import BDF, Radau
from scipy.sparse import csc_matrix

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare
from research.m4_wmri_residual_micro_screen import _groups_from_source, _maps, _score, _trial

OUT = ROOT / "results/m4-bdf-audit-batch-screen-v1"
STATE_INDEX = 100
STEPS = 8
H = 1e-3
AUDIT_RTOL = 1e-11
ATOL_FLOOR = 1e-18

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def audit(case, y0, h, method, rtol=AUDIT_RTOL):
    model, system, scale = case["model"], case["system"], case["scale"]
    stoich = csc_matrix(system.S)
    query = h * np.array((0.5, 1.0))
    check_min = float(y0.min())
    def rhs(t, z):
        del t
        if not np.isfinite(z).all() or np.any(z < -1e-12 * scale):
            raise FloatingPointError("invalid full-ODE audit RHS state")
        return stoich @ model.flux(0.0, z)
    def jac(t, z):
        del t
        return (stoich @ model.flux_jacobian(0.0, z)).tocsc()
    started = time.perf_counter()
    cls = {"BDF": BDF, "Radau": Radau}[method]
    solver = cls(rhs, 0.0, np.array(y0, copy=True), h, rtol=rtol,
                 atol=np.maximum(scale * (0.01 * rtol), ATOL_FLOOR), jac=jac)
    values = np.empty((len(y0), len(query)))
    i = 0
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12 * scale):
            raise FloatingPointError("invalid accepted full-ODE state")
        check_min = min(check_min, float(solver.y.min()))
        right = solver.t
        dense = solver.dense_output()
        stop = np.searchsorted(query, right, side="right")
        if stop > i:
            values[:, i:stop] = dense(query[i:stop])
            i = stop
    if i != len(query) or not np.isfinite(values).all() or np.any(values < -1e-12 * scale[:, None]):
        raise FloatingPointError("invalid full-ODE audit output")
    return dict(values=values, seconds=time.perf_counter()-started, nfev=int(solver.nfev),
                njev=int(solver.njev), nlu=int(solver.nlu), min_state=check_min,
                internal_states_checked=True)

def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, _ = _groups_from_source()
    archive = np.load(case["reference_path"])
    initial = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    states = [initial]
    curves = []
    candidate_rows = []
    started = time.perf_counter()
    for index in range(STEPS):
        y0 = states[-1]
        maps = _maps(case, y0, groups)
        tick = time.perf_counter()
        candidate, stats = _trial(case, maps, y0, H)
        endpoint = candidate[:, -1].copy()
        curves.append(candidate.copy())
        states.append(endpoint)
        candidate_rows.append(dict(index=index, seconds=time.perf_counter()-tick, stats=stats,
                                   start_state_sha256=sha(y0.tolist()), end_state_sha256=sha(endpoint.tolist())))
    candidate_seconds = time.perf_counter()-started
    save(OUT/"registration.json", dict(date="2026-11-16", model="k-ecoli457", state_index=STATE_INDEX,
        h=H, steps=STEPS, audit_rtol=AUDIT_RTOL, methods=["BDF","Radau"], repeats=3,
        semantics=dict(independent_full_nonlinear_audit=True, internal_states_checked=True,
                       no_clipping=True, candidate_generation_sequential=True),
        scope="Short-window audit backend batch screen; no full-window or M4 qualification.",
        source_sha256={p:digest(ROOT/p) for p in ("research/m4_bdf_audit_batch_screen.py","metabolism/kecoli.py","metabolism/splitting.py","results/m4-growth-factor-screen-v3/runs.json")},
        environment=dict(python=platform.python_version(), numpy=np.__version__)))
    save(OUT/"candidate-runs.json", dict(rows=candidate_rows, candidate_seconds=candidate_seconds))
    np.savez_compressed(OUT/"candidate-states.npz", states=np.column_stack(states), h=H, groups=groups)
    rows=[]
    for method in ("BDF","Radau"):
        for repeat in range(3):
            warm = audit(case, states[0], H, method)
            del warm
            tick=time.perf_counter(); results=[]
            for i in range(STEPS):
                results.append(audit(case, states[i], H, method))
            elapsed=time.perf_counter()-tick
            rows.append(dict(method=method, repeat=repeat, seconds=elapsed,
                             per_task_seconds=[r["seconds"] for r in results],
                             outputs_sha256=[sha(r["values"].tolist()) for r in results],
                             internal_states_checked=all(r["internal_states_checked"] for r in results),
                             nfev=sum(r["nfev"] for r in results), nlu=sum(r["nlu"] for r in results)))
    # Cross-check every BDF output against the corresponding independently run Radau output.
    truth=[]
    for i in range(STEPS):
        truth.append(audit(case, states[i], H, "Radau", rtol=1e-12))
    comparisons=[]
    for i in range(STEPS):
        b=audit(case, states[i], H, "BDF")
        comparisons.append(dict(index=i, max_abs=float(np.max(np.abs(b["values"]-truth[i]["values"]))),
                                 metrics={str(target): _score(case, states[i], curves[i], b["values"], target) for target in (1e-3,1e-5)}))
    save(OUT/"runs.json", dict(rows=rows, comparisons=comparisons))
    med={m:float(np.median([r["seconds"] for r in rows if r["method"]==m])) for m in ("BDF","Radau")}
    summary=dict(scope="Short-window independent audit backend batch screen", candidate_seconds=candidate_seconds,
        median_seconds=med, speedup_radau_over_bdf=med["Radau"]/med["BDF"], comparisons=comparisons,
        all_bdf_targets_pass=all(c["metrics"]["0.001"]["passes"] and c["metrics"]["1e-05"]["passes"] for c in comparisons),
        max_bdf_vs_tight_radau=max(c["max_abs"] for c in comparisons), M4_admission=False,
        interpretation="Backend-only evidence; independent full audits retained.")
    save(OUT/"summary.json", summary)
    files=[OUT/p for p in ("registration.json","candidate-runs.json","candidate-states.npz","runs.json","summary.json")]
    save(OUT/"manifest.json", {str(p.relative_to(ROOT)):digest(p) for p in files})
    (OUT/"README.md").write_text("# BDF audit batch screen\n\nEvery candidate has an independent complete nonlinear audit with internal states checked.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
