"""Micro-screen for a slow-implicit W-MRI stage.

The proposed stage keeps the selected fast reaction columns nonlinear and adds
the slow reaction tangent to the same implicit inner solve:

    z' = f_F(z) + f_S(y) + J_S(y) @ (z-y),  z(0)=y.

This is deliberately a bounded falsification experiment.  It does not alter
the production controller or any historical M4 evidence.  The full state is
integrated because the slow tangent couples inactive coordinates; this makes
the cost consequence explicit rather than silently dropping cross-couplings.

Run with::

    bash scripts/run-macos.sh -m research.m4_wmri_micro_screen
"""

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import Radau
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare


OUT = ROOT / "results/m4-wmri-micro-screen-v1"
SOURCE = ROOT / "results/m4-growth-factor-screen-v3"
STATE_INDEX = 61
TARGETS = (1e-3, 1e-5)
H_VALUES = (1e-6, 1e-5, 3e-5)
REPEATS = (0, 1, 2)
AUDIT_RTOL = 1e-12
CANDIDATE_RTOL = 1e-10


def _groups_from_source():
    data = json.loads((SOURCE / "runs.json").read_text())
    row = next(
        r for r in data["rows"]
        if r["kind"] == "candidate"
        and r["target"] == 1e-3
        and r["growth_safety"] == 0.9
        and r["repeat"] == 0
    )
    groups = next(a["groups"] for a in row["stats"]["attempts"] if a.get("groups"))
    return np.asarray(groups, dtype=int), row


def _affine_formula_check():
    """Verify that the frozen slow tangent is exact for an affine system."""
    fast = np.array([[-2.0, 0.4, 0.0], [0.0, -1.0, 0.2], [0.0, 0.0, -0.5]])
    slow = np.array([[-0.1, 0.0, 0.0], [0.2, -0.3, 0.0], [0.0, 0.1, -0.2]])
    y0 = np.array([0.8, 0.6, 0.4])
    h = 0.7
    truth = expm((fast + slow) * h) @ y0
    slow_base = slow @ y0

    def rhs(t, z):
        del t
        return fast @ z + slow_base + slow @ (z - y0)

    sol = Radau(rhs, 0.0, y0, h, rtol=1e-12, atol=1e-14)
    while sol.status == "running":
        sol.step()
    if sol.status == "failed":
        raise RuntimeError(sol.message or "Affine W-MRI check failed")
    error = float(np.max(np.abs(sol.y - truth)))
    return dict(max_abs_error=error, passed=bool(error <= 1e-10),
                dimension=3, h=h, expected="expm((A_F+A_S)H)y0")


def _manual_radau(rhs, jac, y0, end, query, scale, rtol, timeout=120.0):
    started = time.perf_counter()
    atol = np.maximum(scale * (0.01 * rtol), 1e-18)
    solver = Radau(rhs, 0.0, y0.copy(), float(end), rtol=rtol, atol=atol, jac=jac)
    values = np.empty((len(y0), len(query)))
    index = 0
    minimum = float(y0.min())
    rhs_minimum = float(y0.min())
    internal_states = 0
    while solver.status == "running":
        if time.perf_counter() - started > timeout:
            raise TimeoutError("W-MRI micro-screen solve timeout")
        solver.step()
        if solver.status == "failed":
            raise RuntimeError(solver.message or "Radau failed")
        internal_states += 1
        minimum = min(minimum, float(solver.y.min()))
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12 * scale):
            raise FloatingPointError("negative or nonfinite accepted state")
        dense = solver.dense_output()
        stop = np.searchsorted(query, solver.t, side="right")
        if stop > index:
            values[:, index:stop] = dense(query[index:stop])
            index = stop
    if index != len(query):
        raise RuntimeError("Radau did not cover all requested output points")
    return values, dict(
        seconds=time.perf_counter() - started,
        accepted_steps=int(internal_states),
        nfev=int(solver.nfev),
        njev=int(solver.njev),
        nlu=int(solver.nlu),
        min_internal_state=minimum,
        min_rhs_argument=rhs_minimum,
        internal_states_checked=True,
    )


def _score(case, y0, values, truth, target):
    model, scale, options = case["model"], case["scale"], case["options"]
    flux = np.column_stack([model.flux(0.0, y) for y in values.T])
    truth_flux = np.column_stack([model.flux(0.0, y) for y in truth.T])
    concentration = float(np.max(np.abs(values - truth) /
                                 (scale[:, None] * (0.05 * target))))
    flux_error = float(np.max(np.abs(flux - truth_flux) /
                              (options["flux_scale"][:, None] * (0.05 * target))))
    net_flux = float(np.max(np.abs(options["net_map"] @ (flux - truth_flux)) /
                          (options["net_scale"][:, None] * (0.05 * 1e-3))))
    pools = options["pools"]
    pool_error = float(np.max(np.abs(pools @ (values - y0[:, None]))) /
                       max(1e-12, float(np.max(np.abs(pools @ y0)))))
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(np.all(values >= -1e-12 * scale[:, None]))
    return dict(
        concentration_error=concentration,
        flux_error=flux_error,
        net_flux_error=net_flux,
        pool_error=pool_error,
        finite=finite,
        nonnegative=nonnegative,
        passes=bool(finite and nonnegative and pool_error <= 1e-10 and
                    max(concentration, flux_error, net_flux) <= 1.0),
    )


def _build_maps(case, y0, groups):
    system, model = case["system"], case["model"]
    stoich = csc_matrix(system.S)
    columns = np.concatenate([system.groups[g] for g in groups])
    all_columns = np.arange(stoich.shape[1])
    slow_columns = np.setdiff1d(all_columns, columns)
    fast_matrix = stoich[:, columns]
    slow_matrix = stoich[:, slow_columns]
    gradient = model.flux_jacobian(0.0, y0)
    fast_jacobian = (fast_matrix @ gradient[columns]).tocsc()
    slow_jacobian = (slow_matrix @ gradient[slow_columns]).tocsc()
    slow_base = slow_matrix @ model.flux(0.0, y0)[slow_columns]
    return dict(
        columns=columns,
        slow_columns=slow_columns,
        fast_matrix=fast_matrix,
        slow_matrix=slow_matrix,
        fast_jacobian=fast_jacobian,
        slow_jacobian=slow_jacobian,
        slow_base=np.asarray(slow_base).ravel(),
    )


def _trial(case, maps, y0, h, rtol):
    model, scale = case["model"], case["scale"]
    fast_matrix = maps["fast_matrix"]
    slow_base = maps["slow_base"]
    slow_jacobian = maps["slow_jacobian"]
    fast_jacobian = maps["fast_jacobian"]
    columns = maps["columns"]
    query = h * np.array((0.5, 1.0))
    rhs_calls = [0]
    minimum_rhs = [float(y0.min())]

    def rhs(t, z):
        del t
        if not np.isfinite(z).all() or np.any(z < -1e-12 * scale):
            raise FloatingPointError("negative or nonfinite W-MRI RHS argument")
        rhs_calls[0] += 1
        minimum_rhs[0] = min(minimum_rhs[0], float(z.min()))
        return (fast_matrix @ model.flux(0.0, z)[columns]
                + slow_base + slow_jacobian @ (z - y0))

    def jac(t, z):
        del t
        # The slow tangent is frozen at y0 by construction, while the selected
        # fast reaction block remains genuinely nonlinear throughout the stage.
        return (fast_matrix @ model.flux_jacobian(0.0, z)[columns]).tocsc() + slow_jacobian

    values, stats = _manual_radau(rhs, jac, y0, h, query, scale, rtol)
    stats.update(rhs_calls=rhs_calls[0], min_rhs_argument=minimum_rhs[0],
                 slow_jacobian_nnz=int(slow_jacobian.nnz),
                 fast_jacobian_nnz=int(fast_jacobian.nnz))
    return values, stats


def _audit(case, y0, h):
    model, system, scale = case["model"], case["system"], case["scale"]
    stoich = csc_matrix(system.S)
    query = h * np.array((0.5, 1.0))
    rhs_calls = [0]
    minimum_rhs = [float(y0.min())]

    def rhs(t, z):
        rhs_calls[0] += 1
        if not np.isfinite(z).all() or np.any(z < -1e-12 * scale):
            raise FloatingPointError("negative or nonfinite audit RHS argument")
        minimum_rhs[0] = min(minimum_rhs[0], float(z.min()))
        return stoich @ model.flux(t, z)

    def jac(t, z):
        return stoich @ model.flux_jacobian(t, z)

    values, stats = _manual_radau(rhs, jac, y0, h, query, scale, AUDIT_RTOL)
    stats.update(rhs_calls=rhs_calls[0], min_rhs_argument=minimum_rhs[0])
    return values, stats


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    affine_check = _affine_formula_check()
    if not affine_check["passed"]:
        raise AssertionError(f"Affine W-MRI identity check failed: {affine_check}")
    groups, source_row = _groups_from_source()
    archived = np.load(case["reference_path"])
    y0 = np.asarray(archived["concentrations"][:, STATE_INDEX], dtype=float).copy()
    maps = _build_maps(case, y0, groups)
    baselines = json.loads((SOURCE / "runs.json").read_text())["rows"]
    local_baseline = {
        str(target): float(np.median([
            row["seconds"] for row in baselines
            if row["kind"] == "baseline" and row["target"] == target and not row["warmup"]
        ])) for target in TARGETS
    }
    source_files = (
        "research/m4_wmri_micro_screen.py",
        "metabolism/m3_continuation.py",
        "metabolism/splitting.py",
        "metabolism/kecoli.py",
        "results/m4-growth-factor-screen-v3/runs.json",
    )
    registration = dict(
        date="2026-11-15",
        model="k-ecoli457",
        state_index=STATE_INDEX,
        scope="Bounded single-state W-MRI slow-tangent micro-screen; full-state implicit stage and independent full-ODE Radau audit. No trajectory or M4 qualification.",
        hypothesis="Adding the slow reaction tangent J_S(z-y) to the nonlinear fast stage can stabilize a larger macro interval while preserving the real fast reaction partition.",
        formula="z'=f_F(z)+f_S(y)+J_S(y)(z-y), J_S=S_S Dv_S(y)",
        affine_formula_check=affine_check,
        groups=groups.tolist(),
        fast_reaction_columns=int(len(maps["columns"])),
        total_reaction_columns=int(case["system"].S.shape[1]),
        fast_group_count=int(len(groups)),
        targets=list(TARGETS),
        h_values=list(H_VALUES),
        repeats=list(REPEATS),
        candidate_rtol=CANDIDATE_RTOL,
        audit_method="Radau",
        audit_rtol=AUDIT_RTOL,
        gates=dict(full_state_slow_tangent=True, every_candidate_independently_audited=True,
                   no_clipping=True, rhs_domain_checked=True, pools=True,
                   concentration_flux_net_flux_against_tight_truth=True,
                   costs_include_candidate_and_audit=True),
        source_candidate="results/m4-growth-factor-screen-v3, target=1e-3, growth_safety=0.9, repeat=0",
        source_candidate_seconds=float(source_row["seconds"]),
        local_baselines=local_baseline,
        source_sha256={path: digest(ROOT / path) for path in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    save(OUT / "registration.json", registration)
    rows = []
    for h, repeat in itertools.product(H_VALUES, REPEATS):
        row = dict(model="kecoli", state_index=STATE_INDEX, h=h, repeat=repeat,
                   warmup=False, status="failed")
        started = time.perf_counter()
        try:
            candidate, candidate_stats = _trial(case, maps, y0, h, CANDIDATE_RTOL)
            audit, audit_stats = _audit(case, y0, h)
            row.update(
                status="success",
                seconds=time.perf_counter() - started,
                candidate_seconds=candidate_stats["seconds"],
                audit_seconds=audit_stats["seconds"],
                candidate_stats=candidate_stats,
                audit_stats=audit_stats,
                metrics={str(target): _score(case, y0, candidate, audit, target)
                         for target in TARGETS},
                projected_speedup_vs_baseline_over_1_5={
                    str(target): local_baseline[str(target)] / 1.5 /
                    (candidate_stats["seconds"] + audit_stats["seconds"])
                    for target in TARGETS
                },
            )
            artifact = OUT / f"wmri-h-{h:.17g}-rep-{repeat}.npz"
            np.savez_compressed(artifact, initial=y0, query=h * np.array((0.5, 1.0)),
                                candidate=candidate, audit=audit,
                                scale=case["scale"], groups=groups)
            row["npz"] = artifact.name
            row["npz_sha256"] = digest(artifact)
        except (FloatingPointError, RuntimeError, ValueError, TimeoutError) as exc:
            row.update(seconds=time.perf_counter() - started, reason=str(exc))
        rows.append(row)
        save(OUT / "runs.json", dict(rows=rows))
        print(h, repeat, row["status"], round(row.get("seconds", 0.0), 6),
              {target: row.get("metrics", {}).get(str(target), {}).get("passes")
               for target in TARGETS}, flush=True)

    summary = []
    for h in H_VALUES:
        samples = [r for r in rows if r["h"] == h and r["status"] == "success"]
        summary.append(dict(
            h=h,
            repeats=len(samples),
            candidate_seconds=[r["candidate_seconds"] for r in samples],
            audit_seconds=[r["audit_seconds"] for r in samples],
            total_seconds=[r["seconds"] for r in samples],
            median_total_seconds=float(np.median([r["seconds"] for r in samples])) if samples else None,
            all_targets_pass=len(samples) == len(REPEATS) and all(
                all(r["metrics"][str(target)]["passes"] for target in TARGETS) for r in samples),
            metrics={str(target): [r["metrics"][str(target)] for r in samples] for target in TARGETS},
            speedup_vs_baseline_over_1_5={
                str(target): float(np.median([
                    r["projected_speedup_vs_baseline_over_1_5"][str(target)] for r in samples
                ])) if samples else None for target in TARGETS
            },
        ))
    result = dict(
        scope=registration["scope"],
        formula=registration["formula"],
        affine_formula_check=registration["affine_formula_check"],
        sparse_structure=dict(fast_jacobian_nnz=int(maps["fast_jacobian"].nnz),
                              slow_jacobian_nnz=int(maps["slow_jacobian"].nnz),
                              full_state_dimension=int(len(y0))),
        summary=summary,
        M4_admission=False,
        interpretation=(
            "The slow tangent is mathematically exact for frozen affine systems, "
            "but a real nonlinear stage must integrate all state coordinates because "
            "J_S couples inactive coordinates. This micro-screen records whether the "
            "larger step offsets its full-state solve plus mandatory audit cost; it "
            "does not establish full-window or seven-repeat admission."
        ),
        baseline_over_1_5={str(target): local_baseline[str(target)] / 1.5 for target in TARGETS},
        local_baselines=local_baseline,
    )
    save(OUT / "summary.json", result)
    files = [OUT / "registration.json", OUT / "runs.json", OUT / "summary.json"]
    files.extend(OUT / row["npz"] for row in rows if row.get("npz"))
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    (OUT / "README.md").write_text(
        "# W-MRI slow-tangent micro-screen\n\n"
        "This bounded experiment adds the slow reaction tangent to the nonlinear fast stage. "
        "Every candidate is independently audited with a full nonlinear Radau solve; all "
        "NPZs, failed rows and hashes are retained. The result is diagnostic evidence only "
        "and cannot qualify M4.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
