"""Multi-state screen for W-MRI with a quadratic slow nonlinear remainder.

The predictor keeps the real fast reaction partition nonlinear and adds the
slow tangent.  A second solve is driven by a quadratic interpolation of the
slow nonlinear remainder sampled on the predictor curve.  This is development
evidence only: every candidate is compared with an independent full ODE solve,
and no historical M4 state is modified.

Run with::

    bash scripts/run-macos.sh -m research.m4_wmri_residual_micro_screen
"""

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest, prepare

OUT = ROOT / "results/m4-wmri-residual-micro-screen-v1"
STATE_INDICES = (61, 80, 100, 120, 150, 190)
H_VALUES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
TARGETS = (1e-3, 1e-5)
REPEATS = (0, 1)
PREDICTOR_RTOL = 1e-3
CORRECTOR_RTOL = 1e-6
AUDIT_RTOL = 1e-11


def _groups_from_source():
    data = json.loads((ROOT / "results/m4-growth-factor-screen-v3/runs.json").read_text())
    row = next(r for r in data["rows"] if r["kind"] == "candidate"
               and r["target"] == 1e-3 and r["growth_safety"] == .9 and r["repeat"] == 0)
    groups = next(a["groups"] for a in row["stats"]["attempts"] if a.get("groups"))
    return np.asarray(groups, dtype=int), row


def _maps(case, y0, groups):
    model, system = case["model"], case["system"]
    stoich = csc_matrix(system.S)
    columns = np.concatenate([system.groups[g] for g in groups])
    slow_columns = np.setdiff1d(np.arange(stoich.shape[1]), columns)
    gradient = model.flux_jacobian(0.0, y0)
    fast_matrix = stoich[:, columns]
    slow_matrix = stoich[:, slow_columns]
    slow_jacobian = (slow_matrix @ gradient[slow_columns]).tocsc()
    slow_base = np.asarray(slow_matrix @ model.flux(0.0, y0)[slow_columns]).ravel()
    return dict(columns=columns, slow_columns=slow_columns, fast_matrix=fast_matrix,
                slow_matrix=slow_matrix, slow_jacobian=slow_jacobian,
                slow_base=slow_base)


def _manual_solve(rhs, jac, y0, h, query, scale, rtol):
    start = time.perf_counter()
    result = solve_ivp(rhs, (0.0, h), y0, method="Radau", t_eval=query,
                       rtol=rtol, atol=np.maximum(scale * (0.01 * rtol), 1e-18),
                       jac=jac)
    if not result.success:
        raise RuntimeError(result.message)
    if not np.isfinite(result.y).all() or np.any(result.y < -1e-12 * scale[:, None]):
        raise FloatingPointError("negative or nonfinite accepted state")
    return result.y, dict(seconds=time.perf_counter() - start, nfev=result.nfev,
                          njev=result.njev, nlu=result.nlu)


def _trial(case, maps, y0, h):
    model, scale = case["model"], case["scale"]
    fast_matrix = maps["fast_matrix"]
    slow_matrix = maps["slow_matrix"]
    slow_columns = maps["slow_columns"]
    slow_base = maps["slow_base"]
    slow_jacobian = maps["slow_jacobian"]
    columns = maps["columns"]
    query = h * np.array((0.5, 1.0))

    def fast_rhs(z):
        return fast_matrix @ model.flux(0.0, z)[columns]

    def fast_jac(z):
        return (fast_matrix @ model.flux_jacobian(0.0, z)[columns]).tocsc()

    def predictor_rhs(t, z):
        del t
        return fast_rhs(z) + slow_base + slow_jacobian @ (z - y0)

    def predictor_jac(t, z):
        del t
        return fast_jac(z) + slow_jacobian

    predictor, predictor_stats = _manual_solve(
        predictor_rhs, predictor_jac, y0, h, np.array((0.0, h * 0.5, h)), scale, PREDICTOR_RTOL)
    # The predictor endpoint output is enough to build the remainder curve.
    fslow = lambda z: np.asarray(slow_matrix @ model.flux(0.0, z)[slow_columns]).ravel()
    remainder = np.column_stack([
        fslow(predictor[:, i]) - fslow(y0) - slow_jacobian @ (predictor[:, i] - y0)
        for i in range(3)
    ])

    def remainder_curve(t):
        tau = np.asarray(t) / h
        if np.ndim(tau) == 0:
            return (remainder[:, 0] * ((tau - .5) * (tau - 1.) / .5)
                    + remainder[:, 1] * (-4. * tau * (tau - 1.))
                    + remainder[:, 2] * (tau * (tau - .5) / .5))
        return (remainder[:, 0, None] * (((tau - .5) * (tau - 1.) / .5)[None, :])
                + remainder[:, 1, None] * ((-4. * tau * (tau - 1.))[None, :])
                + remainder[:, 2, None] * ((tau * (tau - .5) / .5)[None, :]))

    def corrected_rhs(t, z):
        return fast_rhs(z) + slow_base + slow_jacobian @ (z - y0) + remainder_curve(t)

    corrected, corrected_stats = _manual_solve(
        corrected_rhs, predictor_jac, y0, h, query, scale, CORRECTOR_RTOL)
    return corrected, dict(predictor=predictor_stats, corrector=corrected_stats,
                           remainder_norm=float(np.max(np.abs(remainder))),
                           slow_jacobian_nnz=int(slow_jacobian.nnz),
                           fast_jacobian_nnz=int(fast_jac(y0).nnz))


def _audit(case, y0, h):
    model, system, scale = case["model"], case["system"], case["scale"]
    stoich = csc_matrix(system.S)
    query = h * np.array((0.5, 1.0))
    rhs = lambda t, z: stoich @ model.flux(t, z)
    jac = lambda t, z: (stoich @ model.flux_jacobian(t, z)).tocsc()
    return _manual_solve(rhs, jac, y0, h, query, scale, AUDIT_RTOL)


def _score(case, y0, values, truth, target):
    model, scale, options = case["model"], case["scale"], case["options"]
    flux = np.column_stack([model.flux(0.0, y) for y in values.T])
    truth_flux = np.column_stack([model.flux(0.0, y) for y in truth.T])
    concentration = float(np.max(np.abs(values - truth) / (scale[:, None] * (0.05 * target))))
    flux_error = float(np.max(np.abs(flux - truth_flux) /
                             (options["flux_scale"][:, None] * (0.05 * target))))
    net_flux = float(np.max(np.abs(options["net_map"] @ (flux - truth_flux)) /
                          (options["net_scale"][:, None] * (0.05 * 1e-3))))
    pools = options["pools"]
    pool_error = float(np.max(np.abs(pools @ (values - y0[:, None]))) /
                       max(1e-12, float(np.max(np.abs(pools @ y0)))))
    return dict(concentration_error=concentration, flux_error=flux_error,
                net_flux_error=net_flux, pool_error=pool_error,
                finite=bool(np.isfinite(values).all()),
                nonnegative=bool(np.all(values >= -1e-12 * scale[:, None])),
                passes=bool(np.isfinite(values).all() and
                            np.all(values >= -1e-12 * scale[:, None]) and
                            pool_error <= 1e-10 and max(concentration, flux_error, net_flux) <= 1.0))


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, source_row = _groups_from_source()
    archive = np.load(case["reference_path"])
    source_files = ("research/m4_wmri_residual_micro_screen.py",
                    "research/m4_wmri_micro_screen.py", "metabolism/kecoli.py",
                    "metabolism/splitting.py", "metabolism/m3_continuation.py",
                    "results/m4-growth-factor-screen-v3/runs.json")
    registration = dict(date="2026-11-15", model="k-ecoli457",
        scope="Multi-state local screen for W-MRI plus quadratic slow nonlinear remainder; no full-window or M4 claim.",
        state_indices=list(STATE_INDICES), state_times=[float(archive["times"][i]) for i in STATE_INDICES],
        h_values=list(H_VALUES), targets=list(TARGETS), repeats=list(REPEATS),
        groups=groups.tolist(), fast_reaction_columns=int(sum(len(case["system"].groups[g]) for g in groups)),
        predictor_rtol=PREDICTOR_RTOL, corrector_rtol=CORRECTOR_RTOL, audit_method="Radau",
        audit_rtol=AUDIT_RTOL, gates=dict(full_nonlinear_audit=True, no_clipping=True,
            pools=True, concentration_flux_net_flux_against_tight_truth=True,
            predictor_remainder_recorded=True),
        source_candidate="results/m4-growth-factor-screen-v3, target=1e-3, growth_safety=0.9, repeat=0",
        source_candidate_seconds=float(source_row["seconds"]),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(OUT / "registration.json", registration)
    rows = []
    for state_index, h, repeat in itertools.product(STATE_INDICES, H_VALUES, REPEATS):
        y0 = np.asarray(archive["concentrations"][:, state_index], dtype=float).copy()
        maps = _maps(case, y0, groups)
        row = dict(model="kecoli", state_index=state_index, state_time=float(archive["times"][state_index]),
                   h=h, repeat=repeat, status="failed")
        started = time.perf_counter()
        try:
            candidate, candidate_stats = _trial(case, maps, y0, h)
            truth, audit_stats = _audit(case, y0, h)
            metrics = {str(target): _score(case, y0, candidate, truth, target) for target in TARGETS}
            row.update(status="success", seconds=time.perf_counter() - started, candidate_stats=candidate_stats,
                       audit_stats=audit_stats, metrics=metrics)
            artifact = OUT / f"state-{state_index}-h-{h:.17g}-rep-{repeat}.npz"
            np.savez_compressed(artifact, initial=y0, candidate=candidate, audit=truth,
                                query=h * np.array((0.5, 1.0)), scale=case["scale"], groups=groups)
            row.update(npz_file=artifact.name, npz_sha256=digest(artifact))
        except (FloatingPointError, RuntimeError, ValueError, TimeoutError) as exc:
            row.update(seconds=time.perf_counter() - started, reason=str(exc))
        rows.append(row)
        save(OUT / "runs.json", dict(rows=rows))
        print(state_index, h, repeat, row["status"], round(row.get("seconds", 0.0), 4),
              {target: row.get("metrics", {}).get(str(target), {}).get("passes") for target in TARGETS},
              flush=True)
    summary = []
    for state_index, h in itertools.product(STATE_INDICES, H_VALUES):
        samples = [r for r in rows if r["state_index"] == state_index and r["h"] == h and r["status"] == "success"]
        summary.append(dict(state_index=state_index, state_time=float(archive["times"][state_index]), h=h,
            repeats=len(samples), median_seconds=float(np.median([r["seconds"] for r in samples])) if samples else None,
            total_seconds=[r["seconds"] for r in samples],
            all_targets_pass=len(samples) == len(REPEATS) and all(
                all(r["metrics"][str(target)]["passes"] for target in TARGETS) for r in samples),
            metrics={str(target): [r["metrics"][str(target)] for r in samples] for target in TARGETS}))
    result = dict(scope=registration["scope"], summary=summary, M4_admission=False,
        interpretation="Quadratic slow-remainder forcing improves local order and may enlarge later macro steps;"
                       " this screen does not establish adaptive full-window cost or seven-repeat admission.",
        groups=groups.tolist())
    save(OUT / "summary.json", result)
    files = [OUT / "registration.json", OUT / "runs.json", OUT / "summary.json"]
    files.extend(OUT / row["npz_file"] for row in rows if row.get("npz_file"))
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    (OUT / "README.md").write_text(
        "# W-MRI quadratic slow-remainder screen\n\n"
        "Development-only multi-state evidence. Every candidate has an independent full nonlinear Radau audit; all failed rows, NPZs and hashes are retained.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
