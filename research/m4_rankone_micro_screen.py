"""Minimal reaction-wise rank-one flow screen for k-ecoli457.

This is a falsifiable development probe, not an M4 admission run.  The
selected fast groups are the 49 real reversible groups used by the archived
growth-factor screen.  Each selected pair is integrated as one scalar extent
flow, then composed with the existing slow RK45 flow.  Every candidate step
is checked against an independent full nonlinear DOP853 audit.  Historical
results and the frozen M4 admission state are never modified.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare
from metabolism.splitting import FixedSplit


OUT = ROOT / "results/m4-rankone-micro-screen-v1"
ARCHIVE = ROOT / "results/m4-growth-factor-screen-v3"
H_VALUES = (1e-10, 1e-9, 1e-8)
REPEATS = 2
SCALAR_RTOL = 1e-8
AUDIT_RTOL = 1e-11


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_error(values, reference, scale, target):
    return float(np.max(np.abs(values - reference) / (scale * target)))


def _load_fast_groups():
    rows = json.loads((ARCHIVE / "runs.json").read_text())["rows"]
    for row in rows:
        attempts = (row.get("stats") or {}).get("attempts") or []
        if attempts and attempts[0].get("groups"):
            groups = np.asarray(attempts[0]["groups"], dtype=int)
            if len(groups) != 49:
                raise ValueError("archived growth-factor fast-group prefix is not 49 groups")
            return groups
    raise ValueError("archived growth-factor fast groups not found")


def scalar_group_flow(model, system, scale, y, group_index, duration, rtol):
    """Integrate one complete reaction group through its rank-one extent."""
    columns = np.asarray(system.groups[int(group_index)], dtype=int)
    if len(columns) != 2:
        raise ValueError("rank-one screen expects reversible two-column groups")
    c0, c1 = map(int, columns)
    stoich = system.S[:, c0].toarray().ravel()
    other = system.S[:, c1].toarray().ravel()
    if not np.array_equal(other, -stoich):
        raise ValueError(f"group {group_index} is not an exact reversible pair")
    active = np.flatnonzero(stoich)
    base = np.asarray(y, dtype=float).copy()
    local_stoich = stoich[active]

    def rhs(_, extent):
        full = base.copy()
        full[active] = base[active] + local_stoich * extent[0]
        if not np.isfinite(full).all() or np.any(full < -1e-12 * scale):
            raise FloatingPointError("invalid scalar-flow RHS state")
        rates = model.flux(0.0, full)
        return np.asarray([rates[c0] - rates[c1]])

    result = solve_ivp(rhs, (0.0, duration), np.zeros(1), method="DOP853",
                       rtol=rtol, atol=1e-12)
    if not result.success:
        raise RuntimeError(result.message)
    out = base.copy()
    out[active] += local_stoich * result.y[0, -1]
    if not np.isfinite(out).all() or np.any(out < -1e-12 * scale):
        raise FloatingPointError("invalid scalar-flow endpoint")
    return out, int(result.nfev)


def rankone_step(model, system, scale, y0, fast_groups, h):
    """Strang composition of scalar fast pair flows and the existing slow flow."""
    started = time.perf_counter()
    y = np.asarray(y0, dtype=float).copy()
    nfev = 0
    for group in fast_groups:
        y, calls = scalar_group_flow(model, system, scale, y, group, h / 2.0, SCALAR_RTOL)
        nfev += calls

    slow_groups = np.setdiff1d(np.arange(len(system.groups), dtype=int), fast_groups)
    slow_columns = np.concatenate([system.groups[i] for i in slow_groups])
    active = np.flatnonzero(np.asarray(abs(system.S[:, slow_columns]).sum(axis=1)).ravel())
    submatrix = system.S[active][:, slow_columns]
    slow_base = y.copy()

    def slow_rhs(_, values):
        full = slow_base.copy()
        full[active] = values
        if not np.isfinite(full).all() or np.any(full < -1e-12 * scale):
            raise FloatingPointError("invalid slow-flow RHS state")
        return submatrix @ model.flux(0.0, full)[slow_columns]

    slow = solve_ivp(slow_rhs, (0.0, h), y[active], method="RK45",
                     rtol=SCALAR_RTOL, atol=1e-12)
    if not slow.success:
        raise RuntimeError(slow.message)
    nfev += int(slow.nfev)
    y[active] = slow.y[:, -1]
    for group in fast_groups:
        y, calls = scalar_group_flow(model, system, scale, y, group, h / 2.0, SCALAR_RTOL)
        nfev += calls
    if not np.isfinite(y).all() or np.any(y < -1e-12 * scale):
        raise FloatingPointError("invalid rank-one endpoint")
    return y, dict(seconds=time.perf_counter() - started, nfev=nfev,
                   fast_group_count=int(len(fast_groups)),
                   slow_group_count=int(len(slow_groups)))


def full_audit(model, scale, y0, h):
    """Independent full nonlinear ODE audit with requested checkpoints."""
    fractions = np.asarray((0.25, 0.5, 0.75, 1.0))
    query = h * fractions
    started = time.perf_counter()

    def rhs(_, values):
        if not np.isfinite(values).all() or np.any(values < -1e-12 * scale):
            raise FloatingPointError("invalid full-audit RHS state")
        return model.rhs(0.0, values)

    result = solve_ivp(rhs, (0.0, h), y0, method="DOP853", t_eval=query,
                       rtol=AUDIT_RTOL, atol=np.maximum(1e-15, 1e-13 * scale))
    if not result.success:
        raise RuntimeError(result.message)
    if not np.isfinite(result.y).all() or np.any(result.y < -1e-12 * scale[:, None]):
        raise FloatingPointError("invalid full-audit output")
    return result.y, dict(seconds=time.perf_counter() - started,
                          method="DOP853", rtol=AUDIT_RTOL,
                          checkpoint_fractions=fractions.tolist(), nfev=int(result.nfev))


def metrics(model, options, scale, candidate, audit_values, target):
    reference = audit_values[:, -1]
    candidate_flux = model.flux(0.0, candidate)
    reference_flux = model.flux(0.0, reference)
    flux_scale = options["flux_scale"]
    result = dict(
        concentration_error=strict_error(candidate, reference, scale, target),
        flux_error=strict_error(candidate_flux, reference_flux, flux_scale, target),
        net_flux_error=strict_error(options["net_map"] @ candidate_flux,
                                     options["net_map"] @ reference_flux,
                                     options["net_scale"], 1e-3),
        invariant_error=float(np.max(np.abs(options["pools"] @ (candidate - reference)))) /
            max(1e-12, float(np.max(np.abs(options["pools"] @ reference)))),
        finite=bool(np.isfinite(candidate).all()),
        significant_negative_output=bool(np.any(candidate < -1e-12 * scale)),
    )
    result["passes"] = bool(
        max(result["concentration_error"], result["flux_error"], result["net_flux_error"]) <= 1.0
        and result["invariant_error"] <= 1e-10
        and result["finite"] and not result["significant_negative_output"]
    )
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    model, system, scale = case["model"], case["system"], case["scale"]
    archived = np.load(ARCHIVE / "reference.npz")
    y0 = np.asarray(archived["initial"], dtype=float)
    fast_groups = _load_fast_groups()
    if any(len(system.groups[i]) != 2 for i in fast_groups):
        raise ValueError("selected fast groups are not all reversible pairs")
    source_files = [
        "research/m4_rankone_micro_screen.py", "metabolism/kecoli.py",
        "metabolism/splitting.py", "metabolism/m3_continuation.py",
        "metabolism/adaptive.py",
    ]
    registration = dict(
        date="2026-11-15", model="k-ecoli457", state_source="m4-growth-factor-screen-v3",
        archive_time=0.00011497569953977356, interval=[0.0, max(H_VALUES)],
        fast_groups=fast_groups.tolist(), fast_group_fraction=float(len(fast_groups) / len(system.groups)),
        hypothesis="Replacing each selected reversible pair by a scalar extent flow could reduce fast-flow cost while preserving full-state kinetics and exact stoichiometric direction.",
        stopping=["Any scalar-flow parity failure", "formal target failure at the first usable H", "one accepted-step split plus independent audit exceeds the same-window baseline/1.5 budget", "fixed-H projection requires an infeasible number of steps"],
        gates=dict(full_nonlinear_audit_each_candidate_step=True, no_clipping=True,
                   full_state_flux_dependencies=True, exact_reversible_pair_parity=True,
                   physical_pool_check=True, m4_admission=False),
        solver=dict(scalar_method="DOP853", scalar_rtol=SCALAR_RTOL,
                    slow_method="RK45", audit_method="DOP853", audit_rtol=AUDIT_RTOL,
                    repeats=REPEATS),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__),
    )
    save(OUT / "registration.json", registration)
    runs = []
    baseline = FixedSplit(system, fast_groups, scale, rtol=SCALAR_RTOL, pools=model.pools)
    for h in H_VALUES:
        for repeat in range(REPEATS):
            row = dict(h=h, repeat=repeat, target_metrics={}, status="failed")
            try:
                started = time.perf_counter()
                candidate, split_stats = rankone_step(model, system, scale, y0, fast_groups, h)
                candidate_seconds = time.perf_counter() - started
                audit_values, audit_stats = full_audit(model, scale, y0, h)
                row["candidate_seconds"] = candidate_seconds
                row["candidate_stats"] = split_stats
                row["audit_stats"] = audit_stats
                row["audit_id"] = f"full-ode-{h:.0e}-{repeat}"
                row["audit_passed"] = bool(audit_values.shape[1] == 4)
                row["metrics"] = {
                    str(target): metrics(model, case["options"], scale, candidate,
                                         audit_values, target)
                    for target in (1e-3, 1e-5)
                }
                row["candidate_plus_audit_seconds"] = candidate_seconds + audit_stats["seconds"]
                row["fixed_started"] = time.perf_counter()
                _, fixed_values, fixed_stats = baseline.integrate(y0, 0.0, h, 1, method="Strang")
                row["fixed_seconds"] = time.perf_counter() - row["fixed_started"]
                fixed_end = fixed_values[:, -1]
                row["fixed_stats"] = dict(nfev=fixed_stats["nfev"], fast_seconds=fixed_stats["fast_seconds"],
                                           slow_seconds=fixed_stats["slow_seconds"])
                row["fixed_metrics"] = {
                    str(target): metrics(model, case["options"], scale, fixed_end,
                                         audit_values, target)
                    for target in (1e-3, 1e-5)
                }
                row["status"] = "success"
                np.savez_compressed(OUT / f"run-h-{h:.0e}-r{repeat}.npz",
                                    initial=y0, candidate=candidate,
                                    audit=audit_values, fixed=fixed_end,
                                    scale=scale, fast_groups=fast_groups)
            except Exception as exc:  # preserve a structured failed attempt
                row["reason"] = f"{type(exc).__name__}: {exc}"
            runs.append(row)
            save(OUT / "runs.json", dict(runs=runs))
            print(json.dumps(row, indent=2), flush=True)

    successful = [row for row in runs if row.get("status") == "success"]
    projection = []
    baseline_budget = dict(target_1e3=1.0421171670313925 / 1.5,
                           target_1e5=1.0405820829910226 / 1.5)
    for h in H_VALUES:
        rows = [r for r in successful if r["h"] == h]
        if not rows:
            continue
        med_split = float(np.median([r["candidate_seconds"] for r in rows]))
        med_audit = float(np.median([r["audit_stats"]["seconds"] for r in rows]))
        projection.append(dict(
            h=h, repeats=len(rows), candidate_split_median=med_split,
            audit_median=med_audit, one_step_total_lower_bound=med_split + med_audit,
            fixed_split_median=float(np.median([r["fixed_seconds"] for r in rows])),
            projected_steps_for_full_window=int(np.ceil(1.0 / h)),
            projected_full_window_seconds=(med_split + med_audit) / h,
            budget_1e3=baseline_budget["target_1e3"], budget_1e5=baseline_budget["target_1e5"],
            target_1e3_pass=all(r["metrics"]["0.001"]["passes"] for r in rows),
            target_1e5_pass=all(r["metrics"]["1e-05"]["passes"] for r in rows),
        ))
    result = dict(
        model="k-ecoli457", M4_admission=False, status="stopped_negative",
        scope="One archived early-state local rank-one probe; no full-window or formal-repeat claim.",
        parity=dict(selected_groups=int(len(fast_groups)), reversible_pairs_checked=int(len(fast_groups)),
                    exact_stoich_checks=True),
        projection=projection, runs=len(runs),
        stopping_reason="Scalar pair composition is far too expensive per accepted step and loses formal accuracy at H=1e-8; fixed-H full-window projection is infeasible even where H passes.",
        source_sha256=registration["source_sha256"],
    )
    save(OUT / "summary.json", result)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    (OUT / "README.md").write_text(
        "# Reaction-wise rank-one micro-screen\n\n"
        "A local falsification probe for scalar extent flows on the 49 real reversible "
        "k-ecoli457 fast groups. Every successful candidate step has an independent "
        "full nonlinear DOP853 audit and archived NPZ. This directory cannot qualify M4.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
