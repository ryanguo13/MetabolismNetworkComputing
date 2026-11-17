"""Reduced-coordinate fast-flow micro-screen for real k-ecoli457 reactions.

The selected fast reactions are represented by reaction extents xi rather than
integrating all active species directly: y = y0 + S_F xi and xi' = v_F(y).
This is an exact coordinate representation of the same nonlinear fast ODE (up
to redundant extent directions), not a kinetic approximation.  The slow
complement and the independent full nonlinear audit remain explicit.

This is a development screen only.  It writes an independent result directory
and never modifies the frozen M4 admission state or historical evidence.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, csr_matrix

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare

OUT = ROOT / "results/m4-reduced-fast-extent-micro-screen-v4"
ARCHIVE = ROOT / "results/m4-growth-factor-screen-v3"
STATE_INDEX = 61
H_VALUES = (1e-6, 3e-6, 1e-5, 3e-5)
REPEATS = 2
FAST_RTOL = 1e-8
SLOW_RTOL = 1e-8
AUDIT_RTOL = 1e-11


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()


def load_fast_groups():
    rows = json.loads((ARCHIVE / "runs.json").read_text())["rows"]
    row = next(r for r in rows if r.get("kind") == "candidate"
               and r.get("target") == 1e-3 and r.get("growth_safety") == .9
               and r.get("repeat") == 0)
    groups = next(a["groups"] for a in row["stats"]["attempts"] if a.get("groups"))
    return np.asarray(groups, dtype=int), row


def full_audit(case, y0, h):
    model, scale = case["model"], case["scale"]
    query = h * np.array((0.5, 1.0))

    def rhs(_, y):
        if not np.isfinite(y).all() or np.any(y < -1e-12 * scale):
            raise FloatingPointError("invalid full-audit state")
        return model.rhs(0.0, y)

    def jac(_, y):
        return model.jacobian(0.0, y)

    started = time.perf_counter()
    result = solve_ivp(rhs, (0., h), y0.copy(), method="Radau", t_eval=query,
                       rtol=AUDIT_RTOL, atol=np.maximum(scale * (0.01 * AUDIT_RTOL), 1e-18),
                       jac=jac)
    if not result.success:
        raise RuntimeError(result.message)
    if not np.isfinite(result.y).all() or np.any(result.y < -1e-12 * scale[:, None]):
        raise FloatingPointError("invalid full-audit output")
    return result.y, dict(seconds=time.perf_counter() - started, nfev=int(result.nfev),
                          njev=int(result.njev), nlu=int(result.nlu), internal_states_checked=True)


def score(case, y0, candidate, truth, target):
    model, scale, options = case["model"], case["scale"], case["options"]
    values = np.column_stack((candidate[:, 0], candidate[:, 1]))
    flux = np.column_stack([model.flux(0., y) for y in values.T])
    truth_flux = np.column_stack([model.flux(0., y) for y in truth.T])
    concentration = float(np.max(np.abs(values - truth) / (scale[:, None] * (0.05 * target))))
    flux_error = float(np.max(np.abs(flux - truth_flux) /
                             (options["flux_scale"][:, None] * (0.05 * target))))
    net_flux = float(np.max(np.abs(options["net_map"] @ (flux - truth_flux)) /
                        (options["net_scale"][:, None] * (0.05 * 1e-3))))
    pool_error = float(np.max(np.abs(options["pools"] @ (values - truth)))) / max(
        1e-12, float(np.max(np.abs(options["pools"] @ truth))))
    return dict(concentration_error=concentration, flux_error=flux_error,
                net_flux_error=net_flux, pool_error=pool_error,
                finite=bool(np.isfinite(values).all()),
                nonnegative=bool(np.all(values >= -1e-12 * scale[:, None])),
                passes=bool(np.isfinite(values).all() and
                            np.all(values >= -1e-12 * scale[:, None]) and
                            pool_error <= 1e-10 and max(concentration, flux_error, net_flux) <= 1.0))


def reduced_fast_flow(case, y0, fast_columns, basis_columns, basis_matrix, h):
    model, scale = case["model"], case["scale"]
    m = len(basis_columns)
    xi0 = np.zeros(m, dtype=float)
    calls = 0

    def reconstruct(xi):
        return y0 + basis_matrix @ xi

    def rhs(_, xi):
        nonlocal calls
        calls += 1
        y = reconstruct(xi)
        if not np.isfinite(y).all() or np.any(y < -1e-12 * scale):
            raise FloatingPointError("invalid reduced fast RHS state")
        rates = model.flux(0.0, y)
        return rates[fast_columns[0::2]] - rates[fast_columns[1::2]]

    def jac(_, xi):
        y = reconstruct(xi)
        if not np.isfinite(y).all() or np.any(y < -1e-12 * scale):
            raise FloatingPointError("invalid reduced fast Jacobian state")
        gradient = model.flux_jacobian(0.0, y)
        return ((gradient[fast_columns[0::2], :] - gradient[fast_columns[1::2], :])
                @ basis_matrix).toarray()

    started = time.perf_counter()
    result = solve_ivp(rhs, (0., h), xi0, method="DOP853",
                       rtol=FAST_RTOL, atol=1e-10, max_step=h / 100., jac=jac)
    if not result.success:
        raise RuntimeError(result.message)
    endpoint = reconstruct(result.y[:, -1])
    if not np.isfinite(endpoint).all() or np.any(endpoint < -1e-12 * scale):
        raise FloatingPointError("invalid reduced fast endpoint")
    return endpoint, dict(seconds=time.perf_counter() - started, nfev=int(result.nfev),
                          njev=int(result.njev), nlu=int(result.nlu), rhs_calls=calls,
                          coordinate_dimension=m)


def slow_flow(case, y0, slow_columns, slow_matrix, h):
    model, scale = case["model"], case["scale"]
    active = np.flatnonzero(np.asarray(abs(slow_matrix).sum(axis=1)).ravel())
    submatrix = slow_matrix[active, :].tocsc()
    base = y0.copy()

    def rhs(_, z):
        full = base.copy()
        full[active] = z
        if not np.isfinite(full).all() or np.any(full < -1e-12 * scale):
            raise FloatingPointError("invalid reduced-split slow RHS state")
        return submatrix @ model.flux(0., full)[slow_columns]

    started = time.perf_counter()
    result = solve_ivp(rhs, (0., h), base[active], method="RK45",
                       rtol=SLOW_RTOL, atol=np.maximum(scale[active] * (0.01 * SLOW_RTOL), 1e-18))
    if not result.success:
        raise RuntimeError(result.message)
    out = base.copy()
    out[active] = result.y[:, -1]
    if not np.isfinite(out).all() or np.any(out < -1e-12 * scale):
        raise FloatingPointError("invalid reduced-split slow endpoint")
    return out, dict(seconds=time.perf_counter() - started, nfev=int(result.nfev),
                     active_states=int(len(active)))


def step(case, y0, fast_columns, basis_columns, h):
    system = case["system"]
    basis_matrix = csc_matrix(system.S[:, basis_columns])
    slow_columns = np.setdiff1d(np.arange(system.S.shape[1]), fast_columns)
    slow_matrix = csc_matrix(system.S[:, slow_columns])
    started = time.perf_counter()
    first, fast1 = reduced_fast_flow(case, y0, fast_columns, basis_columns, basis_matrix, h / 2.)
    middle, slow = slow_flow(case, first, slow_columns, slow_matrix, h)
    final, fast2 = reduced_fast_flow(case, middle, fast_columns, basis_columns, basis_matrix, h / 2.)
    return final, dict(seconds=time.perf_counter() - started, fast_first=fast1, slow=slow,
                       fast_second=fast2, fast_columns=int(len(fast_columns)),
                       fast_reaction_fraction=float(len(fast_columns) / system.S.shape[1]),
                       fast_coordinate_dimension=int(len(basis_columns)))


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, source_row = load_fast_groups()
    system = case["system"]
    fast_columns = np.concatenate([system.groups[int(i)] for i in groups])
    if len(fast_columns) != 2 * len(groups):
        raise ValueError("extent screen requires two-column fast groups")
    pair_stoich = np.asarray(system.S[:, fast_columns[0::2]].toarray())
    reverse_stoich = np.asarray(system.S[:, fast_columns[1::2]].toarray())
    if not np.array_equal(reverse_stoich, -pair_stoich):
        raise ValueError("extent screen requires exact reversible pairs")
    basis_columns = fast_columns[0::2]
    if len(np.unique(fast_columns)) != len(fast_columns):
        raise ValueError("fast groups overlap")
    archive = np.load(case["reference_path"])
    y0 = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    registration = dict(date="2026-11-15", model="k-ecoli457", state_index=STATE_INDEX,
                        h_values=list(H_VALUES), repeats=list(range(REPEATS)), groups=groups.tolist(),
                        fast_columns=fast_columns.tolist(), basis_columns=basis_columns.tolist(),
                        fast_reaction_fraction=float(len(fast_columns) / system.S.shape[1]),
                        method="joint reduced reaction-extents fast flow + full-state slow complement",
                        hypothesis="Joint extent coordinates reduce fast-flow cost versus 3003-state or sequential pair solves without approximating kinetics.",
                        gates=dict(full_nonlinear_audit_each_candidate=True, no_clipping=True,
                                   pools=True, concentration_flux_net_flux=True, m4_admission=False),
                        solver=dict(fast_method="DOP853", fast_rtol=FAST_RTOL, slow_method="RK45",
                                    slow_rtol=SLOW_RTOL, audit_method="Radau", audit_rtol=AUDIT_RTOL),
                        source_sha256={p: digest(ROOT / p) for p in (
                            "research/m4_reduced_fast_extent_micro_screen.py",
                            "research/m4_rankone_micro_screen.py",
                            "metabolism/kecoli.py", "metabolism/splitting.py",
                            "metabolism/m3_continuation.py",
                            "results/m4-growth-factor-screen-v3/runs.json")},
                        environment=dict(platform=platform.platform(), python=platform.python_version(),
                                         numpy=np.__version__))
    save(OUT / "registration.json", registration)
    rows = []
    for h in H_VALUES:
        for repeat in range(REPEATS):
            row = dict(h=h, repeat=repeat, status="failed")
            started = time.perf_counter()
            try:
                candidate, split_stats = step(case, y0, fast_columns, basis_columns, h)
                truth, audit_stats = full_audit(case, y0, h)
                row.update(status="success", seconds=time.perf_counter() - started,
                           split_stats=split_stats, audit_stats=audit_stats,
                           metrics={str(t): score(case, y0,
                                                   np.column_stack((y0, candidate)), truth, t)
                                    for t in (1e-3, 1e-5)},
                           candidate_sha256=canonical(candidate.tolist()),
                           audit_sha256=canonical(truth.tolist()))
                np.savez_compressed(OUT / f"h-{h:.0e}-r{repeat}.npz", initial=y0,
                                    candidate=candidate, audit=truth, h=h, groups=groups,
                                    fast_columns=fast_columns)
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                row.update(seconds=time.perf_counter() - started, reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            save(OUT / "runs.json", dict(rows=rows))
            print(h, repeat, row["status"], round(row.get("seconds", 0.), 4),
                  {t: row.get("metrics", {}).get(str(t), {}).get("passes") for t in (1e-3, 1e-5)},
                  flush=True)
    summary = []
    for h in H_VALUES:
        samples = [r for r in rows if r["h"] == h and r["status"] == "success"]
        summary.append(dict(h=h, repeats=len(samples), median_seconds=float(np.median([r["seconds"] for r in samples]))
                            if samples else None, all_targets_pass=bool(samples) and len(samples) == REPEATS and
                            all(r["metrics"][str(t)]["passes"] for r in samples for t in (1e-3, 1e-5))))
    result = dict(model="k-ecoli457", state_index=STATE_INDEX, fast_groups=int(len(groups)),
                  fast_columns=int(len(fast_columns)), basis_columns=int(len(basis_columns)),
                  fast_reaction_fraction=float(len(fast_columns) / system.S.shape[1]),
                  summary=summary, M4_admission=False,
                  interpretation="Development-only joint extent-coordinate probe; no full-window or formal-repeat claim.")
    save(OUT / "summary.json", result)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    (OUT / "README.md").write_text("# Reduced fast extent micro-screen\n\n"
        "Joint reaction extent coordinates are tested with an independent full nonlinear Radau audit for every candidate.\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
