"""Prototype same-origin dense audit reuse across nested rejected retries.

This is an offline calibration experiment only. It does not alter MacroSplit or
claim that one dense trajectory satisfies the formal per-candidate audit rule.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.m3_continuation import ROOT, digest, prepare

OUT = ROOT / "results/m4-rejection-audit-cache-v2"
STATE_INDEX = 61
WINDOW = 1e-6
RETRY_FRACTIONS = (1.0, 0.5, 0.25, 0.125)
METHOD = "Radau"
RTOL = 1e-10
TRUTH_RTOL = 1e-13


def solve_dense(case, y0, end, rtol):
    system = case["system"]
    model = case["model"]
    scale = case["scale"]
    def rhs(t, y):
        return system.S @ model.flux(0.0, y)
    def jac(t, y):
        return (system.S @ model.flux_jacobian(0.0, y)).tocsc()
    result = solve_ivp(
        rhs, (0.0, end), y0, method=METHOD, rtol=rtol,
        atol=np.maximum(scale * (0.01 * rtol), 1e-18), dense_output=True,
        jac=jac,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def fluxes(case, values):
    model = case["model"]
    return np.column_stack([model.flux(0.0, y) for y in values.T])


def max_scaled_error(a, b, scale):
    return float(np.max(np.abs(a - b) / np.maximum(scale[:, None], 1e-300)))


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    with np.load(case["reference_path"]) as source:
        y0 = source["concentrations"][:, STATE_INDEX].copy()
        archive_time = float(source["times"][STATE_INDEX])
    fractions = np.asarray(RETRY_FRACTIONS, dtype=float)
    ends = WINDOW * fractions
    query = np.unique(np.r_[0.0, ends])
    source_files = [
        "research/m4_rejection_audit_cache.py",
        "metabolism/m3_continuation.py",
        "metabolism/kecoli.py",
        "metabolism/cvode.py",
    ]
    registration = dict(
        date="2026-11-15", model="kecoli", state_index=STATE_INDEX,
        archive_time=archive_time, window=WINDOW,
        retry_fractions=list(RETRY_FRACTIONS), method=METHOD,
        audit_rtol=RTOL, truth_rtol=TRUTH_RTOL,
        source_sha256={path: digest(ROOT / path) for path in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        scope=("Same committed state and nested retry horizons; compares one "
               "independent dense full-ODE trajectory with fresh solves. "
               "Diagnostic only; not M4 evidence."),
    )
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    np.savez_compressed(OUT / "initial.npz", y0=y0)

    truth = solve_dense(case, y0, WINDOW, TRUTH_RTOL)
    truth_values = truth.sol(query)
    np.savez_compressed(OUT / "truth.npz", query=query, values=truth_values)

    fresh_rows = []
    fresh_values = {}
    (OUT / "fresh-rows.json").write_text("[]\n")
    for fraction, end in zip(fractions, ends):
        started = time.perf_counter()
        result = solve_dense(case, y0, float(end), RTOL)
        values = result.sol(query[query <= end])
        elapsed = time.perf_counter() - started
        fresh_values[str(fraction)] = values
        truth_at = truth.sol(query[query <= end])
        fresh_rows.append(dict(
            fraction=float(fraction), end=float(end), seconds=elapsed,
            nfev=int(result.nfev), njev=int(result.njev), nlu=int(result.nlu),
            state_truth_error=max_scaled_error(values, truth_at, case["scale"]),
        ))
        np.savez_compressed(OUT / f"fresh-{fraction}.npz", query=query[query <= end], values=values)
        (OUT / "fresh-rows.json").write_text(json.dumps(fresh_rows, indent=2) + "\n")

    started = time.perf_counter()
    cached = solve_dense(case, y0, WINDOW, RTOL)
    cache_seconds = time.perf_counter() - started
    cached_values = cached.sol(query)
    np.savez_compressed(OUT / "cached.npz", query=query, values=cached_values)
    cached_rows = []
    (OUT / "cache-rows.json").write_text("[]\n")
    for fraction, end in zip(fractions, ends):
        mask = query <= end
        truth_at = truth.sol(query[mask])
        cached_at = cached_values[:, mask]
        fresh_at = fresh_values[str(fraction)]
        cached_rows.append(dict(
            fraction=float(fraction), end=float(end),
            cache_setup_seconds=cache_seconds,
            cache_vs_fresh_state_error=max_scaled_error(cached_at, fresh_at, case["scale"]),
            cache_vs_truth_state_error=max_scaled_error(cached_at, truth_at, case["scale"]),
            fresh_vs_truth_state_error=max_scaled_error(fresh_at, truth_at, case["scale"]),
            cache_vs_fresh_flux_error=max_scaled_error(
                fluxes(case, cached_at), fluxes(case, fresh_at), case["options"]["flux_scale"]),
            cache_vs_truth_flux_error=max_scaled_error(
                fluxes(case, cached_at), fluxes(case, truth_at), case["options"]["flux_scale"]),
        ))
        (OUT / "cache-rows.json").write_text(json.dumps(cached_rows, indent=2) + "\n")
    report = dict(
        registration=registration,
        fresh_rows=fresh_rows,
        cached_rows=cached_rows,
        fresh_total_seconds=float(sum(row["seconds"] for row in fresh_rows)),
        cached_total_seconds=float(cache_seconds),
        cached_plus_scoring_seconds=float(cache_seconds),
        speed_ratio=float(sum(row["seconds"] for row in fresh_rows) / cache_seconds),
        note=("The cached trajectory starts at the same committed state and is "
              "independently compared against fresh solves and tighter truth. "
              "This does not establish formal per-candidate audit admissibility."),
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
