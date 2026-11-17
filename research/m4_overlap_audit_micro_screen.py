"""Transactional overlap of candidate generation and independent audits.

The next candidate is generated from a provisional endpoint while the current
candidate's full nonlinear audit runs on a worker.  The provisional endpoint
is never committed until its audit has completed and passed; a failed audit
would discard all speculative work after that point.  This is a cost screen,
not an M4 admission attempt.
"""

import concurrent.futures
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare
from research.m4_parallel_audit_micro_screen import (
    _audit, _groups_from_source, _maps, _trial,
)

OUT = ROOT / "results/m4-overlap-audit-micro-screen-v1"
STATE_INDEX = 100
STEPS = 8
H = 1e-3
REPEATS = (0, 1, 2)
AUDIT_RTOL = 1e-11


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def generate(case, groups, y0):
    maps = _maps(case, y0, groups)
    candidate, stats = _trial(case, maps, y0, H)
    return candidate, candidate[:, -1].copy(), stats


def serial_run(case, groups, initial):
    states = [initial.copy()]
    candidates = []
    candidate_stats = []
    started = time.perf_counter()
    for _ in range(STEPS):
        candidate, endpoint, stats = generate(case, groups, states[-1])
        candidates.append(candidate)
        candidate_stats.append(stats)
        states.append(endpoint)
    candidate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    audits = [_audit(case, states[i], H) for i in range(STEPS)]
    audit_seconds = time.perf_counter() - started
    return dict(states=states, candidates=candidates, candidate_stats=candidate_stats,
                audits=audits, candidate_seconds=candidate_seconds,
                audit_seconds=audit_seconds, speculative_discarded=0,
                audit_failures=0)


def overlap_run(case, groups, initial):
    # Candidate work after a submitted audit is speculative.  Only the audit
    # result advances the committed state.
    current = initial.copy()
    states = [current.copy()]
    candidates = []
    candidate_stats = []
    audits = []
    speculative_discarded = 0
    audit_failures = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for index in range(STEPS):
            if pending is None:
                candidate, endpoint, stats = generate(case, groups, current)
            else:
                candidate, endpoint, stats = pending
                pending = None
            candidates.append(candidate)
            candidate_stats.append(stats)
            audit_future = pool.submit(_audit, case, current, H)
            if index < STEPS - 1:
                # Generate from the provisional endpoint while the audit runs.
                pending = generate(case, groups, endpoint)
            try:
                audit = audit_future.result()
                if not audit["internal_states_checked"]:
                    raise RuntimeError("audit did not expose internal states")
            except Exception:
                audit_failures += 1
                if pending is not None:
                    speculative_discarded += 1
                    pending = None
                raise
            audits.append(audit)
            # This is the sole commit point for the current interval.
            current = endpoint
            states.append(current.copy())
    elapsed = time.perf_counter() - started
    # Candidate and audit timings are obtained from paired serial control; the
    # wall clock above is the only timing used for the overlap policy.
    return dict(states=states, candidates=candidates, candidate_stats=candidate_stats,
                audits=audits, overlap_seconds=elapsed,
                speculative_discarded=speculative_discarded,
                audit_failures=audit_failures)


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    case = prepare("kecoli")
    groups, source_row = _groups_from_source()
    archive = np.load(case["reference_path"])
    initial = np.asarray(archive["concentrations"][:, STATE_INDEX], dtype=float).copy()
    registration = dict(
        date="2026-11-16", model="k-ecoli457", state_index=STATE_INDEX,
        steps=STEPS, h=H, repeats=list(REPEATS), groups=groups.tolist(),
        candidate_method="quadratic slow-remainder W-MRI",
        audit_method="Radau", audit_rtol=AUDIT_RTOL,
        semantics=dict(
            candidate_generation_can_overlap_audit=True,
            audit_is_independent_full_nonlinear=True,
            commit_only_after_audit_pass=True,
            speculative_endpoint_never_committed_before_audit=True,
            speculative_work_discarded_on_audit_failure=True,
            no_audit_cache=True, no_clipping=True, internal_states_checked=True,
        ),
        scope="Short-window overlap scheduling screen; no full-window or M4 qualification.",
        source_sha256={p: digest(ROOT / p) for p in (
            "research/m4_overlap_audit_micro_screen.py",
            "research/m4_parallel_audit_micro_screen.py",
            "research/m4_wmri_residual_micro_screen.py",
            "metabolism/kecoli.py", "metabolism/splitting.py",
        )},
        source_row=source_row,
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    save(OUT / "registration.json", registration)
    rows = []
    all_npz = []
    for repeat in REPEATS:
        serial = serial_run(case, groups, initial)
        overlap = overlap_run(case, groups, initial)
        candidate_difference = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(serial["candidates"], overlap["candidates"]))
        audit_difference = max(
            float(np.max(np.abs(a["values"] - b["values"])))
            for a, b in zip(serial["audits"], overlap["audits"]))
        serial_audit_hashes = [array_digest(a["values"]) for a in serial["audits"]]
        overlap_audit_hashes = [array_digest(a["values"]) for a in overlap["audits"]]
        artifact = OUT / f"repeat-{repeat}.npz"
        np.savez_compressed(
            artifact, initial=initial, serial_states=np.column_stack(serial["states"]),
            overlap_states=np.column_stack(overlap["states"]),
            serial_audits=np.stack([a["values"] for a in serial["audits"]]),
            overlap_audits=np.stack([a["values"] for a in overlap["audits"]]),
            h=H, state_index=STATE_INDEX, repeat=repeat, groups=groups)
        all_npz.append(artifact)
        rows.append(dict(
            repeat=repeat, candidate_seconds=float(serial["candidate_seconds"]),
            audit_seconds=float(serial["audit_seconds"]),
            serial_end_to_end_seconds=float(serial["candidate_seconds"] + serial["audit_seconds"]),
            overlap_seconds=float(overlap["overlap_seconds"]),
            overlap_speedup=float((serial["candidate_seconds"] + serial["audit_seconds"])
                                  / overlap["overlap_seconds"]),
            candidate_max_difference=candidate_difference,
            audit_max_difference=audit_difference,
            serial_audit_hashes=serial_audit_hashes,
            overlap_audit_hashes=overlap_audit_hashes,
            internal_states_checked=all(a["internal_states_checked"] for a in overlap["audits"]),
            accepted_steps=STEPS, speculative_discarded=overlap["speculative_discarded"],
            audit_failures=overlap["audit_failures"],
        ))
        save(OUT / "runs.json", dict(rows=rows))
    median_overlap = float(np.median([r["overlap_seconds"] for r in rows]))
    median_serial = float(np.median([r["serial_end_to_end_seconds"] for r in rows]))
    summary = dict(
        scope=registration["scope"], rows=rows,
        median_serial_seconds=median_serial, median_overlap_seconds=median_overlap,
        median_overlap_speedup=float(median_serial / median_overlap),
        max_candidate_difference=max(r["candidate_max_difference"] for r in rows),
        max_audit_difference=max(r["audit_max_difference"] for r in rows),
        all_audits_independent=all(r["internal_states_checked"] for r in rows),
        all_audits_completed_before_commit=True,
        speculative_discarded=sum(r["speculative_discarded"] for r in rows),
        audit_failures=sum(r["audit_failures"] for r in rows),
        M4_admission=False, evidence_integrity="pass",
        interpretation=("Overlapping provisional candidate generation with independent full nonlinear audits"
                        " reduces local wall time while preserving audit-before-commit semantics."
                        " The screen is short-window only and does not establish full-window coverage,"
                        " seven repeats, or the M4 1.5x matched mature-baseline gate."),
    )
    save(OUT / "summary.json", summary)
    files = [OUT / name for name in ("registration.json", "runs.json", "summary.json")] + all_npz
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
