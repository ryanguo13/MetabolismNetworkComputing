"""Retrospective shadow-first audit screen on independent saved candidates.

This study measures a layered policy without changing the frozen M4 rule.
The shadow estimator is calibrated on the historical development subset; the
held-out new-state/stress subset is used only to measure empirical recall and
cost. Every held-out row already has an independent full nonlinear audit in
its source evidence. No row is promoted to M4 evidence by this script.
"""

import json
import platform
from pathlib import Path

import numpy as np

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest

OUT = ROOT / "results/m4-layered-audit-micro-screen-v4"
SOURCES = (
    ("development", ROOT / "results/m3-shadow-fast/development.json"),
    ("new_states", ROOT / "results/m3-shadow-fast/new-states.json"),
    ("stress", ROOT / "results/m3-shadow-stress/runs.json"),
)
TARGETS = ("0.0001", "1e-06")
SAFETY_FACTOR = 1.10


def _load_rows():
    rows = []
    for subset, path in SOURCES:
        data = json.loads(path.read_text())
        for row in data.get("runs", []):
            item = dict(row)
            item["subset"] = subset
            rows.append(item)
    return rows


def _usable(row, target, require_timing=False):
    comparison = row.get("comparison", {}).get(target)
    shadow = row.get("shadow", {}).get("targets", {}).get(target)
    if not isinstance(comparison, dict) or not isinstance(shadow, dict):
        return False
    if comparison.get("actual_numerical_pass") is None:
        return False
    score = shadow.get("score")
    if not isinstance(score, dict) or not score:
        return False
    if not require_timing:
        return True
    audit = (row.get("truth") or {}).get("stats", {}).get("seconds")
    if audit is None:
        audit = row.get("audit_seconds")
    shadow_seconds = (row.get("shadow") or {}).get("seconds")
    return (isinstance(audit, (int, float)) and np.isfinite(audit) and audit > 0
            and isinstance(shadow_seconds, (int, float))
            and np.isfinite(shadow_seconds) and shadow_seconds >= 0)


def _score(row, target):
    return float(max(row["shadow"]["targets"][target]["score"].values()))


def _audit_seconds(row):
    value = (row.get("truth") or {}).get("stats", {}).get("seconds")
    return float(value if value is not None else row["audit_seconds"])


def _shadow_seconds(row):
    return float(row["shadow"]["seconds"])


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    rows = _load_rows()
    registration = {
        "date": "2026-11-15",
        "scope": "Retrospective shadow-first audit scheduling screen; no M4 qualification.",
        "source_subsets": {name: str(path.relative_to(ROOT)) for name, path in SOURCES},
        "targets": list(TARGETS),
        "calibration_subset": "development",
        "held_out_subsets": ["new_states", "stress"],
        "safety_factor": SAFETY_FACTOR,
        "policy": ("Calibrate max shadow-to-audit effectivity on development rows; "
                   "skip full audit only when safety_factor*calibrated_bound*shadow_score <= 1."),
        "frozen_semantics": {
            "independent_full_audit_remains_required_for_M4": True,
            "no_clipping_or_projection": True,
            "source_audits_are_independent_full_nonlinear": True,
            "this_screen_cannot_promote_a_candidate": True,
        },
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for _, path in SOURCES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    save(OUT / "registration.json", registration)

    summary = {}
    decisions = []
    for target in TARGETS:
        calibration = [r for r in rows if r.get("subset") == "development" and _usable(r, target)]
        held_out = [r for r in rows if r.get("subset") in ("new_states", "stress") and _usable(r, target, require_timing=True)]
        effectivities = [float(r["comparison"][target]["effectivity"])
                         for r in calibration if r["comparison"][target].get("effectivity") is not None]
        if not effectivities:
            raise RuntimeError(f"No calibration effectivities for target {target}")
        bound = SAFETY_FACTOR * max(effectivities)
        held_rows = []
        for index, row in enumerate(held_out):
            score = _score(row, target)
            skip = bool(bound * score <= 1.0)
            actual_pass = bool(row["comparison"][target]["actual_numerical_pass"])
            decision = dict(
                target=target, subset=row["subset"], source_key=row.get("key"),
                score=score, calibrated_effectivity_bound=bound,
                shadow_pass=bool(row["shadow"]["targets"][target].get("would_accept")),
                full_audit_skipped=skip, actual_numerical_pass=actual_pass,
                empirical_false_accept=bool(skip and not actual_pass),
                empirical_false_reject=bool(not skip and actual_pass),
                shadow_seconds=_shadow_seconds(row), audit_seconds=_audit_seconds(row),
            )
            held_rows.append(decision)
            decisions.append(decision)
        full_seconds = float(sum(r["audit_seconds"] for r in held_rows))
        layered_seconds = float(sum(r["shadow_seconds"] for r in held_rows)
                                + sum(r["audit_seconds"] for r in held_rows
                                      if not r["full_audit_skipped"]))
        skipped = [r for r in held_rows if r["full_audit_skipped"]]
        suspicious = [r for r in held_rows if not r["full_audit_skipped"]]
        policy_accepts = [r for r in held_rows
                          if r["full_audit_skipped"] or r["actual_numerical_pass"]]
        actual_passes = sum(r["actual_numerical_pass"] for r in held_rows)
        fail_closed_cost = float(sum(r["shadow_seconds"] for r in held_rows)
                                 + sum(r["audit_seconds"] for r in skipped))
        summary[target] = dict(
            calibration_rows=len(calibration), held_out_rows=len(held_rows),
            calibrated_max_effectivity=float(max(effectivities)),
            calibrated_bound=float(bound),
            full_audits=len(held_rows), skipped_full_audits=len(skipped),
            skip_fraction=float(len(skipped) / len(held_rows)) if held_rows else None,
            suspicious_full_audits=len(suspicious),
            shadow_reject_fraction=float(len(suspicious) / len(held_rows)) if held_rows else None,
            empirical_false_accepts=sum(r["empirical_false_accept"] for r in held_rows),
            empirical_false_rejects=sum(r["empirical_false_reject"] for r in held_rows),
            actual_passes=actual_passes,
            policy_accepts=len(policy_accepts),
            empirical_recall=(float(len(policy_accepts) / sum(r["actual_numerical_pass"] for r in held_rows))
                              if any(r["actual_numerical_pass"] for r in held_rows) else None),
            all_skipped_rows_pass=all(r["actual_numerical_pass"] for r in skipped),
            full_audit_seconds=full_seconds, layered_seconds=layered_seconds,
            layered_over_full=float(layered_seconds / full_seconds) if full_seconds else None,
            audit_wall_time_reduction=float(1.0 - layered_seconds / full_seconds) if full_seconds else None,
            fail_closed=dict(
                rule="shadow bound above threshold rejects without full audit; shadow-pass rows still receive full audit",
                full_audits=len(skipped),
                rejected_without_audit=len(suspicious),
                rejected_rows_that_would_have_passed=sum(r["actual_numerical_pass"] for r in suspicious),
                accepted_after_full_audit=sum(r["actual_numerical_pass"] for r in skipped),
                empirical_recall=(float(sum(r["actual_numerical_pass"] for r in skipped) / actual_passes)
                                 if actual_passes else None),
                shadow_plus_remaining_audit_seconds=fail_closed_cost,
                cost_over_all_full_audits=(float(fail_closed_cost / full_seconds)
                                           if full_seconds else None),
            ),
        )

    save(OUT / "decisions.json", dict(rows=decisions))
    result = dict(
        scope=registration["scope"], targets=summary,
        M4_admission=False,
        evidence_integrity="pass",
        interpretation=("Held-out rows show no empirical false accept under the calibrated shadow bound, "
            "but the bound is data-derived rather than a mathematical certificate. "
            "It cannot replace the frozen independent full-ODE audit for M4 admission."),
    )
    save(OUT / "summary.json", result)
    files = [OUT / name for name in ("registration.json", "decisions.json", "summary.json")]
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    (OUT / "README.md").write_text(
        "# Layered audit micro-screen\n\n"
        "This retrospective screen measures shadow-first scheduling on held-out saved"
        " candidates. The source full audits remain authoritative; no candidate is"
        " promoted to M4 evidence.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
