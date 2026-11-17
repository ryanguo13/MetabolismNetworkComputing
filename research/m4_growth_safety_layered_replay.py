"""Offline fail-closed replay of a cheap structural shadow bound.

The growth-safety screen already recorded `slow_scaled_norm_times_h` for every
candidate attempt and an independent full-audit outcome. This script replays
the legal policy: attempts above a registered threshold are rejected before a
full audit; attempts below it still receive the original full audit, and only
audited passing attempts may be accepted. It is a cost/coverage analysis only.
"""

import json
import platform
from pathlib import Path

import numpy as np

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, digest

OUT = ROOT / "results/m4-growth-safety-layered-replay-v1"
SOURCE = ROOT / "results/m4-growth-safety-screen-v2/runs.json"
THRESHOLDS = (0.01, 0.03, 0.05, 0.07, 0.10)


def _attempts():
    rows = json.loads(SOURCE.read_text())["runs"]
    output = []
    for row in rows:
        if row.get("method") != "candidate" or row.get("warmup"):
            continue
        for index, attempt in enumerate(row.get("stats", {}).get("attempts", [])):
            if not attempt.get("groups"):
                continue
            output.append(dict(
                model=row["model"], target=row["target"],
                safety=row.get("growth_safety"), repeat=row.get("repeat"),
                attempt_index=index, h=float(attempt.get("h", 0.0)),
                shadow=float(attempt.get("slow_scaled_norm_times_h", np.nan)),
                status=attempt.get("status"),
                audit_max=float(attempt.get("audit_max", np.nan)),
                audit_flux_max=float(attempt.get("audit_flux_max", np.nan)),
                source_seconds=float(row["seconds"]),
                source_audit_seconds=float(row["stats"].get("audit_seconds", 0.0)),
                source_full_calls=int(row["stats"].get("full_calls", 0)),
                source_coverage=float(row.get("split_coverage_recomputed", 0.0)),
            ))
    return output


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    rows = _attempts()
    registration = dict(
        date="2026-11-15", source=str(SOURCE.relative_to(ROOT)),
        scope="Offline fail-closed shadow-bound replay; no M4 qualification.",
        thresholds=list(THRESHOLDS),
        shadow="slow_scaled_norm_times_h recorded by the existing controller",
        policy="shadow above threshold rejects without full audit; shadow-pass attempts retain full audit and commit only on audit pass",
        gates=dict(accepted_attempts_still_full_audited=True, no_clipping=True,
                   coverage_recomputed=True, no_historical_state_change=True),
        source_sha256={str(SOURCE.relative_to(ROOT)): digest(SOURCE)},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    save(OUT / "registration.json", registration)
    decisions = []
    summaries = []
    for model in sorted({r["model"] for r in rows}):
        for target in sorted({r["target"] for r in rows if r["model"] == model}):
            for threshold in THRESHOLDS:
                subset = [r for r in rows if r["model"] == model and r["target"] == target]
                if not subset:
                    continue
                grouped = {}
                for r in subset:
                    grouped.setdefault((r["safety"], r["repeat"]), []).append(r)
                rejected = [r for r in subset if r["shadow"] > threshold]
                accepted_source = [r for r in subset if r["status"] == "accepted"]
                rejected_accepted = [r for r in rejected if r["status"] == "accepted"]
                # Each source run reports one audit time and full-call count.
                run_keys = {(r["safety"], r["repeat"]): r for r in subset}
                calls = sum(r["source_full_calls"] for r in run_keys.values())
                audit_seconds = sum(r["source_audit_seconds"] for r in run_keys.values())
                saved_audits = sum(1 for r in rejected)
                mean_audit = (audit_seconds / calls) if calls else 0.0
                candidate_seconds = sum(r["source_seconds"] -
                                        (r["source_audit_seconds"] if r["source_full_calls"] else 0.0)
                                        for r in run_keys.values())
                replay_seconds = candidate_seconds + max(0, calls - saved_audits) * mean_audit
                accepted_duration = sum(r["h"] for r in accepted_source)
                rejected_accepted_duration = sum(r["h"] for r in rejected_accepted)
                summaries.append(dict(
                    model=model, target=target, threshold=threshold,
                    attempts=len(subset), source_full_calls=calls,
                    shadow_rejected_attempts=len(rejected),
                    replay_full_calls=max(0, calls - saved_audits),
                    full_call_reduction=float(saved_audits / calls) if calls else None,
                    source_candidate_seconds=candidate_seconds,
                    source_audit_seconds=audit_seconds,
                    replay_candidate_plus_shadow_plus_audit_seconds=replay_seconds,
                    estimated_wall_time_reduction=(float(1.0 - replay_seconds /
                                                         (candidate_seconds + audit_seconds))
                                                   if candidate_seconds + audit_seconds else None),
                    source_accepted_attempts=len(accepted_source),
                    accepted_attempts_rejected_by_shadow=len(rejected_accepted),
                    accepted_duration=accepted_duration,
                    accepted_duration_rejected_by_shadow=rejected_accepted_duration,
                    accepted_duration_retained=(accepted_duration - rejected_accepted_duration),
                    retained_acceptance_fraction=(float((accepted_duration - rejected_accepted_duration) / accepted_duration)
                                                 if accepted_duration else None),
                ))
                decisions.extend(dict(r, threshold=threshold, shadow_rejected=bool(r["shadow"] > threshold),
                                      source_accepted=bool(r["status"] == "accepted")) for r in subset)
    save(OUT / "decisions.json", dict(rows=decisions))
    result = dict(scope=registration["scope"], summaries=summaries, M4_admission=False,
                  interpretation="A fail-closed bound can reduce audit calls only by rejecting some attempts; retained accepted attempts still require independent full audits.")
    save(OUT / "summary.json", result)
    files = [OUT / name for name in ("registration.json", "decisions.json", "summary.json")]
    save(OUT / "manifest.json", {str(path.relative_to(ROOT)): digest(path) for path in files})
    (OUT / "README.md").write_text(
        "# Growth safety layered replay\n\n"
        "Offline replay only. It does not rewrite candidate trajectories or M4 evidence.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
