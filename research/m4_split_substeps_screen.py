"""Bounded short-prefix screen for internally subcycled reaction splitting.

Run with: bash scripts/run-macos.sh -m research.m4_split_substeps_screen
This is development evidence only; it is not an M4 admission run.
"""
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, metrics, prepare
from metabolism.splitting import SplitFailure


OUT = ROOT / 'results/m4-split-substeps-screen-v3'
SUBSTEPS = (1, 2, 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration_path = OUT / 'registration.json'
    if registration_path.exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = (
        'research/m4_split_substeps_screen.py',
        'metabolism/continuation.py',
        'metabolism/adaptive.py',
        'metabolism/splitting.py',
        'metabolism/m3_continuation.py',
    )
    registration = dict(
        date='2026-11-15',
        hypothesis='Internal repeated Strang substeps reduce local splitting error enough to increase accepted reaction-split coverage on a short prefix without changing the frozen audit or correctness gates.',
        scope='One bounded short-prefix run for each model, target and substep count; original M1 initial state, public scales and prefix of the registered grid. Development screen only; no full-window, seven-repeat or speedup claim.',
        prefix_end=1e-4,
        substeps=list(SUBSTEPS),
        gates=dict(
            full_ode_audit_every_attempt=True,
            all_attempts_preserved=True,
            no_clipping=True,
            rejected_trial_rollback=True,
            original_metrics_recomputed=True,
            split_coverage_recomputed=True,
            costs_include_split_and_audit=True,
        ),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    registration_path.write_text(json.dumps(registration, indent=2) + '\n')
    runs = []
    (OUT / 'runs.json').write_text(json.dumps(dict(runs=runs), indent=2) + '\n')

    for model, target, substeps in itertools.product(('teusink', 'kecoli'), TARGETS, SUBSTEPS):
        case = prepare(model)
        prefix_mask = case['times'] <= registration['prefix_end']
        prefix = case['times'][prefix_mask]
        prefix_case = dict(case, times=case['times'][prefix_mask],
                           reference=case['reference'][:, prefix_mask],
                           flux_reference=case['flux_reference'][:, prefix_mask])
        config = ContinuationConfig(
            target=target,
            local_fraction=1.0,
            initial_step=1e-5,
            max_step=.1 if model == 'teusink' else .01,
            max_attempts=512,
            max_rejections=16,
            timeout=180.,
            group_every=1,
            audit_method='Radau' if model == 'teusink' else 'CVODE-KLU',
            audit_rtol=1e-11,
            fallback_method='BDF' if model == 'teusink' else 'CVODE-KLU',
            fallback_rtol=1e-11,
            audit_control_only=True,
            split_substeps=substeps,
        )
        row = dict(model=model, target=target, substeps=substeps,
                   prefix_end=float(prefix[-1]), status='failed')
        started = time.perf_counter()
        try:
            controller = MacroSplit(case['system'], case['scale'], config, **case['options'])
            values, stats = controller.integrate(prefix)
            row.update(status='success', seconds=time.perf_counter() - started,
                       metrics=metrics(values, prefix_case, target, stats),
                       stats=stats,
                       split_coverage_recomputed=sum(
                           a['h'] for a in stats['attempts']
                           if a['status'] == 'accepted' and a.get('groups')
                       ) / float(prefix[-1] - prefix[0]),
                       every_attempt_audited=(stats['full_calls'] == len(stats['attempts'])))
        except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
            row.update(seconds=time.perf_counter() - started, reason=str(exc),
                       stats=getattr(exc, 'stats', None))
        runs.append(row)
        (OUT / 'runs.json').write_text(json.dumps(dict(runs=runs), indent=2) + '\n')
        print(model, target, substeps, row['status'], round(row['seconds'], 3),
              row.get('split_coverage_recomputed'), flush=True)

    summary = []
    for model, target in itertools.product(('teusink', 'kecoli'), TARGETS):
        for substeps in SUBSTEPS:
            row = next(r for r in runs if r['model'] == model and r['target'] == target
                       and r['substeps'] == substeps)
            stats = row.get('stats') or {}
            summary.append(dict(
                model=model,
                target=target,
                substeps=substeps,
                status=row['status'],
                seconds=row.get('seconds'),
                accuracy_pass=row.get('metrics', {}).get('passes', False),
                split_coverage=row.get('split_coverage_recomputed', 0.),
                accepted_full_system_fraction=sum(
                    a['h'] for a in stats.get('attempts', [])
                    if a.get('status') == 'accepted' and not a.get('groups')
                ) / float(row.get('prefix_end', 0.) or 1.),
                accepted_steps=stats.get('accepted_steps', 0),
                rejected_steps=stats.get('rejected_steps', 0),
                first_accepted_h=next((a['h'] for a in stats.get('attempts', [])
                                       if a.get('status') == 'accepted'), None),
                full_calls=stats.get('full_calls', 0),
                every_attempt_audited=row.get('every_attempt_audited', False),
            ))
    result = dict(summary=summary, M4_admission=False,
                  interpretation='Bounded short-prefix mechanism screen only; no full-window, seven-repeat or formal speedup claim.',
                  source_sha256=registration['source_sha256'])
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# Internal Strang substep screen\n\n'
        'Short-prefix development evidence for explicit FixedSplit subcycling. '
        'Every trial remains independently audited and all costs are recorded. '
        'This directory cannot qualify M4.\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
