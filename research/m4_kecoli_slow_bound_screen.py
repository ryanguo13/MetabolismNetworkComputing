"""Bounded Kecoli slow-bound controller screen.

Run with: bash scripts/run-macos.sh -m research.m4_kecoli_slow_bound_screen
Development evidence only; not an M4 admission run.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, digest, metrics, prepare
from metabolism.splitting import SplitFailure


OUT = ROOT / 'results/m4-kecoli-slow-bound-screen-v3'
LIMITS = (2.0, 4.0, 8.0)
PREFIX_END = 1e-6
TARGET = 1e-3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration_path = OUT / 'registration.json'
    if registration_path.exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = (
        'research/m4_kecoli_slow_bound_screen.py',
        'metabolism/continuation.py',
        'metabolism/adaptive.py',
        'metabolism/splitting.py',
        'metabolism/m3_continuation.py',
    )
    registration = dict(
        date='2026-11-15',
        hypothesis='Kecoli accepted macro steps are pinned by slow_bound_limit=2 rather than audited splitting error; a higher bound limit may increase genuine reaction-split coverage.',
        scope='One Kecoli 1e-3 run per slow_bound_limit on the original public grid prefix <=1e-6. Development screen only; no full-window, seven-repeat or speedup claim.',
        model='kecoli', target=TARGET, prefix_end=PREFIX_END,
        slow_bound_limits=list(LIMITS),
        gates=dict(accepted_steps_full_ode_audited=True,
                   executed_split_trials_full_ode_audited=True,
                   preaudit_controller_rejections_recorded=True,
                   all_attempts_preserved=True, no_clipping=True,
                   rejected_trial_rollback=True, original_metrics_recomputed=True,
                   costs_include_split_and_audit=True),
        artifact_schema='npz: times, values, reference, flux_reference, scale, y0',
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    registration_path.write_text(json.dumps(registration, indent=2) + '\n')
    runs = []
    (OUT / 'runs.json').write_text(json.dumps(dict(runs=runs), indent=2) + '\n')

    case = prepare('kecoli')
    mask = case['times'] <= PREFIX_END
    prefix = case['times'][mask]
    prefix_case = dict(case, times=case['times'][mask],
                       reference=case['reference'][:, mask],
                       flux_reference=case['flux_reference'][:, mask])
    for limit in LIMITS:
        config = ContinuationConfig(
            target=TARGET, local_fraction=1.0, initial_step=1e-8,
            max_step=PREFIX_END, max_attempts=512, max_rejections=16,
            timeout=300., group_every=1, slow_bound_limit=limit,
            audit_method='CVODE-KLU', audit_rtol=1e-11,
            fallback_method='CVODE-KLU', fallback_rtol=1e-11,
            audit_control_only=True,
        )
        row = dict(model='kecoli', target=TARGET, slow_bound_limit=limit,
                   prefix_end=float(prefix[-1]), status='failed')
        started = time.perf_counter()
        try:
            controller = MacroSplit(case['system'], case['scale'], config, **case['options'])
            values, stats = controller.integrate(prefix)
            attempts = stats['attempts']
            artifact_name = f'run-{len(runs):02d}-limit-{limit:g}.npz'
            artifact_path = OUT / artifact_name
            np.savez_compressed(
                artifact_path, times=prefix, values=values,
                reference=prefix_case['reference'],
                flux_reference=prefix_case['flux_reference'],
                scale=case['scale'], y0=case['system'].y0,
            )
            row.update(
                status='success', seconds=time.perf_counter() - started,
                metrics=metrics(values, prefix_case, TARGET, stats), stats=stats,
                npz_file=artifact_name, npz_sha256=digest(artifact_path),
                npz_shape=list(values.shape),
                split_coverage_recomputed=sum(
                    a['h'] for a in attempts
                    if a.get('status') == 'accepted' and a.get('groups')
                ) / float(prefix[-1] - prefix[0]),
                accepted_full_system_fraction=sum(
                    a['h'] for a in attempts
                    if a.get('status') == 'accepted' and not a.get('groups')
                ) / float(prefix[-1] - prefix[0]),
                executed_split_trials_audited=(stats['full_calls'] == sum(
                    a.get('full_ode_audit') is not False for a in attempts
                )),
                all_attempts_recorded=True,
            )
        except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
            row.update(seconds=time.perf_counter() - started,
                       reason=str(exc), stats=getattr(exc, 'stats', None))
        runs.append(row)
        (OUT / 'runs.json').write_text(json.dumps(dict(runs=runs), indent=2) + '\n')
        print('kecoli', limit, row['status'], round(row['seconds'], 3),
              row.get('split_coverage_recomputed'),
              (row.get('metrics') or {}).get('passes'), flush=True)

    summary = []
    for row in runs:
        stats = row.get('stats') or {}
        attempts = stats.get('attempts') or []
        summary.append(dict(
            slow_bound_limit=row['slow_bound_limit'], status=row['status'],
            seconds=row.get('seconds'), accuracy_pass=(row.get('metrics') or {}).get('passes', False),
            split_coverage=row.get('split_coverage_recomputed', 0.),
            accepted_full_system_fraction=row.get('accepted_full_system_fraction', 0.),
            attempts=len(attempts), accepted_steps=stats.get('accepted_steps', 0),
            rejected_steps=stats.get('rejected_steps', 0),
            first_accepted_h=next((a['h'] for a in attempts if a.get('status') == 'accepted'), None),
            unique_accepted_h=sorted({a['h'] for a in attempts if a.get('status') == 'accepted'}),
            max_slow_scaled_norm_times_h=max(
                (a.get('slow_scaled_norm_times_h', 0.) for a in attempts), default=0.),
            full_calls=stats.get('full_calls', 0),
            executed_split_trials_audited=row.get('executed_split_trials_audited', False),
            all_attempts_recorded=row.get('all_attempts_recorded', False),
            preaudit_controller_rejections=sum(
                a.get('full_ode_audit') is False for a in attempts
            ),
            npz_file=row.get('npz_file'), npz_sha256=row.get('npz_sha256'),
        ))
    result = dict(summary=summary, M4_admission=False,
                  interpretation='Bounded Kecoli controller screen only; no full-window, seven-repeat or formal speedup claim.',
                  source_sha256=registration['source_sha256'])
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# Kecoli slow-bound screen\n\n'
        'Short-prefix development evidence for the adaptive slow-bound limit. '
        'Every successful run stores its candidate curve and reference inputs in '
        'an independently hashed NPZ. Every executed split trial is audited; '
        'pre-audit controller rejections are retained but are not claimed as '
        'full ODE solves. This directory cannot qualify M4.\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
