"""Bounded real-model screen of controller control from mandatory audit error.

Run with: bash scripts/run-macos.sh -m research.m3_audit_control_only
Existing results are never overwritten.
"""
import itertools
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, prepare
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m3-audit-control-only-v2'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration = OUT / 'registration.json'
    if registration.exists():
        raise FileExistsError('Preserve existing evidence')
    reg = dict(
        date='2026-11-15',
        hypothesis='Mandatory full-ODE audit error may replace the coarse/fine split estimator as a controller signal, eliminating one coarse split per candidate without reducing audit points or other acceptance guards.',
        scope='Short initial-window development screen only; two models x two targets x original and audit-control-only modes x three interleaved repeats. Not formal M4 evidence.',
        protocol=dict(repeats=3, independent_full_reference=True, no_clipping=True,
                      all_attempts_preserved=True, audit_points='all requested public points plus quarter/half/three-quarter/end',
                      candidate_audit='full original ODE solve from the same committed start for every split attempt',
                      controls='same system, tolerances, initial state, public grid and diagnostics; original uses coarse+fine split; variant uses fine split only; both retain complete per-candidate audit'),
        source_sha256={p: digest(ROOT / p) for p in ('research/m3_audit_control_only.py',
            'metabolism/continuation.py', 'metabolism/adaptive.py', 'metabolism/splitting.py',
            'metabolism/m3_continuation.py', 'checks/test_continuation.py')},
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )
    registration.write_text(json.dumps(reg, indent=2) + '\n')
    rows = []
    (OUT / 'runs.json').write_text(json.dumps(dict(runs=rows), indent=2) + '\n')
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        times = case['times']
        horizon = min(float(times[-1]), 1e-4)
        short_times = np.array([float(times[0]), horizon])
        # Independent complete-system trajectory is the shared endpoint/path reference.
        reference_cfg = AdaptiveConfig(target=1e-5, initial_step=min(1e-5, horizon / 10),
            max_step=horizon / 20, fallback_method='Radau' if name == 'teusink' else 'CVODE-KLU',
            fallback_rtol=1e-12, audit_rtol=1e-12, timeout=120.)
        reference_controller = AdaptiveSplit(case['system'], case['scale'], reference_cfg,
            fixed_groups=np.arange(len(case['system'].groups)), **case['options'])
        tick = time.perf_counter()
        reference_values, reference_stats = reference_controller.integrate(short_times)
        reference_seconds = time.perf_counter() - tick
        local_case = dict(case, times=short_times, reference=reference_values,
                          flux_reference=np.column_stack([case['model'].flux(0., y) for y in reference_values.T]))
        for target in TARGETS:
            base = ContinuationConfig(target=target, initial_step=1e-3, max_step=.01 if name == 'teusink' else .001,
                max_attempts=512, max_rejections=16, timeout=120., group_every=1,
                audit_method='Radau' if name == 'teusink' else 'CVODE-KLU',
                audit_rtol=1e-11 if name == 'teusink' else 1e-11,
                fallback_method='BDF' if name == 'teusink' else 'CVODE-KLU', fallback_rtol=1e-11)
            for repeat in range(3):
                modes = [False, True] if repeat % 2 == 0 else [True, False]
                for audit_control_only in modes:
                    config = replace(base, audit_control_only=audit_control_only)
                    identity = 'audit-control-only' if audit_control_only else 'coarse-fine'
                    row = dict(model=name, target=target, repeat=repeat, mode=identity,
                               reference_seconds=reference_seconds)
                    tick = time.perf_counter()
                    try:
                        controller = MacroSplit(case['system'], case['scale'], config, **case['options'])
                        values, stats = controller.integrate(short_times)
                        seconds = time.perf_counter() - tick
                        # Score on the independent reference sampled at this exact short grid.
                        from metabolism.m3_continuation import metrics
                        scored = metrics(values, local_case, target, stats)
                        endpoint = float(np.max(np.abs(values[:, -1] - reference_values[:, -1]) /
                            (target * (.01 * case['scale'] + np.maximum(abs(reference_values[:, -1]), case['scale'])))))
                        row.update(status='success', seconds=seconds, metrics=scored,
                                   independent_endpoint_error=endpoint, stats=stats,
                                   independent_reference_nfev=reference_stats['nfev'])
                    except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                        row.update(status='failed', seconds=time.perf_counter() - tick,
                                   reason=str(exc), stats=getattr(exc, 'stats', None))
                    rows.append(row)
                    (OUT / 'runs.json').write_text(json.dumps(dict(runs=rows), indent=2) + '\n')
                    print(name, target, repeat, identity, row['status'], round(row['seconds'], 3),
                          row.get('independent_endpoint_error'), flush=True)
    summary = []
    for name, target, mode in itertools.product(('teusink', 'kecoli'), TARGETS,
                                                 ('coarse-fine', 'audit-control-only')):
        subset = [r for r in rows if r['model'] == name and r['target'] == target and r['mode'] == mode]
        passed = [r for r in subset if r['status'] == 'success']
        summary.append(dict(model=name, target=target, mode=mode, successes=len(passed),
            median_seconds=float(np.median([r['seconds'] for r in passed])) if passed else None,
            iqr_seconds=float(np.ptp(np.quantile([r['seconds'] for r in passed], [.25, .75]))) if passed else None,
            median_endpoint_error=float(np.median([r['independent_endpoint_error'] for r in passed])) if passed else None,
            median_split_coverage=float(np.median([r['stats']['split_coverage'] for r in passed])) if passed else 0.,
            audits_preserved=bool(passed) and all(r['stats']['full_calls'] >= r['stats']['accepted_steps'] + r['stats']['rejected_steps'] for r in passed),
            zero_coarse_calls=mode != 'audit-control-only' or (bool(passed) and all(r['stats']['coarse_split_calls'] == 0 for r in passed))))
    result = dict(summary=summary, M4_admission=False,
        interpretation='Exploratory short-prefix evidence only; no full-window, >=50% coverage, seven-candidate-repeat or formal speedup claim.',
        source_sha256=reg['source_sha256'])
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'README.md').write_text('# Audit-control-only screening\n\n'
        'Bounded short-prefix development experiment. Full per-attempt ODE audits and all original acceptance guards remain enabled. '
        'This evidence cannot qualify M4. See registration.json, runs.json and summary.json.\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
