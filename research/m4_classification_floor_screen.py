"""Post-failure development screen for a step-aware grouping probe.

The floor changes only the step used to classify candidate fast groups. Actual
integration, independent full ODE audits, acceptance, rollback and coverage
still use the true attempted macro step. This is not formal M4 admission.
"""
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, metrics, prepare
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m4-classification-floor-screen'
FLOORS = (0., 1e-4, 1e-3)


def main():
    if (OUT / 'registration.json').exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = ('research/m4_classification_floor_screen.py', 'metabolism/continuation.py',
                    'metabolism/adaptive.py', 'metabolism/splitting.py', 'metabolism/m3_continuation.py')
    registration = dict(date='2026-11-15',
        hypothesis='A nonzero classification horizon can preserve candidate fast groups after audited step rejection while actual integration and audits remain tied to the true step.',
        scope='Post-failure development screen; original model initial states/windows/grids, one candidate and same-wrapper complete control per model x target x classification floor; no formal claim.',
        floors=list(FLOORS), candidate=dict(audit_control_only=True, local_fraction=1.0,
            initial_step=1e-5, max_step={'teusink': .1, 'kecoli': .01}, max_attempts=512,
            max_rejections=16, timeout=180., audit_rtol=1e-11, fallback_rtol=1e-11),
        gates=dict(full_ode_audit_every_attempt=True, no_clipping=True, rollback=True,
                   coverage_recomputed=True, cost_includes_full_fallback=True),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    OUT.mkdir(parents=True)
    (OUT / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    runs = []
    (OUT / 'runs.json').write_text(json.dumps({'runs': runs}, indent=2) + '\n')
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        large = name == 'kecoli'
        for target in TARGETS:
            baseline_cfg = AdaptiveConfig(target=target, initial_step=1e-5,
                                          max_step=.01 if large else .1,
                                          fallback_method='CVODE-KLU' if large else 'Radau',
                                          fallback_rtol=1e-11, timeout=180.)
            for floor in FLOORS:
                candidate_cfg = ContinuationConfig(target=target, initial_step=1e-5,
                    max_step=.01 if large else .1, max_attempts=512, max_rejections=16,
                    timeout=180., audit_method='CVODE-KLU' if large else 'Radau',
                    audit_rtol=1e-11, fallback_method='CVODE-KLU' if large else 'BDF',
                    fallback_rtol=1e-11, audit_control_only=True,
                    classification_step_floor=floor)
                for method, config, fixed in (
                    ('baseline', baseline_cfg, np.arange(len(case['system'].groups))),
                    ('candidate', candidate_cfg, None)):
                    row = dict(model=name, target=target, method=method, classification_floor=floor,
                               parameters=asdict(config))
                    tick = time.perf_counter()
                    try:
                        controller = (AdaptiveSplit(case['system'], case['scale'], config,
                                      fixed_groups=fixed, **case['options']) if method == 'baseline' else
                                      MacroSplit(case['system'], case['scale'], config, **case['options']))
                        values, stats = controller.integrate(case['times'])
                        row.update(status='success', seconds=time.perf_counter() - tick,
                                   metrics=metrics(values, case, target, stats), stats=stats)
                        if method == 'candidate':
                            row['split_coverage_recomputed'] = sum(
                                item['h'] for item in stats['attempts'] if item['status'] == 'accepted'
                            ) / (case['times'][-1] - case['times'][0])
                            row['every_attempt_audited'] = stats['full_calls'] == len(stats['attempts'])
                    except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                        row.update(status='failed', seconds=time.perf_counter() - tick,
                                   reason=str(exc), stats=getattr(exc, 'stats', None))
                    runs.append(row)
                    (OUT / 'runs.json').write_text(json.dumps({'runs': runs}, indent=2) + '\n')
                    print(name, target, floor, method, row['status'], round(row['seconds'], 3),
                          (row.get('stats') or {}).get('split_coverage'), flush=True)
    result = []
    for name in ('teusink', 'kecoli'):
        for target in TARGETS:
            for floor in FLOORS:
                cand = next(row for row in runs if row['model'] == name and row['target'] == target
                            and row['method'] == 'candidate' and row['classification_floor'] == floor)
                base = next(row for row in runs if row['model'] == name and row['target'] == target
                            and row['method'] == 'baseline' and row['classification_floor'] == floor)
                result.append(dict(model=name, target=target, classification_floor=floor,
                    candidate_status=cand['status'], baseline_status=base['status'],
                    candidate_seconds=cand.get('seconds'), baseline_seconds=base.get('seconds'),
                    speedup=base['seconds'] / cand['seconds'] if cand['status'] == base['status'] == 'success' else None,
                    accuracy_pass=cand.get('metrics', {}).get('passes', False),
                    split_coverage=cand.get('split_coverage_recomputed', 0.),
                    audit_all_attempts=cand.get('every_attempt_audited', False),
                    split_coverage_pass=cand.get('split_coverage_recomputed', 0.) >= .5))
    summary = dict(M4_admission=False, post_failure_candidate=True, summary=result,
                   interpretation='Development-only post-failure candidate screen; no formal admission.',
                   source_sha256=registration['source_sha256'])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
