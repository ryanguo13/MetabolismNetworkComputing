"""Short-window post-failure screen for audited macro-step growth control."""
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, metrics, prepare
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m4-growth-safety-screen'
SAFETIES = (.9, 1.8)
WINDOW = 1e-6


def main():
    if (OUT / 'registration.json').exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = ('research/m4_growth_safety_screen.py', 'metabolism/continuation.py',
                    'metabolism/adaptive.py', 'metabolism/splitting.py',
                    'metabolism/m3_continuation.py')
    registration = dict(date='2026-11-15',
        hypothesis='A larger bounded safety factor can grow accepted macro steps faster when full audit errors are below their local budget; failed audits still halve and roll back.',
        scope='Post-failure short-window development screen on all four model/target combinations; no formal admission claim.',
        interval=[0., WINDOW], safeties=list(SAFETIES), warmup=1, repeats=3,
        candidate=dict(audit_control_only=True, local_fraction=1., initial_step=1e-8,
                       max_step=WINDOW, max_attempts=256, max_rejections=16,
                       audit_rtol=1e-12, fallback_rtol=1e-12),
        gates=dict(full_ode_audit_every_attempt=True, no_clipping=True,
                   rollback=True, coverage_recomputed=True, costs_include_audit=True),
        source_sha256={p: digest(ROOT / p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    OUT.mkdir(parents=True)
    (OUT / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    rows = []
    (OUT / 'runs.json').write_text(json.dumps({'runs': rows}, indent=2) + '\n')
    for name in ('teusink', 'kecoli'):
        case = prepare(name)
        end = min(float(case['times'][-1]), WINDOW)
        grid = np.array([0., end])
        reference_cfg = dict(rtol=1e-13, atol=np.maximum(case['scale']*1e-15, 1e-18),
                             method='Radau', dense_output=True)
        def rhs(t, y):
            return case['system'].S @ case['model'].flux(t, y)
        ref = solve_ivp(rhs, (0., end), case['model'].y0, **reference_cfg)
        if not ref.success:
            raise RuntimeError(f'{name}: independent short reference failed: {ref.message}')
        reference = ref.sol(grid)
        reference_flux = np.column_stack([case['model'].flux(t, y) for t, y in zip(grid, reference.T)])
        local_case = dict(case, times=grid, reference=reference, flux_reference=reference_flux)
        large = name == 'kecoli'
        for target in TARGETS:
            baseline_cfg = AdaptiveConfig(target=target, initial_step=1e-8, max_step=end,
                max_attempts=256, max_rejections=16, fallback_method='CVODE-KLU' if large else 'Radau',
                fallback_rtol=1e-12, timeout=180.)
            entries = []
            for safety in SAFETIES:
                candidate_cfg = ContinuationConfig(target=target, initial_step=1e-8, max_step=end,
                    max_attempts=256, max_rejections=16, timeout=180.,
                    audit_method='CVODE-KLU' if large else 'Radau', audit_rtol=1e-12,
                    fallback_method='CVODE-KLU' if large else 'BDF', fallback_rtol=1e-12,
                    audit_control_only=True, growth_safety=safety, classification_step_floor=end)
                entries.append(('candidate', safety, candidate_cfg, None))
            entries.append(('baseline', None, baseline_cfg, np.arange(len(case['system'].groups))))
            order = [('candidate', .9), ('baseline', None), ('candidate', 1.8),
                     ('candidate', 1.8), ('baseline', None), ('candidate', .9),
                     ('baseline', None), ('candidate', .9)]
            for repeat, (method, safety) in enumerate(order):
                if method == 'baseline':
                    _, _, config, fixed = next(e for e in entries if e[0] == 'baseline')
                else:
                    _, _, config, fixed = next(e for e in entries if e[0] == method and e[1] == safety)
                identity = dict(model=name, target=target, method=method, growth_safety=safety,
                                repeat=repeat, warmup=False, parameters=asdict(config))
                tick = time.perf_counter()
                try:
                    controller = (AdaptiveSplit(case['system'], case['scale'], config,
                                  fixed_groups=fixed, **case['options']) if method == 'baseline' else
                                  MacroSplit(case['system'], case['scale'], config, **case['options']))
                    values, stats = controller.integrate(grid)
                    row = dict(**identity, status='success', seconds=time.perf_counter()-tick,
                        metrics=metrics(values, local_case, target, stats), stats=stats)
                    if method == 'candidate':
                        row['split_coverage_recomputed'] = sum(
                            a['h'] for a in stats['attempts'] if a['status'] == 'accepted') / end
                        row['every_attempt_audited'] = stats['full_calls'] == len(stats['attempts'])
                except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                    row = dict(**identity, status='failed', seconds=time.perf_counter()-tick,
                               reason=str(exc), stats=getattr(exc, 'stats', None))
                rows.append(row)
                (OUT / 'runs.json').write_text(json.dumps({'runs': rows}, indent=2) + '\n')
                print(name, target, method, safety, repeat, row['status'],
                      round(row['seconds'], 4), (row.get('stats') or {}).get('split_coverage'), flush=True)
    print(f'completed {len(rows)} rows; independent reference and raw rows preserved in {OUT}', flush=True)


if __name__ == '__main__':
    main()
