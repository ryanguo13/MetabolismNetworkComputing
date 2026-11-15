"""Screen local audit-budget allocation on the k-ecoli457 initial layer.

Run with: bash scripts/run-macos.sh -m research.m3_audit_budget_allocation
This is a development screen, not formal M4 evidence.
"""
import itertools
import json
import platform
import time
from dataclasses import replace

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, prepare, metrics
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m3-audit-budget-allocation'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg_path = OUT / 'registration.json'
    if reg_path.exists():
        raise FileExistsError('Preserve existing evidence')
    files = ('research/m3_audit_budget_allocation.py', 'metabolism/continuation.py',
             'metabolism/adaptive.py', 'metabolism/splitting.py', 'metabolism/m3_continuation.py')
    registration = dict(date='2026-11-15',
        hypothesis='Increasing the local per-attempt audit budget from 0.1x to 1.0x the registered global target may permit larger real split steps; only independent short-trajectory global errors decide whether cumulative accuracy remains acceptable.',
        scope='k-ecoli457, target 1e-3, initial interval [0,1e-6], audit-only and coarse/fine controllers, local_fraction 0.1 and 1.0, two interleaved exploratory repeats. Not formal M4 evidence.',
        gates=dict(full_independent_ode_audit_each_attempt=True, no_clipping=True,
                   same_split_and_audit_methods=True, global_error_scored_against_independent_reference=True,
                   preserve_failures=True),
        source_sha256={p: digest(ROOT / p) for p in files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    reg_path.write_text(json.dumps(registration, indent=2) + '\n')
    case = prepare('kecoli')
    times = np.array([0., 1e-6])
    ref_cfg = AdaptiveConfig(target=1e-5, initial_step=1e-7, max_step=1e-7,
        fallback_method='CVODE-KLU', fallback_rtol=1e-13, audit_rtol=1e-13, timeout=180.)
    ref_controller = AdaptiveSplit(case['system'], case['scale'], ref_cfg,
        fixed_groups=np.arange(len(case['system'].groups)), **case['options'])
    tick = time.perf_counter()
    reference, ref_stats = ref_controller.integrate(times)
    reference_seconds = time.perf_counter() - tick
    local_case = dict(case, times=times, reference=reference,
        flux_reference=np.column_stack([case['model'].flux(0., y) for y in reference.T]))
    rows = []
    configs = list(itertools.product((.1, 1.), (False, True), range(2)))
    for fraction, audit_only, repeat in configs:
        row = dict(model='kecoli', target=1e-3, local_fraction=fraction,
                   mode='audit-control-only' if audit_only else 'coarse-fine', repeat=repeat,
                   reference_seconds=reference_seconds)
        cfg = ContinuationConfig(target=1e-3, local_fraction=fraction, initial_step=5e-5,
            max_step=1e-6, max_attempts=256, max_rejections=16, timeout=180., group_every=1,
            audit_method='CVODE-KLU', audit_rtol=1e-11,
            fallback_method='CVODE-KLU', fallback_rtol=1e-11, audit_control_only=audit_only)
        tick = time.perf_counter()
        try:
            values, stats = MacroSplit(case['system'], case['scale'], cfg, **case['options']).integrate(times)
            row.update(status='success', seconds=time.perf_counter()-tick,
                metrics=metrics(values, local_case, 1e-3, stats), stats=stats,
                endpoint_error=float(np.max(np.abs(values[:, -1]-reference[:, -1]) /
                    (1e-3*(.01*case['scale']+np.maximum(abs(reference[:, -1]),case['scale']))))),
                independent_reference_nfev=ref_stats['nfev'])
        except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
            row.update(status='failed', seconds=time.perf_counter()-tick,
                       reason=str(exc), stats=getattr(exc, 'stats', None))
        rows.append(row)
        (OUT / 'runs.json').write_text(json.dumps(dict(runs=rows), indent=2) + '\n')
        print(f"fraction={fraction} mode={row['mode']} repeat={repeat} {row['status']} "
              f"seconds={row['seconds']:.3f} coverage={((row.get('stats') or {}).get('split_coverage',0)):.4g} "
              f"endpoint={row.get('endpoint_error')}", flush=True)
    summary=[]
    for fraction, mode in itertools.product((.1,1.),('coarse-fine','audit-control-only')):
        subset=[r for r in rows if r['local_fraction']==fraction and r['mode']==mode]
        good=[r for r in subset if r['status']=='success']
        summary.append(dict(local_fraction=fraction,mode=mode,successes=len(good),
            median_seconds=float(np.median([r['seconds'] for r in good])) if good else None,
            median_coverage=float(np.median([r['stats']['split_coverage'] for r in good])) if good else 0.,
            median_global_error=float(np.median([r['endpoint_error'] for r in good])) if good else None,
            all_audited=bool(good) and all(r['stats']['full_calls'] >= len(r['stats']['attempts']) for r in good),
            all_no_clipping=bool(good) and all(r['stats']['clipping_count']==0 for r in good)))
    result=dict(summary=summary,M4_admission=False,
        interpretation='Initial-layer-only controller screen. A larger local audit allocation is not qualified unless independent global error and subsequent full-window metrics remain within the original targets.',
        source_sha256=registration['source_sha256'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'README.md').write_text('# Local audit-budget allocation screen\n\nInitial-layer development screen only; not M4 evidence. See registration, runs and summary.\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
