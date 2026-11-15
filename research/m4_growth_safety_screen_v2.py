"""Corrected, balanced short-window screen for audited step-growth safety."""
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

OUT = ROOT / 'results/m4-growth-safety-screen-v2'
SAFETIES = (.9, 1.8)
WINDOW = 1e-6
REPEATS = 2


def main():
    if (OUT / 'registration.json').exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = ('research/m4_growth_safety_screen_v2.py', 'metabolism/continuation.py',
                    'metabolism/adaptive.py', 'metabolism/splitting.py',
                    'metabolism/m3_continuation.py')
    candidate_parameters = dict(audit_control_only=True, local_fraction=1., group_every=1,
        initial_step=1e-8, max_attempts=256, max_rejections=16, audit_rtol=1e-12,
        fallback_rtol=1e-12, max_step=WINDOW, classification_step_floor=WINDOW)
    registration = dict(date='2026-11-15',
        hypothesis='With correct local_fraction=1 and group refresh every attempt, safety=1.8 may reduce attempt count versus .9 without losing accuracy or coverage.',
        scope='Post-failure exploratory short-window screen only; no formal M4 claim.',
        interval=[0., WINDOW], safeties=list(SAFETIES), warmups=1, repeats=REPEATS,
        candidate_parameters=candidate_parameters,
        order='per combo: warmup baseline, warmup candidate .9, warmup candidate 1.8; then repeats 0..1 each baseline/.9/1.8 interleaved',
        gates=dict(full_ode_audit_every_attempt=True, no_clipping=True, rollback=True,
                   coverage_recomputed=True, costs_include_audit=True),
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
        large = name == 'kecoli'
        method = 'CVODE-KLU' if large else 'Radau'
        atol = np.maximum(case['scale']*1e-15, 1e-18)
        ref = solve_ivp(lambda t,y: case['system'].S @ case['model'].flux(t,y),
                        (0., end), case['model'].y0, method='Radau', rtol=1e-13,
                        atol=atol, dense_output=True)
        if not ref.success:
            raise RuntimeError(f'{name}: short reference failed: {ref.message}')
        reference = ref.sol(grid)
        flux_reference = np.column_stack([case['model'].flux(t,y) for t,y in zip(grid,reference.T)])
        local_case = dict(case, times=grid, reference=reference, flux_reference=flux_reference)
        for target in TARGETS:
            base = AdaptiveConfig(target=target, initial_step=1e-8, max_step=end,
                group_every=1, max_attempts=256, max_rejections=16,
                fallback_method=method, fallback_rtol=1e-12, timeout=240.)
            candidates = {
                safety: ContinuationConfig(target=target, initial_step=1e-8, max_step=end,
                    group_every=1, local_fraction=1., max_attempts=256, max_rejections=16,
                    timeout=240., audit_method=method, audit_rtol=1e-12,
                    fallback_method='CVODE-KLU' if large else 'BDF', fallback_rtol=1e-12,
                    audit_control_only=True, growth_safety=safety,
                    classification_step_floor=end)
                for safety in SAFETIES}
            sequence = [('baseline', None, True), ('candidate', .9, True), ('candidate', 1.8, True)]
            for repeat in range(REPEATS):
                sequence.extend([('baseline', None, False), ('candidate', .9, False),
                                 ('candidate', 1.8, False)])
            for index, (which, safety, warmup) in enumerate(sequence):
                config = base if which == 'baseline' else candidates[safety]
                identity = dict(model=name, target=target, method=which, growth_safety=safety,
                                repeat=-1 if warmup else (index-len(SAFETIES))//3,
                                warmup=warmup, parameters=asdict(config))
                tick = time.perf_counter()
                try:
                    controller = (AdaptiveSplit(case['system'],case['scale'],config,
                                    fixed_groups=np.arange(len(case['system'].groups)),**case['options'])
                                  if which == 'baseline' else
                                  MacroSplit(case['system'],case['scale'],config,**case['options']))
                    values, stats = controller.integrate(grid)
                    row = dict(**identity,status='success',seconds=time.perf_counter()-tick,
                                metrics=metrics(values,local_case,target,stats),stats=stats)
                    if which == 'candidate':
                        row['split_coverage_recomputed'] = sum(a['h'] for a in stats['attempts']
                            if a['status']=='accepted')/end
                        row['every_attempt_audited'] = stats['full_calls']==len(stats['attempts'])
                except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    row = dict(**identity,status='failed',seconds=time.perf_counter()-tick,
                               reason=str(exc),stats=getattr(exc,'stats',None))
                rows.append(row)
                (OUT/'runs.json').write_text(json.dumps({'runs':rows},indent=2)+'\n')
                print(name,target,which,safety,'warmup' if warmup else identity['repeat'],
                      row['status'],round(row['seconds'],4),
                      (row.get('stats') or {}).get('split_coverage'),flush=True)
    print(f'completed {len(rows)} rows; raw results at {OUT}',flush=True)


if __name__ == '__main__':
    main()
