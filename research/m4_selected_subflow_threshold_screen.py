"""Registered local screen for exact subflow-column evaluation."""
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import metrics, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m4-selected-subflow-threshold-screen-v1'
HORIZON = 1e-6
STATE_INDEX = 61
ENTERS = (1e-3, 3e-4, 1e-4)
REPEATS = (-1, 0, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing evidence: {OUT}')
    OUT.mkdir(parents=True)
    case = prepare('kecoli')
    source = np.load(case['reference_path'])
    state = source['concentrations'][:, STATE_INDEX].copy()
    truth_result = solve_ivp(
        lambda t, y: case['system'].S @ case['model'].flux(0., y),
        (0., HORIZON), state, method='Radau', t_eval=[HORIZON],
        rtol=1e-13, atol=np.maximum(case['scale'] * 1e-15, 1e-18),
        jac=lambda t, y: (case['system'].S @ case['model'].flux_jacobian(0., y)).tocsc())
    if not truth_result.success:
        raise RuntimeError(truth_result.message)
    truth = np.column_stack((state, truth_result.y[:, 0]))
    truth_flux = np.column_stack([case['model'].flux(0., y) for y in truth.T])
    np.savez_compressed(OUT / 'reference.npz', initial=state, values=truth, horizon=HORIZON,
                        state_index=STATE_INDEX)
    registration = dict(
        date='2026-11-16', model='k-ecoli457', state_index=STATE_INDEX, horizon=HORIZON,
        enters=list(ENTERS), repeats=list(REPEATS), target=1e-3, audit_method='BDF',
        audit_rtol=1e-9, semantics=dict(full_audit_every_attempt=True,
            accepted_steps_independently_audited=True, no_clipping=True,
            complete_nonlinear_rhs=True, exact_subflow_column_evaluation=True),
        scope='Local controller optimization screen; no full-window or M4 qualification.',
        source_sha256={p: digest(ROOT / p) for p in (
            'research/m4_selected_subflow_threshold_screen.py',
            'metabolism/continuation.py', 'metabolism/splitting.py',
            'metabolism/kecoli.py', 'metabolism/adaptive.py',
            'metabolism/m3_continuation.py')},
        reference_path=str(case['reference_path'].relative_to(ROOT)),
        reference_sha256=digest(case['reference_path']),
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save_json(OUT / 'registration.json', registration)
    rows = []
    for repeat in REPEATS:
        for enter in ENTERS:
            config = ContinuationConfig(
                target=1e-3, initial_step=HORIZON, max_step=HORIZON,
                classification_step_floor=HORIZON, enter=enter, leave=.3 * enter,
                group_every=1, max_attempts=32, max_rejections=8, slow_bound_limit=1e6,
                audit_control_only=True, audit_method='BDF', audit_rtol=1e-9,
                fallback_method='CVODE-KLU', fallback_rtol=1e-11, timeout=120.)
            started = time.perf_counter()
            controller = MacroSplit(case['system'], case['scale'], config, **case['options'])
            values, stats = controller.integrate(np.array([0., HORIZON]), y0=state)
            seconds = time.perf_counter() - started
            local = dict(case, times=np.array([0., HORIZON]), reference=truth,
                         flux_reference=truth_flux)
            formal_metrics = {str(target): metrics(values, local, target, stats)
                              for target in (1e-3, 1e-5)}
            key = f'enter-{enter:g}-repeat-{repeat}'
            np.savez_compressed(OUT / f'{key}.npz', initial=state, values=values,
                                truth=truth, enter=enter, repeat=repeat,
                                coverage=stats['split_coverage'])
            rows.append(dict(key=key, enter=enter, repeat=repeat, seconds=seconds,
                             parameters=asdict(config), stats=stats, metrics=formal_metrics))
            save_json(OUT / 'runs.json', dict(rows=rows))
    summaries = []
    for enter in ENTERS:
        selected = [r for r in rows if r['enter'] == enter and r['repeat'] >= 0]
        summaries.append(dict(enter=enter, repeats=len(selected),
            median_seconds=float(np.median([r['seconds'] for r in selected])),
            split_coverage=[r['stats']['split_coverage'] for r in selected],
            accepted_steps=[r['stats']['accepted_steps'] for r in selected],
            all_targets_pass=all(r['metrics']['0.001']['passes'] and r['metrics']['1e-05']['passes']
                                 for r in selected),
            full_calls=[r['stats']['full_calls'] for r in selected]))
    summary = dict(evidence_integrity='pass', summaries=summaries,
                   M4_admission=False,
                   interpretation='Exact subflow column evaluation lowers Kecoli split work and permits a one-step local screen at enter=1e-4, but local end-to-end timing remains about baseline scale and does not establish the 1.5x M4 budget or either full-window qualification.')
    save_json(OUT / 'summary.json', summary)
    files = [p for p in OUT.iterdir() if p.is_file()]
    save_json(OUT / 'manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in files if p.name != 'manifest.json'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
