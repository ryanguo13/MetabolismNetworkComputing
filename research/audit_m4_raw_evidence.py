"""Independent raw-evidence audit for the completed formal M4 attempt.

This deliberately does not import or call the admission validator. It audits
raw JSON rows, execution order, counters, and recomputed timing/coverage facts.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'results/m4-formal-audit-control-only'
BASELINES = ROOT / 'results/m4-same-machine-baselines-v2'
MODELS = ('teusink', 'kecoli')
TARGETS = (0.001, 1e-5)
WINDOWS = {'teusink': 10.0, 'kecoli': 1.0}


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def iqr(values):
    quartiles = quantiles(values, n=4, method='inclusive')
    return quartiles[2] - quartiles[0]


def expect(condition, failures, message):
    if not condition:
        failures.append(message)


def formal_combo(rows, model, target, failures):
    combo = [row for row in rows if row.get('model') == model and row.get('target') == target]
    expect(len(combo) == 16, failures, f'{model}/{target}: expected 16 rows, got {len(combo)}')
    for method in ('baseline', 'candidate'):
        warmups = [row for row in combo if row.get('method') == method and row.get('warmup') is True]
        formal = [row for row in combo if row.get('method') == method and row.get('warmup') is False]
        expect(len(warmups) == 1 and warmups[0].get('repeat') == -1,
               failures, f'{model}/{target}/{method}: warmup is not exactly one repeat -1')
        ids = [row.get('repeat') for row in formal]
        expect(len(ids) == 7 and set(ids) == set(range(7)),
               failures, f'{model}/{target}/{method}: formal repeat ids are {ids}')
        expect(all(row.get('status') == 'success' and finite_positive(row.get('seconds')) for row in formal),
               failures, f'{model}/{target}/{method}: failed or invalid formal timing')
    ordered = [row for row in combo if row.get('warmup') is False]
    expected = []
    for repeat in range(7):
        expected.extend(('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline'))
    actual = [row.get('method') for row in ordered]
    expected_repeats = [repeat for repeat in range(7) for _ in (0, 1)]
    actual_repeats = [row.get('repeat') for row in ordered]
    expect(actual == expected and actual_repeats == expected_repeats, failures,
           f'{model}/{target}: interleaving is {list(zip(actual_repeats, actual))}, '
           f'expected {list(zip(expected_repeats, expected))}')

    candidates = [row for row in combo if row.get('method') == 'candidate' and row.get('warmup') is False]
    details = []
    for row in sorted(candidates, key=lambda item: item['repeat']):
        stats = row.get('stats') or {}
        attempts = stats.get('attempts')
        expect(isinstance(attempts, list) and attempts, failures,
               f'{model}/{target}/candidate/{row.get("repeat")}: missing attempts')
        if not isinstance(attempts, list) or not attempts:
            continue
        expect(stats.get('full_calls') == len(attempts), failures,
               f'{model}/{target}/candidate/{row["repeat"]}: full_calls != attempts')
        accepted = [item for item in attempts if item.get('status') == 'accepted']
        rejected = [item for item in attempts if item.get('status') == 'rejected']
        expect(stats.get('accepted_steps') == len(accepted), failures,
               f'{model}/{target}/candidate/{row["repeat"]}: accepted_steps mismatch')
        expect(all(item.get('rollback_verified') is True for item in rejected), failures,
               f'{model}/{target}/candidate/{row["repeat"]}: rejected rollback evidence missing')
        intervals = sorted((item.get('time'), item.get('time') + item.get('h'))
                           for item in accepted
                           if finite_positive(item.get('h')) and isinstance(item.get('time'), (int, float)))
        duration = 0.0
        previous = 0.0
        valid_intervals = True
        for start, stop in intervals:
            if not math.isfinite(start) or start < previous - 1e-12 or stop > WINDOWS[model] + 1e-12:
                valid_intervals = False
                break
            duration += stop - start
            previous = stop
        expect(valid_intervals, failures,
               f'{model}/{target}/candidate/{row["repeat"]}: accepted intervals overlap or exceed window')
        coverage = duration / WINDOWS[model] if valid_intervals else None
        expect(coverage is not None and math.isclose(coverage, row.get('split_coverage_recomputed', -1), rel_tol=1e-9, abs_tol=1e-12),
               failures, f'{model}/{target}/candidate/{row["repeat"]}: recomputed coverage mismatch')
        expect(coverage is not None and coverage < 0.5, failures,
               f'{model}/{target}/candidate/{row["repeat"]}: expected failed coverage was not observed')
        details.append(dict(repeat=row['repeat'], seconds=row['seconds'], attempts=len(attempts),
                            full_calls=stats.get('full_calls'), accepted_steps=len(accepted),
                            rejected_steps=len(rejected), coverage=coverage,
                            metrics_passes=(row.get('metrics') or {}).get('passes')))
    candidates_seconds = [row['seconds'] for row in candidates]
    baselines = [row for row in combo if row.get('method') == 'baseline' and row.get('warmup') is False]
    baseline_seconds = [row['seconds'] for row in baselines]
    return dict(model=model, target=target, candidate_median=median(candidates_seconds),
                candidate_iqr=iqr(candidates_seconds), baseline_median=median(baseline_seconds),
                baseline_iqr=iqr(baseline_seconds), historical_baseline_config=baselines[0].get('config'),
                historical_speedup=median(baseline_seconds) / median(candidates_seconds),
                candidates=details)


def local_baseline_combo(rows, model, target, failures):
    combo = [row for row in rows if row.get('model') == model and row.get('target') == target]
    configs = sorted({row.get('config') for row in combo})
    qualified = []
    for config in configs:
        config_rows = [row for row in combo if row.get('config') == config and row.get('warmup') is False]
        warmups = [row for row in combo if row.get('config') == config and row.get('warmup') is True]
        ids = [row.get('repeat') for row in config_rows]
        expect(len(warmups) == 1 and warmups[0].get('repeat') == -1,
               failures, f'{model}/{target}/{config}: warmup is incomplete')
        expect(len(ids) == 7 and set(ids) == set(range(7)),
               failures, f'{model}/{target}/{config}: local repeat ids are {ids}')
        expect(all(row.get('status') == 'success' and finite_positive(row.get('seconds')) for row in config_rows),
               failures, f'{model}/{target}/{config}: local baseline failure/timing')
        expect(all((row.get('metrics') or {}).get('passes') is True for row in config_rows),
               failures, f'{model}/{target}/{config}: local baseline metrics do not pass')
        if len(config_rows) == 7 and all((row.get('metrics') or {}).get('passes') is True for row in config_rows):
            seconds = [row['seconds'] for row in config_rows]
            qualified.append(dict(config=config, median=median(seconds), iqr=iqr(seconds), seconds=seconds))
    expect(bool(qualified), failures, f'{model}/{target}: no locally qualified baseline')
    return min(qualified, key=lambda item: item['median']) if qualified else None


def main():
    failures = []
    formal_registration = load(FORMAL / 'registration.json')
    formal_rows = load(FORMAL / 'runs.json').get('runs', [])
    local_registration = load(BASELINES / 'registration.json')
    local_rows = load(BASELINES / 'runs.json').get('runs', [])
    expect(len(formal_rows) == 64, failures, f'formal raw row count is {len(formal_rows)}, expected 64')
    expect(len(local_rows) == 80, failures, f'local baseline raw row count is {len(local_rows)}, expected 80')

    formal_manifest = load(FORMAL / 'manifest.json')
    local_manifest = load(BASELINES / 'manifest.json')
    expect(formal_manifest.get('runs_sha256') == sha256(FORMAL / 'runs.json'), failures, 'formal runs manifest hash mismatch')
    expect(local_manifest.get('runs_sha256') == sha256(BASELINES / 'runs.json'), failures, 'local baseline runs manifest hash mismatch')
    expect(formal_manifest.get('registration_sha256') == sha256(FORMAL / 'registration.json'), failures, 'formal registration hash mismatch')
    expect(local_manifest.get('registration_sha256') == sha256(BASELINES / 'registration.json'), failures, 'local baseline registration hash mismatch')

    source_checks = {}
    for relative, expected in formal_registration.get('source_sha256', {}).items():
        actual = sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        source_checks[relative] = actual == expected
        expect(actual == expected, failures, f'formal source hash mismatch: {relative}')
    for model in MODELS:
        for target in TARGETS:
            formal_combo(formal_rows, model, target, failures)
            local_baseline_combo(local_rows, model, target, failures)

    formal_summary = [formal_combo(formal_rows, model, target, []) for model in MODELS for target in TARGETS]
    local_summary = [local_baseline_combo(local_rows, model, target, []) for model in MODELS for target in TARGETS]
    result = dict(date='2026-11-15', evidence_integrity='pass' if not failures else 'fail',
                  raw_rows=dict(formal=len(formal_rows), same_machine_baselines=len(local_rows)),
                  source_hash_checks=source_checks, combinations=formal_summary,
                  same_machine_fastest_baselines=local_summary,
                  failures=failures,
                  conclusion='formal four-combination attempt is complete; candidate is not_admitted')
    out = FORMAL / 'raw_evidence_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
