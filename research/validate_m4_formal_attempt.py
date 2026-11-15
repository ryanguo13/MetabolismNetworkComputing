"""Independent fail-closed validator for the registered formal M4 attempt."""
import json
import math
from pathlib import Path
from statistics import median, quantiles

from metabolism.m3_continuation import ROOT, TARGETS

OUT = ROOT / 'results/m4-formal-audit-control-only'
MODELS = ('teusink', 'kecoli')
WINDOWS = {'teusink': (0.0, 10.0), 'kecoli': (0.0, 1.0)}
GROUP_COUNTS = {'teusink': 17, 'kecoli': 2666}


def finite(value, lower=0.0, upper=math.inf):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lower <= value <= upper)


def metrics_pass(row, model):
    metrics = row.get('metrics') or {}
    required = {'concentration_error': 1.0, 'flux_error': 1.0,
                'invariant_error': 1e-10}
    if model == 'teusink':
        required['relative_ledger_error'] = 1e-10
    else:
        required['net_flux_error'] = 1.0
    return (row.get('status') == 'success' and metrics.get('passes') is True
            and metrics.get('finite') is True
            and metrics.get('significant_negative_output') is False
            and all(finite(metrics.get(key), upper=limit)
                    for key, limit in required.items())
            and finite(row.get('seconds'), lower=math.nextafter(0.0, 1.0)))


def split_evidence(row, model):
    stats = row.get('stats') or {}
    attempts = stats.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        return False, 'missing attempts'
    accepted = [item for item in attempts if item.get('status') == 'accepted']
    if (not accepted or stats.get('accepted_steps') != len(accepted)
            or not finite(stats.get('split_seconds'), lower=math.nextafter(0.0, 1.0))
            or stats.get('clipping_count') != 0
            or stats.get('projection_count') != 0):
        return False, 'missing accepted-step or structural evidence'
    intervals = []
    for item in attempts:
        status = item.get('status')
        if status == 'rejected':
            if item.get('rollback_verified') is not True:
                return False, 'rejected trial lacks rollback evidence'
            continue
        if status != 'accepted':
            return False, 'unknown attempt status'
        if not all(finite(item.get(key), upper=1.0) for key in
                   ('audit_max', 'audit_flux_max', 'audit_net_flux_max', 'estimator_max')):
            return False, 'accepted attempt lacks bounded audit fields'
        start, step = item.get('time'), item.get('h')
        groups = item.get('groups')
        if not finite(start) or not finite(step, lower=math.nextafter(0.0, 1.0)):
            return False, 'invalid accepted interval'
        if not isinstance(groups, list) or not groups:
            return False, 'empty fast group is not split coverage'
        if (len(groups) >= GROUP_COUNTS[model] or len(set(groups)) != len(groups)
                or any(type(group) is not int or not 0 <= group < GROUP_COUNTS[model]
                       for group in groups)):
            return False, 'invalid reaction-group evidence'
        intervals.append((start, start + step))
    intervals.sort()
    begin, end = WINDOWS[model]
    previous = begin
    duration = 0.0
    for start, stop in intervals:
        if start < previous - 1e-12 or stop > end + 1e-12:
            return False, 'accepted intervals overlap or exceed window'
        duration += stop - start
        previous = stop
    coverage = duration / (end - begin)
    stored = stats.get('split_coverage')
    if (not finite(stored, upper=1.0)
            or not math.isclose(coverage, stored, rel_tol=1e-9, abs_tol=1e-12)):
        return False, 'stored coverage disagrees with accepted intervals'
    if coverage < 0.5:
        return False, f'coverage {coverage:.9g} < 0.5'
    if stats.get('full_calls') != len(attempts):
        return False, 'full audit count does not equal candidate attempt count'
    return True, 'pass'


def fixed_parameters(registration, historical, row):
    if row['method'] == 'candidate':
        candidate = registration['candidate']
        model = row['model']
        return dict(candidate, max_step=candidate['max_step'][model], target=row['target'],
                    audit_method='CVODE-KLU' if model == 'kecoli' else 'Radau',
                    fallback_method='CVODE-KLU' if model == 'kecoli' else 'BDF')
    config = row['config']
    matches = [item for item in historical
               if item.get('model') == row['model']
               and item.get('config') == config
               and (item.get('parameters') or {}).get('target') == row['target']
               and item.get('repeat') >= 0]
    return matches[0].get('parameters') if matches else None


def main():
    registration_path = OUT / 'registration.json'
    runs_path = OUT / 'runs.json'
    historical_path = ROOT / 'results/m3-continuation/baselines.json'
    local_baseline_path = ROOT / 'results/m4-same-machine-baselines-v2/runs.json'
    if not all(path.is_file() for path in (registration_path, runs_path, historical_path)):
        result = dict(M4_admission='not_admitted', evidence_integrity='fail',
                      reason='formal registration, runs, or historical baselines missing')
        (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    registration = json.loads(registration_path.read_text())
    rows = json.loads(runs_path.read_text()).get('runs', [])
    historical = json.loads(historical_path.read_text()).get('runs', [])
    local_baselines = (json.loads(local_baseline_path.read_text()).get('runs', [])
                       if local_baseline_path.is_file() else [])
    source_hash_checks = {
        relative: (ROOT / relative).is_file()
        and __import__('hashlib').sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        for relative, expected in registration.get('source_sha256', {}).items()
    }
    source_integrity = bool(source_hash_checks) and all(source_hash_checks.values())
    combinations = []
    integrity = source_integrity
    for model in MODELS:
        for target in TARGETS:
            candidate_rows = [row for row in rows if row.get('model') == model
                              and row.get('target') == target
                              and row.get('method') == 'candidate'
                              and row.get('warmup') is False]
            baseline_rows = [row for row in rows if row.get('model') == model
                             and row.get('target') == target
                             and row.get('method') == 'baseline'
                             and row.get('warmup') is False]
            ids = [row.get('repeat') for row in candidate_rows]
            repeat_check = len(ids) == 7 and set(ids) == set(range(7))
            config_check = (len({row.get('config') for row in candidate_rows}) == 1
                            and bool(candidate_rows)
                            and all(row.get('config') == registration['candidate'].get('identity',
                                      'audit-control-only-local1-frozen')
                                    for row in candidate_rows))
            parameter_values = [fixed_parameters(registration, historical, row)
                                for row in candidate_rows]
            parameter_check = bool(parameter_values) and all(value == parameter_values[0]
                                                              for value in parameter_values)
            accuracy_check = bool(candidate_rows) and all(metrics_pass(row, model)
                                                           for row in candidate_rows)
            split_results = [split_evidence(row, model) for row in candidate_rows]
            split_check = bool(split_results) and all(ok for ok, _ in split_results)
            baseline_ids = [row.get('repeat') for row in baseline_rows]
            formal_baseline_check = (len(baseline_ids) == 7 and set(baseline_ids) == set(range(7))
                                     and all(row.get('status') == 'success' for row in baseline_rows))
            locally_qualified = []
            for config in sorted({row.get('config') for row in local_baselines
                                  if row.get('model') == model and row.get('target') == target
                                  and row.get('warmup') is False}):
                local_rows = [row for row in local_baselines if row.get('model') == model
                              and row.get('target') == target and row.get('config') == config
                              and row.get('warmup') is False]
                local_ids = [row.get('repeat') for row in local_rows]
                if (len(local_rows) == 7 and set(local_ids) == set(range(7))
                        and all(metrics_pass(row, model) for row in local_rows)):
                    seconds = [row['seconds'] for row in local_rows]
                    locally_qualified.append(dict(config=config, rows=local_rows,
                                                  median_seconds=median(seconds), seconds=seconds))
            fastest_local = min(locally_qualified, key=lambda item:item['median_seconds']) if locally_qualified else None
            candidate_seconds = [row['seconds'] for row in candidate_rows if finite(row.get('seconds'), lower=math.nextafter(0.0, 1.0))]
            baseline_seconds = fastest_local['seconds'] if fastest_local else []
            baseline_check = formal_baseline_check and fastest_local is not None
            speedup = (median(baseline_seconds) / median(candidate_seconds)
                       if len(candidate_seconds) == 7 and len(baseline_seconds) == 7 else None)
            checks = dict(seven_formal_repeats=repeat_check, one_candidate_config=config_check,
                          fixed_parameters=parameter_check, per_repeat_accuracy=accuracy_check,
                          per_repeat_split_audit=split_check, seven_formal_baseline_repeats=formal_baseline_check,
                          same_machine_fastest_qualified_baseline=fastest_local is not None,
                          median_speedup_is_1_5_or_more=speedup is not None and speedup >= 1.5)
            integrity = integrity and all(checks.values())
            combinations.append(dict(model=model, target=target, checks=checks,
                candidate_median_seconds=median(candidate_seconds) if len(candidate_seconds) == 7 else None,
                candidate_iqr_seconds=(quantiles(candidate_seconds, n=4, method='inclusive')[2]
                                      - quantiles(candidate_seconds, n=4, method='inclusive')[0])
                                     if len(candidate_seconds) == 7 else None,
                fastest_local_baseline_config=fastest_local['config'] if fastest_local else None,
                baseline_median_seconds=median(baseline_seconds) if len(baseline_seconds) == 7 else None,
                baseline_iqr_seconds=(quantiles(baseline_seconds, n=4, method='inclusive')[2]
                                     - quantiles(baseline_seconds, n=4, method='inclusive')[0])
                                    if len(baseline_seconds) == 7 else None,
                speedup=speedup,
                split_failure_reasons=[reason for ok, reason in split_results if not ok]))
    result = dict(date='2026-11-15', scope='Independent fail-closed validation of registered formal M4 attempt',
        M4_admission='pass' if integrity else 'not_admitted',
        evidence_integrity='pass' if rows and source_integrity else 'fail',
        source_hash_checks=source_hash_checks, combinations=combinations,
        limitations=['Candidate fixed parameters are reconstructed from registration because formal rows store the registration identity rather than a full parameters object.',
                      'No fresh ODE solve is performed by this validator; it validates saved formal-run evidence.',
                      'A failed candidate configuration cannot qualify even when the solver returned status=success.'])
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
