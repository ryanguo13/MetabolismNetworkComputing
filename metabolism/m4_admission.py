"""Read-only M4 status audit for the existing M3 continuation evidence.

This is deliberately not a general admission API for future method schemas.
It never solves an ODE or modifies an historical acceptance/manifest file.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('teusink', 'kecoli')
TARGETS = (1e-3, 1e-5)
WINDOWS = {'teusink': (0., 10.), 'kecoli': (0., 1.)}
# Frozen adapters: one group per Teusink net reaction; original k-ecoli pairs.
GROUP_COUNTS = {'teusink': 17, 'kecoli': 2666}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, lower=0., upper=math.inf):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lower <= value <= upper)


def metrics_pass(row, model):
    m = row.get('metrics') or {}
    required = {'concentration_error': 1., 'flux_error': 1.,
                'invariant_error': 1e-10}
    if model == 'teusink':
        required['relative_ledger_error'] = 1e-10
    else:
        required['net_flux_error'] = 1.
    return (row.get('status') == 'success' and m.get('passes') is True
            and m.get('finite') is True and m.get('significant_negative_output') is False
            and all(finite(m.get(k), upper=limit) for k, limit in required.items())
            and finite(row.get('seconds'), lower=math.nextafter(0., 1.)))


def split_evidence(row, model):
    """Validate existing MacroSplit attempt records, not self-labelled coverage."""
    s = row.get('stats') or {}
    attempts = s.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        return False
    accepted = [a for a in attempts if a.get('status') == 'accepted']
    if (not accepted or s.get('accepted_steps') != len(accepted)
            or not finite(s.get('split_seconds'), lower=math.nextafter(0., 1.))
            or not finite(s.get('partition_builds'), lower=1)
            or s.get('clipping_count') != 0 or s.get('projection_count') != 0):
        return False
    intervals = []
    for a in attempts:
        if a.get('status') == 'rejected':
            if a.get('rollback_verified') is not True:
                return False
        elif a.get('status') == 'accepted':
            if not all(finite(a.get(k), upper=1.) for k in
                       ('audit_max', 'audit_flux_max', 'audit_net_flux_max', 'estimator_max')):
                return False
            t, h = a.get('time'), a.get('h')
            if not finite(t) or not finite(h, lower=math.nextafter(0., 1.)):
                return False
            groups = a.get('groups')
            if not isinstance(groups, list) or not groups:
                # Empty fast partition is one complete slow flow, not real splitting.
                return False
            if (len(groups) >= GROUP_COUNTS[model] or len(set(groups)) != len(groups)
                    or any(type(g) is not int or not 0 <= g < GROUP_COUNTS[model] for g in groups)):
                return False
            intervals.append((t, t+h))
        else:
            return False
    begin, end = WINDOWS[model]
    intervals.sort()
    previous = begin
    duration = 0.
    for t, stop in intervals:
        if t < previous - 1e-12 or stop > end + 1e-12:
            return False
        duration += stop-t
        previous = stop
    coverage = duration/(end-begin)
    return (finite(s.get('split_coverage'), upper=1.)
            and math.isclose(coverage, s['split_coverage'], rel_tol=1e-9, abs_tol=1e-12)
            and coverage >= .5)


def qualify_repeats(rows, model, target, split=False):
    """Check one fixed historical configuration; reject mixed/duplicate repeats."""
    formal = [r for r in rows if type(r.get('repeat')) is int and r['repeat'] >= 0]
    ids = [r['repeat'] for r in formal]
    checks = dict(seven_distinct_formal_repeats=len(ids) == 7 and set(ids) == set(range(7)),
                  one_config=len({r.get('config') for r in formal}) == 1,
                  fixed_parameters=bool(formal) and all(r.get('parameters') == formal[0].get('parameters') for r in formal),
                  model_target=bool(formal) and all(r.get('model') == model and
                      (r.get('parameters') or {}).get('target') == target for r in formal),
                  per_repeat_accuracy=bool(formal) and all(metrics_pass(r, model) for r in formal))
    if split:
        checks['per_repeat_split_coverage_and_audits'] = bool(formal) and all(split_evidence(r, model) for r in formal)
    good = all(checks.values())
    seconds = [r['seconds'] for r in formal] if good else []
    qs = quantiles(seconds, n=4, method='inclusive') if seconds else None
    return dict(qualified=good, checks=checks,
                median_seconds=median(seconds) if seconds else None,
                iqr_seconds=qs[2]-qs[0] if qs else None,
                formal_repeat_ids=ids)


def check_hashes(root, document):
    hashes = document.get('evidence_sha256')
    if not isinstance(hashes, dict) or not hashes:
        return False
    return all((root / path).is_file() and digest(root / path) == expected
               for path, expected in hashes.items())


def evaluate(root=ROOT):
    root = Path(root)
    paths = ['results/m3/acceptance.json', 'results/m3-continuation/acceptance.json',
             'results/m3-continuation/candidates.json', 'results/m3-continuation/baselines.json']
    missing = [p for p in paths if not (root/p).is_file()]
    if missing:
        return dict(scope='Existing M3 continuation evidence only', M4_admission='not_admitted',
                    evidence_integrity='fail', missing=missing, combinations=[])
    historical, saved, candidates, baselines = [read(root/p) for p in paths]
    checks = dict(historical_hashes=check_hashes(root, historical),
                  continuation_hashes=check_hashes(root, saved),
                  historical_correctness=historical.get('correctness') == 'pass',
                  continuation_checks=bool(saved.get('checks')) and all(v is True for v in saved['checks'].values()),
                  all_model_target_selections=all(sum(1 for r in
                      candidates.get('selected', []) + candidates.get('skipped', [])
                      if r.get('model') == model and r.get('target') == target) == 1
                      for model in MODELS for target in TARGETS))
    combinations = []
    for model in MODELS:
        reg = baselines.get('registration', {}).get(model, {})
        checks[f'{model}_registered_window'] = reg.get('window') == list(WINDOWS[model])
        for target in TARGETS:
            def relevant(records):
                return [r for r in records if r.get('model') == model and
                        (r.get('parameters') or {}).get('target') == target]
            cb = relevant(candidates.get('runs', []))
            bb = relevant(baselines.get('runs', []))
            candidate = qualify_repeats(cb, model, target, split=True)
            configs = sorted({r.get('config') for r in bb if isinstance(r.get('config'), str)})
            options = [dict(config=config, **qualify_repeats(
                [r for r in bb if r.get('config') == config], model, target)) for config in configs]
            eligible = [r for r in options if r['qualified']]
            fastest = min(eligible, key=lambda r:r['median_seconds']) if eligible else None
            speedup = fastest['median_seconds']/candidate['median_seconds'] if fastest and candidate['qualified'] else None
            combinations.append(dict(model=model, target=target, candidate=candidate,
                fastest_qualified_baseline=fastest, baseline_configurations=options,
                median_speedup=speedup, passes=bool(speedup is not None and speedup >= 1.5)))
    evidence_ok = all(checks.values())
    # Re-aggregation cannot promote unverified new data beyond the saved acceptance.
    admitted = evidence_ok and saved.get('M4_admission') == 'pass' and all(r['passes'] for r in combinations)
    studies = []
    for relative in ('results/m3-audit-budget/summary.json',
                     'results/m3-audit-calibration/summary.json',
                     'results/m3-formal-audit-reanalysis/summary.json',
                     'results/m3-audit-finite-validation/summary.json',
                     'results/m3-selected-cost/summary.json',
                     'results/m3-audit-output-cost/summary.json',
                     'results/m3-audit-output-corrected/summary.json',
                     'results/m3-audit-output-integrity/summary.json',
                     'results/m3-slow-tangent/summary.json',
                     'results/m3-slow-tangent-integrity/summary.json',
                     'results/m3-reusable-audit-corrected/summary.json'):
        if (root/relative).is_file():
            summary = read(root/relative)
            studies.append(dict(path=relative, sha256=digest(root/relative),
                                M4_admission=summary.get('M4_admission'),
                                role='Development evidence; never substitutes for formal trajectory repeats'))
    return dict(date='2026-11-15', scope='Read-only aggregation of existing M3 continuation evidence; not a future-method certification API',
                M4_admission='pass' if admitted else 'not_admitted',
                evidence_integrity='pass' if evidence_ok else 'fail', checks=checks,
                formal_concentration_flux_targets=list(TARGETS), formal_kecoli_net_flux_target=1e-3,
                required_split_coverage=.5, required_median_speedup=1.5,
                combinations=combinations, development_studies=studies,
                missing_for_admission=['Qualifying genuine reaction-split trajectories for every model/target with seven distinct formal repeats and >=1.5 median speedup'] if not admitted else [],
                audit_policy='Runtime complete nonlinear ODE audit remains required. Per-step paired solves are not an original hard gate; reliable independent tightening calibration is required.',
                inputs_sha256={p:digest(root/p) for p in paths},
                source_sha256={p:digest(root/p) for p in
                    ('metabolism/m4_admission.py', 'research/test_m4_admission.py')
                    if (root/p).is_file()},
                limitations=['No ODE rerun or fresh trajectory metrics recomputation.',
                    'Existing acceptance hashes protect previously audited curves; new method output requires its own independent validation.',
                    'Development timing is not pooled with historical formal baselines.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write only results/m4-admission/state.json')
    args = parser.parse_args()
    result = evaluate()
    if args.write:
        out = ROOT/'results/m4-admission/state.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
