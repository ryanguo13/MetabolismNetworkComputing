"""Independent raw evidence audit for the growth-safety short-window screen."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m4-growth-safety-screen'
MODELS = ('teusink', 'kecoli')
TARGETS = (0.001, 1e-5)
SAFETIES = (0.9, 1.8)
WINDOW = 1e-6
EXPECTED_ORDER = [
    ('candidate', .9), ('baseline', None), ('candidate', 1.8),
    ('candidate', 1.8), ('baseline', None), ('candidate', .9),
    ('baseline', None), ('candidate', .9),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    failures = []
    def check(condition, message):
        if not condition:
            failures.append(message)

    registration = json.loads((OUT / 'registration.json').read_text())
    runs = json.loads((OUT / 'runs.json').read_text()).get('runs', [])
    check(registration.get('interval') == [0., WINDOW], 'registered interval mismatch')
    check(tuple(registration.get('safeties', ())) == SAFETIES, 'registered safety factors mismatch')
    check(len(runs) == 32, f'expected 32 rows, got {len(runs)}')
    sources = {}
    for relative, expected in registration.get('source_sha256', {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        sources[relative] = actual == expected
        check(actual == expected, f'source hash mismatch: {relative}')

    deviations = [
        'registration declares one warmup, but no separate warmup row was executed',
        'growth_safety=1.8 has two exploratory runs per combination, not three',
    ]
    summaries = []
    for model in MODELS:
        for target in TARGETS:
            combo = [r for r in runs if r.get('model') == model and r.get('target') == target]
            actual = [(r.get('method'), r.get('growth_safety')) for r in combo]
            check(actual == EXPECTED_ORDER, f'{model}/{target}: order mismatch: {actual}')
            candidates = {s: [r for r in combo if r.get('method') == 'candidate'
                              and r.get('growth_safety') == s] for s in SAFETIES}
            baseline = [r for r in combo if r.get('method') == 'baseline']
            check(len(baseline) == 3, f'{model}/{target}: expected 3 baseline controls')
            check(all(r.get('status') == 'success' and finite(r.get('seconds')) and r['seconds'] > 0
                      for r in baseline), f'{model}/{target}: invalid baseline rows')
            baseline_median = sorted(r['seconds'] for r in baseline)[len(baseline)//2] if baseline else math.nan
            for safety in SAFETIES:
                group = candidates[safety]
                expected_n = 3 if safety == .9 else 2
                check(len(group) == expected_n,
                      f'{model}/{target}/{safety}: expected {expected_n} candidates')
                positions = [i for i, item in enumerate(combo)
                             if item.get('method') == 'candidate' and item.get('growth_safety') == safety]
                for position, row in zip(positions, group):
                    check(row.get('repeat') == position and row.get('warmup') is False,
                          f'{model}/{target}/{safety}: repeat metadata mismatch')
                    repeat = position
                    params = row.get('parameters') or {}
                    check(params.get('growth_safety') == safety, f'{model}/{target}/{safety}: config mismatch')
                    check(params.get('audit_control_only') is True and params.get('local_fraction') == 1.,
                          f'{model}/{target}/{safety}: audit-control config mismatch')
                    check(row.get('status') == 'success' and finite(row.get('seconds')) and row['seconds'] > 0,
                          f'{model}/{target}/{safety}: candidate failed')
                    stats = row.get('stats') or {}
                    attempts = stats.get('attempts') or []
                    check(attempts and stats.get('full_calls') == len(attempts),
                          f'{model}/{target}/{safety}: missing full audit per attempt')
                    check(row.get('every_attempt_audited') is True,
                          f'{model}/{target}/{safety}: stored audit marker false')
                    rejected = [a for a in attempts if a.get('status') == 'rejected']
                    check(all(a.get('rollback_verified') is True for a in rejected),
                          f'{model}/{target}/{safety}: rejection rollback missing')
                    accepted = [a for a in attempts if a.get('status') == 'accepted']
                    intervals = sorted((a.get('time'), a.get('h')) for a in accepted)
                    previous = 0.
                    covered = 0.
                    for start, h in intervals:
                        check(finite(start) and finite(h) and h > 0 and start >= previous - 1e-14,
                              f'{model}/{target}/{safety}: invalid or overlapping accepted interval')
                        if finite(start) and finite(h) and h > 0 and start >= previous - 1e-14:
                            covered += h
                            previous = start + h
                    check(previous <= WINDOW + 1e-14, f'{model}/{target}/{safety}: interval past window')
                    coverage = covered / WINDOW
                    stored_coverage = row.get('split_coverage_recomputed')
                    check(finite(stored_coverage) and math.isclose(coverage, stored_coverage,
                          rel_tol=1e-9, abs_tol=1e-12), f'{model}/{target}/{safety}: coverage mismatch')
                    metrics = row.get('metrics') or {}
                    check(metrics.get('finite') is True and metrics.get('significant_negative_output') is False,
                          f'{model}/{target}/{safety}: finite/domain gate failed')
                    check(metrics.get('invariant_error', math.inf) <= 1e-10
                          and metrics.get('relative_ledger_error', math.inf) <= 1e-10,
                          f'{model}/{target}/{safety}: invariant/ledger gate failed')
                    control = max(metrics.get('concentration_error', math.inf),
                                  metrics.get('flux_error', math.inf),
                                  metrics.get('net_flux_error', 0.))
                    summaries.append(dict(model=model, target=target, safety=safety,
                        repeat=repeat, candidate_seconds=row.get('seconds'),
                        baseline_median_seconds=baseline_median,
                        speedup=baseline_median / row['seconds'] if finite(row.get('seconds')) else None,
                        coverage=coverage, accuracy_control=control,
                        accuracy_pass=bool(control <= 1), attempts=len(attempts),
                        accepted=len(accepted), rejected=len(rejected), full_calls=stats.get('full_calls')))

    grouped = {}
    for row in summaries:
        grouped.setdefault((row['model'], row['target'], row['safety']), []).append(row)
    plausible = True
    for key, values in grouped.items():
        speed = sorted(v['speedup'] for v in values if finite(v.get('speedup')))
        coverage = [v['coverage'] for v in values]
        accuracy = [v['accuracy_pass'] for v in values]
        plausible &= (len(speed) == 3 and speed[1] > 1 and min(coverage) >= .5 and all(accuracy))
    result = dict(evidence_integrity='pass' if not failures else 'fail',
                  source_hash_checks=sources, summaries=summaries,
                  protocol_deviations=deviations,
                  candidate_worth_full_window=bool(plausible) and not deviations, failures=failures,
                  conclusion=('growth candidate merits a full-window screen' if plausible else
                              'growth candidate does not merit full-window or formal screening'))
    (OUT / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
