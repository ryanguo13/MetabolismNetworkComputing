"""Independent audit of the post-failure classification-floor screen.

This reads raw screen records only. It does not import the screen or admission
validator, and it reports whether the candidate is worth a second formal run.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/m4-classification-floor-screen'
MODELS = ('teusink', 'kecoli')
TARGETS = (0.001, 1e-5)
FLOORS = (0.0, 1e-4, 1e-3)
WINDOWS = {'teusink': 10.0, 'kecoli': 1.0}


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value):
    return finite(value) and value > 0


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def accepted_coverage(row, failures):
    model = row['model']
    stats = row.get('stats') or {}
    attempts = stats.get('attempts')
    check(isinstance(attempts, list) and attempts, failures,
          f'{model}/{row["target"]}/{row["classification_floor"]}: missing attempts')
    if not isinstance(attempts, list) or not attempts:
        return 0.0
    check(stats.get('full_calls') == len(attempts), failures,
          f'{model}/{row["target"]}/{row["classification_floor"]}: full_calls mismatch')
    accepted = [item for item in attempts if item.get('status') == 'accepted']
    rejected = [item for item in attempts if item.get('status') == 'rejected']
    check(stats.get('accepted_steps') == len(accepted), failures,
          f'{model}/{row["target"]}/{row["classification_floor"]}: accepted count mismatch')
    check(all(item.get('rollback_verified') is True for item in rejected), failures,
          f'{model}/{row["target"]}/{row["classification_floor"]}: rollback evidence missing')
    intervals = []
    for item in accepted:
        start, h = item.get('time'), item.get('h')
        check(finite(start) and positive(h), failures,
              f'{model}/{row["target"]}/{row["classification_floor"]}: invalid accepted interval')
        if finite(start) and positive(h):
            intervals.append((start, start + h))
    intervals.sort()
    covered = 0.0
    previous = 0.0
    for start, stop in intervals:
        check(start >= previous - 1e-12, failures,
              f'{model}/{row["target"]}/{row["classification_floor"]}: overlapping accepted intervals')
        check(stop <= WINDOWS[model] + 1e-12, failures,
              f'{model}/{row["target"]}/{row["classification_floor"]}: interval exceeds window')
        if start >= previous - 1e-12:
            covered += stop - start
            previous = stop
    coverage = covered / WINDOWS[model]
    stored = row.get('split_coverage_recomputed')
    check(finite(stored) and math.isclose(stored, coverage, rel_tol=1e-9, abs_tol=1e-12), failures,
          f'{model}/{row["target"]}/{row["classification_floor"]}: stored coverage mismatch')
    return coverage


def main():
    failures = []
    registration = load(OUT / 'registration.json')
    rows = load(OUT / 'runs.json').get('runs', [])
    check(len(rows) == 24, failures, f'expected 24 rows, got {len(rows)}')
    expected_sources = registration.get('source_sha256', {})
    source_checks = {}
    for relative, expected in expected_sources.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        source_checks[relative] = actual == expected
        check(actual == expected, failures, f'source hash mismatch: {relative}')

    summaries = []
    for model in MODELS:
        for target in TARGETS:
            for floor in FLOORS:
                combo = [row for row in rows if row.get('model') == model
                         and row.get('target') == target
                         and row.get('classification_floor') == floor]
                check(len(combo) == 2, failures, f'{model}/{target}/{floor}: expected baseline+candidate')
                candidate = next((row for row in combo if row.get('method') == 'candidate'), None)
                baseline = next((row for row in combo if row.get('method') == 'baseline'), None)
                check(candidate is not None and baseline is not None,
                      failures, f'{model}/{target}/{floor}: missing method row')
                if candidate is None or baseline is None:
                    continue
                check(candidate.get('status') == 'success', failures,
                      f'{model}/{target}/{floor}: candidate did not complete')
                check(positive(candidate.get('seconds')) and positive(baseline.get('seconds')),
                      failures, f'{model}/{target}/{floor}: invalid timing')
                check(candidate.get('every_attempt_audited') is True, failures,
                      f'{model}/{target}/{floor}: candidate audit counter failed')
                coverage = accepted_coverage(candidate, failures)
                stats = candidate.get('stats') or {}
                attempts = stats.get('attempts') or []
                floor_values = [item.get('classification_step') for item in attempts]
                h_values = [item.get('h') for item in attempts]
                check(all(finite(value) and value >= 0 for value in floor_values), failures,
                      f'{model}/{target}/{floor}: invalid classification steps')
                check(all(finite(h) and finite(floor_value) and floor_value + 1e-15 >= h
                          for h, floor_value in zip(h_values, floor_values)), failures,
                      f'{model}/{target}/{floor}: classification step below actual h')
                summaries.append(dict(model=model, target=target, classification_floor=floor,
                                     candidate_seconds=candidate.get('seconds'),
                                     baseline_seconds=baseline.get('seconds'),
                                     speedup=baseline.get('seconds') / candidate.get('seconds'),
                                     coverage=coverage,
                                     accuracy_pass=(candidate.get('metrics') or {}).get('passes', False),
                                     attempts=len(attempts),
                                     accepted_steps=stats.get('accepted_steps'),
                                     full_calls=stats.get('full_calls')))

    by_combo = {}
    for item in summaries:
        by_combo.setdefault((item['model'], item['target']), []).append(item)
    candidate_worth_formal = True
    for combo, items in by_combo.items():
        # A second formal run is warranted only when at least one floor is both
        # accurate and has a plausible path to the frozen coverage/speed gates.
        candidate_worth_formal &= any(item['accuracy_pass'] and item['coverage'] >= .5
                                     and item['speedup'] >= 1.5 for item in items)
    result = dict(evidence_integrity='pass' if not failures else 'fail',
                  post_failure_candidate=True, source_hash_checks=source_checks,
                  summaries=summaries, candidate_worth_formal=bool(candidate_worth_formal),
                  failures=failures,
                  conclusion=('candidate merits second formal four-combination evaluation'
                              if candidate_worth_formal else
                              'candidate is post-failure evidence but does not merit formal admission rerun'))
    (OUT / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
