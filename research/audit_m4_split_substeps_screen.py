"""Recompute eligible reaction-split coverage for a partial substep screen."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/m4-split-substeps-screen-v2/runs.json'
OUTPUT = ROOT / 'results/m4-split-substeps-screen-v2/coverage-audit.json'


def main():
    data = json.loads(SOURCE.read_text())
    rows = []
    for run in data['runs']:
        stats = run.get('stats') or {}
        attempts = stats.get('attempts') or []
        duration = float(run['prefix_end'])
        accepted = [a for a in attempts if a.get('status') == 'accepted']
        split = sum(float(a['h']) for a in accepted if a.get('groups'))
        full = sum(float(a['h']) for a in accepted if not a.get('groups'))
        rows.append(dict(
            model=run['model'], target=run['target'], substeps=run['substeps'],
            status=run['status'], seconds=run.get('seconds'),
            accuracy_pass=(run.get('metrics') or {}).get('passes', False),
            attempts=len(attempts), accepted=len(accepted),
            eligible_split_coverage=split / duration,
            accepted_full_system_fraction=full / duration,
            all_attempts_audited=stats.get('full_calls') == len(attempts),
            first_accepted_h=next((a['h'] for a in attempts if a.get('status') == 'accepted'), None),
        ))
    result = dict(
        source='results/m4-split-substeps-screen-v2/runs.json',
        source_sha256=__import__('hashlib').sha256(SOURCE.read_bytes()).hexdigest(),
        coverage_rule='Only accepted intervals with a non-empty fast reaction-group selection count as reaction-split coverage; empty fast groups run the complete system.',
        scope='Partial, stopped short-prefix development screen; not M4 evidence.',
        runs=rows,
        M4_admission=False,
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
