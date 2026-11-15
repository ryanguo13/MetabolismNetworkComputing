"""Re-run every historically qualified mature baseline configuration locally.

Run only after the registered candidate formal batch finishes, to avoid
resource contention. Results are write-once and never alter historical data.
"""
import itertools
import json
import platform
import time

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, prepare, metrics
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m4-same-machine-baselines-v2'
REPEATS = 7


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration_path = OUT / 'registration.json'
    if registration_path.exists():
        raise FileExistsError('Preserve existing baseline evidence')
    historical_path = ROOT / 'results/m3-continuation/baselines.json'
    historical = json.loads(historical_path.read_text())['runs']
    sources = ('research/m4_same_machine_baselines.py', 'metabolism/adaptive.py',
               'metabolism/m3_continuation.py', 'metabolism/splitting.py')
    jobs = []
    for model, target in itertools.product(('teusink', 'kecoli'), TARGETS):
        for config in sorted({row['config'] for row in historical
                              if row['model'] == model and row['repeat'] >= 0
                              and row['parameters']['target'] == target}):
            repeats = [row for row in historical if row['model'] == model
                       and row['config'] == config and row['repeat'] >= 0
                       and row['parameters']['target'] == target]
            if (len(repeats) == 7 and {row['repeat'] for row in repeats} == set(range(7))
                    and all(row.get('metrics', {}).get('passes') is True for row in repeats)):
                jobs.append(dict(model=model, target=target, config=config,
                                 parameters=repeats[0]['parameters']))
    registration = dict(date='2026-11-15',
        purpose='Identify the fastest same-machine qualified mature full-system baseline for each frozen model/target.',
        scope='Re-run every historical seven-repeat qualified mature configuration on the original full window and output grid; one warmup and seven formal interleaved repeats per configuration.',
        criteria='Every config is qualified only if all seven local repeats pass original metrics and structural gates; choose the lowest local median among qualified configs.',
        configurations=jobs, formal_repeats=REPEATS, warmups_per_config=1,
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={p:digest(ROOT/p) for p in sources},
        historical_baselines_sha256=digest(historical_path))
    save(registration_path, registration)
    runs_path = OUT / 'runs.json'
    rows = []
    save(runs_path, dict(runs=rows))
    cases = {model: prepare(model) for model in ('teusink', 'kecoli')}
    # One warmup per configuration, then interleave all qualified configurations
    # within each model/target to reduce ordering and thermal bias.
    for job in jobs:
        row = dict(**{k: job[k] for k in ('model', 'target', 'config')}, repeat=-1, warmup=True)
        try:
            case = cases[job['model']]
            controller = AdaptiveSplit(case['system'], case['scale'], AdaptiveConfig(**job['parameters']),
                fixed_groups=np.arange(len(case['system'].groups)), **case['options'])
            start = time.perf_counter(); values, stats = controller.integrate(case['times'])
            row.update(status='success', seconds=time.perf_counter()-start,
                       metrics=metrics(values, case, job['target'], stats), stats=stats)
        except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
            row.update(status='failed', seconds=None, reason=str(exc), stats=getattr(exc, 'stats', None))
        rows.append(row); save(runs_path, dict(runs=rows))
        print(job['model'], job['target'], job['config'], 'warmup', row['status'], flush=True)
    for model, target in itertools.product(('teusink', 'kecoli'), TARGETS):
        subset = [job for job in jobs if job['model'] == model and job['target'] == target]
        for repeat in range(REPEATS):
            order = subset if repeat % 2 == 0 else list(reversed(subset))
            for job in order:
                row = dict(**{k: job[k] for k in ('model', 'target', 'config')},
                           repeat=repeat, warmup=False)
                try:
                    case = cases[model]
                    controller = AdaptiveSplit(case['system'], case['scale'], AdaptiveConfig(**job['parameters']),
                        fixed_groups=np.arange(len(case['system'].groups)), **case['options'])
                    start = time.perf_counter(); values, stats = controller.integrate(case['times'])
                    row.update(status='success', seconds=time.perf_counter()-start,
                               metrics=metrics(values, case, target, stats), stats=stats)
                except (SplitFailure, RuntimeError, ValueError, FloatingPointError, TimeoutError) as exc:
                    row.update(status='failed', seconds=None, reason=str(exc), stats=getattr(exc, 'stats', None))
                rows.append(row); save(runs_path, dict(runs=rows))
                print(model, target, job['config'], 'repeat', repeat, row['status'],
                      round(row['seconds'], 4) if row['seconds'] is not None else None, flush=True)
    save(OUT/'manifest.json', dict(registration_sha256=digest(registration_path),
        runs_sha256=digest(runs_path), source_sha256=registration['source_sha256']))
    print('same-machine baseline comparison complete', flush=True)


if __name__ == '__main__':
    main()
