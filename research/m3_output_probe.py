"""Recompute local evidence and test full-RHS cubic Hermite output offline."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from metabolism.m2_experiments import error, save
from metabolism.m3_continuation import prepare, registration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/m3-research'


def midpoint(a, b, fa, fb, dt):
    return .5*(a+b)+dt/8*(fa-fb)


def run():
    # Exact for cubic data; unlike convex interpolation, not unconditionally positive.
    assert midpoint(1., 8., 3., 12., 1.) == 1.5**3
    assert midpoint(0., 0., -1., 1., 1.) < 0
    data = json.loads((OUT/'probe.json').read_text())
    for path, sha in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    case = prepare('kecoli')
    current = registration(case)
    for key in ('model_sha256', 'reference_sha256', 'window', 'output_count'):
        assert current[key] == data['registration'][key]
    model, scale = case['model'], case['scale']
    arrays = np.load(OUT/'probe.npz')
    def metrics(values, ref):
        vf = np.column_stack([model.flux(0., y) for y in values.T])
        vr = np.column_stack([model.flux(0., y) for y in ref.T])
        return dict(concentration=error(values, ref, scale, 1e-4),
                    flux=error(vf, vr, case['options']['flux_scale'], 1e-4),
                    net_flux=error(model.net_map@vf, model.net_map@vr, case['options']['net_scale'], 1e-4))
    rows = []
    for row in data['runs']:
        if row['status'] != 'success':
            continue
        si = (0.001, .01).index(row['start'])+1
        hi = (1e-6, 1e-4, 1e-3).index(row['h'])
        key = f"{si}-{hi}-{row['inner_rtol']}-{row['method']}"
        values, ref = arrays[key], arrays[f'ref-{si}-{hi}']
        for name, actual in metrics(values, ref).items():
            assert np.isclose(actual, row['grid'][name], rtol=1e-10, atol=1e-10)
        for name, actual in metrics(values[:, -1:], ref[:, -1:]).items():
            assert np.isclose(actual, row['endpoint'][name], rtol=1e-10, atol=1e-10)
        tick = time.perf_counter()
        deriv = [model.rhs(0., values[:, i]) for i in (0, 2, 4)]
        hermite = values.copy()
        for j in (0, 1):
            hermite[:, 2*j+1] = midpoint(values[:, 2*j], values[:, 2*j+2],
                                         deriv[j], deriv[j+1], row['h']/2)
        elapsed = time.perf_counter()-tick
        nonnegative = bool(np.all(hermite >= -1e-12*scale[:, None]))
        finite = bool(np.isfinite(hermite).all())
        drift = float(np.max(abs(model.pools@(hermite-values[:, 0:1]))))
        normalized_drift = drift/max(1e-12, float(np.max(abs(model.pools@values[:, 0]))))
        result = dict(start=row['start'], h=row['h'], inner_rtol=row['inner_rtol'], method=row['method'],
                      output_seconds=elapsed, finite=finite, nonnegative=nonnegative,
                      pool_drift=drift, relative_pool_drift=normalized_drift,
                      linear_grid=row['grid'], endpoint=row['endpoint'])
        if nonnegative and finite:
            result['hermite_grid'] = metrics(hermite, ref)
        result['sampled_pass'] = (nonnegative and finite and normalized_drift <= 1e-10
                                 and row['local_reference_budget_pass']
                                 and max(result.get('hermite_grid', {'failed':2}).values()) <= 1)
        rows.append(result)
    save(OUT/'output-probe.json', dict(scope='Post-hoc development probe: cubic Hermite using full RHS at the two fine-step endpoints and shared midpoint. No reference states used for construction. Only sampled quarter points are audited, not a continuous error or positivity guarantee.',
        original_saved_metrics_recomputed=True, cubic_exactness_check=True,
        positivity_counterexample_check=True, runs=rows,
        evidence_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [OUT/'probe.json', OUT/'probe.npz', Path(__file__), ROOT/'research/m3_coupling_probe.py',
                      *[ROOT/'metabolism'/name for name in ('__init__.py', 'cvode.py', 'kecoli.py',
                           'splitting.py', 'adaptive.py', 'm2_experiments.py', 'm3_continuation.py')]]}))
    for r in rows:
        if r['h'] >= 1e-4:
            print(r['start'], r['h'], r['inner_rtol'], r['method'], r['sampled_pass'], r.get('hermite_grid'))


if __name__ == '__main__':
    run()
