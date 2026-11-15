"""Bounded nonlinear slow-remainder accuracy/cost probe; not M4 evidence."""
import json
import statistics
import time

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.adaptive import GroupScreen
from research.inner_dense import full_curve
from research.slow_tangent import SlowTangent
from research.test_rodas_continuous import nonlinear_case


def run(repeats=5):
    if type(repeats) is not int or repeats < 3:
        raise ValueError('Use at least three timing repeats')
    case, y = nonlinear_case()
    system = case['system']
    h = 0.1
    query = np.linspace(0.0, 1.0, 65)
    rhs = lambda t, z: system.S @ system.flux(0.0, z)
    truth = solve_ivp(rhs, (0.0, h), y, method='DOP853', rtol=3e-14,
                      atol=2e-14, dense_output=True)
    if not truth.success:
        raise RuntimeError(truth.message)
    truth_values = truth.sol(h * query)

    baseline = []
    baseline_errors = []
    baseline_counts = []
    screen = GroupScreen(system, case['scale'])
    for _ in range(repeats):
        start = time.perf_counter()
        result = full_curve(case, y, h, 1e-10, screen)
        elapsed = time.perf_counter() - start
        if not result['stats']['continuous_nonnegative'] or result['stats']['pool_error'] > 1e-10:
            raise RuntimeError('Full-system baseline failed physical guards')
        values = result['pieces'][0](query)
        baseline.append(elapsed)
        baseline_errors.append(float(np.max(np.abs(values - truth_values))))
        baseline_counts.append({k: result['stats'][k] for k in ('nfev', 'njev', 'nlu', 'accepted_steps',
                                                                 'physical_ledger_residual', 'physical_ledger_quadrature')})

    rows = []
    kernel = SlowTangent(case, [0])
    for sweeps in (0, 1, 2):
        timings = []
        errors = []
        counts = []
        correction_sizes = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = kernel.trial(y, h, tol=1e-10, sweeps=sweeps)
            elapsed = time.perf_counter() - start
            curve = result['pieces'][0]
            values = curve(query)
            timings.append(elapsed)
            errors.append(float(np.max(np.abs(values - truth_values))))
            counts.append({k: result['stats'][k] for k in ('nfev', 'njev', 'nlu', 'subflows',
                                                            'nonlinear_nfev', 'nonlinear_njev',
                                                            'ledger_rate_evaluations')})
            correction_sizes.append(float(np.max(np.abs(result['remainders']))) if sweeps else 0.0)
        rows.append(dict(sweeps=sweeps, median_seconds=statistics.median(timings),
                         max_error=statistics.median(errors), median_counts=counts[len(counts)//2],
                         max_slow_remainder=statistics.median(correction_sizes)))

    full_seconds = statistics.median(baseline)
    return dict(scope='single small nonlinear conservative model; local H=0.1; exploratory timings, not M4 qualification',
                repeats=repeats, model='research.test_rodas_continuous.nonlinear_case',
                state_dimension=len(y), reaction_count=system.S.shape[1],
                fast_reactions=2, slow_reactions=2, target_rtol=1e-10,
                truth='DOP853 rtol=3e-14 atol=2e-14; 65 equally spaced continuous-output queries',
                baseline=dict(method='full-system Radau with dense curve, positivity, pool, and physical-ledger checks', median_seconds=full_seconds,
                              median_max_error=statistics.median(baseline_errors),
                              median_counts=baseline_counts[len(baseline_counts)//2]),
                candidates=rows,
                speedups={str(r['sweeps']): full_seconds/r['median_seconds'] for r in rows},
                correction_effect={str(r['sweeps']): r['max_error'] for r in rows})


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
