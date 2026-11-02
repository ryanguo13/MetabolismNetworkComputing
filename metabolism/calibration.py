"""Complete the small-model strict flux tolerance scan without rerunning M0/M1."""

import time

import numpy as np
import roadrunner

from .benchmark import ROOT, cvode_solve, error, scipy_solve, write_json
from .sbml import SBMLModel


def main():
    out = ROOT / 'results/m1-completion'
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for mid in ('BIOMD0000000064', 'BIOMD0000000051'):
        model = SBMLModel(ROOT / f'models/{mid}.xml')
        rr = roadrunner.RoadRunner(str(model.path))
        data = np.load(ROOT / f'results/m1/{mid}-reference.npz')
        times, ref, scale = data['times'], data['Radau_tighter'], data['scale']
        flux = np.column_stack([model.flux(t, y) for t, y in zip(times, ref.T)])
        flux_scale = np.maximum(np.max(abs(flux), axis=1), 1e-6)
        checks = {}
        for method in ('BDF', 'CVODE'):
            y, _ = cvode_solve(rr, model, times, 1e-12, scale) if method == 'CVODE' else scipy_solve(model, times, method, 1e-12, scale)
            f = np.column_stack([model.flux(t, x) for t, x in zip(times, y.T)])
            checks[method] = {'concentration': error(y, ref, scale, 1e-7), 'flux': error(f, flux, flux_scale, 1e-7)}
        if max(value for row in checks.values() for value in row.values()) > .05:
            raise RuntimeError(f'Strict flux reference unresolved: {mid}, {checks}')
        configs = [(method, tol) for tol in (1e-10, 1e-11) for method in ('BDF', 'Radau', 'CVODE')]
        def solve(method, tol):
            return cvode_solve(rr, model, times, tol, scale) if method == 'CVODE' else scipy_solve(model, times, method, tol, scale)
        warm = []
        for method, tol in configs:
            start = time.perf_counter()
            solve(method, tol)
            warm.append(time.perf_counter() - start)
        batch_size = max(1, int(np.ceil(.02 / min(warm))))
        rng = np.random.default_rng(20261102)
        for repetition in range(7):
            for index in rng.permutation(len(configs)):
                method, tol = configs[index]
                row = {'model_id': mid, 'method': method, 'rtol': tol, 'repeat': repetition, 'batch_size': batch_size}
                start = time.perf_counter()
                try:
                    for _ in range(batch_size):
                        y, stats = solve(method, tol)
                    elapsed = time.perf_counter() - start
                    f = np.column_stack([model.flux(t, x) for t, x in zip(times, y.T)])
                    row.update(status='success', batch_seconds=elapsed, seconds=elapsed / batch_size,
                               concentration_error_at_1e7=error(y, ref, scale, 1e-7),
                               flux_error_at_1e7=error(f, flux, flux_scale, 1e-7), counts=stats,
                               min_raw_output=float(y.min()))
                except Exception as exc:
                    row.update(status='failed', failure_reason=repr(exc), seconds=time.perf_counter() - start)
                results.append(row)
                write_json(out / 'strict-calibration-runs.json', results)
        write_json(out / f'{mid}-strict-reference.json', {'reference': 'Saved Radau 1e-12', 'target': 1e-7,
                                                        'checks': checks, 'flux_scale': flux_scale.tolist()})
        print(mid, checks, flush=True)


if __name__ == '__main__':
    main()
