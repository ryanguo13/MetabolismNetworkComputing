"""Original k-ecoli first-step net-flux observable and default steady fluxes."""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
from pathlib import Path

import numpy as np

from .kecoli import KEcoliModel
from .large_benchmark import integrate, weighted_error, save_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/m1-completion'
    model = KEcoliModel(root / 'models/k-ecoli457.zip')
    data = np.load(out / 'kecoli-reference.npz')
    times, ref, scale = data['times'], data['concentrations'], data['scale']
    default = KEcoliModel(model.path, enzyme_factor=1)
    published = default.data['network_data']['unexpanded_model']['Vss']
    reproduced = default.net_map @ default.flux(0, default.y0)
    expected = published.copy()
    # The source aerobic_glucose wrapper explicitly disables these alternative
    # carbon channels, although Data.mat stores nonzero placeholder Vss values.
    expected[445:455] = 0
    steady = float(np.max(abs(reproduced - expected) / np.maximum(1, abs(expected))))
    flux_ref = model.net_map @ np.column_stack([model.flux(t, y) for t, y in zip(times, ref.T)])
    flux_scale = np.maximum(np.max(abs(flux_ref), axis=1), 1e-6)
    reference_checks = {}
    for method in ('BDF', 'Radau', 'CVODE_looser'):
        f = model.net_map @ np.column_stack([model.flux(t, y) for t, y in zip(times, data[method].T)])
        reference_checks[method] = weighted_error(f, flux_ref, flux_scale, 1e-3)
    records = []
    for tol in (1e-4, 1e-6, 1e-8, 1e-9):
        for method in ('BDF', 'Radau', 'CVODE-KLU'):
            y, _ = integrate(model, times, method, tol, scale)
            f = model.net_map @ np.column_stack([model.flux(t, state) for t, state in zip(times, y.T)])
            records.append({'method': method, 'rtol': tol,
                            'net_flux_error': {str(target): weighted_error(f, flux_ref, flux_scale, target) for target in (1e-2, 1e-3)}})
            print(method, tol, records[-1]['net_flux_error'], flush=True)
    report = {'observable': 'Original get_Vnet.m first elementary step per unexpanded reaction, including irreversible transport sign; not all steps are equal during transient enzyme redistribution',
              'default_steady_flux_max_scaled_difference': steady,
              'default_steady_state_rhs_inf_norm': float(np.max(abs(default.rhs(0, default.y0)))),
              'default_source_Vss': published.tolist(), 'expected_after_original_condition_switches': expected.tolist(),
              'condition_switches': 'Original aerobic wrapper zeros reactions 446–455; unadjusted Data.mat Vss retains their alternate-carbon placeholders',
              'reproduced_default_net_fluxes': reproduced.tolist(),
              'net_flux_scale': flux_scale.tolist(), 'reference_checks_at_1e3': reference_checks, 'configs': records,
              'net_flux_targets': [1e-2, 1e-3],
              'target_reason': 'Subtraction of large one-way rates amplifies state uncertainty: the three double-precision references disagree by up to 1.79 target units at 1e-5 net-flux accuracy. At 1e-3 their disagreement is below 5%. Concentration target remains 1e-5, elementary-flux target 1e-5.',
              'reference_pass': max(reference_checks.values()) < .05 and steady < 1e-8}
    save_json(out / 'kecoli-net-flux.json', report)
    if not report['reference_pass']:
        raise RuntimeError(f'Net flux audit gate failed: {reference_checks}; steady={steady}')


if __name__ == '__main__':
    main()
