"""Offline complete reversible-group screening; no species freezing or PE."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

from .kecoli import KEcoliModel


def main():
    root = Path(__file__).resolve().parents[1]
    model = KEcoliModel(root / 'models/k-ecoli457.zip')
    data = np.load(root / 'results/m1-completion/kecoli-reference.npz')
    group_ids = model.data['network_data']['rxn_f_bInd_rxnInd'].astype(int)
    groups = [np.flatnonzero(group_ids == group) for group in np.unique(group_ids)]
    arity = [len(group) for group in groups]
    if min(arity) < 1 or max(arity) > 2 or sum(arity) != len(model.k):
        raise ValueError('Unexpected elementary reversible mapping')
    scale = data['scale']
    snapshots = []
    for index in (0, np.searchsorted(data['times'], 1e-4), np.searchsorted(data['times'], .01), len(data['times']) - 1):
        t, y = float(data['times'][index]), data['concentrations'][:, index]
        started = time.perf_counter()
        gradient = model.flux_jacobian(t, y)
        kappas = np.array([np.max(np.asarray(abs(model.S[:, group] @ gradient[group]).dot(scale)).ravel() / scale)
                            for group in groups])
        classification = []
        for h in (1e-7, 1e-5, 1e-3, .1):
            fast_groups = np.flatnonzero(h * kappas > 1)
            columns = np.concatenate([groups[i] for i in fast_groups]) if len(fast_groups) else np.array([], dtype=int)
            fast_S = model.S[:, columns]
            touched = np.flatnonzero(np.asarray(abs(fast_S).sum(axis=1)).ravel())
            dependencies = np.flatnonzero(np.asarray(model.dependency[columns].sum(axis=0)).ravel())
            connectivity = (fast_S != 0).astype(int)
            graph = (connectivity @ connectivity.T).tocsr()[touched][:, touched]
            nblocks, labels = connected_components(graph) if len(touched) else (0, np.array([], dtype=int))
            classification.append({'h': h, 'fast_groups': len(fast_groups), 'total_groups': len(groups),
                                   'fast_directional_reactions': len(columns), 'updated_states': len(touched),
                                   'rate_dependency_states': len(dependencies), 'connected_blocks': nblocks,
                                   'largest_connected_block': int(np.bincount(labels).max()) if nblocks else 0})
            if h == 1e-5:
                full_jacobian = model.S @ gradient
                slow_jacobian = full_jacobian - fast_S @ gradient[columns]
                full_modes = eigs(full_jacobian, k=3, which='LM', return_eigenvectors=False)
                slow_modes = eigs(slow_jacobian, k=3, which='LM', return_eigenvectors=False)
                classification[-1].update(full_largest_mode_magnitudes=sorted(map(float, abs(full_modes))),
                                           slow_largest_mode_magnitudes=sorted(map(float, abs(slow_modes))),
                                           largest_mode_reduction=float(max(abs(full_modes)) / max(abs(slow_modes))))
        snapshots.append({'time': t, 'group_jacobian_inf_norm_min': float(kappas.min()),
                          'group_jacobian_inf_norm_max': float(kappas.max()),
                          'diagnostic_seconds': time.perf_counter() - started, 'classifications': classification})
    report = {'model': 'k-ecoli457', 'group_policy': 'Original rxn_f_bInd_rxnInd; each elementary forward/reverse pair remains together',
              'groups': len(groups), 'pair_groups': arity.count(2), 'single_groups': arity.count(1),
              'snapshots': snapshots,
              'interpretation': 'Scaled Jacobian norm screening is conservative. At larger macro steps fast coverage can be broad; complete updated coordinates and all rate dependencies are retained. Not an eigenvalue proof or a speedup claim.'}
    (root / 'results/m1-completion/large-reaction-structure.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print('large reaction screening complete', flush=True)


if __name__ == '__main__':
    main()
