"""Prepare known publisher source files and registered Octave audit inputs."""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat

from .kecoli import KEcoliModel


def main():
    root = Path(__file__).resolve().parents[1]
    model = KEcoliModel(root / 'models/k-ecoli457.zip')
    directory = root / 'results/scratch'
    model.source_directory(directory / 'k-ecoli457')
    pilot = solve_ivp(model.rhs, (0, 1), model.y0, method='BDF', jac=model.jacobian,
                      rtol=1e-7, atol=1e-10)
    if not pilot.success:
        raise RuntimeError(pilot.message)
    times = np.unique(np.r_[0, np.geomspace(1e-10, 1, 100), np.linspace(0, 1, 101)])
    states = np.column_stack([model.y0, pilot.y[:, -1]])
    savemat(directory / 'kecoli-input.mat', dict(y0=model.y0, check_states=states, times=times))
    output = root / 'results/m1-completion'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'kecoli-source-inputs.npz', y0=model.y0, check_states=states, times=times)
    print('Prepared original publisher sources and two independent audit states')


if __name__ == '__main__':
    main()
