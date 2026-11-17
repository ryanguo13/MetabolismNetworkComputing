"""Validate SUNDIALS ExtSTS on a stiff scalar split before real-model work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import sundials4py
import sundials4py.arkode as arkode
import sundials4py.core as core

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rhs(rate: float):
    def rhs(t, state, derivative, user_data):
        core.N_VGetNumpyArray(derivative)[:] = rate * core.N_VGetNumpyArray(state)
        return 0
    return rhs


def solve(h: float | None, rtol: float, atol: float) -> dict:
    _, context = core.SUNContext_Create(0)
    initial = core.N_VNew_Serial(1, context)
    core.N_VGetNumpyArray(initial)[:] = 1.0
    fast_rhs, slow_rhs = make_rhs(-1000.0), make_rhs(-0.1)
    view = arkode.MRIStepCreateExtSTS(fast_rhs, slow_rhs, None, 0.0, initial, context)
    memory = view.get()
    _, inner_view = arkode.MRIStepGetSTS(memory)
    inner = inner_view.get()
    guess = core.N_VClone(initial)
    core.N_VConst(1.0, guess)
    estimator = core.SUNDomEigEstimator_Power(guess, 20, 0.01, context)
    core.SUNDomEigEstimator_SetRhs(estimator, fast_rhs)
    arkode.LSRKStepSetDomEigEstimator(inner, estimator)
    arkode.LSRKStepSetSTSMethod(inner, arkode.ARKODE_LSRK_RKC_2)
    arkode.LSRKStepSetMaxNumStages(inner, 10000)
    arkode.ARKodeInit(memory)
    arkode.ARKodeSStolerances(memory, rtol, atol)
    arkode.ARKodeSetMaxNumSteps(memory, 200000)
    if h is None:
        arkode.ARKodeSetInitStep(memory, 0.01)
        arkode.ARKodeSetMaxStep(memory, 0.01)
    else:
        arkode.ARKodeSetFixedStep(memory, h)
    output = core.N_VClone(initial)
    flag, reached = arkode.ARKodeEvolve(memory, 0.01, output, arkode.ARK_NORMAL)
    value = float(core.N_VGetNumpyArray(output)[0])
    expected = math.exp(-1000.1 * 0.01)
    _, steps = arkode.ARKodeGetNumSteps(memory)
    _, attempts = arkode.ARKodeGetNumStepAttempts(memory)
    return dict(step_size=h, rtol=rtol, atol=atol, flag=flag, reached_time=float(reached), value=value,
                expected=expected, relative_error=abs(value - expected) / expected,
                accepted_steps=int(steps), attempts=int(attempts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    registration = dict(
        equation="y' = -1000*y - 0.1*y", initial_value=1.0, interval=[0.0, 0.01],
        exact_solution='exp(-1000.1*t)',
        adaptive_tolerances=[[1e-8, 1e-12], [1e-10, 1e-14], [1e-12, 1e-16]],
        fixed_step_checks=[1e-2, 1e-3, 1e-4], adaptive_initial_and_max_step=1e-2,
        acceptance='at least one adaptive tolerance reaches t=0.01 with endpoint relative error <= 1e-3',
        method='SUNDIALS ARKODE MRIStep ExtSTS with RKC2 and power dominant-eigenvalue estimate')
    (out / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    rows = [solve(h, 1e-8, 1e-12) for h in registration['fixed_step_checks']]
    rows.extend(solve(None, *tolerances) for tolerances in registration['adaptive_tolerances'])
    (out / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    adaptive = rows[-1]
    summary = dict(
        status='pass' if any(row['flag'] == 0 and row['relative_error'] <= 1e-3
                             and row['reached_time'] == 0.01 for row in rows[3:]) else 'fail',
        adaptive_rows=rows[3:], fixed_step_rows=rows[:3],
        environment=dict(python=platform.python_version(), sundials4py=str(sundials4py.__version__)
                         if hasattr(sundials4py, '__version__') else '7.9.0',
                         platform=platform.platform()),
        limitations=['Scalar correctness only; no reaction network, positivity or ledger test.',
                     'No performance comparison or M4 admission evidence.'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    arrays = {key: np.array([np.nan if row[key] is None else row[key] for row in rows])
              for key in ('step_size', 'rtol', 'atol', 'flag', 'reached_time', 'value', 'expected',
                          'relative_error', 'accepted_steps', 'attempts')}
    np.savez_compressed(out / 'scalar-runs.npz', **arrays)
    manifest = {name: sha256(out / name) for name in
                ('registration.json', 'runs.json', 'summary.json', 'scalar-runs.npz')}
    manifest['research/m4_extsts_scalar_probe.py'] = sha256(Path(__file__))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
