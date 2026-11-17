# Kecoli Slow-Bound Controller Screen

日期：2026-11-15（Asia/Shanghai）

## Hypothesis and stopping conditions

Hypothesis: Kecoli accepted macro steps may be limited by `slow_bound_limit=2` rather than by the independently audited splitting error. Increasing the limit to `4` or `8` might increase the accepted reaction-split step size while preserving the formal accuracy checks.

Stopping conditions were registered before the v3 run:

- Run Kecoli at target `1e-3` for `slow_bound_limit={2,4,8}`.
- Use only the original public output grid prefix ending at `<=1e-6`.
- Retain every controller attempt, including pre-audit slow-bound rejections.
- Independently audit every executed split trial with the complete nonlinear ODE.
- Reject clipping, invalid states, invariant drift, failed rollback, or failed original concentration/flux/net-flux metrics.
- Stop before full-window, seven-repeat, or formal speedup work because this is a bounded development screen.

The screen was not allowed to relax any M4 gate and was not a PE experiment.

## Exact configuration

| Field | Value |
|---|---|
| Model | k-ecoli457 |
| Target | `1e-3` concentration/unidirectional flux target |
| Prefix | Original public grid points through `8.697490026177835e-7` |
| Registered prefix limit | `1e-6` |
| Slow-bound limits | `2`, `4`, `8` |
| Initial step | `1e-8` |
| Maximum step | `1e-6` |
| Maximum attempts | `512` |
| Maximum consecutive rejections | `16` |
| Group refresh | Every accepted step (`group_every=1`) |
| Full audit backend | `CVODE-KLU`, `rtol=1e-11` |
| Fallback backend | `CVODE-KLU`, `rtol=1e-11` |
| Audit mode | `audit_control_only=True` |
| Repeats | One run per limit; no formal repeats |

The source registration and raw data are in `results/m4-kecoli-slow-bound-screen-v3/registration.json` and `results/m4-kecoli-slow-bound-screen-v3/runs.json`. Each successful run also stores `times`, `values`, `reference`, `flux_reference`, `scale`, and `y0` in a SHA-256-recorded NPZ.

## Results

| Limit | Seconds | Attempts | Accepted | Pre-audit rejects | Full ODE calls | Split coverage | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 26.1303 | 171 | 87 | 84 | 87 | 1.000000 | pass |
| 4 | 13.8737 | 83 | 44 | 39 | 44 | 1.000000 | pass |
| 8 | 30.1300 | 83 | 44 | 0 | 83 | 1.000000 | pass |

The `limit=4` run confirms that the controller bound was active: it reduced the number of executed split trials and approximately halved the short-prefix runtime relative to `limit=2`. The result is not a complete-window speed result. The fastest mature Kecoli baseline observed in the existing short-window evidence is approximately `0.45 s`, so the `13.8737 s` candidate remains about 30.6 times slower on this prefix, corresponding to a candidate/baseline ratio of about 30.6 and a baseline-over-candidate speedup of about `0.0327x`, far below the M4 requirement of `1.5x`.

## Audit and artifact checks

The following checks passed after the run:

- Every row satisfies `len(attempts) == accepted_steps + rejected_steps`.
- `full_calls` equals the number of attempts that actually executed a split trial.
- Pre-audit slow-bound rejections have `reason='coupled_slow_norm_bound'`, `rollback_verified=true`, and `full_ode_audit=false`.
- Every successful NPZ exists, has the recorded SHA-256, contains the registered arrays, and has finite candidate values with a shape consistent with the time grid.
- All registered source hashes match the current source files.
- The relevant continuation, splitting, independent admission-audit, and historical admission tests pass: 30 tests, 0 failures.

The implementation fix is in `metabolism/continuation.py`: pre-trial slow-bound rejections are now retained in `stats['attempts']` instead of being silently skipped. Regression coverage is in `checks/test_continuation.py::test_slow_bound_rejection_is_retained_before_audit`.

## Decision

`M4_admission: false`.

This screen supports the narrower conclusion that `slow_bound_limit` controls Kecoli step growth on the tested short prefix. It does not provide full-window coverage evidence, seven formal independent repeats, or a positive end-to-end cost budget. The method therefore does not advance to a formal M4 campaign. The negative result, raw attempts, NPZs, source fingerprints, and registration are retained for reproducibility.
