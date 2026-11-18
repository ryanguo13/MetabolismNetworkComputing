# M4 Final Stop Audit

Date: 2026-11-15 (Asia/Shanghai)

This report records the final read-only audit for the frozen M4 protocol. It does not alter historical result files, admission state, accuracy targets, audit semantics, or coverage rules.

## Gatekeeper result

The M4 gate remains `not_admitted`. The required matrix is Teusink and k-ecoli457, two formal concentration/flux targets, full registered windows, at least 50% genuine reaction-split coverage, seven independent formal repeats, and median end-to-end speedup of at least `1.5x` against the fastest qualified mature baseline. No candidate satisfies the four model/target combinations.

The current read-only command `uv run python -m metabolism.m4_admission` reports `M4_admission: not_admitted`. It also reports `evidence_integrity: fail` because the working-tree continuation implementation has changed after the historical registration. The historical acceptance hashes therefore correctly reject treating the current source as the source of the old timing evidence. The historical state is intentionally preserved.

## Hard budget lower bounds

The maximum allowed candidate time is the qualified baseline median divided by `1.5`:

| Model / target | Fastest qualified baseline median | Maximum candidate budget |
|---|---:|---:|
| Teusink / `1e-3` | `0.717806875 s` | `0.478537917 s` |
| Teusink / `1e-5` | `0.727883167 s` | `0.485255444 s` |
| k-ecoli457 / `1e-3` | `1.407938792 s` | `0.938625861 s` |
| k-ecoli457 / `1e-5` | `1.392011000 s` | `0.928007333 s` |

The best measured parallel-audit screen generated candidates in `1.107494792 s` and reduced the eight independent audits to `0.512983750 s`. Its best end-to-end time was `1.620478542 s`, or `2.5461x` the matched short-window baseline. This already exceeds the k-ecoli formal budget by at least `1.73x`, before full-window startup, control, rollback, and coverage costs. Parallelism changes wall time only; it does not remove the required independent audits.

The DOP853 audit screen provides an even stronger stopping bound. Its registered non-audit candidate work was `5.610550248 s`, already `5.98x` the k-ecoli `1e-3` budget and `6.05x` the `1e-5` budget when audit cost is treated as zero. The measured fastest audit was about `2.27 ms` per call; the recorded `256` complete audit calls therefore add an optimistic lower bound of about `0.58 s`.

The W-MRI residual screen reached a locally correct early k-ecoli state only at a median of `0.345151896 s`, above the corresponding short-window budget of roughly `0.303–0.317 s`. Later accepted steps cost seconds. Reaction-wise rank-one integration required approximately `1e9` accepted steps at its best locally passing step size, giving a candidate-plus-audit lower bound of about `3.01e8 s` over the registered `[0,1]` window. Native C RHS evaluation measured only `0.940–0.998x` of the existing NumPy RHS and cannot change these bounds.

## Decision

No remaining screened mechanism has a positive audit-aware cost budget. Starting a seven-repeat full-window campaign would therefore violate the registered stopping rule and consume compute without a credible path to M4. The correct result for this session is a reproducible negative conclusion: retain all existing NPZs, manifests, ledgers, fingerprints, and failed screens; keep `M4_admission` as `not_admitted`; do not start PE, M5, or M6.

Validation performed during this audit: `uv run python -m unittest checks.test_continuation research.test_m4_admission_audit` passed (`18` tests).

## 2026-11-18 Follow-up Provenance Check

An append-only check in `research/audit_m4_source_history.py` compared the 61 paths registered by the two M3 acceptance records with the current worktree. Fifty-six still match. All five mismatched source files have an exact SHA-256 match in reachable Git history: `metabolism/adaptive.py` at commit `78d873600b235bed260a87599f9a257cfb35bf32`; `checks/test_continuation.py`, `metabolism/continuation.py`, `scripts/run-macos.sh`, and `scripts/setup-macos.sh` at `0516865c2da8e09373c7e620885f8edb2e8f2bd5`. The machine-readable result is `results/m4-source-history-audit/summary.json`.

This establishes that the registered source versions remain traceable in Git. It does not reproduce their historical execution environments, make the current worktree match those versions, or change the gatekeeper result. M4 remains `not_admitted`.

## 2026-11-18 Follow-up CVODE Runtime Screen

`research/m4_cvode_runtime_screen.py` compared the project's SUNDIALS 6.4.1 runtime with the installed Homebrew MPI-enabled 7.9.0 runtime on k-ecoli457 reference states 0 and 61 over one `1e-6` interval. Each backend/state had one warmup and seven interleaved measurements using the same Python RHS/Jacobian, CVODE-BDF/KLU tolerance, four query points, and independent tight Radau truth. The registration, reference and initial-state hashes, library hashes, raw NPZs, run ledger and output manifest are retained under `results/m4-cvode-runtime-screen-v1/`; all 32 run rows passed local accuracy, pool and nonnegativity checks, and all 43 manifest entries verified.

The 7.9.0 outputs were bitwise identical to 6.4.1. Median solve speedup was `1.013x` at state 0 (`0.3946 s` vs `0.3998 s`) and `0.987x` at state 61 (`0.1953 s` vs `0.1927 s`). The MPI-enabled 7.9.0 process also required `0.148 s` MPI initialization; model/truth preparation took about `2.86 s`. This installed-build comparison provides no material audit-aware budget improvement and does not justify candidate generation, full-window work or formal repeats. The comparison is runtime-configuration evidence, not an isolated SUNDIALS-version claim.

The admission checker remains `not_admitted` with `evidence_integrity=fail`; this screen does not satisfy any missing frozen M4 candidate, source-integrity, full-window, reaction-split coverage or end-to-end speedup gate. The existing stopping decision therefore stands.

## 2026-11-18 Historical Snapshot Integrity Check

A detached worktree at `0516865c2da8e09373c7e620885f8edb2e8f2bd5` contains all 61 source files registered by the two M3 acceptance records with exact SHA-256 matches (39 historical M3 paths and 22 continuation paths). Evaluating the current read-only gatekeeper against that historical root returned `evidence_integrity=pass`; all registered integrity and window checks passed. This verifies source snapshot identity only, not historical execution reproduction. The same evaluation still returned `M4_admission=not_admitted`: all four candidate combinations were unqualified, with no qualifying median speedups. The active worktree correctly remains `evidence_integrity=fail` because its source files differ from the historical registrations. No acceptance hashes or historical evidence were changed.

## 2026-11-18 Follow-up KLU Symbolic-Reuse Audit

The registered engineer report, .aris/workbench/outputs/engineer/M4_KLU_REUSE.md, was independently checked against the pinned SUNDIALS 6.4.1 source under results/scratch/sundials-6.4.1/. In src/cvode/cvode.c, CVodeReInit resets CVODE history; on the next CVode call, initial setup invokes the attached linear solver's initialize operation. KLU initialization sets FIRSTFACTORIZE=1 in src/sunlinsol/klu/sunlinsol_klu.c; its next setup runs sun_klu_analyze and a fresh numeric factorization. The FULL and PARTIAL branches of SUNLinSol_KLUReInit both free the prior symbolic and numeric factors. Within one uninterrupted solve, later setup calls can retain symbolic analysis and use sun_klu_refactor, with numeric refactor/factorization and the rest of the independent audit still required.

An independent source review confirmed this call chain and found no issue changing the report's conclusion. The supported pinned API therefore provides no route to avoid symbolic re-analysis across independent full-ODE audits; preserving internal KLU factor pointers would bypass its lifecycle contract. Decision: NO-GO for a timed probe of this mechanism. This negative result supplies no positive audit-aware cost budget; M4 remains not_admitted, and the frozen full-window, 50% reaction-split time coverage, seven-repeat and median 1.5x gates remain unchanged.
