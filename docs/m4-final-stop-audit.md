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
