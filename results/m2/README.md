# M2 fixed splitting evidence

Date: 2026-11-03, Asia/Shanghai. See [the report](../../docs/m2-completion-report.md) for the acceptance outcome and limits.

- `b0.json`: 32 exact-flow linear scans (two chains, four stiffness ratios, two initial states, two compositions), two explicit enzyme scans, eight Robertson initial-layer runs, and an intentional negative-domain failure.
- `*-registration.json`: model hash, frozen groups and species scales, bounded short window, reference checks, output policy, runtime, inner tolerances and repeat count.
- `*-warmups.json`, `*-runs.json`: one warmup and seven interleaved repeats for each of 12 configurations per real model. Integration success and accuracy qualification are separate.
- `*-trajectories.npz`: independent full-system references and first-repeat macro-endpoint trajectories. No interpolation between composed maps is asserted.
- `*-summary.json`: all timings, medians, IQRs, concentration/flux errors, invariants and subflow-tolerance sensitivity.
- `tests.json`, `tests.txt`: recorded automated test commands and outcomes.
- `acceptance.json`: recomputed admission checks and SHA-256 attestations, including implementation and test sources.
- `convergence.png`, `restart-cost.png`: convergence/order reduction and subflow costs. Ledger timings are already included in subflow timings.

The real-model windows are 0–0.001 min (Teusink) and 0–0.0001 in the source integration coordinate (k-ecoli457). They are subsets of M1's registered windows. These are correctness/cost probes, not full-window accuracy qualification, held-out validation or evidence of speedup. Chassagnole remains excluded from autonomous splitting. No PE, clipping, projection, adaptive macro steps or dynamic partitions are used.

The source ZIP remains an ignored local cache, fetched and verified with `python -m metabolism.fetch_models`. The published model and original semantic validation from M1 are unchanged.
