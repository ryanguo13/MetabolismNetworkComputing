# M3 audited adaptive splitting evidence

Implemented November 3–4, 2026 (Asia/Shanghai). See [the stage report](../../docs/m3-completion-report.md) and `acceptance.json` for correctness and the separate stopping decision.

- `calibration.json`: 144 local exact-flow/matrix-exponential checks, estimator underestimation, hidden-component RMS counterexample and threshold sensitivity.
- `toy-registration.json`, `toy-runs.json`, `toy-warmups.json`: fixed versus 1/5/20 accepted-step grouping, hysteresis on/off, seven interleaved repeats after warmup.
- `validation.json`: stiff initial layers, explicit nonlinear enzyme pools, Robertson, a rejected stiff step and rollback, known piecewise-constant event, and a controlled chatter trace.
- `sensitivities.json`: development group thresholds and tenfold inner-tolerance tightening with identical frozen maps.
- `teusink-*`, `kecoli-*`: unchanged M1 grids, scales, references and full registered windows at two concentration/flux targets. Each target has seven configurations, one warmup and seven formal repeats.
- `*-full-*`: full-ODE controls using the same diagnostics, preprocessing and strict fallback tolerances. These quantify added attempts, not the fastest solver at a target accuracy.
- `*-trajectories.npz`: references and first-repeat outputs. A record's `trajectory_key` identifies an array if identical outputs are stored once; otherwise the configuration ID is the array key.
- `screening.json`: exact norm agreement at four states and seven measurements per state before/after rank-one reversible-group optimization. Includes default-guard fallback and a separate large-model macro-step probe with a deliberately relaxed slow guard, only to exercise estimator audits. Three audited rejections end in safe fallback. The CLI reuses saved profiling samples unless `--refresh-screening` is supplied.
- `*-summary.json`, `calibration-cost.png`: all timings, medians/IQRs, actual errors, fallback coverage, grouping and audit costs.
- `tests.json`, `tests.txt`: Windows and native WSL checks plus source fingerprints. Historical M1/M2 evidence is preserved.

Attempt logs include frozen group IDs, accepted-state index, step size, WRMS and maximum estimates, independent midpoint/endpoint concentration and flux audits, rejection reasons and verified rollback. Accepted history excludes rejected group proposals. Rejected work still contributes to timing and counters.

No PE, extrapolation, projection, clipping or unaudited acceptance is implemented. Full-window outputs are predominantly obtained by persistent full-ODE fallback after a registered eight-attempt splitting budget; they do not qualify a pure splitting solver. M3's current error/output/restart strategy has no demonstrated net efficiency benefit, so M4 is not admitted. No M5 or M6 work is implied.

Native CVODE exposes RHS arguments and outputs, not every internal accepted state or LU factorization count. The metadata records that limit explicitly. No isolated-worker peak memory measurement is claimed. All comparisons are within the respective runtime; Windows and WSL timings are not combined into cross-model speed claims.
