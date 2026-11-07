# M3 macOS continuation evidence

Run on November 7, 2026 (Asia/Shanghai), on branch `test`. The [Chinese report](../../docs/m3-macos-continuation.md) explains the results and the separate M4 admission decision.

- `environment.json`: Mac/Python/NumPy/SciPy runtime, thread policy and native solver ABI. The project builds SUNDIALS 6.4.1 with KLU; Homebrew SUNDIALS 7 is not substituted into the registered binding.
- `probes.json`, `*-probes.npz`: 16 single-run development probes, including failures and timeouts. Each configuration registers its target, budget, grouping thresholds, startup interval and solver tolerances. Saved successful trajectories use the original M1 grids, scales, references and initial conditions.
- `baseline-calibration.json`: separate scans of five tolerances per mature solver. Both successful and failed calibration runs remain in the file.
- `inner-sensitivity.json`, `inner-sensitivity.npz`: nine local checks with identical frozen groups, three step sizes and three inner tolerances. The initial state is computed by a full-ODE prefix from the original initial conditions. Endpoint and interpolation errors are separate; independently tightened local references are saved. These are local sensitivity measurements, not full-window timing evidence.
- `baselines.json`, `*-baselines.npz`: actual-error-qualified baseline configurations, one warmup and seven cyclically interleaved formal repeats. Teusink uses BDF/Radau; k-ecoli457 also includes same-RHS CVODE-KLU. The same conservation, domain and open-ledger wrapper is used.
- `candidates.json`: explicit selection or exclusion for formal split timing. Single-run probes cannot establish speedup. A candidate must pass actual error checks, cover at least half the window with splitting, and show a timing prospect against a qualified repeated baseline before spending formal repeats.
- `tests.json`, `tests.txt`: the frozen historical test set and new `checks/` tests, with current-source fingerprints. The added native test explicitly runs on macOS even though one historical native test is Linux-only.
- `summary.json`, `acceptance.json`: recomputed concentration, flux, invariant and domain evidence, baseline medians/IQRs, failure accounting, historical evidence integrity and the M4 admission decision. Integrity passing does not mean M4 admission passing.

All splitting work, rejected trials, independent full-ODE audits, classification, partition construction, conservation diagnostics, initial-layer integration and fallback are timed. Model parsing, stored reference loading and offline error analysis are separate. Full-ODE audit states never supply accepted split outputs. Requested output values are convex interpolants of accepted fine split states and are independently audited; this is not a global continuous-error theorem.

No PE, projection, clipping, held-out validation, M5 or M6 is performed. Native CVODE does not expose every internal accepted state or an independent LU count. Unreturned counters on a failed native call are not invented. No isolated-worker peak-memory measurement is claimed.

```bash
bash scripts/setup-macos.sh
uv run python -m metabolism.m3_continuation --case probes
uv run python -m metabolism.m3_continuation --case baselines
uv run python -m metabolism.m3_continuation --case sensitivities
uv run python -m metabolism.m3_continuation --case candidates
uv run python -m metabolism.m3_continuation --case tests
uv run python -m metabolism.m3_continuation --case report
uv run python -m metabolism.m3_continuation --case check
uv run python -m metabolism.m3_report --check
```

The final two commands verify saved evidence without new ODE solves. Historical `results/m1*`, `results/m2`, `results/m3` and their attested reports are preserved.
