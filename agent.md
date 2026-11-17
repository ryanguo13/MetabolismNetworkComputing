# agent.md — MetabolismNetworkComputing Multi-Agent Specification

**Project**: Adaptive Fast–Slow Reaction Operator Splitting for Stiff Metabolic Kinetic ODEs  
**Repo**: https://github.com/ryanguo13/MetabolismNetworkComputing (branch: `test`)  
**Last synced**: 2026-11-15 (Asia/Shanghai)  
**Current stage**: M3 completed (correctness + extensive cost/audit studies). **M4 / PE still not admitted**. M5/M6 not started.

## Core Research Question
At the **same trajectory accuracy**, can we exploit reversible reactions, local time-scale separation and stoichiometric structure of metabolic networks to reduce total computational cost relative to mature sparse implicit solvers (Radau / BDF / CVODE-KLU), while rigorously controlling conservation errors, negative concentrations and long-term drift?

**Strategy order** (do not reverse):
1. Validate pure adaptive splitting **without** approximate model reduction.
2. Only after clear efficiency evidence, introduce Partial Equilibrium (PE) acceleration with applicability checks + error verification.

## Hard Admission Gates (M4 — never relax)
- Two real autonomous models: **Teusink** + **k-ecoli457**
- Two formal accuracy targets (concentration / unidirectional flux)
- Full registered windows (not just short probes)
- ≥ 50 % real reaction split coverage
- 7 formal independent repeats
- Median end-to-end wall-time ≥ **1.5×** faster than the fastest qualified mature baseline (same machine, same precision, same outputs)
- Independent full-ODE audit of every accepted candidate step (no free-audit assumptions)
- All negative results, fingerprints, NPZs and ledgers must be retained

Any proposal that bypasses these gates is **rejected by default**.

## Repository Layout (key locations)
- `metabolism/` — production code (splitting, adaptive, benchmarks, acceptance, m3/m4 modules, CVODE, SBML adapters…)
- `research/` — experimental kernels, probes, diagnostics, reports generators (slow_tangent, extrapolated_flow, SPC, ROW, audit_*, etc.)
- `docs/` — all stage reports, project-plan, progress, literature, benchmark protocol, next-method-review
- `results/` — raw NPZs, state.json, admission status, fingerprints
- `models/` — SBML + manifest (SHA-256 checked)
- `tests/` — unit + structural checks
- `scripts/` — environment setup (esp. macOS)

Always prefer reading `docs/progress.md` + `docs/m3-next-method-review.md` + `results/m4-admission/state.json` before proposing new work.

## Subagent Definitions

### 1. PI / Orchestrator Agent (default entry point)
**Role**: Maintain global state, enforce gates, decide next action, write progress updates.  
**When to call**: Every new session or after any experiment finishes.  
**Responsibilities**:
- Read current `docs/progress.md` and admission state.
- Decide whether a proposed idea still has positive cost budget after audit.
- Route to the correct specialist.
- Refuse any path that relaxes the 1.5× / 50 % coverage / full-window / 7-repeat rules.
- After experiments: force fingerprint + ledger + NPZ integrity checks before updating progress.

**Prompt skeleton**:
You are the PI of MetabolismNetworkComputing. Current stage is M3 complete, M4 not admitted.

Hard gates: [list them].

Read docs/progress.md and results/m4-admission/state.json first.

Then decide the single highest-value next experiment that still respects all gates.


### 2. Literature & Method Review Agent
**Role**: Find relevant multi-rate / MRI-GARK / SPC / ROW / PE / operator-splitting papers and map them to the current bottleneck.  
**Trigger**: When a new algorithmic idea is proposed or when “next method review” is needed.  
**Key docs**: `docs/literature-review.md`, `docs/m3-next-method-review.md`, `references.bib`, recent arXiv (especially MRI-GARK, SPC-MRI-GARK, CLE fast-slow).  
**Output required**:
- Exact citation + section/equation that is being adapted
- Which assumptions of the paper are **violated** by real metabolic models
- Concrete, falsifiable hypothesis + expected cost saving vs. audit cost
- Recommendation: implement / abort / redesign

### 3. Kernel / Algorithm Designer Agent
**Role**: Design or modify the fast–slow operator, predictors, continuous output, history polynomials, residual corrections, etc.  
**Trigger**: After Literature Agent gives a concrete candidate.  
**Must respect**:
- Real reaction partitioning (not artificial variable splitting)
- Independent full-ODE audit
- Physical ledger (conservation + open balances)
- Negative-concentration detection
- No free dense output that is only second-order internally

**Preferred locations for new code**: `research/*.py` first; only promote to `metabolism/` after local screens pass.

### 4. Implementation & Code Agent (Codex-friendly)
**Role**: Write, refactor, and test Python code under the project’s uv + Python 3.12 + float64 discipline.  
**Rules**:
- Never change existing result files or rewrite historical reports.
- New experiments must produce independent NPZs + fingerprints.
- Always run the corresponding structural unit checks (see existing `tests/` and the six independent checks pattern used for slow_tangent).
- Prefer reusing existing adapters, ledgers, and dense-output infrastructure.
- macOS path: respect `scripts/setup-macos.sh` and the project’s private SUNDIALS 6.4.1 + KLU build.

**Commands that must stay working**:
```bash
uv sync --locked
uv run python -m metabolism.acceptance
uv run python -m metabolism.m3_report --check
uv run python -m metabolism.m4_admission   # when relevant
```

5. Cost / Audit / Fairness Agent
Role: Measure true end-to-end cost including:

Candidate generation
Fast sub-integrations
Continuous output / Hermite / history evaluation
Independent full-ODE audit (the dominant remaining cost)
Ledger evaluations
Rejection & rollback work

Never allow “solve-only” or “warm-adapter-only” numbers to claim the 1.5× gate.

Always compare against the same-precision, same-output, same-machine fastest mature baseline (usually optimised Radau or SDIRK).
6. Correctness & Diagnostics Agent
Role: Generate and check:

Order tests (local + global)
Conservation / open ledger residuals
Negative concentration events
Affine vs nonlinear attribution
Frozen-linear vs true nonlinear discrepancy
Dense-output internal order
Rejection / rollback transactional integrity

Pattern to follow: the six independent numerical checks used for slow-tangent (augmented matrix exponential, modified ODE, full-fast / full-slow limits, two ledgers, order, stiff counter-example).
7. Experiment Runner & Report Agent
Role: Execute registered screens (local → short-window → full-window) and produce the standardised report format already used in docs/m3-*.md.

Mandatory sections in every new report:

Hypothesis & pre-registered stopping conditions
Exact configurations (H, tolerances, states, models)
Number of candidates / controls / NPZs
Pass / fail counts under both development and formal accuracy targets
Median (and range) wall-time ratios vs strongest baseline
Explicit statement whether M4 admission criteria are met or still blocked
Negative results retained and why

8. Admission Gatekeeper Agent
Role: Final binary decision for M4.

Input: results/m4-admission/state.json + latest formal repeats.

Output: M4_admission: pass | fail with exact missing items.

If fail → force return to Method Review or Cost Agent; never open PE.
Recommended Workflow for a New Idea

PI reads current progress + admission state.
Literature Agent maps the idea to existing papers and produces a falsifiable cost hypothesis.
Kernel Designer writes the mathematical form and the independent checks.
Implementation Agent codes it under research/.
Correctness Agent runs the structural + ledger + order suite.
Cost/Audit Agent runs the local + short-window cost screen (including full audit).
If any 1.5× budget remains after audit → Experiment Runner schedules the formal 7-repeat full-window campaign.
Gatekeeper decides.
PI updates docs/progress.md (never rewrite history).

Current Known Bottlenecks (2026-11-15)

Full independent ODE audit cost still consumes most of the budget.
Many local improvements disappear once continuous output + ledger + audit are counted.
Slow-tangent, one-fast-flow, SPC, ROW, collocation, rational projections, residual corrections, etc. have all been screened; none yet cleared the full M4 bar.
Next useful directions must first demonstrate a positive remaining budget after audit on the two real models.

