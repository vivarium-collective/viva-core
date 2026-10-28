# viva-core Milestone 3 — native process ports (plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Port the first hot v2ecoli processes from Python to NATIVE RUST so the Rust engine runs them without a Python callback (M1 showed native ≈6× over Python-process-on-Rust-engine), staying BIT-IDENTICAL to the Python oracle — and measure the first real quadrant-2 (all-native) numbers. Establishes the native-port pattern: RNG parity + build-time sim_data→config + native factory registration.

**Approach (from the M3 design note, scratchpad/m3_process_ports_design.md):** bit-identical RNG by REIMPLEMENTING numpy's legacy `RandomState` (MT19937 core + numpy's own per-distribution algorithms + `stochasticRound`) — validated bit-for-bit against a numpy oracle. First port = `ProteinDegradation` (1 Step, poisson only, no numba/ODE). Then metabolism via GLPK FFI (hot, no RNG — the M1 spike proved the FFI). sim_data params extracted once at build-time into the edge config (no live Python on the hot path). scipy-ODE processes (equilibrium/TCS) are OUT of the bit-identical track.

**Tech:** Rust, pyo3 0.29 + rust-numpy; GLPK via `extern "C"` FFI (`/opt/homebrew/include/glpk.h`, `-lglpk`); maturin; the M1/M2 conformance harness as the bit-identical gate.

**Spec:** scratchpad/m3_process_ports_design.md (copied to docs/superpowers/specs/) + the M1 design `docs/superpowers/specs/2026-10-27-rust-core-design.md` (§4 RNG decision).

## Global Constraints
- **Bit-identical to the Python twin** (decoded state, int-exact, float at parity epsilon; integer stochastic draws EXACT given the same seed) — the M1/M2 conformance harness (`conformance.py` decoded-state comparator, `upstream_suite`) + cargo oracle tests gate every port. A port that can't be bit-identical is NOT shipped as bit-identical.
- **Flag-gated native factory:** register each native Rust process factory at its `local:<Name>` address AHEAD of the PyProcess fallback, behind a flag/env, so ONE `.pbg` runs either the Python or the native path (and the conformance harness runs both and diffs).
- **sim_data stays Python** (offline). Native processes get params from the edge config (serialized by v2ecoli's `get_<proc>_config`), deserialized + precomputed once in the Rust constructor.
- **RNG = reimplemented numpy RandomState** (MT19937 + numpy's per-distribution algorithms + stochasticRound), validated bit-for-bit via a `gen_rng_oracle.py` (reuse M1's `gen_oracle.py` pattern). Do NOT use an off-the-shelf MT crate's distributions (they differ from numpy).
- No AI attribution; cargo fmt + clippy clean; maturin builds into the py3.12 venv (pb 1.8.3 + bigraph-schema 1.6.0 + numpy + v2ecoli importable from the shared venv, which is NOT modified).

## Review Focus
- **RNG stream order / edge cases:** vector-`lam` poisson draws must consume the stream in numpy's exact element order; `lam==0` and the `fmin` cap paths must match numpy; a wrong draw order silently diverges only on some seeds — the oracle must cover these.
- **stochasticRound semantics** (the `value + (random_sample < frac)` rounding) exact vs numpy/v2ecoli's helper.
- **sim_data config completeness:** a param the Python process reads but the config doesn't serialize → the native port silently uses a default/wrong value. The conformance diff must exercise the real config.
- **GLPK FFI determinism:** the same LP must give the same solution as swiglpk (solver settings/pivoting) — bit-identical FBA requires matching GLPK parameters; assert the objective + flux vector match the Python metabolism path.
- **Native-factory precedence:** the flag must cleanly select native vs Python at the SAME address; a mis-wire that runs BOTH or neither is a silent correctness/perf bug.

---

### Task 1: `rng` foundation module + numpy oracle

**Files:** `crates/bigraph-schema-rs/src/rng.rs` (or a new `crates/viva-rng` crate); Test `crates/.../tests/rng_parity.rs` + `python/oracle/gen_rng_oracle.py`.

**Interfaces:** `rng::RandomState::new(seed)` reproducing numpy legacy `RandomState`: MT19937 `init_genrand` seeding, `random_sample` (res53), `stochastic_round`, and `poisson` (both regimes + loggam). Bit-for-bit vs numpy.

- [ ] Step 1: `gen_rng_oracle.py` runs numpy `RandomState(seed)` in the shared venv and emits expected streams for random_sample, stochastic_round, poisson (incl. lam=0, large-lam, vector-lam) as committed literals.
- [ ] Step 2: Write failing `rng_parity.rs` asserting the Rust stream == the oracle literals (bit-exact).
- [ ] Step 3: Run → FAIL.
- [ ] Step 4: Implement the rng module (MT19937 + res53 + stochastic_round + poisson).
- [ ] Step 5: Run → PASS; cargo fmt/clippy clean.
- [ ] Step 6: Commit (no attribution).

---

### Task 2: `ProteinDegradation` native Rust process (first native port)

**Files:** `crates/viva-core/src/processes/protein_degradation.rs` (+ native factory registration in lib.rs, flag-gated); Test `python/tests/test_native_protein_degradation.py`.

**Interfaces:** `ProteinDegradationCore` implementing the engine `Process` trait (`inputs/outputs/update`), constructed from the serialized `get_protein_degradation_config` params (rate array, AA-count matrix, id lists, water_id, seed) — precompute the S matrix once. Registered at `local:...ProteinDegradation...` ahead of the PyProcess fallback behind a flag/env. Uses Task-1 `rng` (poisson, per-process seed via the same derivation as Python).

- [ ] Step 1: Write failing `@parity` test `test_native_protein_degradation_bit_identical`: run a composite with ProteinDegradation under (a) the Python process and (b) the native Rust process, same seed, N ticks; assert decoded state bit-identical (bulk deltas exact). Cover the vector-lam / lam=0 draw paths.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `ProteinDegradationCore` + config deserialization + flag-gated native-factory registration.
- [ ] Step 4: Run → PASS; M1/M2 conformance + upstream battery still green (no regression); cargo test --workspace.
- [ ] Step 5: Re-benchmark release: native ProteinDegradation vs Python-process-on-Rust-engine — the FIRST real quadrant-2 per-process number. Record it.
- [ ] Step 6: Commit.

---

### Task 3: metabolism native process via GLPK FFI (hot, no RNG)

**Files:** `crates/viva-core/src/processes/metabolism.rs` + a GLPK FFI module (reuse the M1 spike's `extern "C"` GLPK bindings pattern from ~/code/viva-rs-spike); Test `python/tests/test_native_metabolism.py`.

**Interfaces:** `MetabolismCore` (engine `Process`) that assembles the FBA LP from config (stoichiometry, bounds, objective from sim_data's metabolism config) and solves via GLPK's C API (`glp_simplex`), returning the flux→bulk deltas. Flag-gated native factory at the metabolism address. No RNG (only shared stochasticRound/randint if the real process uses it — check).

- [ ] Step 1: Write failing `@parity` test `test_native_metabolism_bit_identical`: the native FBA solve matches the Python metabolism path (objective + exchange/reaction fluxes + the integer bulk delta) on matched inputs (seed 0, glucose-minimal) — the arch-01 methodology. Set GLPK params to match swiglpk so the solution is identical.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `MetabolismCore` + GLPK FFI + config assembly + flag-gated registration.
- [ ] Step 4: Run → PASS; conformance + workspace green.
- [ ] Step 5: Re-benchmark: native metabolism vs Python — the hot-process quadrant-2 number (metabolism is ~the biggest per-tick compute).
- [ ] Step 6: Commit.

---

### Task 4: quadrant-2 benchmark + M3 results

**Files:** `python/benchmarks/bench_m3.py` (or extend bench_m1.py); Create `docs/milestone-3-results.md`; Test: full M1/M2/M3 suite as the no-regression gate.

- [ ] Step 1: Full conformance (q3/q4, native ProteinDegradation + native metabolism bit-identical, upstream battery, cargo oracle) — all green.
- [ ] Step 2: Benchmark a composite running the NATIVE processes (quadrant 2 — native engine + native processes) vs the all-Python baseline; record the real per-process + composite speedups (release).
- [ ] Step 3: Write `docs/milestone-3-results.md` honestly: the first real quadrant-2 numbers, what's ported (ProteinDegradation, metabolism) vs still-Python, and that the WHOLE-MODEL (full partitioned.pbg) number still needs the remaining ports (elongation + the full model runnable) — deferred to M4.
- [ ] Step 4: Commit.

## Notes
- M4+ (not this plan): elongation (numba `polymerize` → native Rust) + rna_degradation/tf_binding (multinomial/binomial/choice) + complexation (arrowhead.so FFI) + the full-`partitioned.pbg` whole-model benchmark (the headline speedup) + scipy-ODE processes (SUNDIALS FFI or bounded-divergent).
- Stacked on M2 (PR #2); rebase onto main when M2 merges.
- Report the quadrant-2 numbers to the `optimize` session.
