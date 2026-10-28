# viva-core Milestone 4 — the whole-model number (plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Produce the first HONEST whole-model speedup number — run the REAL v2ecoli baseline composite (one generation) on the Rust engine with the already-ported native processes swapped in, bit-identical to pb 1.8.3, and measure ticks/s vs pb. This is the user's actual goal (M1–M3 proved per-process bit-identity + micro-benchmarks only).

**Approach (from the M4 design note, scratchpad/m4_wholemodel_design.md):** option (iii). The full model CANNOT be built from a JSON `.pbg` — 11 process types carry live, non-serializable `sim_data` handles (`'node'`/`'method'` config). So add a **`run_prebuilt` engine entry point** that takes PRE-BUILT Python process instances (constructed by v2ecoli's own `ecoli_baseline.py`, which injects the sim_data) + topology + initial state, schedules them on the Rust engine as **instance-bound `PyProcess` shims**, and swaps NATIVE twins for allow-listed addresses (`VIVA_NATIVE_PROCESSES`). ParCa output is already on disk (`out/cache/sim_data_cache.dill`) — no ParCa run. Metabolism = **hybrid (C)**: wire the already-exposed native `viva_core.fba_solve`/`flux_to_bulk_delta` into the Python assembly (native solve, Python assembly). Elongation full port + division deferred to M5+.

**Tech:** Rust, pyo3 0.29 + rust-numpy; the existing native processes (ProteinDegradation, FBA solve) + flag/feature gates; maturin; the M1–M3 conformance harness as the bit-identical gate.

**Spec:** scratchpad/m4_wholemodel_design.md (copied to docs/superpowers/specs/) + the M1 design.

## Global Constraints
- **Bit-identical to pb 1.8.3** on the real composite: decoded state int-exact, floats within 1e-12, and RNG-exact for the native-swapped processes (same seed → same draws). The conformance comparator (`conformance.py`) + decoded-state diff gate every step.
- **`run_prebuilt` must NOT require JSON-serializing the processes** — it takes already-constructed Python instances (with their live sim_data) + topology + state. Instance-bound shims, not address-reconstructed.
- **Native swap is opt-in + allow-listed** (`VIVA_NATIVE_PROCESSES`): a swapped native process must be bit-identical to the Python instance it replaces, or it's not swapped. Default (no swap) = every process runs as its pre-built Python instance on the Rust engine (quadrant-3 whole-model).
- **Metabolism hybrid:** reuse the EXISTING bit-identical native `fba_solve`/`flux_to_bulk_delta` (M3) inside the Python assembly; do NOT port the LP assembly (it reassembles per-tick from live mass/environment through the pint/Unum bridge — 1.6k+ LOC, high bit-identity risk; rejected).
- **Honesty:** this is ONE GENERATION, baseline media, single seed; division is sidestepped (not run to division). Report the number with these bounds; whole-generation/multi-seed/multi-condition + elongation-native are later. No fabricated numbers.
- No AI attribution; cargo fmt + clippy clean; GLPK feature-gated (default wheel GLPK-free); wheels into a FRESH local venv, shared `/Users/eranagmon/code/v2ecoli/.venv` untouched (read-only for v2ecoli + sim_data on disk).

## Review Focus
- **Instance-bound shim fidelity:** a pre-built Python process scheduled via `run_prebuilt` must receive the SAME inputs (ports/views) each tick as pb gives it, and its update applied identically — any wiring/port-mapping drift silently diverges on a real model. The bit-identical gate on the real composite catches it.
- **Native-swap correctness per process:** ProteinDegradation swapped in must be bit-identical to the pre-built Python instance (not just to a synthetic twin) — i.e. same sim_data-derived params. Verify the native config matches what ecoli_baseline built.
- **Metabolism hybrid seam:** the native `fba_solve` must get the EXACT LP the Python assembly built (bounds/objective/stoich) and return fluxes the Python side applies identically — the solve is bit-identical (M3), but the hand-across must not reorder/retype anything.
- **Whole-model gating/scheduling:** the real composite has gated Steps (perform_update), emitters, and the layer order — run_prebuilt must reproduce pb's execution (M1 Task-4/5 gating lessons) across all 55 processes, not just the few tested in M1.
- **Honest benchmark:** native-swap OFF (pure Python-on-Rust whole model) vs pb is the engine number; native-swap ON adds the ported-process gains. Report BOTH; warm-up excluded; note one-generation scope.

---

### Task 1: `run_prebuilt` engine entry point + instance-bound PyProcess shim

**Files:** `crates/viva-core/src/lib.rs` + `crates/viva-core/src/pyproc.rs` (instance-bound variant); `python/viva_core/prebuilt.py` (the Python handoff helper); Test `python/tests/test_run_prebuilt.py`.

**Interfaces:** `viva_core.run_prebuilt(processes, topology, state, n_ticks, native_addresses=...)` where `processes` is a dict/list of ALREADY-CONSTRUCTED Python process instances (with live sim_data), `topology` the wiring, `state` the initial state. The engine schedules each as an instance-bound `PyProcess` (calls the given instance's `update`/`perform_update`, NOT reconstructing from a JSON address), applying updates via the Rust apply/reconcile; for addresses in `native_addresses` (and enabled) it swaps the native twin.

- [ ] Step 1: Write failing `@parity` test `test_run_prebuilt_bit_identical_small`: take a SMALL real multi-process slice of the baseline (built via v2ecoli's builders with real sim_data), run N ticks under pb `Composite` and under `viva_core.run_prebuilt` (no native swap), assert decoded state bit-identical. Include a gated Step + the emitter.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `run_prebuilt` + the instance-bound shim + `prebuilt.py`.
- [ ] Step 4: Run → PASS; M1–M3 conformance still green (flag off); cargo test --workspace.
- [ ] Step 5: Commit (no attribution).

---

### Task 2: metabolism hybrid (native solve wired into Python assembly)

**Files:** `python/viva_core/metabolism_hybrid.py` (a thin wrapper subclass over v2ecoli's metabolism that calls `viva_core.fba_solve`/`flux_to_bulk_delta`); Test `python/tests/test_metabolism_hybrid.py`.

**Interfaces:** a metabolism process variant that runs v2ecoli's Python per-tick LP ASSEMBLY unchanged, but replaces the swiglpk solve + flux→count with the bit-identical native `viva_core.fba_solve`/`flux_to_bulk_delta`. Opt-in via the native-swap allow-list.

- [ ] Step 1: Write failing `@parity` test `test_metabolism_hybrid_bit_identical`: under the real composite (or a metabolism sub-composite), N ticks, assert the hybrid (native solve) == pure-Python metabolism bit-identical (objective/fluxes/bulk delta exact), seed 0 glucose-minimal.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the hybrid wrapper (reuse M3's exposed native solve).
- [ ] Step 4: Run → PASS; no regression.
- [ ] Step 5: Commit.

---

### Task 3: whole-model benchmark + M4 results

**Files:** `python/benchmarks/bench_m4_wholemodel.py`; Create `docs/milestone-4-results.md`; Test `python/tests/test_wholemodel_conformance.py`.

**Interfaces:** run the REAL baseline composite (built by v2ecoli's `ecoli_baseline.py` with on-disk sim_data) for ONE generation under (a) pb 1.8.3 and (b) `viva_core.run_prebuilt`, native-swap OFF and ON (ProteinDegradation + metabolism-hybrid). Conformance-gate decoded-state bit-identical; benchmark ticks/s.

- [ ] Step 1: Write failing `test_wholemodel_conformance`: the real baseline composite runs N ticks (as many as hold) under both engines bit-identical (int-exact, float-1e-12, RNG-exact for native). If some process can't yet run on run_prebuilt, record exactly which + why (honest — the whole-model gate is "as much of the real model as runs").
- [ ] Step 2: Run → FAIL (or surface the real blocker honestly).
- [ ] Step 3: Implement the harness + benchmark; get as much of the baseline running end-to-end bit-identical as possible.
- [ ] Step 4: Run → record the WHOLE-MODEL number: pb vs Rust-engine native-swap-OFF (the engine number) and native-swap-ON (engine + ported processes), release build, one generation. Write `docs/milestone-4-results.md` honestly (one generation / baseline media / single seed; what runs vs what's still Python/unported; the real speedup vs pb).
- [ ] Step 5: Commit.

## Notes
- M5+: elongation native port (polymerize 577 LOC + the global-numpy-RNG coupling hazard — stochasticRound/multinomial×2/choice + polymerize limit-hit random/shuffle); full metabolism assembly port if the hybrid's Python assembly dominates; functional division; multi-generation/seed/condition; the Slurm/manylinux cluster build + the libm/cross-host RNG re-verification.
- Stacked on nothing (off clean main = M1+M2+M3). Report the whole-model number to the `optimize` session + the user.
