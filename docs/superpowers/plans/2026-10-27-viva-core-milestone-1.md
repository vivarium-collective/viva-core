# viva-core Milestone 1 (engine MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A Rust `viva-core` engine that loads a real v2ecoli `.pbg` composite (JSON contract), runs it with **all processes still in Python** (via PyO3 callbacks), produces results **bit-identical** to the Python process-bigraph engine, and also runs a Rust process **under** the Python engine — proving both interchange quadrants + the day-one engine-swap speedup, on this Mac.

**Architecture:** One Cargo workspace → one pip-installable `viva-core` wheel (maturin/PyO3). Crates: `bigraph-schema-rs` (type grammar + registry + state + apply/reconcile, SUBSET the target `.pbg` uses), `process-bigraph-rs` (scheduler + tick loop + emitters), `viva-core` (PyO3 bindings: Rust-engine-runs-Python-process, and Rust-process-under-Python-engine). No kernel FFI and no RNG porting in M1 — processes stay Python, so numeric kernels run in their existing Python form.

**Tech Stack:** Rust (edition 2021), `serde`/`serde_json` (lenient NaN/Infinity), `pyo3` + `numpy` crate (zero-copy arrays), `ndarray`, `rayon`, `mpi` (rsmpi) against Homebrew Open MPI 5, `maturin`. PEG via `pest` or hand-written recursive descent for the type grammar. Reference: Python `process-bigraph==1.8.3` + `bigraph-schema==1.6.0` (the conformance oracle).

**Spec:** `docs/superpowers/specs/2026-10-27-rust-core-design.md` (esp. §4 decisions, §8 builder-check resolutions).

## Global Constraints
- **JSON contract compatibility:** byte-structural round-trip of bigraph-schema **1.6.0** specs/state; handle both serializers — schema-paired (bigraph-notation type strings) and schema-agnostic `json_codec` tagged-dicts (pint quantity / numpy array / set / bytes) — and **lenient `Infinity`/`NaN`** tokens (Python `allow_nan`).
- **Bit-identical = DECODED STATE** (not serialized bytes): integer molecule counts **exact**; floats within the v2ecoli parity-ladder epsilon (solver/INT). Never assert byte-equality of serialized floats.
- **Subset-driven (YAGNI):** implement only the schema **types + ops** the target `.pbg` actually exercises (measure first); do not port all ~23 ops × ~35 Node types up front.
- **Target this Mac** (macOS, Homebrew Open MPI 5). No manylinux wheel, no Slurm, no cluster in M1.
- **Processes stay Python** in M1 (no kernel FFI, no RNG porting). One `viva-core` wheel from the workspace.
- Pin the Python oracle to `process-bigraph==1.8.3` + `bigraph-schema==1.6.0` in a dedicated venv (do NOT mutate the shared `/Users/eranagmon/code/v2ecoli/.venv`).
- Commits: **no AI attribution** (no `Co-Authored-By`, no "Generated with"). Keep crates `cargo fmt`/`clippy`-clean.

## Review Focus
- **i64 vs Python bigint:** molecule counts/accumulators are Python ints (unbounded) but Rust defaults to i64 — audit every count path for possible overflow; use i128/checked arithmetic where a value can exceed i64, and a test that drives a large-count store proves no silent wrap.
- **Unique structured stores, not just flat bulk:** a real `.pbg` has per-molecule unique structured arrays (fields of mixed dtype); the engine + the zero-copy boundary must handle them, and a test must exercise a unique-store round-trip + a Python process reading/writing one through the boundary.
- **json_codec tagged-dicts:** pint quantities, numpy arrays, sets, bytes serialize as tagged dicts — a test must round-trip each tagged form that the target `.pbg`/state contains (not just plain numbers), or decoded values silently change type.
- **`local:<Name>` address resolution:** process edges carry `address: {protocol, data}` (e.g. `local:MetabolismProcess` importlib); the engine must resolve + import the Python class exactly as pb does, and a test must run a process addressed this way (wrong resolution → the process silently doesn't run).
- **Emitters/listeners run every tick:** RAMEmitter/JSONEmitter are steps in the layers; a test must confirm the engine either drives them identically to pb or explicitly excludes them with matching state — otherwise emitted/listener state diverges and the bit-identical gate is false-green.

---

### Task 1: Workspace + JSON contract data model + lenient codec

**Files:** Create `Cargo.toml` (workspace), `crates/bigraph-schema-rs/{Cargo.toml,src/lib.rs,src/doc.rs,src/codec.rs}`; Test: `crates/bigraph-schema-rs/tests/roundtrip.rs` + a committed real fixture `tests/fixtures/<model>.pbg`.

**Interfaces:**
- Produces: `doc::CompositeSpec` (stores tree + `Vec<ProcessEdge>{address, config, inputs, outputs, interval, _contract}`), `doc::Value` (the tagged-union over number/int/array/tagged-dict), `codec::from_json(&str) -> CompositeSpec`, `codec::to_json(&CompositeSpec) -> String` with lenient NaN/Infinity.

- [ ] Step 1: Write failing test `roundtrip_structural`: load the real `.pbg` fixture, `from_json` then `to_json`, assert structural equivalence (same store tree, same edges/wiring/config), and that bare `Infinity`/`NaN` parse (don't error).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the data model + lenient codec (serde with a custom de/serializer accepting non-finite floats; tagged-dict handling for pint/numpy/set/bytes as opaque-preserving `Value` variants for now).
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 2: Type grammar + registry (subset)

**Files:** Create `crates/bigraph-schema-rs/src/{types.rs,grammar.rs,registry.rs}`; Test `tests/types.rs`.

**Interfaces:** Consumes Task 1's `Value`. Produces: `types::Type` (primitives + `array[dtype]` + tree/map + the structured unique-record type), `grammar::parse_type(&str) -> Type` (the bigraph-notation PEG, subset), `registry::Registry` (name → Type + apply/reconcile dispatch keys). **Decision:** enumerate the types used by the fixture's store schemas; implement exactly those.

- [ ] Step 1: Write failing test `parse_and_resolve_fixture_types`: for each store schema string in the fixture, `parse_type` round-trips the notation and the registry resolves it to a concrete `Type` (including the unique structured-record type).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the grammar subset + registry.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 3: State + apply/reconcile (the hot path)

**Files:** Create `crates/bigraph-schema-rs/src/{state.rs,apply.rs}`; Test `tests/apply.rs`.

**Interfaces:** Consumes Tasks 1–2. Produces: `state::State` (bulk → `ndarray` flat arrays; unique → structured record arrays; nested per the store tree), `state::from_initial(&CompositeSpec, initial_json) -> State`, `apply::apply_update(&mut State, &[Update])` and `apply::reconcile(...)` implementing the SUBSET of ops the model uses (accumulate-then-apply; the reconcile/bulk semantics). **Review-focus:** i64/bigint audit here; unique-store set/add/delete semantics.

- [ ] Step 1: Write failing tests: `apply_bulk_matches_python` (apply a fixed update set to a bulk store, compare decoded state to the Python `apply` output on the same inputs — int-exact) and `apply_unique_set_add_delete` (unique store ops) and `no_i64_overflow_on_large_counts`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement state + apply/reconcile for the subset ops (i128/checked where counts can overflow).
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 4: Scheduler + tick loop + emitters

**Files:** Create `crates/process-bigraph-rs/{Cargo.toml,src/lib.rs,src/schedule.rs,src/engine.rs,src/emitter.rs}`; Test `tests/engine.rs`.

**Interfaces:** Consumes bigraph-schema-rs. Produces: `schedule::wire_step_layers(&CompositeSpec) -> Vec<Layer>` (dependency-ordered edges, mirroring pb 1.8.3's layering incl. the layer-0 `_triggers`={global_time} rule), `engine::Engine{state, layers}` with `engine.run(n_ticks)` (each tick: per layer, run each process → accumulate triples → apply/reconcile; advance global_time clock), a `Process` trait (`inputs()/outputs()/update(state) -> updates`), and a RAM/JSON emitter step. **Decision:** M1 runs with a stub/native test `Process` here; the Python-process path is Task 5.

- [ ] Step 1: Write failing tests: `layer_order_matches_pb` (a small spec's layers match pb 1.8.3's order), `run_advances_clock_and_applies`, `emitter_captures_each_tick`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement scheduler + tick loop + emitter.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 5: PyO3 — Rust engine runs Python processes (Quadrant 3)

**Files:** Create `crates/viva-core/{Cargo.toml,src/lib.rs}` (PyO3 cdylib) + `pyproject.toml` (maturin) at workspace root; Test `python/tests/test_quadrant3_parity.py` + dedicated `.venv-m1` with `process-bigraph==1.8.3 bigraph-schema==1.6.0 numpy`.

**Interfaces:** Consumes Task 4. Produces a Python module `viva_core` with `run(pbg_path|spec_json, n_ticks) -> state`, where for each process edge the engine resolves its `local:<Name>` address, imports the Python class, and each tick calls its `update(state)` with **zero-copy numpy views** of the input stores (bulk AND unique structured), then applies the returned deltas via Task 3. `maturin develop` into `.venv-m1`.

- [ ] Step 1: Write failing `@parity` test `test_quadrant3_bit_identical`: run a REAL (start small — a few-process) v2ecoli `.pbg` for N ticks under (a) the Python pb 1.8.3 engine and (b) `viva_core.run`; assert decoded state bit-identical (int-exact, float-epsilon). Include a case where a Python process reads/writes a UNIQUE store through the boundary.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the PyO3 bridge (address resolution, zero-copy numpy views, per-tick callback, apply).
- [ ] Step 4: Run → PASS. Scale the fixture up toward the full `partitioned.pbg` as far as it holds; record where it breaks if it does.
- [ ] Step 5: Commit.

---

### Task 6: Python engine runs a Rust process (Quadrant 4)

**Files:** `crates/viva-core/src/lib.rs` (+ a `#[pyclass]` Rust process + a generated/hand-written Python `Process` shell in `python/viva_core/rust_process.py`); Test `python/tests/test_quadrant4_parity.py`.

**Interfaces:** A Rust process exposed via PyO3 with `inputs()/outputs()/update(state[,interval])`, wrapped in the ~10-line Python `Process` shell so pb's registry can address it by `local:<Name>`. Consumes nothing from Tasks 4–5 beyond the shared types.

- [ ] Step 1: Write failing test `test_quadrant4_bit_identical`: build a tiny composite whose step layer references the Rust-backed process by address, run it under the REAL `process_bigraph.Composite` (1.8.3), assert bit-identical to a pure-Python twin process on the same inputs.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the `#[pyclass]` process + the Python shell + registration.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 7: Parallelism — rayon ensemble + rsmpi smoke (this Mac)

**Files:** `crates/process-bigraph-rs/src/parallel.rs` + `crates/viva-core` exposure; `crates/process-bigraph-rs/src/bin/mpi_smoke.rs`; Test `tests/parallel.rs` + a shell check for the MPI binary.

**Interfaces:** `parallel::run_ensemble(specs_or_seeds, n_ticks) -> Vec<State>` using `rayon`. A standalone `mpi_smoke` binary (links Homebrew Open MPI 5) that scatters seeds across ranks, runs the engine per rank, gathers a scalar; run via `mpirun -n 2`.

- [ ] Step 1: Write failing tests: `ensemble_bit_identical_to_serial` (rayon ensemble of the real composite == running each serially) and a smoke assertion that `mpirun -n 2 mpi_smoke` prints the gathered result.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement rayon ensemble + rsmpi smoke binary.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

---

### Task 8: Conformance harness + honest day-one benchmark

**Files:** `python/viva_core/conformance.py` + `python/benchmarks/bench_m1.py`; Test `python/tests/test_conformance_suite.py`.

**Interfaces:** A cross-language conformance runner (reuse the v2ecoli `distilled_compare.py` methodology) that runs the chosen real `.pbg` under both engines and reports per-store/per-observable divergence; and a benchmark reporting the **honest day-one number**: Rust engine + Python processes vs Python engine + Python processes, same `.pbg`, warm-up excluded, ticks/sec + ratio.

- [ ] Step 1: Write failing test `test_conformance_full_model`: the largest `.pbg` that Task 5 reached runs bit-identical under both engines across N ticks (int-exact, float-epsilon).
- [ ] Step 2: Run → FAIL (or surface the real divergence honestly).
- [ ] Step 3: Implement the conformance runner + benchmark.
- [ ] Step 4: Run → PASS; record the measured day-one ticks/sec + ratio in `docs/` (honest — this is the number the design promised to produce).
- [ ] Step 5: Commit.

---

## Notes
- After M1: process-port milestones (M2 = elongation port with exact MT19937+stochasticRound RNG parity; metabolism via GLPK FFI), then the Slurm/Linux cluster + manylinux wheel milestone. Each port is a bit-identical gate.
- Coordinate with the `optimize` session when the M1 day-one number lands (they want it for the overhead-ceiling memory).
- The `~/code/viva-rs-spike` throwaway spike is the reference implementation for each mechanism (codec, PyO3 zero-copy, rsmpi, Q3/Q4) — mine it, don't re-derive.
