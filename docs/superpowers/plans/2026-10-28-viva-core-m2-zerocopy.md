# viva-core Milestone 2 — zero-copy structured-array boundary (plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make the Python↔Rust structured-array boundary ZERO-COPY so a Python process reads the Rust-owned store buffers as genuine numpy (structured) arrays with no per-tick copy — flipping M1's net-slowdown (≈0.066× on the structured-array hot path) up toward the 1.27× engine ceiling, while staying bit-identical.

**Approach (from the M2 design note, scratchpad/m2_zerocopy_design.md): A + C, staged.** Back the crossing stores with a buffer in numpy record layout (A) and hand Python a READ-ONLY numpy view over it (C); keep WRITES on M1's existing `Op`/delta path (already bit-identical). Justified because the real v2ecoli hot path reads field-wise (`counts`=`states["count"][idx]`, `attrs`=`states[field][mask]`) and returns update dicts — it never mutates the view. Option B (per-column separate buffers) is impossible (numpy structured-dtype fields are offsets into ONE buffer).

**Tech:** Rust, pyo3 0.29 + rust-numpy 0.29, ndarray; maturin; the M1 conformance harness as the bit-identical gate.

**Spec:** scratchpad/m2_zerocopy_design.md + the M1 design `docs/superpowers/specs/2026-10-27-rust-core-design.md`.

## Global Constraints
- **Bit-identical unchanged:** every M1 conformance gate must stay green — `cargo test --workspace` (oracle tests), `.venv-m1/bin/python -m pytest python/tests/` (quadrant-3/4 + conformance), `python -m viva_core.upstream_suite` (6/6 run-level battery). Zero-copy is correctness-neutral by definition.
- **Read-only across the boundary:** views handed to Python are `writeable=False`; writes continue through `Op::BulkDelta`/unique Op path. Do NOT make the Python-facing view writable in M2.
- **No-copy is the deliverable:** each task adds a `np.shares_memory(view, <rust buffer>)` / no-realloc assertion so a reintroduced copy FAILS the test.
- Lifetime safety: either numpy owns the persistent array (Rust holds `Py<PyArray>`, mutates under the GIL) or Rust owns the `Vec<u8>` and hands a per-tick borrowed view with a `base` container; realloc only in flush, after the view is dropped. Cache the `PyArrayDescr`.
- No AI attribution; cargo fmt + clippy clean; maturin builds into a py3.12 venv with pb 1.8.3 + bigraph-schema 1.6.0 + numpy (rebuild with `maturin develop` in this worktree).

## Review Focus
- A view that LOOKS zero-copy but silently copies (dtype/stride mismatch forcing numpy to copy) — the `np.shares_memory` assertion must actually hold, not be vacuous.
- Lifetime/use-after-free: Python holding a borrowed view past a Rust realloc (flush grows the buffer). Prove the ownership model forbids it.
- The interleaved-layout rewrite (Task 2) silently breaking apply/reconcile bit-identity (field offsets/record stride wrong) — the M1 unique parity + q3 tests must still pass exactly.
- Endianness/alignment of the record dtype vs what numpy expects (derive offsets from a built `PyArrayDescr`, never hand-compute).

---

### Task 1: Zero-copy BULK read view (no layout change)

**Files:** `crates/viva-core/src/convert.rs` (the bulk view path in/around `build_structured_array`); Test `python/tests/test_zerocopy_bulk.py`; a throwaway `crates/.../probe` is fine but don't commit it.

**Interfaces:** hand Python a single-field numpy structured array `[("count","<i8")]` (or the real bulk dtype) that VIEWS `BulkStore.count` zero-copy, read-only — replacing the `np.zeros(length,dtype)` + per-column copy. Writes stay on `Op::BulkDelta`.

- [ ] Step 1: ~30-line Rust probe (throwaway) confirming rust-numpy borrow-from-slice-with-custom-descr is actually zero-copy (`shares_memory` true). Record the result.
- [ ] Step 2: Write failing test `test_bulk_view_is_zero_copy`: a Python process reads `bulk["count"]` through the boundary bit-identical to pb AND `np.shares_memory` holds (no copy); would fail if the copy path were used.
- [ ] Step 3: Run → FAIL.
- [ ] Step 4: Implement the zero-copy bulk read view (read-only).
- [ ] Step 5: Run → PASS; re-run the M1 bulk parity cases (bit-identical) + `cargo test --workspace`.
- [ ] Step 6: Re-run `python/benchmarks/bench_m1.py` case A; record the new bulk-path ratio (expect ≥1×, toward 1.27×).
- [ ] Step 7: Commit (no attribution).

---

### Task 2: Zero-copy UNIQUE read view (interleaved buffer)

**Files:** `crates/bigraph-schema-rs/src/state.rs` (`UniqueStore` columnar → interleaved `Vec<u8>` + offset table), `crates/bigraph-schema-rs/src/apply.rs` (flush_unique/active_indices/get_free_indices → field-offset/record-stride arithmetic), `crates/viva-core/src/convert.rs` (the unique view path); Test `python/tests/test_zerocopy_unique.py`.

**Interfaces:** `UniqueStore` backed by one interleaved byte buffer in numpy record layout (offsets derived from a built `PyArrayDescr`, not hand-computed); hand Python a read-only zero-copy structured-array view; apply/reconcile rewritten to operate on the interleaved buffer, bit-identical.

- [ ] Step 1: Write failing tests: `test_unique_view_is_zero_copy` (a Python process reads a unique store's fields through the boundary bit-identical to pb + `np.shares_memory`); and confirm the M1 unique parity + quadrant-3 tests still pass (they become the interleaved-layout regression gate).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the interleaved `UniqueStore` + the apply/reconcile offset/stride rewrite + the zero-copy view.
- [ ] Step 4: Run → PASS: new zero-copy tests + ALL M1 conformance (unique parity, q3/q4, upstream battery, cargo oracle) bit-identical.
- [ ] Step 5: Commit.

---

### Task 3: Re-benchmark + conformance + M2 results

**Files:** `python/benchmarks/bench_m1.py` (reuse/extend); Create `docs/milestone-2-results.md`; Test: the full M1 suite as the no-regression gate.

- [ ] Step 1: Run the full conformance (quadrant-3/4, conformance suite, upstream battery, cargo oracle) — all still bit-identical.
- [ ] Step 2: Re-run the day-one benchmark (structured-array composite + scalar) with zero-copy; record the new honest ratio.
- [ ] Step 3: Write `docs/milestone-2-results.md` with the before (0.066×)→after number, honestly framed (did zero-copy flip the hot path positive? by how much? any residual gap vs the 1.27× ceiling).
- [ ] Step 4: Commit.

## Notes
- Writes stay on the Op path in M2 (read-only views). A future milestone may make the boundary writable if a process needs in-place mutation (none in v2ecoli's current hot path).
- After M2, report the new day-one number to the `optimize` session (they asked).
- Then the process-port milestone (elongation + RNG parity; GLPK FFI).
