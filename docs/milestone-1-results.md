# viva-core — Milestone 1 results (the honest day-one numbers)

**Date:** 2026-10-28
**Branch:** `m1-engine-mvp`
**Scope:** engine MVP — the Rust bigraph-schema + process-bigraph core running a
real v2ecoli `.pbg` with **all processes still Python** (quadrants 1–3),
bit-identical to the Python engine. Measured on this macOS machine (spec A4).
Pinned conformance target: **bigraph-schema 1.6.0 / process-bigraph 1.8.3**.

This is the Task-8 capstone record. Everything below is measured, not hoped.
The honesty mandate (spec §7): the engine swap alone is a **modest** win whose
real point is interchangeability + parallelism + the migration path; the
dramatic speedup needs porting hot processes, which is post-M1.

---

## 1. Cross-language conformance (decoded-state, bit-identical)

Both engines run the **same** composite; comparison is on decoded state —
integer counts **exact**, floats within `1e-12` (spec A2: bytes reformat, state
doesn't). Harness: `python/viva_core/conformance.py` (mirrors v2ecoli's
`distilled_compare.py` across the language boundary).

The largest composite (`test_conformance_full_model`) is the real v2ecoli
unique-store + gated-deriver + emitter model from Tasks 5/6, **plus the Task-5
carry-forward**: a Python process that reads and writes a **bulk** store
(`bulk["count"]`, v2ecoli's hot path) across the PyO3 boundary.

| Store / aspect | Result |
|---|---|
| `global_time` (float add) + `GlobalClock.calculate_timestep` | bit-identical |
| `unique` stores (set/add/delete + flush, `unique_index` metadata) | bit-identical |
| `listeners` overwrite tree (int/float/bool leaves) | bit-identical |
| step gating (`update_condition`) decisions | match (ON at t=0, OFF at t=6) |
| RAMEmitter row counts | match (2 == 2) |
| **`bulk` store — Python process reads+writes `count` across boundary** | **bit-identical** |

### The BULK-path carry-forward — resolved

Task 5 proved quadrant-3 only for **unique** stores. Task 8 drove a **bulk**
store end-to-end through a Python process, both shapes of `BulkNumpyUpdate`:

* **vectorized** `(index_array, delta_array)` — v2ecoli's real evolver shape;
* **scalar** `[(index, delta), …]` list.

Both are bit-identical to pb. Two findings surfaced and were fixed (see §4);
neither is a residual bug.

**Important modelling note (real finding, not hidden):** the bulk driver is a
**Process**, not a Step. The Rust engine runs every Step **each cascade**, while
pb triggers Steps **reactively on input change**. For idempotent derivers
(v2ecoli's actual Steps) this is invisible; for a *non-idempotent* Step it
diverges on run-count. Bulk writes in v2ecoli come from Processes, which both
engines schedule identically (`calculate_timestep`), so the realistic path is
bit-identical. The Step/Process triggering difference is documented here as a
known semantic boundary, not a silent divergence.

---

## 2. Day-one benchmark (ticks/sec + ratio) — the honest number

`python/benchmarks/bench_m1.py`, 2000 ticks, best-of-5, warm-up excluded, this
Mac. Same composite under **Rust engine + Python processes** vs **Python engine
+ Python processes** (quadrant 3 vs quadrant 1).

### A. Structured-array composite (per-tick `build_structured_array` copy)

One Python bulk-evolver, bulk n=200:

| | ticks/sec |
|---|---|
| Rust engine + Python processes | **≈ 6,760** |
| Python engine + Python processes | ≈ 101,700 |
| **day-one ratio (Rust / Python)** | **≈ 0.066× — a NET SLOWDOWN** |

This is the **pessimistic upper bound on overhead / lower bound on speedup.**
Zero-copy is **not** achieved for structured arrays: `convert::build_structured_array`
rebuilds a fresh numpy structured array for every bulk/unique view **every
tick**, whereas pb passes the array by reference. For an array-heavy process the
copy dominates and **inverts** the result. This is the single most important
honest caveat of M1: the headline engine-swap number is NOT a win yet for the
real hot path — closing the zero-copy gap is the first post-M1 engine task.

### B. Scalar-only composite (no array crosses the boundary) — pure-engine delta

One Python scalar-evolver (isolates engine + PyO3-callback overhead, no copy):

| | ticks/sec |
|---|---|
| Rust engine + Python processes | **≈ 186,000** |
| Python engine + Python processes | ≈ 147,000 |
| **ratio (Rust / Python)** | **≈ 1.27×** |

The pure engine + boundary path is **~1.27× faster**, matching the spike's
~1.2–1.26× day-one prediction (spec §7). So the engine itself delivers the
predicted modest win; the structured-array copy (A) is what currently erases it.

### Not measurable in M1

The **full-native ceiling** (Rust engine + Rust processes, spec's ~208k tps /
~6.2× over day-one) is **not** measurable through the M1 `run()` surface:
`viva_core.run` drives Python processes only; there is no Python entry point that
registers a native Rust process into a `run()` composite in M1 (quadrant 2 is a
post-M1 process-port milestone).

---

## 3. Upstream-suite conformance (passing fraction + a/b/c)

`python/viva_core/upstream_suite.py` (`python -m viva_core.upstream_suite`).

### Part 1 — literal drop-in of the real upstream pytest suites: INFEASIBLE in M1

| suite (pinned tag) | test fns | runnable as drop-in |
|---|---|---|
| bigraph-schema **v1.6.0** `tests.py` | 176 | **0** |
| process-bigraph **v1.8.3** `tests.py` | 174 (86 build a `Composite`, 68 call `.run`) | **0** |

(counts re-derivable via `scripts/count_upstream_tests.sh`.)

**Why 0 (honest, scoped):** a literal `import viva_core as bigraph_schema` /
`as process_bigraph` cannot run these suites because the M1 `viva_core` wheel
exposes only `run()` / `run_report()` on a JSON spec — **no `TypeSystem`,
`Composite`, or `register_link` Python API** that the suites instantiate
(`core.access/realize/render/resolve/admits/check/apply/serialize…`;
`Composite({...}, core).run()`). This is **category (a) by construction**: M1
deliberately implements only the engine-hot op × node-type subset (spec §8-A1,
YAGNI), so the full upstream API surface is unimplemented-by-design, **not**
"failing". A Composite-shim that would run the pb run()-level subset as a true
drop-in is deferred past M1.

### Part 2 — runnable run()-level battery (the scoped north-star): 6/6 = 100%

Real v2ecoli composites, each run under **both** engines, decoded-state compared:

| case | category | result |
|---|---|---|
| apply(float) global_time-add + `calculate_timestep` | pass | bit-identical |
| RAMEmitter row-count / native-emitter parity | pass | match |
| apply(bulk_array) vectorized `(idx_array, delta_array)` via Process | pass | bit-identical |
| apply(bulk_array) scalar `(idx, delta)` list via Process | pass | bit-identical |
| apply(unique) set/add/delete/flush + step gating + overwrite tree | pass | bit-identical |
| FULL MODEL: bulk Process + unique + gating + emitter + clock | pass | bit-identical |

**pass_fraction = 1.000 (6/6); category (c) real divergences = 0.**

Category breakdown for M1:
* **(a) unimplemented-subset (expected):** the entire upstream type-system API
  beyond the engine-hot op×type subset — the 176 + 174 upstream tests' use of
  `access/realize/render/resolve/admits/check/fill_sites/serialize/…` and the
  Composite/register_link surface. Large and by design.
* **(b) implementation-coupled:** 0 in the battery (it's built to avoid Python
  exception-type / object-identity / internal-attr coupling). The battery
  runner categorizes any plumbing exception as (b) if one arises.
* **(c) real contract divergence:** **0**. (Two were found and FIXED during
  Task 8 — see §4 — so they are not residual.)

### Part 3 — Rust oracle conformance (schema core): `cargo test --workspace`

The bigraph-schema Rust pieces (apply / reconcile / codec / type grammar) are
checked **bit-identical against oracles generated by the real Python
bigraph-schema 1.6.0 + v2ecoli apply**
(`crates/bigraph-schema-rs/tests/oracle/gen_oracle.py`) — the Rust analogue of
running the upstream apply/reconcile/codec test cases. 9 oracle-backed tests;
`cargo test --workspace` green.

---

## 4. Conformance issues found and fixed in Task 8

1. **Bare `bulk_array` / `unique_array` type didn't parse.** The grammar only
   accepted the parameterized form (`bulk_array[…]`); a v2ecoli process
   `outputs()` declares the **bare** `bulk_array`, which fell through to
   `Named` → the update was treated as an `Overwrite` and hit a store-type
   mismatch. Fixed in `grammar.rs` (bare forms → `BulkArray{fields:[]}` /
   `UniqueArray{fields:[]}`), with a regression test.
2. **`bulk_op` only handled scalar `(index, delta)` pairs.** v2ecoli's real
   evolver emits one **vectorized** `(index_array, delta_array)` pair. Extended
   `convert::bulk_op` to zip array/array pairs (and broadcast a scalar against
   an array), matching numpy's `count[idx] += value`.

---

## 5. Status

* Conformance: full-model bit-identical incl. the **bulk** carry-forward. ✅
* Day-one benchmark: scalar/pure-engine **1.27×** (matches spike); structured-
  array **0.066×** (copy-dominated, pessimistic bound). Honest. ✅
* Upstream suite: run()-level battery **6/6 = 100%**, 0 real divergences;
  literal drop-in infeasible in M1 and scoped honestly. ✅
* Tests: `cargo test --workspace` green; `.venv-m1 pytest python/tests/` 9/9. ✅

**First post-M1 engine task (highest leverage):** make the structured-array
boundary zero-copy (share the numpy buffer instead of rebuilding it each tick),
which is what currently turns the real-hot-path day-one number negative.
