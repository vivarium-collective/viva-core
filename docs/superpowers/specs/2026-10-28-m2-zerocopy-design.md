# viva-core M2 — Zero-copy Python↔Rust structured-array boundary — DESIGN NOTE

**Status:** read-only design investigation (no production code written). Date 2026-10-28.
Repo: `/Users/eranagmon/code/viva-core` @ `main` (M1 merged, e867be7). Probed in `.venv-m1`
(numpy 2.4.6, py3.12). Throwaway probe:
`/private/tmp/claude-502/-Users-eranagmon-code/26ece8ca-3de6-4989-b582-2e8a485a0f4a/scratchpad/probe_zerocopy.py`.

---

## 1. The problem, restated precisely

`convert::build_structured_array` (`crates/viva-core/src/convert.rs:202`) is called **every tick**
from `build_view` for every bulk/unique input port. It does `np.zeros(length, dtype)` then one
`arr.set_item(field, column)` per field — i.e. it **allocates a fresh interleaved numpy record array
and copies every column into it, every tick**. That copy is what inverts the hot-path day-one number:

| M1 bench (bulk n=200, `docs/milestone-1-results.md` §2A) | ticks/sec |
|---|---|
| Rust engine + Python processes | ≈ 6,760 |
| Python engine + Python processes | ≈ 101,700 |
| **ratio** | **≈ 0.066× (net slowdown)** |

The scalar-only composite (no array crosses the boundary) is **1.27×** (§2B) — so the engine itself
already wins; the structured-array copy is the entire deficit.

**Why a naive view doesn't work:** numpy structured arrays are **interleaved** (record layout: all
fields of row 0, then row 1, … in ONE contiguous buffer). The Rust `State` stores each unique store
**column-oriented** — `UniqueStore.columns: Map<Column>`, one `Array1`/`Vec` per field
(`crates/bigraph-schema-rs/src/state.rs:150`) — and each `BulkStore` as separate `count`/`ids`/`floats`
(`state.rs:123`). A zero-copy numpy structured **view** requires the backing bytes to already be in
numpy's interleaved layout; column-oriented Rust storage cannot be viewed as one record array.

---

## 2. The real v2ecoli access pattern (this decides the design)

Checked the actual hot-path helpers in `/Users/eranagmon/code/v2ecoli/v2ecoli/library/schema.py`
and the M1 process bodies (`python/viva_core/conformance_processes.py`):

- **Bulk read** — `counts(states, idx)` (schema.py:227): `if states.dtype.names: return states["count"][idx]`.
  Always **field-wise** (`states["count"]`), and it branches on `states.dtype.names`.
- **Unique read** — `attrs(states, attrs)` (schema.py:398):
  `mask = states["_entryState"].view(np.bool_)` then `[states[a][mask] for a in attrs]`.
  Again **field-wise**, plus the int8-field-viewed-as-bool idiom.
- **Writes** — processes **do not mutate the view**; they **return update dicts**:
  bulk → `{"bulk": [(idx_array, delta_array)]}` (a `BulkNumpyUpdate`),
  unique → `{"mol": {"set":…, "add":…, "delete":…}}`. The grow/`np.append`/`MetadataArray`
  machinery (`get_free_indices`, `UniqueNumpyUpdater`) runs in the **apply**, which in M1 is the
  Rust `flush_unique`/bulk-apply path (`crates/bigraph-schema-rs/src/apply.rs`) and is already
  **bit-identical** (results §1, §3).

**Two consequences that shape everything:**
1. Processes access stores **by field**, never as whole records → a true record array is needed only
   so `states["field"]` and `states.dtype.names`/`.view(bool)` work; the *interleaving* is incidental
   to how numpy must lay the bytes out, not something the process logic wants.
2. The realistic hot path is **read-only across the boundary** — writes go back as update dicts on the
   already-bit-identical Op/delta path. So **zero-copy READ alone recovers the entire deficit**;
   writable views are not required for the v2ecoli pattern.

---

## 3. Options analysis

### (B) Per-column views — REJECTED as the primary mechanism
numpy structured-dtype fields are **byte offsets into ONE buffer** (probe claim 2:
offsets `{_entryState:0, count:1, mass:9}`). numpy has **no** API to back two fields by two separate
allocations, so "a structured array whose fields view separate Rust column buffers" is **impossible**.
The fallback — hand the process a plain `dict`/object of per-column 1-D views — **breaks the real
access pattern**: `counts()` does `states.dtype.names` and a dict has no `.dtype` (probe claim 4). A
numpy-subclass/`__getitem__` shim could fake it, but it's fragile and still fails `.view()`/`.dtype`
code. Not viable as the boundary type.

### (A) Interleaved-buffer State for the crossing stores — the required layout for unique
Store each unique store (and bulk) as **one contiguous byte buffer in numpy record layout**, so numpy
views it zero-copy as a genuine structured array. **Validated** (probe claims 1a–1e):
- `np.frombuffer(buf, dtype=record_dt)` is a zero-copy view; `.dtype.names` truthy.
- `arr["count"]` is a **strided** view that **shares memory** (stride = itemsize 17, non-contiguous) —
  `counts()`/`attrs()` field access works on it.
- in-place writes reach the raw buffer (so a writable view is *available* if ever needed).
- `arr["_entryState"].view(np.bool_)` works **on the strided field view** (same itemsize) — the exact
  `attrs()` idiom.
- attrs-style masked gather (`arr["count"][mask]`) returns the right rows.

Cost: the hot data structure changes from columnar to interleaved for these stores, and
`flush_unique` / `active_indices` / `get_free_indices` / bulk `count[idx]+=` must operate on the
interleaved buffer via **field offset + record-stride arithmetic** and stay bit-identical. This is the
real work and the real risk.

### (C) Hybrid: zero-copy READ + writes via the existing Op/delta path — RECOMMENDED posture
Because the v2ecoli hot path is read-only across the boundary (§2), do **read zero-copy** and leave
writes on the M1 Op/delta path that is **already bit-identical**. This keeps the dangerous mutation
code unchanged and still removes 100% of the per-tick copy (the copy is on the *view-construction*
side). The view can be handed **read-only** (`arr.flags.writeable = False`) to make "don't write
through the view" a hard invariant, not a convention.

---

## 4. RECOMMENDATION — **A (layout) + C (read-only view), staged; bulk first**

Adopt the **interleaved-buffer backing (A)** so Python gets a genuine numpy structured array, but use
it **read-only with writes via the Op path (C)**. Stage it so the first increment is tiny and proves
the end-to-end win before the risky unique rewrite:

- **Stage 1 (bulk, minimal, proves the pattern):** the bulk hot field is `count` (flat `Array1<i64>`,
  `state.rs:123`). A **single-field** record dtype `[("count","<i8")]` over that flat buffer is
  **byte-identical to the flat i64 buffer** → zero-copy with **no layout change at all** (probe claims
  3a–3c: `np.shares_memory` true, `.dtype.names`=`('count',)` truthy, `states["count"]` correct). This
  alone satisfies `counts()`/`state["bulk"]["count"]`, which is exactly what the M1 bench + bulk
  parity tests exercise. (If a process also reads `id`/submass, add those as further fields — but that
  needs interleaving; the hot evolvers read only `count`, and `id`/mass are emit-path only via
  `np.ascontiguousarray`, which copies regardless.)
- **Stage 2 (unique, the real layout change):** convert `UniqueStore` to an interleaved byte buffer
  (+ cached `PyArrayDescr`) and rewrite `flush_unique`/`active_indices`/`get_free_indices` to
  offset/stride arithmetic on it. Bit-identical-guarded by the oracle + conformance harness.

Rationale: matches the real access pattern (field-wise reads, dict writes); gives a **true** numpy
structured array so `counts`/`attrs`/`.view(bool)`/`.dtype.names` "just work" unmodified; keeps the
bit-identical write path untouched; and front-loads the cheap, decisive bulk proof before the unique
rewrite.

---

## 5. Layout change to State + apply/reconcile

- **Stage 1 (bulk):** *no* change to `BulkStore` storage. In `convert.rs`, replace the bulk arm of
  `build_structured_array` with a zero-copy wrap of `BulkStore.count` under a cached single-field
  (or count-first) record dtype. `apply`/`reconcile` untouched (deltas already apply to `count`).
- **Stage 2 (unique):** `UniqueStore.columns: Map<Column>` → one interleaved `Vec<u8>` (len
  `rows * record_itemsize`) + a field table `[(name, dtype, offset, size)]` + cached `PyArrayDescr`.
  - `active_indices` / `get_free_indices`: read `_entryState` at `base + row*stride + es_offset`.
  - `flush_unique` set/add/delete: write cells at computed byte offsets instead of `col.write(row,..)`;
    grow = extend the `Vec<u8>` by `n_new*stride` zeros (mirrors `np.append(.., np.zeros(...))`).
  - `reconcile` is unchanged (it merges update **dicts**, not storage).
  - Endianness/alignment: pack to **little-endian, numpy default field offsets** (packed, as
    `np.dtype([... ])` lays out — itemsize 17 in the probe, i.e. **unaligned/packed**, not C-struct
    aligned). Match numpy's offset table exactly (derive offsets from a built `PyArrayDescr`, don't
    hand-compute) or field reads will silently corrupt.

---

## 6. PyO3 / numpy mechanism + lifetime/safety (pyo3 0.29, numpy 0.29 = rust-numpy)

Two sound ownership models; recommend the first for M1:

- **(Recommended) numpy owns the canonical buffer, Rust holds `Py<PyArray>` and mutates in place.**
  Create the record array once (numpy allocates); store the `Py<...>` in the Rust store. Each tick the
  view handed to the process is just a borrowed ref to that persistent array — **no alloc, no copy**.
  Rust apply mutates it under the GIL (raw-ptr/offset writes, or keep applying to a shadow and sync).
  Lifetime is trivially safe: **numpy's refcount owns the bytes**, so "Python outlives the Rust
  buffer" cannot happen. Downside: that store needs the GIL to touch, so the future *native-only /
  rayon* path can't use it lock-free — acceptable for M2 (Python-process-on-Rust-engine), revisit for
  native processes.

- **(Alternative) Rust owns the `Vec<u8>`, per-tick zero-copy view with a `base` container.** Wrap the
  slice in a numpy array whose `base` is a Py object pinning the Rust store (rust-numpy borrow path),
  so numpy can't read freed memory. Safe because **growth/realloc happens only in `flush_unique`
  (Rust apply), strictly after the process has returned and its view is dropped** — no live view ever
  survives a realloc. Keeps the store GIL-free for native use. More `unsafe` to get right.

Either way: hand the process the view with **`writeable = False`** (writes go via the update dict →
Op), make the `PyArrayDescr` **cached per store** (don't rebuild the dtype each tick), and keep the
GIL held across the tick (M1 already does — results §E notes "no per-call acquire/release").

**Validate the exact rust-numpy borrow constructor with a Rust probe** before Stage 1 lands (the
Python-side layout is already proven; the Rust-side zero-copy wrap of a borrowed slice under a custom
`PyArrayDescr` is the one unproven mechanic). Stage E of the spike already proved zero-copy for a
**flat** `PyArray1`, so Stage 1 (single-field over the flat count buffer) is the smallest extension of
a proven mechanism.

---

## 7. Bit-identical risk + guard

Risks: (a) wrong field offsets/endianness → silent corruption; (b) strided-view surprises in a
process; (c) the unique grow/flush rewrite diverging from `np.append`/`UniqueNumpyUpdater` ordering.

Guards — **reuse the M1 conformance harness unchanged** (it already asserts bit-identical vs the real
Python bigraph-schema 1.6.0 / pb 1.8.3):
- `python/viva_core/conformance.py` + the run()-level battery `python -m viva_core.upstream_suite`
  (6/6, 0 real divergences — results §3 Part 2), especially the **bulk** and **FULL MODEL** cases.
- `cargo test --workspace` — the 9 oracle-backed apply/reconcile/flush tests generated from the real
  Python apply (`crates/bigraph-schema-rs/tests/oracle/gen_oracle.py`). The unique rewrite (Stage 2)
  must keep these green.
- Add a **shares_memory assertion** in the parity test: `np.shares_memory(view["count"], <prior>)` /
  check no per-tick allocation, so a regression that silently reintroduces the copy fails the test,
  not just the benchmark.

---

## 8. CONCRETE FIRST TASK (smallest change that proves zero-copy read end-to-end)

**Make the BULK view zero-copy over `BulkStore.count`, remove the copy, re-prove parity + re-benchmark.**

1. In `convert.rs`, add a bulk-specific zero-copy path: wrap `BulkStore.count` (flat `Array1<i64>`) as
   a numpy array with a cached single-field record dtype `[("count","<i8")]` (zero-copy; proven to
   share memory and satisfy `.dtype.names` + `states["count"]`), instead of calling
   `build_structured_array` for the bulk arm. Hand it `writeable=False`.
2. Keep writes exactly as-is (`bulk_op` → `Op::BulkDelta` → `count[idx]+=`, already bit-identical).
3. **Parity gate:** run the M1 bulk cases — `python -m viva_core.upstream_suite` (the two
   `apply(bulk_array)` cases + FULL MODEL) and `python/tests/test_quadrant3_parity.py` — must stay
   bit-identical. Add a `np.shares_memory`/no-copy assertion.
4. **Re-benchmark:** `./.venv-m1/bin/python python/benchmarks/bench_m1.py` case A (bulk n=200, and
   sweep n up to ~16k). Expect the **0.066×** to move to **≥1×** (toward the 1.27× scalar ceiling as
   the copy vanishes and only the engine delta remains).
5. If needed first, a ~30-line Rust probe confirming the rust-numpy borrow-from-slice-with-custom-descr
   wrap is zero-copy (`np.shares_memory` true from the Python side) — then land step 1.

Success = bulk parity still bit-identical **and** bench case A no longer a net slowdown. That unblocks
Stage 2 (the unique interleaved-buffer rewrite) on a proven mechanism.
