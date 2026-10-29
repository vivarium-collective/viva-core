# Milestone 5 — the whole model RUNS on the Rust engine

**Scope:** the REAL v2ecoli baseline composite — one generation, baseline
("minimal") media, single seed (0), no functional division — against v2ecoli
`@main` (`20c69d78`, the sim_data-decoupled + schema-v4 code), sim_data hydrated
from the shipped ParCa fixture (`scripts/build_cache.py`, SCHEMA_VERSION 4; no
ParCa run). Wheel built with maturin into a fresh local venv; conformed to
process-bigraph 1.8.3 / bigraph-schema 1.6.0.

## Headline

The whole realized baseline agent (46 edges, ~55 processes) now **RUNS
end-to-end on the Rust engine** via `viva_core.run_prebuilt`, native-swap OFF —
every process executes as its pre-built Python instance on the Rust scheduler,
through the realized-`Composite` handoff (`build_payload(comp, path=("agents",
"0"))`). M4 could not even build the payload; M5 makes the whole model run.

The run is **deterministic** (repeated run_prebuilt runs are byte-for-byte
identical) and is now **bit-identical to pb 1.8.3** in all core model state — bulk
counts int-exact, unique stores (including the next-unique-index counter)
int-exact, every float leaf within 1e-12, every pint Quantity equal in magnitude
and units. The earlier "float-precision residual" diagnosis was **wrong**: the
divergence was a deterministic **scheduler-ordering** defect (plus a unique-index
counter gap), both now fixed (see "Root cause" below). One narrow, precisely
located residual remains in six additive-store diagnostic listener leaves (the
schema-projection apply gap), tracked as a strict-xfail — it does not perturb the
model trajectory.

This is an **enablement / correctness-plumbing** milestone, not a speed one — see
the number.

## What it took (beyond the four designed blockers)

The four blockers the design named were all resolved as planned:
1. **pint units** at the PyO3 boundary (Task 1, Option A — opaque tag + wrap/unwrap).
2. **Live process-as-data store** + **`allocator_rng`** (Task 2, Option 1 — Python-owned handle table).
3. **Realized-`Composite` handoff** (Task 3 — accept a `Composite` + `path`).

Running the *whole* model surfaced a further set of representation gaps that only
the real baseline exposes; M5 Task 4 added each (all general, no v2ecoli change):

- **Emitter fallback** — the baseline's `ParquetEmitter` edge has no native Rust
  twin; the registry now maps any emitter-addressed edge to the RAM emitter (a
  read-only sink — state-equivalent).
- **Nested-map update merge** (closes known-unknown **(b)**) — a dict written to a
  container-typed port (`map[T]` / `tree[T]` / untyped `node`) is merged
  per-leaf, not wholesale-overwritten at the node: `environment.exchange`
  (`map[float]`) applies **additively** per key; `environment.exchange_data` and
  the nested listener trees **overwrite per leaf**. (Previously the whole dict hit
  the `_ -> Op::Overwrite` arm → op/store type mismatch.)
- **Schema-absent-path drop** — pb projects every update through the composition
  schema, silently dropping writes to paths the schema does not have (e.g.
  `ecoli-equilibrium` returns `listeners.equilibrium_listener.*`, which is not a
  field of `listeners`). The engine now drops writes whose target path is absent
  from the state (dynamic `map` keys such as `environment.exchange.*` are
  pre-seeded, so they are kept).
- **Empty-dict updates** are no-ops (a unique-molecule port returning `{}` on a
  tick must not clobber the store).
- **Wholesale subtree overwrite** — `overwrite[map[...]]` (the partition `request`
  store, rewritten each tick) replaces the whole subtree, rebuilt from the value.
- **Plain ndarray leaves** preserved as ndarrays (closes known-unknown **(a)**) —
  a `__numpy__` tag through the view + final state, so a reader doing array math /
  fancy-indexing (`basal_prob[TU_index]`) gets an ndarray, not a Python list, and
  the final state compares as arrays. A **0-d** array lowers to a scalar (pb
  serves it as a scalar; a 0-d array breaks numba kernels). Scoped to stored leaf
  values (`Op::Overwrite`), NOT the arrays inside bulk/unique op payloads.
- **Unique stores as `MetadataArray`** — the view carries the molecule-type-
  namespaced next-unique-index counter as `.metadata` (read by
  `create_unique_indices`); the counter is preserved verbatim through
  serialization (it is NOT `max(unique_index)+1` for an empty store, e.g.
  `chromosomal_segment`), and advances by the number added.
- **Bulk submass columns** — the bulk view now also answers the per-molecule
  submass float columns (`rRNA_submass`, …), single- and multi-field
  (`bulk[[...]]`, MassDeriver), with `count` still the zero-copy column.
- **`BulkView` subclass degradation** — the zero-copy bulk view degrades to a
  plain `ndarray` through numpy ufuncs / reductions / field reads (an M2 design
  gap the whole model exposed: a bulk-derived value like `aa_conc` would stay a
  `BulkView`, so `(x/f).min()` returned a 0-d `BulkView` that numba rejects). The
  raw `count` read stays zero-copy; only derived results degrade — matching pb,
  which hands plain ndarrays.
- **Edge name on each instance** — `Requester.update_condition` reads `self.name`;
  pb sets the edge name on realized instances, so the handoff does too.

## Known-unknowns — closed

- **(a) ndarray / MetadataArray listener leaves.** Resolved: ndarray leaves
  round-trip as ndarrays (view + final state) and unique stores are
  `MetadataArray`s. A reader doing array math sees an ndarray, not a list; the
  final state compares equal in type. Gated by
  `test_known_unknown_a_ndarray_leaf_is_ndarray`.
- **(b) non-`overwrite` nested output merge.** Resolved: `map[float]` merges
  additively per key and generic `node`/`map[overwrite[…]]` trees merge
  overwrite-per-leaf, matching pb — not a wholesale node overwrite. Gated by
  `test_known_unknown_b_nested_map_merges_additively`.

## Speed (the honest enablement number)

Whole model, release wheel, warm-up excluded, per-tick via the differential
method (`bench_m5_wholemodel.py`):

| engine | ms/tick | ticks/s | vs pb |
|---|---|---|---|
| pb 1.8.3 | 152 | 6.58 | 1.00x |
| `run_prebuilt` native-OFF | 353 | 2.83 | **0.43x** |

As expected and designed: **< 1x**. M5 makes the whole model *run* on the Rust
engine; it does not make it faster. Every process still pays a PyO3 view/update
crossing per tick on top of pb's in-process scheduling. The whole-model speedup
is the **M6+ native-port program** (porting the hot processes to Rust twins; the
one ported process, `ProteinDegradation`, is already > 1x native-ON in M3/M4).

## Root cause of the former divergence (bit-identity now achieved)

The ~3e-7 bulk-total divergence after two ticks was **NOT** float/numba noise. It
was a deterministic **step-scheduler ordering** defect, with a secondary
unique-index counter gap. Both are fixed; the core model is now bit-identical.

1. **Scheduler ordering (the primary cause).** The realized v2ecoli baseline is
   wired by pb's `wire_step_layers` into a linear flow-token chain — every step
   reads `_layer_token_{N-1}` and writes `_layer_token_N`, and that chain is the
   data dependency pb's `build_step_network` / `determine_steps` order the start
   cascade by. The handoff (`prebuilt.build_payload`, via `_strip_layer_ports`)
   was **stripping those tokens**. The whole model's real stores (bulk / unique /
   timestep) are read-and-written by nearly every step, so they are self-loops
   that create no ordering edges; with the tokens gone the Rust scheduler had no
   total order and ran steps in document order — e.g. `ProteinDegradation` *before*
   `Equilibrium`, where pb runs `Equilibrium` first. `stochasticRound` and the
   near-degenerate FBA LP then amplified that integer-count seed into the drift.
   **Fix:** pass the `_layer*` flow tokens (and `priority`) through the handoff so
   the Rust scheduler reproduces pb's exact layer order. The whole start cascade
   now matches pb step-for-step and the bulk/unique state is bit-identical.

2. **Unique-index counter (secondary).** When a process pre-numbers an `add`
   (supplies `unique_index`, e.g. via `create_unique_indices`), pb does NOT bump
   the store counter on apply, and `create_unique_indices`'s in-place `.metadata`
   bump on the shared array is seen by the auto-numbered add that follows. The
   handoff view is detached, so the Rust engine (a) wrongly bumped the counter on
   pre-numbered adds and (b) lost the in-place bump. **Fix:** `flush_unique` bumps
   only on auto-numbered adds (matching `UniqueNumpyUpdater`), and each process's
   `.metadata` delta is read back off its view and replayed against the store
   (`Process::take_metadata_writebacks`). `active_RNAP` and all unique indices now
   match pb exactly.

Also note the benign **+1 global_time label offset**: `pb.run(N)` ends labelled
global_time N, `run_prebuilt(n_ticks=N)` at N+1, while both run every process the
same number of times (the documented duration-vs-iteration convention). The clocks
and the one purely-clock-derived listener (`expected_mass_fold_change`) carry this
offset; it is not a state divergence.

## Remaining residual — the schema-projection apply gap (narrow, tracked)

Six **additive-store diagnostic listener** leaves still differ:
`listeners/atp/{atp_requested,atp_allocated_initial}`,
`listeners/rnap_data/{rna_init_event_per_cistron,did_initialize}`,
`listeners/rna_synth_prob/{total_rna_init,n_actual_bound}`. pb applies every update
through the **store's** composition schema, so a store typed `integer` /
`array[integer]` / a `*_vec` labeled array accumulates additively across ticks —
even when the writer declares `overwrite[integer]` or returns a plain int array.
The Rust engine instead picks the apply op from the **writer's** declared
output-port type (Integer -> additive, but Array / Named / Overwrite -> overwrite)
and has no element-wise integer-array additive op, so these leaves are overwritten
(kept at the last tick) rather than accumulated. They are diagnostic sinks — the
core trajectory is unaffected (bulk/unique/masses/fluxes are bit-identical), which
is why this was masked until the ordering defect was fixed. Closing it needs the
composition schema's per-store apply-types carried into the handoff so the engine
projects updates through them (the same architectural gap as the schema-absent-path
map-vs-node drop). This is beyond the M5 scheduler/counter scope and is pinned by
the strict-xfail `test_listener_schema_projection_residual`.

## Scope recorded

One generation, baseline media, single seed, no functional division. The whole
model runs on the Rust engine, is deterministic, and is bit-identical to pb 1.8.3
in all core model state (one tracked listener residual above); it is not yet faster
(PyO3 tax; speedup is M6+).

## Tests

- `python/tests/test_wholemodel_runs.py` — the whole model runs end-to-end
  native-OFF, is deterministic, exercises the `Composite`+`path` handoff with the
  real partitioned processes + handle table (no wire-symmetry false-positive),
  and closes known-unknowns (a) and (b); `test_wholemodel_bit_identical_to_pb`
  hard-asserts bit-identity over all core state (bulk/unique/float/Quantity), and
  the six-leaf schema-projection residual is pinned by the strict-xfail
  `test_listener_schema_projection_residual`.
- M1–M4 conformance + Tasks 1–3 stay green in both native-flag modes.

## Reproduce

```
# schema-v4 cache from the shipped fixture (minutes, no ParCa)
git -C ~/code/v2ecoli worktree add ~/code/v2ecoli--m5cache origin/main
<v2ecoli venv python> ~/code/v2ecoli--m5cache/scripts/build_cache.py \
    --cache ~/code/v2ecoli--m5cache/out/cache

# build the wheel into a fresh venv, then:
export V2ECOLI_ROOT=~/code/v2ecoli--m5cache
export V2ECOLI_CACHE=~/code/v2ecoli--m5cache/out/cache
export PYTHONPATH=~/code/v2ecoli--m5cache:<v2ecoli venv site-packages>
<venv>/bin/python -m pytest python/tests/test_wholemodel_runs.py
<venv>/bin/python python/benchmarks/bench_m5_wholemodel.py
```
