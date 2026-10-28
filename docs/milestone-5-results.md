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
identical) but is **NOT yet bit-identical to pb 1.8.3**: a small, reproducible,
float-precision-level divergence remains in the numerically-sensitive early
processes and amplifies over ticks (the honest residual, below).

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

## The honest residual — NOT bit-identical

The whole-model run is deterministic but diverges from pb at the float-precision
level. Localized: at `ecoli-metabolism`'s **first** update (tick 0) the bulk
total already differs by ~8 counts out of ~3e10 (a ~1e-10 relative difference) —
i.e. an **upstream** numerically-sensitive process (an ODE/stochastic step such
as equilibrium / complexation / rna-degradation, run before metabolism in the
step order) produces a slightly different output. The metabolism FBA and the
downstream stochastic steps then **amplify** it over ticks (e.g. `rna_synth_prob`
init counts, `environment.exchange` fluxes). After two ticks the bulk-total
relative divergence is ~3e-7.

Both engines are internally deterministic, so this is a **reproducible numerical
-sensitivity residual, not randomness and not a structural blocker** — the model
runs *correctly* (same magnitude, same trajectory shape), just not bit-for-bit.
The likely root is a sub-tick representation difference (numpy operation ordering
/ list-vs-ndarray / float round-trip) in a hot ODE or numba kernel feeding FBA;
closing it to true bit-identity is follow-up work. The aspirational gate
`test_wholemodel_bit_identical_to_pb` is `xfail(strict=True)` and flips green the
day the residual is closed.

## Scope recorded

One generation, baseline media, single seed, no functional division. The whole
model runs on the Rust engine and is deterministic; it is not yet bit-identical
(float-precision residual above) and not yet faster (PyO3 tax; speedup is M6+).

## Tests

- `python/tests/test_wholemodel_runs.py` — the whole model runs end-to-end
  native-OFF, is deterministic, exercises the `Composite`+`path` handoff with the
  real partitioned processes + handle table (no wire-symmetry false-positive),
  and closes known-unknowns (a) and (b); the bit-identity gate is `xfail-strict`
  and the divergence is bounded (< 1e-6 relative) by a hard assert.
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
