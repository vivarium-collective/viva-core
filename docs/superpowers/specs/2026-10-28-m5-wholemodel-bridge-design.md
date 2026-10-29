# viva-core Milestone 5 — the whole model RUNS on the Rust engine (design)

**Companion investigation (the evidence base):** `2026-10-28-m5-investigation.md`
(exact pint wire shapes, the 23-leaf unit survey, the strict read, is-identity of
the process store, the allocator_rng find, ranked design options with costs/risks).

## Intent

M4 proved the real v2ecoli baseline does **not** run end-to-end on the Rust
engine's `run_prebuilt` yet, and that the engine alone is net-negative (every
Python process pays a PyO3 crossing per tick). The user chose to **make the whole
model run bit-identical on the Rust engine first**, then port processes for speed.

M5's deliverable: the REAL baseline composite (one generation, baseline media,
seed 0) runs for N ticks under `viva_core.run_prebuilt`, native-swap OFF,
**bit-identical to process-bigraph 1.8.3** — i.e. every process in the model runs
as its pre-built Python instance on the Rust engine, with no store the handoff
cannot carry. This is an **enablement / correctness** milestone, not a speed one:
the whole-model number stays <1× vs pb until processes are ported (M6+). No
fabricated speedup.

## What blocks it (from the investigation) — four things, each low-risk

1. **pint units at the PyO3 boundary.** Only ~23 `pint.Quantity` leaves exist in
   the whole agent (22 `listeners.mass.*`, 1 `effective_elongation_rate`). The Rust
   codec *already* carries the pint tag opaquely through state and `Op::Overwrite`;
   the loss is purely in `convert.rs` (`value_to_py` hands back a plain dict;
   `py_to_value` turns a `Quantity` into `Null` or infinite-recurses on an
   `np.float64` magnitude).
2. **The live process-as-data store.** The partition `process` store holds
   `(instance,)` — the *same* live object the Requester/Evolver read via
   `self.parameters["process"]`. Live Python objects cannot cross the JSON state
   channel.
3. **`allocator_rng`** — a live `numpy.random.RandomState` the Allocator mutates in
   place (missed by M4; same failure class as #2).
4. **Realized-`Composite` handoff** — the raw `baseline()` doc has edges with no
   realized `instance` (e.g. `shape_step`), so `build_payload` must start from a
   realized `process_bigraph.Composite` state.

## Chosen designs (ranked #1 in the investigation for each)

- **Units → Option A (opaque tagged `Value` in Rust + wrap/unwrap at the PyO3
  boundary).** Hot path (`apply`/`reconcile`/scheduler) untouched — it never
  inspects a unit. Changes confined to `crates/viva-core/src/convert.rs`
  (`py_to_value` gains a `pint.Quantity` arm that emits
  `Value::Tagged(Pint, {..})` and records the scalar magnitude dtype as an
  extension key; `value_to_py` rebuilds a `Quantity` via the shared
  `get_quantity_registry()` with a cached unit-string parse), plus a
  `pint.Quantity` case in `python/viva_core/prebuilt.py::_serialize`, and
  `state_to_py` reusing the rebuild so the final state compares equal to pb.
  **The scalar `np.float64`→`float` demotion must be prevented** (the extension
  dtype key): `1/Quantity(0.0)` raises `ZeroDivisionError` for a Python-float
  magnitude but yields `inf` for `np.float64`, and mass divisors hit this.
  Rejected: a first-class Rust `Value::Quantity` (Option B, overkill until native
  twins consume units) and strip-and-restore via a unit registry (Option C,
  carries a second source of truth + loses the dtype).

- **Process store + `allocator_rng` → Option 1 (Python-owned handle table consulted
  at the view boundary; Rust never serializes it).** Extend the engine-side
  per-path object table (same pattern as `DtypeMap.ids`) with
  `handles: HashMap<Vec<String>, Py<PyAny>>`. `build_payload` passes the live
  objects for `["process", <name>]` and `["allocator_rng"]`; the state JSON holds a
  placeholder at those paths so wiring + the Requester→Evolver ordering edge are
  unchanged; `storenode_to_py` returns the live object for a handle path (the
  `(instance,)` tuple for `process`); updates targeting a handle path are dropped
  (same object, already mutated in place). **No v2ecoli change.** Rejected:
  restructuring `partition.py`/`allocator.py` (Option 2 — cleaner long-term but
  edits the shared v2ecoli + the pb path, needs its own pb regression) and
  pickling the instance (Option 3 — non-serializable sim_data/GLPK handles, breaks
  shared identity).

- **Realized handoff:** `build_payload`/`run_prebuilt` accept a realized
  `Composite` state (all edges carry a live `instance`), rather than the raw
  `baseline()` doc.

## Known-unknowns to close before declaring M5 done (validation, not assumed clean)

- Listeners hold **93 `ndarray` + 8 `MetadataArray`** leaves that `_serialize`
  turns into plain lists — a type change at the view boundary that any reader doing
  array math on a listener input would see. Diff vs pb on the real baseline.
- Non-`overwrite` nested listener output types (`map[map[overwrite[float]]]`, …)
  fall into the `_ → Op::Overwrite` arm in `op_from_type`, which may differ from
  pb's merge semantics for nested listener dicts. Diff vs pb.

These are gated by Task 4's whole-model conformance; if either diverges, it becomes
an explicit M5 task or an honestly-recorded residual.

## Global constraints

- **Bit-identical to pb 1.8.3** on the real baseline: decoded state int-exact,
  floats ≤1e-12, RNG-exact, and Quantity-typed leaves equal in BOTH magnitude and
  units (not just magnitude). The conformance comparator gates every tick.
- **No v2ecoli change** (Option 1/A both avoid it). The shared
  `/Users/eranagmon/code/v2ecoli/.venv` and canonical `~/code/v2ecoli` checkout are
  READ-ONLY; wheels build into a fresh local venv; sim_data cache is on disk
  (`out/cache/`), no ParCa run.
- The hot path stays unit-unaware (Option A keeps units opaque in Rust state).
- Pre-flight: re-verify the M4 `run_prebuilt` tests against current v2ecoli main
  (@20c69d78, post sim_data-decouple) before adding M5 code — `run_prebuilt` takes
  post-resolve pre-built instances, so the decoupled config shapes are expected not
  to affect it, but confirm.
- No AI attribution; `cargo fmt` + `clippy -D warnings` clean; GLPK feature-gated
  (default wheel GLPK-free); M1–M4 conformance stays green.

## Honest scope

One generation, baseline media, single seed, no functional division. M5 makes the
whole model *run* bit-identical on the Rust engine; it does **not** make it faster
(still <1× vs pb — the PyO3-boundary tax remains until processes are ported). The
whole-model *speedup* is the M6+ native-port program.
