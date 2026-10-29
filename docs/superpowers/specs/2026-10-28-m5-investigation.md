# M5 de-risk: unit bridge + live process-as-data store

Read-only investigation. Code read from `/Users/eranagmon/code/viva-core--m4-wholemodel`
(the M4 worktree; the canonical `~/code/viva-core` checkout does NOT contain
`run_prebuilt`/`prebuilt.py`) and the shared venv `/Users/eranagmon/code/v2ecoli/.venv`
(bigraph-schema 1.6.0). Empirical scans ran the real baseline (`out/cache`, seed 0) via
scratch scripts in this scratchpad (`scan.py`, `scan2.py`, `run3.py`, `t*.py`); nothing
outside the scratchpad was written.

---------------------------------------------------------------------------------

## 0. Headline corrections to the M4 framing

1. **Units are far narrower than "29/46 edges".** After running the real baseline 8 ticks
   under pb, exactly **23 leaves** in the whole agent are `pint.Quantity`, all under
   `listeners`:
   - 22 under `listeners.mass.*` (21 are `femtogram`, `volume` is `femtoliter`,
     `instantaneous_growth_rate` is `1 / second`), magnitude `np.float64`
     (`growth` initially a Python `float`);
   - 1 at `listeners.ribosome_data.effective_elongation_rate`, unit `amino_acid / second`.
   `environment` holds **one** Quantity at build time
   (`environment.exchange_data.constrained.GLC[p]`, `millimole / gram / hour`, Python
   float magnitude) and **zero after tick 1** (exchange_data/media_update rewrite them as
   bare floats). Every other store (`process`, `next_update_time`, `boundary`, ...) has none.
   So the 29/46 + 11/46 numbers are edges that are *wired to* a store that contains a few
   Quantity leaves, not edges that all exchange Quantities.
2. **The `process` store is not carrying data the edges lack.** Verified `is`-identity:
   for all 3 partitioned processes, `agent["process"][name][0] is
   Requester.parameters["process"] is Evolver.parameters["process"]`. The store value is a
   1-tuple `(instance,)`.
3. **A second live-object store exists that M4 did not list:** `allocator_rng` holds a live
   `numpy.random.RandomState` (read and mutated in place by `Allocator` via
   `states["allocator_rng"]` -> `calculatePartition`). `prebuilt._serialize` has no case
   for it (json.dumps TypeError). Same class of problem as `process`; the same solution
   covers both. (`process_state` is an ndarray/float store, fine.)
4. **`shape_step` has no realized `instance`** in the as-built agent
   (`build_payload` raises ValueError) -> the handoff must start from a *realized*
   `process_bigraph.Composite` state, not the raw `baseline()` doc. (Already noted in
   the M4 test docstring; restating because it is a third precondition.)

---------------------------------------------------------------------------------

## 1. (a) How a pint Quantity is represented in the JSON contract today

### Schema-agnostic codec: `bigraph_schema/json_codec.py` (1.6.0)

Encode (`BigraphJSONEncoder.default`), quoted:

```python
if isinstance(obj, pint.Quantity):
    mag = obj.magnitude
    if isinstance(mag, np.ndarray):
        return {'__pint__': True,
                'magnitude': mag.tolist(),
                '__magnitude_dtype__': str(mag.dtype),
                'units': str(obj.units)}
    if isinstance(mag, (int, float)):
        return {'__pint__': True, 'magnitude': mag, 'units': str(obj.units)}
    return {'__pint__': True, 'magnitude': float(mag), 'units': str(obj.units)}
```

Decode (`bigraph_json_hook`), quoted:

```python
if obj.get('__pint__') or obj.get('__pint_array__'):
    ureg = get_quantity_registry()          # bigraph_schema.units shared registry
    mag = obj.get('magnitude', obj.get('value'))
    dtype = obj.get('__magnitude_dtype__')
    if obj.get('__pint_array__') or dtype is not None:
        mag = np.array(mag, dtype=dtype) if dtype else np.array(mag)
    return ureg.Quantity(mag, obj['units'])
```

Exact wire shapes (verified by running them):

```
scalar : {"__pint__": true, "magnitude": 3.5, "units": "femtogram"}
rate   : {"__pint__": true, "magnitude": 2.0, "units": "1 / second"}
aa/s   : {"__pint__": true, "magnitude": 1.2345..., "units": "amino_acid / second"}
array  : {"__pint__": true, "magnitude": [..], "__magnitude_dtype__": "int64", "units": "count"}
legacy : {"__pint_array__": true, "magnitude": [..], "units": "..."}   (alias on decode)
```

Round-trip facts (tested): unit string `str(q.units)` re-parses to an equal unit; magnitude is
bit-exact. Two lossy points: (i) the scalar branch has **no dtype tag**, so an `np.float64`
magnitude comes back as a Python `float`; (ii) the registry used on decode is
`bigraph_schema.units.get_quantity_registry()`; v2ecoli sets
`pint.set_application_registry(ureg)` to that same registry (`v2ecoli/types/quantity.py:31`),
so decoded Quantities are registry-coherent with v2ecoli's.

Unum: **not distinct in state.** Unum only lives inside process internals / sim_data
(`library/unit_bridge.py: pint_to_unum / unum_to_pint`); stores carry pint only. No Unum
leaf was found in any agent store. So the bridge is pint-only.

### Schema-paired path: `methods/serialize.py` / `realize.py`

`serialize(schema: Quantity, state)` (serialize.py:992) emits
`{'units': {unit_name: exponent,...}, 'magnitude': float(state.magnitude)}` using
`dict(state.unit_items())`; `realize(core, schema: Quantity, encode)` (realize.py:851)
accepts an existing Quantity, `{units: {...}, magnitude}` or a bare number (units from
schema) and builds `ureg.Quantity.from_tuple`. v2ecoli overrides serialize to
`schema._serialize_state` (`types/__init__.py:20`) -> `{'units': ..., 'magnitude': ...}`.
This is a *different* shape from the tagged dict (`units` is a dict of exponents, no
`__pint__` key) and `float()` truncates array magnitudes. The Rust `.pbg` JSON path
(`prebuilt.py` -> `run_prebuilt`) uses neither; it should use the **tagged-dict** shape
(that is what the Rust codec already recognizes, below).

---------------------------------------------------------------------------------

## 2. (b) What the Rust side does today

JSON-in is already fine; the Python<->Rust boundary is where it is lost.

- `doc.rs`: `TagKind::Pint` (keys `__pint__` and legacy `__pint_array__`);
  `Value::Tagged(TagKind, Map<Value>)` "preserved opaquely (all keys, in order)".
  `codec.rs` `object()` builds `Tagged` when a tag key is `true` and `write_value` re-emits
  it. So a Pint tagged dict **survives JSON parse -> state -> JSON write byte-for-byte**.
- `state.rs::from_initial`: only `StructuredArray`/`NumpyStructured` tags are special-cased
  (lifted to Bulk/Unique); a Pint tag stays a `StoreNode::Leaf(Value::Tagged(Pint, ..))`.
- `apply.rs`: `Op::Overwrite(Value)` -> `replace_leaf` stores *any* `Value` verbatim.
  `reconcile(Overwrite)` = last-non-None. The hot path never inspects the value. **Every
  Quantity-typed output in the baseline is `overwrite[quantity[...]]`** (mass listener,
  `effective_elongation_rate`), so apply/reconcile need no unit awareness.
- `types.rs`/`grammar.rs` already parse `quantity[...]` -> `Type::Quantity{inner,units}`;
  but `convert.rs::op_from_type` has **no `Type::Quantity` arm** (falls to `_` ->
  `Op::Overwrite`). Fine for `overwrite[quantity[..]]` (it is `Type::Overwrite`); a
  non-overwrite `quantity[..]` *output* would be wrongly overwritten instead of added
  (none exist in the baseline outputs I saw; only inputs, e.g. polypeptide_initiation).

The two lossy boundary functions in `crates/viva-core/src/convert.rs`:

- **view (Rust -> Python), `value_to_py`**: `Value::Map(m) | Value::Tagged(_, m)` are the
  same arm -> a **plain `dict`** `{"__pint__": True, "magnitude":..., "units":...}`. It is
  never re-wrapped as a Quantity. A process doing `as_quantity(d, units.fg)` gets
  `dict.to`/`dict * unit` -> AttributeError/TypeError.
- **update (Python -> Rust), `py_to_value`**: a Quantity matches none of the exact
  int/float/str/list/tuple/dict arms, then:
  - `np.float64` magnitude: `hasattr(q,'tolist')` is True and `q.tolist()` **returns a
    Quantity again** (verified in Python) -> `py_to_value` recurses on an equal object ->
    unbounded recursion (Rust stack overflow / abort; reasoned from the verified Python
    behavior, not executed against the wheel).
  - Python-`float` magnitude: `q.tolist()` raises AttributeError; `extract::<i64>`/`<f64>`
    raise DimensionalityError -> falls through to **`Value::Null`**: silent loss of value
    and units.
  - array magnitude: `tolist()` returns a list of Quantities -> list of Nulls/recursion.
- **initial state, `prebuilt._serialize`**: no Quantity case; falls through to
  `return value` and `json.dumps` raises TypeError (the M4 "blocked" assertion; the
  M4 test comment says AttributeError for the `_serialize`-on-Unum case).

---------------------------------------------------------------------------------

## 3. (c) Read-critical vs write-mostly (the key question)

**Mostly tolerant reads, ONE strict read.** v2ecoli is mid-"units-on-ports migration" and
reads through `library/quantity_helpers.py`:

```python
def as_quantity(value, unit):   # bare number -> value*unit ; Quantity -> value.to(unit)
def fg_magnitude(value):        # float(magnitude_in(value, fg)); bare passes through
```

Writers (all emit Quantities): `MassDeriver` (`steps/derivers/mass_deriver.py:99-123`,
`overwrite[quantity[float,fg]]`, wraps at the emit point), `PolypeptideElongation`
(`processes/polypeptide_elongation.py:743`: `currElongRate * units.amino_acid / units.s`).

Readers of `listeners.mass.*` and how strictly they need a Quantity:

| reader | access | needs real Quantity? |
|---|---|---|
| metabolism:671-672, rna_degradation:380, equilibrium:424, two_component_system:265, chromosome_replication:343, polypeptide_elongation:523/1081/1199-1200, ppgpp_initiation:163, dnaa_box_binding:397, mass_conservation_deriver:140, transcript_elongation:492 | `as_quantity(x, units.fg)` / `fg_magnitude(x)` | **No** - bare float is wrapped with the expected unit (`.to(fg)` on same unit is bit-exact, tested) |
| cell_shape.py:132 (`shape_step`) | `float(getattr(raw,'magnitude',raw))` | No |
| metabolism:713-714 | `rna_mass / protein_mass` -> `rp_ratio` | No; bare/bare float vs dimensionless Quantity gives the same number |
| mass_deriver:243 | `fg_magnitude(dry_mass)` | No |
| **polypeptide_initiation.py:313-318** | `eff_rate = ...["effective_elongation_rate"]; if eff_rate == 0: ...; eff_rate.to(units.aa/units.s).magnitude` | **YES - strict.** A bare float (or the tag dict) raises AttributeError. Not covered by a helper. |
| metabolism:689-696 (`environment.exchange_data.constrained`) | `unum_to_pint(val)`; `if not hasattr(q,'magnitude'): q = q*constraint_unit` | No - explicitly accepts bare floats |

Conclusions:
- Units must be **preserved end-to-end for `effective_elongation_rate`** (write by
  PolypeptideElongation -> read by PolypeptideInitiation next tick; Quantity in, `.to()` out).
- For the 22 `listeners.mass.*` fields a bare-float view would *work* for the current
  readers, but only because of the helpers; re-wrapping on read is still the safe/uniform
  choice (and the tolerant path is exactly what `shape_step` demonstrates).
- Beyond reads, the *final state* the conformance harness compares against pb contains
  Quantities, so the write side must preserve them or the bit-identity diff is
  type-different even if magnitude-equal.
- Bit-identity hazards to design around:
  1. scalar `np.float64` -> `float` demotion on the tagged round-trip: `1/Quantity(0.0)`
     raises `ZeroDivisionError` for a Python-float magnitude but gives `inf` for
     `np.float64` (verified). `cell_mass`/`dry_mass` are divisors in several processes.
     Magnitude dtype must be preserved (see option A).
  2. Python `float` vs `np.float64` arithmetic otherwise agrees bit-for-bit (IEEE double).
  3. `volume` is `femtoliter`, `instantaneous_growth_rate` is `1 / second`: unit strings
     must re-parse to the *same registry unit* (they do; tested incl. `amino_acid / second`).

---------------------------------------------------------------------------------

## 4. (d) Unit-bridge design options (ranked)

### Option A (RECOMMENDED): opaque tagged `Value` in Rust + wrap/unwrap at the PyO3 boundary

- Rust state keeps `Value::Tagged(Pint, {...})` untouched (already works for JSON in/out
  and for `Op::Overwrite`/`replace_leaf`). **No change to `apply.rs`, `reconcile`, scheduler,
  or the hot path.**
- `convert.rs::py_to_value`: add a `pint.Quantity` arm *before* the `tolist` fallback ->
  `Value::Tagged(Pint, {"__pint__": true, "magnitude": <f64|Array>, ["__magnitude_dtype__"], "units": str(q.units)})`.
  Scalar magnitude: also record `"__magnitude_dtype__": "float64"` (or `"python_float"`)
  as an *extension key* so the view can rebuild `np.float64` vs `float` (closes the
  zero-division hazard). The Rust codec preserves unknown keys in tagged maps, and
  `bigraph_json_hook` ignores the extra key for scalars, so JSON compatibility holds.
- `convert.rs::value_to_py`: split the `Tagged` arm; `Tagged(Pint, ..)` ->
  `get_quantity_registry().Quantity(mag, units)` (mag via `np.float64(...)`/`np.array(.., dtype)`).
  Cache the registry handle + parsed units (the same ~23 unit strings recur every tick; a
  `PyOnceLock` registry + small unit-string cache like `_parsed_pint_unit`).
- `prebuilt._serialize`: add a `pint.Quantity` case (emit the tagged dict directly, or call
  `BigraphJSONEncoder().default(value)`).
- `state_to_py` (final state): reuse the same `value_to_py` arm so the returned state has
  Quantities and compares equal to pb.
- Cost: small. ~60-100 LOC Rust (two arms + registry cache) + ~10 LOC Python + tests
  (round-trip, `np.float64` zero-div, `effective_elongation_rate` read, final-state parity).
  Per-tick cost: ~23 Quantity constructions per consumer view (the readers only touch a
  few; could be lazy by wrapping only on ports a process reads, but all listeners-wired
  views build the whole `listeners` dict — measure; mass leaves are the only cost).
- Bit-identity risk: **low**. f64 stored as f64 (no text round-trip in-process);
  unit strings round-trip exactly; only real risk is the scalar dtype demotion, handled by
  the extension key. Hot path never sees units.
- Does the hot path need to understand units? **No - fully opaque.**

### Option B: first-class Rust `Value::Quantity { mag: Mag, unit: String }` (+ `Type::Quantity` arm)

- Adds a variant to `Value`, `PartialEq`, writer, `read_value` etc.; `op_from_type` gets a
  real `Type::Quantity` arm; unlocks Rust-side unit arithmetic/validation, native twins that
  read mass directly in typed form.
- Cost: medium-high (touch `doc.rs`, `codec.rs`, `state.rs`, `apply.rs` for non-overwrite
  quantity apply, every `match Value`); must also re-implement dtype preservation.
- Risk: low-medium (more code = more places to diverge from pb; unit-algebra semantics
  would have to match pint if Rust ever *combines* quantities, which nothing needs today).
- Hot path: yes, but only in `match` exhaustiveness unless native processes consume units.
- Verdict: only worth it later, when native Rust twins of mass-consuming processes need
  typed mass. Do not do this for M5.

### Option C: strip-and-restore via a per-field unit registry

- Strip Quantities to bare floats at `_serialize`/`py_to_value`, keep `{path -> unit}` in
  `DtypeMap` (like the `ids` table), re-attach unit on view for known paths.
- Cost: low-medium, but needs a *schema-keyed* registry (units live on the port schema, e.g.
  `overwrite[quantity[float,fg]]`; `parse_leaf_type` already yields `Type::Quantity{units}`).
- Risk: **medium-high.** (i) final-state type diff vs pb (Quantity vs float) unless
  restored on `state_to_py`; (ii) magnitude of mixed-unit conversions: a field can change
  units between writes only if a writer emits a different unit than the registry
  recorded (none today, but silent if it does); (iii) works for the tolerant mass readers
  but needs correct re-wrap for `effective_elongation_rate`, i.e. the registry must be right
  for the one strict read; (iv) still loses the np.float64/float distinction unless also
  recorded. Option A carries the unit *with the value*, which removes (ii) entirely.
- Hot path: opaque (floats), but the unit registry becomes a second source of truth.

Ranking: **A >> C > B for M5.**

---------------------------------------------------------------------------------

## 5. (e) Process-store reads: can the existing instance-binding serve them?

How `run_prebuilt` binds instances today (`lib.rs::build_prebuilt_engine` +
`pyproc.rs::PyProcess::from_instance` + `engine.rs::with_prebuilt`):
- `prebuilt.build_payload` splits **edges** from **stores**; for each non-emitter edge it
  puts the live instance in `processes[name]` (key = edge path joined `/`).
- Rust builds `prebuilt: HashMap<edge-path, Box<dyn Process>>`; each `PyProcess` holds a
  `Py<PyAny>` and drives it via `update/perform_update/calculate_timestep`.
- The map is keyed by **edge path** and consumed by `Engine::with_prebuilt` to *schedule*
  the edge. Nothing in this mechanism exposes an instance to a *view*: views are built from
  `StoreNode`s by `build_view`/`storenode_to_py`, which only know `Value`s.

So: **the same mechanism cannot serve data-store reads as-is** (it binds instances to
edges, not to store paths), but the *pattern* (Python-owned object, handed in by reference,
Rust just holds a `Py<PyAny>`) is exactly what is needed and there is a ready precedent in
the same boundary: `DtypeMap.ids: HashMap<Vec<String>, Py<PyAny>>`, a per-store-path table
of Python objects consulted in `storenode_to_py`.

What reads/writes `process` at runtime (`steps/partition.py`):
- `Requester`/`Evolver` declare port `process: 'quote'` in both `inputs()` and `outputs()`.
- `update()`: `proc_state = states.get("process")`; `process = proc_state[0]`; then
  `process.calculate_request(...)` / `process.evolve_state(...)`; `process.request_set` is read
  and set; **writes back `(process,)`** (a no-op overwrite of the same object).
- They need the **whole live instance** (two methods + a mutable attribute + cached
  indices on `self`), not a config or a single method.
- The state-store round-trip carries no information beyond (1) the **data dependency
  edge** that orders Requester -> Evolver, and (2) object identity. The instance in the
  tuple `is` the one in `parameters["process"]` (verified).
- Rust today: `process` leaf is a Python tuple -> `py_to_value` -> `[Null]` (instances
  have no `tolist`, no numeric coercion). View gives back `[None]`; `proc_state[0]` is
  `None` -> AttributeError on `.calculate_request`. (Initial serialization already fails
  earlier with TypeError.)

---------------------------------------------------------------------------------

## 6. (f) Process-store options (ranked)

### Option 1 (RECOMMENDED): Python-owned handle table consulted at the view boundary; Rust never serializes it

- Add `Py<PyAny>` handle entries to the engine-side table (extend `DtypeMap` with a
  `handles: HashMap<Vec<String>, Py<PyAny>>`, same lifetime/Arc as `ids`): for each store
  path whose value is a live object (`["process", name]`, `["allocator_rng"]`).
  `build_payload` passes `handles={path: python_value}` next to `processes`.
- State JSON gets a *placeholder* at those paths (e.g. `null` or a tiny `{"__handle__": name}`
  Tagged-like marker) so the store/wiring exists and the dependency graph is unchanged
  (Requester still writes `process`, Evolver still reads it -> same ordering as pb).
- `storenode_to_py`: if `path` is in `handles`, return the **handle object itself**
  (`(instance,)` for `process` to keep `proc_state[0]` working; the `RandomState` itself for
  `allocator_rng`) instead of reading the `Value`.
- Update boundary: when a PortUpdate targets a handle path, drop it (the object is the same;
  mutation already happened in place), or assert identity. Needs the same port->absolute
  path resolution the view uses (`view.path`); `translate_update` today only has the
  port-relative path, so add absolute-path resolution there (small).
- `state_to_py` (final state): re-insert the handle (so the returned state holds the same
  live objects and the conformance diff vs pb passes on that store too, or exclude it).
- Cost: low-medium; ~80-120 LOC Rust + ~30 LOC Python + 3 tests. No v2ecoli change.
- Bit-identity risk: **very low.** Same Python object, same in-place mutations
  (`request_set`, cached bulk indices, RNG state) as under pb. The allocator `RandomState`
  stays a single shared mutable (no copy -> no RNG stream divergence). Scheduler order is
  preserved by wiring.
- Constraint to note: handle objects must stay on the Python side; if M5+ ever moves
  Requester/Evolver (or Allocator) to native Rust twins, those twins would need their own
  representation (the `allocator_rng` is already ported as `rng.rs` MT19937; the partition
  instance is not). That is a later port concern, not an M5 blocker.

### Option 2: restructure v2ecoli so Requester/Evolver read `self.parameters["process"]`, and the `process` port carries a token

- Edit `steps/partition.py`: `process = self.parameters["process"]`; write
  `request["process"] = (name,)` (or a bool) just to keep the ordering edge; same for
  `allocator_rng` (hold the `RandomState` on the Allocator instance, not in state).
- Cost: very low code (~10 lines) **but it changes v2ecoli behavior in the shared
  checkout** (read-only for this job), changes the pb path too (needs its own re-validation
  of the pb baseline, `.test_durations`, snapshots), and does not address `allocator_rng`
  without a second edit (Allocator reads the rng from state, and the *same* RandomState is
  possibly read elsewhere).
- Bit-identity risk: low if done right (identical object), but moves a contract the pb side
  relies on (`process` store contents appear in saved states/emitters); needs a v2ecoli PR
  and a pb-vs-pb regression.
- Good long-term cleanliness (data stores should not hold live objects), poor as an M5 gate.

### Option 3 (not recommended): pickle/serialize the instance into state

- Instances carry live `sim_data`/GLPK handles (M4 already established they are
  non-serializable); a copy would also break shared identity. Reject.

Ranking: **1 > 2 >> 3.**

---------------------------------------------------------------------------------

## 7. Recommended M5 scope and ordering

1. Unit bridge, Option A (new `py_to_value` Quantity arm with dtype extension key;
   `value_to_py` Pint arm; `_serialize` Quantity case; `state_to_py` parity). Gate with a
   focused test: PolypeptideElongation -> PolypeptideInitiation `effective_elongation_rate`
   across 2+ ticks (the strict read), plus mass-listener round-trip with a zero-divisor
   `np.float64` mass.
2. Handle table (Option 1) for `process` and `allocator_rng`. Gate: Requester/Evolver for
   the 3 partitioned processes produce the same bulk/listener state as pb for N ticks.
3. Start from a realized `Composite` state (needed for `shape_step`'s instance).
4. Still-open items worth a spike *before* declaring M5 done (not blockers found here, but
   untested): listeners hold 93 `ndarray` + 8 `MetadataArray` leaves that `_serialize`
   turns into plain lists (type change at the view boundary: any reader doing array math on
   a listener input would see a list); and the non-`overwrite` output types
   (`map[map[overwrite[float]]]`, etc.) hit the `_` arm and become `Op::Overwrite`, which
   may differ from pb's merge semantics for nested listener dicts. Both should be diffed
   against pb on the real baseline once the units/handles land.

## 8. Key file references

- `/Users/eranagmon/code/v2ecoli/.venv/lib/python3.12/site-packages/bigraph_schema/json_codec.py`
- `.../bigraph_schema/methods/serialize.py:983-1002`, `.../methods/realize.py:851-871`
- `/Users/eranagmon/code/viva-core--m4-wholemodel/crates/bigraph-schema-rs/src/doc.rs` (TagKind, Value::Tagged)
- `.../crates/bigraph-schema-rs/src/{codec.rs,state.rs,apply.rs,grammar.rs,types.rs}`
- `.../crates/viva-core/src/convert.rs` (`value_to_py` Tagged arm, `py_to_value`, `op_from_type`, `build_view`, `storenode_to_py`)
- `.../crates/viva-core/src/{lib.rs (build_prebuilt_engine, run_prebuilt), pyproc.rs (from_instance)}`
- `.../python/viva_core/prebuilt.py` (`_serialize`, `build_payload`)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/steps/partition.py` (Requester/Evolver/PartitionedProcess)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/steps/allocator.py:59,150` (`allocator_rng`)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/library/quantity_helpers.py` (`as_quantity`, `fg_magnitude`)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/processes/polypeptide_initiation.py:308-318` (the strict read)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/processes/polypeptide_elongation.py:743` (the write)
- `/Users/eranagmon/code/v2ecoli/v2ecoli/steps/derivers/mass_deriver.py:99-123,243,386-396`
- `/Users/eranagmon/code/v2ecoli/v2ecoli/composites/ecoli_baseline.py:440-510,1153` (process store construction)
