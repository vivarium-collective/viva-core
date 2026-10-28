"""Importable Python processes used by the cross-language conformance harness.

These live in the installed `viva_core` package so BOTH engines can load them
through the identical `local:viva_core.conformance_processes.<Class>` address:

  * the Python pb 1.8.3 engine resolves `local:` via the core's link_registry
    (the harness registers these classes under their dotted path);
  * the Rust `viva_core` engine resolves `local:` via PyO3 importlib.

They subclass v2ecoli's :class:`EcoliStep` (not the raw pb `Step`) so a single
positional-config constructor works under both engines — the raw pb `Step`
demands a `core` kwarg the Rust `PyProcess` shim doesn't pass, whereas
`EcoliStep` picks up the current core set by `set_current_core`, exactly as the
quadrant-3 v2ecoli classes do.

The star of the suite is :class:`BulkStepper` — the carry-forward from Task 5:
a Python step that READS a bulk store's `count` column through the boundary and
WRITES a vectorized `(index_array, delta_array)` `BulkNumpyUpdate` back through
it, every cascade. Task 5 only ever drove *unique* stores across the boundary;
the bulk hot path (v2ecoli's dominant store) was never exercised end-to-end.
This step does exactly that, bit-identically against pb.

Steps are invoked WITHOUT an interval by pb but WITH `interval=-1.0` by the Rust
engine, so every delta here is interval-independent (or the engines would
disagree).
"""

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep, EcoliProcess


class BenchScalarProcess(EcoliProcess):
    """A fixed-timestep SCALAR evolver for the benchmark: no structured array
    crosses the boundary, so it isolates the pure engine + PyO3-callback overhead
    WITHOUT the per-tick `build_structured_array` copy — the 'pure-engine hot-path
    delta' the brief asks for."""

    name = "bench_scalar_process"
    config_schema = {}

    def inputs(self):
        return {"value": "float"}

    def outputs(self):
        return {"value": "float"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        return {"value": 1.0}


class TickTrigger(EcoliProcess):
    """Fixed-timestep driver that re-writes `timestep` (same value) every tick,
    so a v2ecoli Step reading `timestep` (e.g. ProteinDegradation) is re-triggered
    each scheduler iteration under BOTH engines (a Step runs reactively, not on an
    interval)."""

    name = "tick_trigger"
    config_schema = {}

    def inputs(self):
        return {}

    def outputs(self):
        return {"timestep": "overwrite[integer]"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        return {"timestep": int(self.parameters.get("value", 1))}


class BenchBulkProcess(EcoliProcess):
    """A fixed-timestep bulk evolver for the day-one benchmark: it runs EVERY
    tick (constant `calculate_timestep`), reads the whole `bulk` structured
    array across the boundary (the per-tick `build_structured_array` copy — the
    cost that makes the day-one number a PESSIMISTIC upper bound), and writes a
    vectorized `BulkNumpyUpdate`. Representative of a v2ecoli evolver's
    boundary traffic without needing a clock to advance time."""

    name = "bench_bulk_process"
    config_schema = {}

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        counts = np.asarray(state["bulk"]["count"])
        n = counts.shape[0]
        idx = np.arange(n, dtype=np.int64)
        deltas = ((counts % 7) - 3).astype(np.int64)
        return {"bulk": [(idx, deltas)]}


class BenchUniqueProcess(EcoliProcess):
    """A fixed-timestep UNIQUE evolver for the M2 Task-2 benchmark: it runs
    EVERY tick and READS a (large) unique structured array across the boundary
    via the exact v2ecoli `attrs` idiom — `_entryState.view(bool)` mask then a
    masked gather of a data field — which is the dominant per-tick unique copy
    (`active_ribosome` ~13k rows). It returns `{}` (no write), so the number
    isolates the READ-side cost: `build_structured_array` copy (before M2 Task 2)
    vs. the zero-copy interleaved-buffer view (after)."""

    name = "bench_unique_process"
    config_schema = {}

    def inputs(self):
        return {"active": "node"}

    def outputs(self):
        return {}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        view = state["active"]
        mask = view["_entryState"].view(np.bool_)
        _ = view["unique_index"][mask]
        _ = view["data0"][mask]
        return {}


class BulkProcess(EcoliProcess):
    """A v2ecoli-shaped *Process* (evolver) that reads `bulk["count"]` across the
    boundary and writes a vectorized `BulkNumpyUpdate` back — the carry-forward
    from Task 5 in its realistic form (bulk writes come from processes, not
    derivers). As a Process it is scheduled by `calculate_timestep` identically
    under both engines (the Rust `tick` mirrors pb's `run_process`), so the run
    count — and thus the accumulated state — matches exactly, unlike a Step
    (whose reactive triggering differs between the engines).

    It shares GlobalClock's clock-locked timestep so it fires in lockstep with
    the clock, and its delta is read-dependent + interval-independent + integer
    (bit-identical regardless of float interval).
    """

    name = "bulk_process"
    config_schema = {}

    def inputs(self):
        return {"bulk": "bulk_array", "global_time": "float",
                "next_update_time": "map[float]"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def calculate_timestep(self, interval_or_states, states=None):
        view = interval_or_states if states is None else states
        return min(nt - view["global_time"]
                   for nt in view["next_update_time"].values())

    def update(self, state, interval=None):
        counts = np.asarray(state["bulk"]["count"])
        n = counts.shape[0]
        m = int(self.parameters.get("modulus", 7))
        deltas = ((counts % m) - (m // 2)).astype(np.int64)
        if self.parameters.get("shape", "vectorized") == "scalar":
            # List of scalar (index, delta) pairs — the other BulkNumpyUpdate
            # shape (covers the scalar branch of the Rust bulk_op).
            return {"bulk": [(int(i), int(deltas[i])) for i in range(n)]}
        # One vectorized (index_array, delta_array) pair — v2ecoli's real shape.
        return {"bulk": [(np.arange(n, dtype=np.int64), deltas)]}


class BulkStepper(EcoliStep):
    """Reads `bulk["count"]` across the boundary and returns a vectorized
    `BulkNumpyUpdate` whose deltas DEPEND on the read counts — so a wrong read
    OR a wrong write diverges.

    The update is a single `(np.arange(n), deltas)` pair: the exact shape the
    real v2ecoli evolvers emit (one pair of parallel index/value arrays), which
    exercises numpy fancy-indexed apply in pb and the zipped vectorized
    `bulk_op` in Rust. Indices are unique (`arange`), so there is no
    duplicate-index ambiguity between numpy `+=` and the Rust accumulate loop.
    """

    config_schema = {}

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def update(self, state, interval=None):
        counts = np.asarray(state["bulk"]["count"])
        n = counts.shape[0]
        idx = np.arange(n, dtype=np.int64)
        m = int(self.parameters.get("modulus", 7))
        # Integer-exact, deterministic, READ-DEPENDENT delta:
        #   delta_i = (count_i % m) - (m // 2)
        deltas = ((counts % m) - (m // 2)).astype(np.int64)
        return {"bulk": [(idx, deltas)]}


class BulkScalarStepper(EcoliStep):
    """Same intent as :class:`BulkStepper` but emits a *list of scalar*
    `(index, delta)` pairs instead of one vectorized pair — the other shape
    `bulk_numpy_updater` accepts. Covers the scalar branch of `bulk_op`."""

    config_schema = {}

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def update(self, state, interval=None):
        counts = np.asarray(state["bulk"]["count"])
        pairs = [(int(i), int(counts[i] % 3) - 1) for i in range(counts.shape[0])]
        return {"bulk": pairs}


class IntAccumulator(EcoliStep):
    """Minimal integer-delta step (exercises `apply(integer)` across the
    boundary): adds a fixed integer to a scalar store each cascade."""

    config_schema = {}

    def inputs(self):
        return {"counter": "integer"}

    def outputs(self):
        return {"counter": "integer"}

    def update(self, state, interval=None):
        return {"counter": int(self.parameters.get("step", 3))}


class FloatAccumulator(EcoliStep):
    """Minimal float-delta step (exercises `apply(float)` across the boundary):
    adds a fixed `rate` to a scalar store each cascade."""

    config_schema = {}

    def inputs(self):
        return {"value": "float"}

    def outputs(self):
        return {"value": "float"}

    def update(self, state, interval=None):
        return {"value": float(self.parameters.get("rate", 1.0))}


class SubArrayUpdater(EcoliProcess):
    """Drives the two unique stores that carry per-row SUB-ARRAY fields —
    ``chromosome_domain`` (``child_domains: array[2,integer]``) and ``promoter``
    (``bound_TF: array[23,boolean]``) — across the boundary, emitting a combined
    ``set``/``add``/``delete``/``update`` unique update every tick. It READS each
    store's view (so a wrong sub-array decode diverges) and WRITES deterministic,
    interval-independent sub-array values keyed off the live active count (so both
    a wrong read and a wrong write diverge).

    A Process (not a Step): it shares the clock-locked timestep (``min(next -
    global_time)``), so both engines invoke it the same number of times — a Step
    that writes a store it reads would re-trigger a different number of times
    across the engines (see :class:`BulkProcess`)."""

    name = "subarray_updater"
    config_schema = {}

    def inputs(self):
        return {"chromosome_domain": "node", "promoter": "node",
                "global_time": "float", "next_update_time": "map[float]"}

    def outputs(self):
        # The real v2ecoli unique types (UniqueNumpyUpdate apply), so pb
        # accumulates + flushes set/add/delete identically to the Rust updater.
        from v2ecoli.library.schema_types import UNIQUE_TYPES
        return {"chromosome_domain": UNIQUE_TYPES["chromosome_domain"],
                "promoter": UNIQUE_TYPES["promoter"]}

    def calculate_timestep(self, interval_or_states, states=None):
        view = interval_or_states if states is None else states
        return min(nt - view["global_time"]
                   for nt in view["next_update_time"].values())

    def update(self, state, interval=None):
        out = {}
        for mol in ("chromosome_domain", "promoter"):
            view = state.get(mol)
            if view is None:
                continue
            act = int(np.asarray(view["_entryState"]).view(np.bool_).sum())
            upd = {"update": True}
            if mol == "chromosome_domain":
                # set child_domains per active molecule; add 2; delete the first.
                upd["set"] = {"child_domains": [[100 + i, 200 + i]
                                                for i in range(act)]}
                upd["add"] = {"domain_index": [7, 8],
                              "child_domains": [[1, 2], [3, 4]]}
            else:
                upd["set"] = {"bound_TF": [[(i + j) % 2 == 0 for j in range(23)]
                                           for i in range(act)]}
                upd["add"] = {"TU_index": [5, 6],
                              "bound_TF": [[j % 3 == 0 for j in range(23)],
                                           [j % 2 == 0 for j in range(23)]]}
            if act >= 1:
                upd["delete"] = [0]
            out[mol] = upd
        return out


# Module-level capture for the zero-copy test: every tick the process records
# facts about the `bulk` view it was handed (it runs in-process under both
# engines, so the test can read this list afterwards).
VIEW_LOG: list = []


class ViewProbeProcess(EcoliProcess):
    """Records, each tick, whether the bulk view it receives is the SAME memory
    as the previous tick's view (zero-copy) and whether it is read-only. Also
    checks the previous view still reflects the live store (a genuine view, not
    a snapshot). Writes the same delta as :class:`BenchBulkProcess`."""

    name = "view_probe_process"
    config_schema = {}
    _prev = None

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        view = state["bulk"]
        counts = np.asarray(view["count"])
        rec = {
            "names": view.dtype.names,
            "writeable": bool(view.flags.writeable),
            "counts": counts.copy(),
        }
        if ViewProbeProcess._prev is not None:
            prev = ViewProbeProcess._prev
            rec["shares_prev"] = bool(np.shares_memory(prev, view))
            rec["prev_live"] = bool(np.array_equal(prev["count"], counts))
        else:
            rec["shares_prev"] = None
            rec["prev_live"] = None
        ViewProbeProcess._prev = view
        VIEW_LOG.append(rec)
        idx = np.arange(counts.shape[0], dtype=np.int64)
        return {"bulk": [(idx, ((counts % 7) - 3).astype(np.int64))]}


# Module-level capture for the M2 Task-2 UNIQUE zero-copy tests.
UNIQUE_VIEW_LOG: list = []
UNIQUE_REALLOC_LOG: list = []

# Module-level capture for the M4 sub-array realloc-safety test.
SUBARRAY_VIEW_LOG: list = []


class SubArrayViewProbe(EcoliProcess):
    """Sub-array realloc-safety probe. Tick 0: STASH the view handed to it and a
    copy of its ``child_domains`` sub-array field, then emit a ``set`` (mutating
    existing rows' sub-array) plus an ``add`` that exhausts the free rows and
    forces the backing buffer to GROW on flush. Tick 1+: verify the STASHED
    (pre-mutation) view still reads its ORIGINAL sub-array bytes element-for-
    element — the copy-on-write (`Arc::make_mut`) left the pinned allocation
    untouched — while the LIVE view reflects the mutation."""

    name = "subarray_view_probe"
    config_schema = {}
    _stash = None
    _snapshot = None

    def inputs(self):
        return {"active": "node"}

    def outputs(self):
        return {"active": "node"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        view = state["active"]
        if SubArrayViewProbe._stash is None:
            SubArrayViewProbe._stash = view
            SubArrayViewProbe._snapshot = {
                "nrows": int(len(view)),
                "child_domains": view["child_domains"].copy(),
                "unique_index": view["unique_index"].copy(),
            }
            act = int(view["_entryState"].view(np.bool_).sum())
            n_add = int(self.parameters.get("n_add", 5))
            return {"active": {
                "set": {"child_domains": [[i, i] for i in range(act)]},
                "add": {"domain_index": list(range(n_add)),
                        "child_domains": [[900 + i, 800 + i] for i in range(n_add)]},
                "update": True,
            }}
        prev = SubArrayViewProbe._stash
        snap = SubArrayViewProbe._snapshot
        SUBARRAY_VIEW_LOG.append({
            "grew": int(len(view)) > snap["nrows"],
            "new_nrows": int(len(view)),
            "old_nrows": snap["nrows"],
            # The RETAINED pre-mutation view still reads its ORIGINAL sub-array
            # bytes (no use-after-free, no in-place overwrite of the pinned buf).
            "stash_child_domains_ok": bool(
                np.array_equal(prev["child_domains"], snap["child_domains"])),
            "stash_unique_index_ok": bool(
                np.array_equal(prev["unique_index"], snap["unique_index"])),
            # ... and the LIVE view reflects the set mutation on the old rows.
            "live_changed": not bool(np.array_equal(
                view["child_domains"][:snap["nrows"]], snap["child_domains"])),
        })
        return {}


class UniqueViewProbe(EcoliProcess):
    """Records, each tick, facts about the UNIQUE view it is handed: whether it
    is the SAME memory as the previous tick's (zero-copy, non-vacuous), whether
    it is read-only, its `dtype.names`, and its field reads (via the `attrs`
    `_entryState.view(bool)` idiom). It writes NO unique update, so the store's
    backing buffer is never mutated between ticks -> a genuine zero-copy view is
    literally the same memory tick-over-tick (a per-tick copy would not be)."""

    name = "unique_view_probe"
    config_schema = {}
    _prev = None

    def inputs(self):
        return {"active": "node"}

    def outputs(self):
        return {}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        view = state["active"]
        mask = view["_entryState"].view(np.bool_)
        rec = {
            "names": view.dtype.names,
            "writeable": bool(view.flags.writeable),
            "field_shares_base": bool(np.shares_memory(view["unique_index"], view)),
            "unique_index": view["unique_index"].tolist(),
            "active_data0": view["data0"][mask].tolist(),
        }
        if UniqueViewProbe._prev is not None:
            rec["shares_prev"] = bool(np.shares_memory(UniqueViewProbe._prev, view))
            rec["prev_live"] = bool(
                np.array_equal(UniqueViewProbe._prev["unique_index"],
                               view["unique_index"])
            )
        else:
            rec["shares_prev"] = None
            rec["prev_live"] = None
        UniqueViewProbe._prev = view
        UNIQUE_VIEW_LOG.append(rec)
        return {}


class UniqueReallocProbe(EcoliProcess):
    """Realloc-safety probe. Tick 0: STASH the view handed to it and a copy of
    its fields, then emit an `add` that exhausts the free rows and forces the
    backing buffer to GROW on flush (a realloc). Tick 1+: verify the STASHED
    (pre-grow) view is still valid and still equals the pre-grow snapshot — i.e.
    the grow did NOT free the memory the retained view points at (no
    use-after-free). A wrong pin would surface here as garbage / a crash."""

    name = "unique_realloc_probe"
    config_schema = {}
    _stash = None
    _snapshot = None

    def inputs(self):
        return {"active": "node"}

    def outputs(self):
        return {"active": "node"}

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        view = state["active"]
        if UniqueReallocProbe._stash is None:
            UniqueReallocProbe._stash = view
            UniqueReallocProbe._snapshot = {
                "nrows": int(len(view)),
                "unique_index": view["unique_index"].copy(),
                "data0": view["data0"].copy(),
                "active_sum": int(view["_entryState"].view(np.bool_).sum()),
            }
            n_add = int(self.parameters.get("n_add", 5))
            add = {"data0": [1000 + i for i in range(n_add)],
                   "data1": [0.0] * n_add}
            return {"active": {"add": add, "update": True}}
        prev = UniqueReallocProbe._stash
        snap = UniqueReallocProbe._snapshot
        UNIQUE_REALLOC_LOG.append({
            "grew": int(len(view)) > snap["nrows"],
            "new_nrows": int(len(view)),
            "old_nrows": snap["nrows"],
            "stash_len_ok": int(len(prev)) == snap["nrows"],
            "stash_unique_index_ok": bool(
                np.array_equal(prev["unique_index"], snap["unique_index"])),
            "stash_data0_ok": bool(np.array_equal(prev["data0"], snap["data0"])),
            "stash_active_sum": int(prev["_entryState"].view(np.bool_).sum()),
            "snap_active_sum": snap["active_sum"],
        })
        return {}


# --------------------------------------------------------------------------- #
# M5 unit bridge: a process that WRITES pint Quantities (exercising the
# py_to_value Quantity arm via overwrite ops) and READS them back across ticks
# (exercising the value_to_py Pint arm on an input view) — the write->read
# round-trip that the `effective_elongation_rate` strict read depends on.
# --------------------------------------------------------------------------- #

# Per-tick record of what QuantityCycler READ from its (previously-written)
# quantity stores: type, is-Quantity, magnitude dtype, and whether the strict
# `.to(<unit>)` read succeeded. Cleared by the test before each run.
QUANTITY_READ_LOG: list = []


class QuantityCycler(EcoliProcess):
    """Fixed-timestep process that each tick READS four quantity stores (as an
    input view) and WRITES fresh pint Quantities back to the SAME stores.

    * The writes are ``pint.Quantity`` values returned on ``overwrite[...]``
      ports -> the Rust ``op_from_type`` routes each to ``Op::Overwrite`` and
      calls ``py_to_value`` on the Quantity (the arm that, pre-M5, either
      infinite-recursed on an ``np.float64`` magnitude or dropped a Python-float
      magnitude to ``Null``).
    * Because inputs and outputs wire to the same stores, on tick ``t+1`` the
      input view is exactly what was written on tick ``t`` -> each read goes
      through ``value_to_py`` on a ``Value::Tagged(Pint, ..)`` leaf. This is the
      write->read-across-ticks path the strict ``effective_elongation_rate`` read
      (PolypeptideElongation -> PolypeptideInitiation) relies on.

    Magnitudes are fixed np.float64 values (plus one Python-float store) so the
    final state is deterministic and the ``np.float64`` vs ``float`` distinction
    is observable. The ``zero_mass`` store is zero so ``1 / q`` must give ``inf``
    (np.float64) rather than raising ``ZeroDivisionError`` (Python float).

    Output schema units: ``femtogram`` is a bare symbol and parses as
    ``quantity[float,femtogram]``; the non-bare units (``1 / second``,
    ``amino_acid / second``) are declared ``overwrite[float]`` since the Rust
    ``op_from_type`` ignores an Overwrite's inner type — the real unit travels on
    the Quantity value, not the schema.
    """

    name = "quantity_cycler"
    config_schema = {}

    # port -> the unit string the written Quantity carries (and the unit the
    # strict read converts to via `.to(...)`).
    UNITS = {
        "mass": "femtogram",
        "rate": "1 / second",
        "eff_rate": "amino_acid / second",
        "zero_mass": "femtogram",
        "py_mass": "femtogram",
    }

    def inputs(self):
        return {port: "float" for port in self.UNITS}

    def outputs(self):
        return {
            "mass": "overwrite[quantity[float,femtogram]]",
            "zero_mass": "overwrite[quantity[float,femtogram]]",
            "py_mass": "overwrite[quantity[float,femtogram]]",
            "rate": "overwrite[float]",
            "eff_rate": "overwrite[float]",
        }

    def calculate_timestep(self, interval_or_states, states=None):
        return float(self.parameters.get("dt", 1.0))

    def update(self, state, interval=None):
        import pint
        from bigraph_schema.units import get_quantity_registry

        ureg = get_quantity_registry()
        rec = {}
        for port, unit in self.UNITS.items():
            v = state.get(port)
            info = {"type": type(v).__name__,
                    "is_quantity": isinstance(v, pint.Quantity)}
            if info["is_quantity"]:
                info["mag_dtype"] = type(v.magnitude).__name__
                try:
                    info["mag"] = float(v.to(unit).magnitude)  # the strict read
                    info["to_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    info["to_ok"] = False
                    info["err"] = repr(exc)
            rec[port] = info
        QUANTITY_READ_LOG.append(rec)

        return {
            "mass": ureg.Quantity(np.float64(5.0), "femtogram"),
            "rate": ureg.Quantity(np.float64(2.0), "1 / second"),
            "eff_rate": ureg.Quantity(np.float64(1.2345), "amino_acid / second"),
            "zero_mass": ureg.Quantity(np.float64(0.0), "femtogram"),
            "py_mass": ureg.Quantity(3.0, "femtogram"),  # Python-float magnitude
        }
