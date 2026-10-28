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
