"""Quadrant-3 parity: the Rust `viva_core` engine runs a REAL v2ecoli `.pbg`
with Python processes, bit-identical to the Python process-bigraph 1.8.3 engine.

For each composite we (a) build it in Python with REAL v2ecoli process classes +
real numpy state, run it under `process_bigraph.Composite`; (b) serialize the
SAME initial state to a `.pbg`-shaped spec (bulk/unique as `__structured_array__`
tagged dicts, configs inline) and run it under `viva_core.run`; then assert the
decoded states are bit-identical (ints exact, floats at epsilon) and the
per-tick gating / emitter row counts match.

Real v2ecoli classes exercised (none need ParCa sim_data):
  * GlobalClock  — process; variable timestep via calculate_timestep; additive
    global_time write.  (scalar + map ports)
  * UniqueUpdate — step; reads the unique stores through the boundary and flushes
    the per-store updater.  (unique structured-array decode + boundary)
  * ReplicationData — GATED step (update_condition: global_time % timestep == 0);
    reads unique stores via `attrs`, writes an overwrite tree to listeners.
  * RAMEmitter   — native Rust emitter (row-count parity).
"""

import json

import numpy as np
import pytest

import viva_core

from process_bigraph import Composite
from process_bigraph.emitter import gather_emitter_results
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core
from v2ecoli.library.schema import MetadataArray
from v2ecoli.steps.derivers.replication_data import ReplicationData

EPS = 1e-12

_SUBMASS = [
    (f"massDiff_{x}", "float64")
    for x in [
        "rRNA", "tRNA", "mRNA", "miscRNA", "nonspecific_RNA",
        "protein", "metabolite", "water", "DNA",
    ]
]
# Real v2ecoli unique arrays carry int8 _entryState (attrs views it as bool).
_TAIL = _SUBMASS + [("_entryState", "int8"), ("unique_index", "int64")]


def _mk_unique(head, rows):
    """Build a real MetadataArray unique store (metadata = max unique_index + 1)."""
    dt = np.dtype(head + _TAIL)
    arr = np.zeros(len(rows), dtype=dt)
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[k][i] = v
        arr["_entryState"][i] = 1
    meta = int(arr["unique_index"].max()) + 1 if len(rows) else 0
    return MetadataArray(arr, metadata=meta)


def _unique_stores():
    oriC = _mk_unique(
        [("domain_index", "int32")],
        [{"domain_index": 1, "unique_index": 10},
         {"domain_index": 2, "unique_index": 11}],
    )
    dnaa = _mk_unique(
        [("coordinates", "int64"), ("domain_index", "int32"),
         ("DnaA_bound", "bool"), ("pool_label", "int64"),
         ("DnaA_bound_form", "int64")],
        [{"coordinates": 100, "domain_index": 0, "DnaA_bound": True,
          "pool_label": 0, "DnaA_bound_form": 1, "unique_index": 20},
         {"coordinates": 200, "domain_index": 0, "DnaA_bound": False,
          "pool_label": 1, "DnaA_bound_form": 0, "unique_index": 21},
         {"coordinates": 300, "domain_index": 1, "DnaA_bound": True,
          "pool_label": 2, "DnaA_bound_form": 1, "unique_index": 22}],
    )
    repl = _mk_unique(
        [("domain_index", "int32"), ("right_replichore", "bool"),
         ("coordinates", "int64")],
        [],
    )
    return {"oriC": oriC, "DnaA_box": dnaa, "active_replisome": repl}


def _replication_defaults():
    """Pre-seed listeners.replication_data leaves from the output schema defaults
    (both engines start with identical leaves; the overwrite tree then replaces
    them on each tick ReplicationData runs)."""
    rd = ReplicationData({}).outputs()["listeners"]["replication_data"]
    out = {}
    for field, spec in rd.items():
        out[field] = spec.get("_default", 0)
    return out


def _build_doc(timestep):
    topo = {"oriC": "", "DnaA_box": "", "active_replisome": ""}
    return {
        "global_time": 0.0,
        "timestep": float(timestep),
        "next_update_time": {"p": 3.0},
        "unique": _unique_stores(),
        "listeners": {"replication_data": _replication_defaults()},
        "global_clock": {
            "_type": "process",
            "address": "local:v2ecoli.steps.global_clock.GlobalClock",
            "config": {},
            "inputs": {"global_time": ["global_time"],
                       "next_update_time": ["next_update_time"]},
            "outputs": {"global_time": ["global_time"]},
        },
        "replication_data_listener": {
            "_type": "step",
            "address": "local:v2ecoli.steps.derivers.replication_data.ReplicationData",
            "config": {},
            "inputs": {"oriCs": ["unique", "oriC"],
                       "DnaA_boxes": ["unique", "DnaA_box"],
                       "active_replisomes": ["unique", "active_replisome"],
                       "global_time": ["global_time"], "timestep": ["timestep"]},
            "outputs": {"listeners": ["listeners"]},
        },
        "unique_update": {
            "_type": "step",
            "address": "local:v2ecoli.steps.unique_update.UniqueUpdate",
            "config": {"unique_topo": topo},
            "inputs": {"oriC": ["unique", "oriC"],
                       "DnaA_box": ["unique", "DnaA_box"],
                       "active_replisome": ["unique", "active_replisome"]},
            "outputs": {"oriC": ["unique", "oriC"],
                        "DnaA_box": ["unique", "DnaA_box"],
                        "active_replisome": ["unique", "active_replisome"]},
        },
        "emitter": {
            "_type": "step",
            "address": "local:process_bigraph.emitter.RAMEmitter",
            "config": {"emit": {"global_time": "float"}},
            "inputs": {"global_time": ["global_time"]},
            "outputs": {},
        },
    }


# ---- serialize the Python doc to a .pbg-shaped spec for viva_core -----------

def _serialize(value):
    if isinstance(value, MetadataArray) or (hasattr(value, "dtype") and getattr(value.dtype, "names", None)):
        return {
            "__structured_array__": True,
            "dtype": [[name, str(value.dtype[name])] for name in value.dtype.names],
            "columns": {name: value[name].tolist() for name in value.dtype.names},
            "length": int(len(value)),
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _spec_json(doc):
    return json.dumps({k: _serialize(v) for k, v in doc.items()})


# ---- bit-identical comparison ----------------------------------------------

def _norm(x):
    """Normalize a value (numpy -> python) for comparison."""
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _assert_equal(ref, got, path=""):
    ref, got = _norm(ref), _norm(got)
    if isinstance(ref, dict):
        assert isinstance(got, dict), f"{path}: type mismatch {type(got)}"
        for k, v in ref.items():
            assert k in got, f"{path}/{k}: missing in viva_core output"
            _assert_equal(v, got[k], f"{path}/{k}")
        return
    if isinstance(ref, (list, tuple)):
        assert isinstance(got, (list, tuple)), f"{path}: {ref!r} vs {got!r}"
        assert len(ref) == len(got), f"{path}: length {len(ref)} vs {len(got)}"
        for i, (a, b) in enumerate(zip(ref, got)):
            _assert_equal(a, b, f"{path}[{i}]")
        return
    if isinstance(ref, bool) or isinstance(got, bool):
        assert bool(ref) == bool(got), f"{path}: {ref!r} vs {got!r}"
        return
    if isinstance(ref, int) and isinstance(got, int):
        assert ref == got, f"{path}: int {ref} vs {got}"
        return
    if isinstance(ref, (int, float)) and isinstance(got, (int, float)):
        assert abs(float(ref) - float(got)) <= EPS, f"{path}: float {ref} vs {got}"
        return
    assert ref == got, f"{path}: {ref!r} vs {got!r}"


def _run_reference(doc):
    """Run the pb Composite reference; return (final_state, ran_log, emitter_rows)."""
    core = build_core()
    set_current_core(core)

    ran = []
    orig = ReplicationData.update_condition
    def patched(self, ts, st):
        r = orig(self, ts, st)
        ran.append(("replication_data_listener", float(st.get("global_time")), bool(r)))
        return r
    ReplicationData.update_condition = patched
    try:
        sim = Composite({"state": doc}, core=core)
        sim.run(6.0)  # one scheduler iteration (global_time 0 -> 6)
    finally:
        ReplicationData.update_condition = orig

    emitter_rows = sum(len(v) for v in gather_emitter_results(sim).values())
    return sim.state, ran, emitter_rows


def _project_state(state):
    """Pull the comparable fields out of a (pb or viva_core) state mapping."""
    out = {
        "global_time": float(state["global_time"]),
        "timestep": float(state["timestep"]),
        "next_update_time": {k: float(v) for k, v in state["next_update_time"].items()},
        "unique": {},
        "listeners": {k: _norm(v) for k, v in state["listeners"]["replication_data"].items()},
    }
    for mol, store in state["unique"].items():
        cols = {}
        if hasattr(store, "dtype") and store.dtype.names:  # pb MetadataArray
            for f in store.dtype.names:
                cols[f] = store[f].tolist()
            cols["_metadata"] = int(store.metadata)
        else:  # viva_core dict-of-columns
            for f, v in store.items():
                if f == "_metadata":
                    cols[f] = int(v)
                else:
                    cols[f] = _norm(v)
        out["unique"][mol] = cols
    return out


@pytest.fixture(scope="module")
def results():
    doc = _build_doc(timestep=4.0)
    ref_state, ref_ran, ref_rows = _run_reference(doc)
    spec = _spec_json(doc)
    report = viva_core.run_report(spec, 1)
    return {
        "ref_state": _project_state(ref_state),
        "got_state": _project_state(report["state"]),
        "ref_ran": ref_ran,
        "got_runs": report["step_runs"],
        "ref_rows": ref_rows,
        "got_emitters": report["emitter_counts"],
    }


def test_quadrant3_bit_identical(results):
    """Decoded final state is bit-identical (ints exact, floats at epsilon)."""
    _assert_equal(results["ref_state"], results["got_state"])
    # Sanity: the run actually did the real thing.
    assert results["got_state"]["global_time"] == 6.0
    assert results["got_state"]["listeners"]["number_of_oric"] == 2
    assert results["got_state"]["listeners"]["total_DnaA_boxes"] == 3


def test_unique_store_metadata_and_boundary(results):
    """Unique structured-array decode + metadata init (= max unique_index + 1)
    survive the boundary bit-identically."""
    ru = results["ref_state"]["unique"]
    gu = results["got_state"]["unique"]
    assert gu["oriC"]["_metadata"] == ru["oriC"]["_metadata"] == 12
    assert gu["DnaA_box"]["_metadata"] == ru["DnaA_box"]["_metadata"] == 23
    _assert_equal(ru["oriC"], gu["oriC"], "unique/oriC")
    _assert_equal(ru["DnaA_box"], gu["DnaA_box"], "unique/DnaA_box")


def test_differential_gating(results):
    """The SET of (step, global_time) at which each step RAN matches pb's — a
    size-equal state can't hide a gating mismatch — and emitter rows match."""
    # pb ran ReplicationData at the global_times where its gate was True.
    ref_on = {gt for (_name, gt, ran) in results["ref_ran"] if ran}
    got_on = {gt for (name, gt, ran) in results["got_runs"]
              if name == "replication_data_listener" and ran}
    assert got_on == ref_on, f"ReplicationData ran at {got_on}, pb at {ref_on}"
    # With timestep=4: ON at global_time 0 (0%4==0), OFF at 6 (6%4==2).
    assert got_on == {0.0}
    got_off = {gt for (name, gt, ran) in results["got_runs"]
               if name == "replication_data_listener" and not ran}
    assert 6.0 in got_off, "ReplicationData should be gated OFF at global_time 6"
    # UniqueUpdate is never gated: it runs at both cascades (t=0 and t=6).
    uu_on = {gt for (name, gt, ran) in results["got_runs"]
             if name == "unique_update" and ran}
    assert uu_on == {0.0, 6.0}
    # Emitter row counts match.
    assert results["got_emitters"]["emitter"] == results["ref_rows"] == 2
