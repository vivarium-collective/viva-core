"""M4 Task 1 — `run_prebuilt` + the instance-bound PyProcess shim, on a REAL
v2ecoli baseline slice built with live `sim_data`.

The baseline's processes carry live, non-JSON-serializable `sim_data` handles
baked in at construction (config stays `{}` — `viva_core.run`'s JSON path can
never rebuild them). This test proves the Python->Rust handoff: v2ecoli's own
builder constructs the real instances, and the Rust engine schedules those
*pre-built* instances identically to pb 1.8.3.

Slice (all built by `ecoli_baseline.baseline` with real sim_data):
  * GlobalClock            — real process, variable timestep (calculate_timestep),
    additive `global_time` write.
  * ProteinDegradation     — real STEP carrying live sim_data (degradation rates)
    + a seeded `random_state`; reads/writes `bulk` (the sim_data/RNG-driven proof
    — it mutates ~100+ bulk counts per tick). This instance CANNOT be rebuilt
    from its (empty) JSON config.
  * ReplicationData        — real GATED deriver step (`update_condition`:
    global_time % timestep); reads the unique stores via `attrs`, writes the
    replication_data listener tree.
  * RAMEmitter             — native Rust emitter (handed over as an edge).

Reference = pb `Composite`; candidate = `viva_core.run_prebuilt` (no native
swap). Two independent `baseline(seed=0)` builds give two identical instance sets
(so pb's run can't advance the instances the handoff then runs). Decoded states
must be bit-identical (ints exact, floats within 1e-12).
"""

import os

import numpy as np
import pytest

import viva_core
from viva_core import prebuilt

from process_bigraph import Composite
from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core

EPS = 1e-12
CACHE = os.environ.get(
    "V2ECOLI_CACHE", "/Users/eranagmon/code/v2ecoli/out/cache"
)
# Simple unique stores only — the Rust zero-copy unique decode does not yet
# handle sub-array fields (`promoter.bound_TF`, `chromosome_domain.child_domains`),
# a Task-3 scale-up concern; these stores have none.
SIMPLE_UNIQUE = ["oriC", "DnaA_box", "active_replisome", "full_chromosome"]
EDGES = ["global_clock", "ecoli-protein-degradation", "replication_data_listener"]
STORES = ["global_time", "timestep", "bulk"]


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE, "sim_data_cache.dill")),
    reason=f"v2ecoli sim_data cache not found under {CACHE}",
)


def _strip_layers(w):
    if isinstance(w, dict):
        return {k: _strip_layers(v) for k, v in w.items() if not str(k).startswith("_layer")}
    return w


def _build_slice(seed, bundle, core):
    """A small, self-consistent, serializable slice of the REAL baseline agent."""
    agent = baseline(core, seed=seed, cache_dir=CACHE, bundle=bundle)["state"]["agents"]["0"]
    sl = {s: agent[s] for s in STORES}
    sl["global_time"] = 0.0
    # One scheduled event so GlobalClock advances exactly once (no process in the
    # slice reschedules next_update_time). pb run(1.0) == one scheduler iteration.
    sl["next_update_time"] = {"clock": 1.0}
    sl["unique"] = {k: agent["unique"][k] for k in SIMPLE_UNIQUE if k in agent["unique"]}
    # Seed listeners with ONLY the replication_data defaults (from the real
    # instance's output schema) — the baseline listeners store carries pint
    # Quantities in other subtrees (mass/atp) the slice does not touch, and
    # ReplicationData creates/overwrites replication_data each tick anyway.
    rd_schema = agent["replication_data_listener"]["instance"].outputs()["listeners"]["replication_data"]
    sl["listeners"] = {
        "replication_data": {f: spec.get("_default", 0) for f, spec in rd_schema.items()}
    }
    for name in EDGES:
        edge = dict(agent[name])
        edge["inputs"] = _strip_layers(edge.get("inputs", {}))
        edge["outputs"] = _strip_layers(edge.get("outputs", {}))
        sl[name] = edge
    sl["emitter"] = {
        "_type": "step",
        "address": "local:process_bigraph.emitter.RAMEmitter",
        "config": {"emit": {"global_time": "float"}},
        "inputs": {"global_time": ["global_time"]},
        "outputs": {},
    }
    return sl


# ---- comparison -------------------------------------------------------------

def _norm(x):
    return x.tolist() if hasattr(x, "tolist") else x


def _assert_equal(ref, got, path=""):
    ref, got = _norm(ref), _norm(got)
    if isinstance(ref, dict):
        assert isinstance(got, dict), f"{path}: type mismatch {type(got)}"
        for k, v in ref.items():
            assert k in got, f"{path}/{k}: missing in run_prebuilt output"
            _assert_equal(v, got[k], f"{path}/{k}")
        return
    if isinstance(ref, (list, tuple)):
        assert isinstance(got, (list, tuple)), f"{path}: {type(ref)} vs {type(got)}"
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


def _project(state):
    """Comparable fields from a pb or run_prebuilt state mapping."""
    out = {
        "global_time": float(state["global_time"]),
        "timestep": float(state["timestep"]),
        "bulk_id": list(_norm(state["bulk"]["id"])),
        "bulk_count": list(_norm(state["bulk"]["count"])),
        "unique": {},
        "replication_data": {k: _norm(v) for k, v in state["listeners"]["replication_data"].items()},
    }
    for mol, store in state["unique"].items():
        cols = {}
        if hasattr(store, "dtype") and store.dtype.names:  # pb MetadataArray
            for f in store.dtype.names:
                cols[f] = store[f].tolist()
            cols["_metadata"] = int(store.metadata)
        else:  # run_prebuilt dict-of-columns
            for f, v in store.items():
                cols[f] = int(v) if f == "_metadata" else _norm(v)
        out["unique"][mol] = cols
    return out


@pytest.fixture(scope="module")
def results():
    core = build_core()
    set_current_core(core)
    bundle = load_cache_bundle(CACHE)

    # Reference: pb Composite on an independent build.
    ref_slice = _build_slice(0, bundle, core)
    init_bulk = np.array(ref_slice["bulk"]["count"], copy=True)
    ref_sim = Composite({"state": ref_slice}, core=core)
    ref_sim.run(1.0)  # one scheduler iteration (global_time 0 -> 2.0)

    # Candidate: the Python->Rust handoff on a second, identical build. Fresh
    # instances (seed 0) so pb's run above did not advance them.
    got_slice = _build_slice(0, bundle, core)
    got_state = prebuilt.run_from_state(got_slice, n_ticks=1)

    return {
        "ref": _project(ref_sim.state),
        "got": _project(got_state),
        "init_bulk": init_bulk,
    }


def test_run_prebuilt_bit_identical_small(results):
    """Decoded state after the handoff is bit-identical to pb on the real slice."""
    _assert_equal(results["ref"], results["got"])
    # Clock advanced identically (GlobalClock additive write + scheduler step).
    assert results["got"]["global_time"] == results["ref"]["global_time"] == 2.0


def test_protein_degradation_actually_ran(results):
    """The ProteinDegradation instance (live sim_data + seeded RNG) truly drove
    bulk through the shim — the test is not vacuously comparing unchanged state."""
    final = np.array(results["got"]["bulk_count"])
    changed = int(np.count_nonzero(final - results["init_bulk"]))
    assert changed > 50, f"expected sim_data/RNG-driven bulk change, got {changed}"
    # And that change is exactly what pb produced.
    assert results["got"]["bulk_count"] == results["ref"]["bulk_count"]


def test_gated_step_and_unique_identical(results):
    """The GATED ReplicationData deriver ran identically (its replication_data
    listener output matches pb), and the unique-store decode round-trips exactly
    (metadata = max unique_index + 1)."""
    _assert_equal(results["ref"]["replication_data"], results["got"]["replication_data"],
                  "replication_data")
    assert results["got"]["replication_data"]["number_of_oric"] == \
        results["ref"]["replication_data"]["number_of_oric"]
    for mol in SIMPLE_UNIQUE:
        _assert_equal(results["ref"]["unique"][mol], results["got"]["unique"][mol], f"unique/{mol}")
