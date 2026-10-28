"""M5 Task 2 — the Python-owned handle table for the live `process` store +
`allocator_rng` (Option 1).

Two store classes hold LIVE Python objects the JSON state channel cannot carry:

  * ``["process", <name>]`` holds the 1-tuple ``(instance,)`` — the SAME live
    object (is-identity) a Requester mutates in place and an Evolver then reads
    (v2ecoli ``steps/partition.py``). They call methods on the whole instance and
    write ``(process,)`` back — a no-op overwrite of the same object.
  * ``["allocator_rng"]`` holds a live ``numpy.random.RandomState`` the Allocator
    draws from IN PLACE (v2ecoli ``steps/allocator.py``). A copy would diverge the
    RNG stream → not bit-identical.

The design (Option 1): ``build_payload`` splits these live-object stores into a
``handles`` map; the JSON state carries a null placeholder at those paths (so the
store exists and the Requester→Allocator→Evolver step ordering is unchanged); the
Rust engine holds the objects in a per-path handle table and hands the SAME object
back at the view boundary, never serializing it; a write targeting a handle path
is dropped (same object, already mutated in place).

This drives a small, sim_data-FREE slice — a ``GlobalClock`` (real, config-free)
plus the three handle steps (``viva_core.conformance_processes``) — under pb 1.8.3
``Composite`` (reference) and ``viva_core.run_prebuilt`` (candidate), on two
independent builds (so pb's run can't advance the candidate's live objects), and
asserts:

  (a) the observable state (``bulk`` counts + the ``draw_sum`` scalar) is
      bit-identical to pb;
  (b) the handle object handed to a view ``is`` the one passed in — the Requester
      and Evolver were handed the EXACT ``(instance,)`` object (incl. the
      ``allocator_rng`` RandomState), not a decoded copy;
  (c) a write to a handle path is a no-op — the live object survives the run with
      its in-place mutations accumulated (``counter`` == number of Requester
      calls), and the returned final state holds the same object;
  (d) the RNG stream is unchanged — the Allocator's per-tick draws match pb
      exactly (the shared mutable RandomState was never copied).
"""

import numpy as np
import pytest

import viva_core
from viva_core import conformance_processes as cp
from viva_core import prebuilt

from process_bigraph import Composite
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core
from v2ecoli.steps.global_clock import GlobalClock

SEED = 0
N_TICKS = 3
CLOCK_ADDR = "local:v2ecoli.steps.global_clock.GlobalClock"
REQ_ADDR = "local:viva_core.conformance_processes.HandleRequester"
ALLOC_ADDR = "local:viva_core.conformance_processes.HandleAllocator"
EVO_ADDR = "local:viva_core.conformance_processes.HandleEvolver"


def _build_slice(seed):
    """A realized composite slice with the two live-object stores. Returns
    ``(slice, obj, rng)`` so the test can check handle identity against the exact
    objects placed in the stores."""
    obj = cp.PartitionedHandle()
    rng = np.random.RandomState(seed)

    dt = np.dtype([("id", "<U4"), ("count", "int64")])
    bulk = np.zeros(3, dtype=dt)
    bulk["id"] = ["A", "B", "C"]
    bulk["count"] = [100, 200, 300]

    sl = {
        "global_time": 0.0,
        "next_update_time": {"clock": 1.0},
        "request": 0,
        "allocate": 0,
        "draw_sum": 0,
        "bulk": bulk,
        # The two live-object stores:
        "process": {"p0": (obj,)},
        "allocator_rng": rng,
        "global_clock": {
            "_type": "process",
            "address": CLOCK_ADDR,
            "config": {},
            "inputs": {"global_time": ["global_time"],
                       "next_update_time": ["next_update_time"]},
            "outputs": {"global_time": ["global_time"]},
            "instance": GlobalClock({}),
            "interval": 1.0,
        },
        "requester": {
            "_type": "step",
            "address": REQ_ADDR,
            "config": {},
            "inputs": {"process": ["process", "p0"],
                       "global_time": ["global_time"]},
            "outputs": {"request": ["request"],
                        "process": ["process", "p0"],
                        "next_update_time": ["next_update_time", "clock"]},
            "instance": cp.HandleRequester({}),
        },
        "allocator": {
            "_type": "step",
            "address": ALLOC_ADDR,
            "config": {},
            "inputs": {"request": ["request"], "bulk": ["bulk"],
                       "allocator_rng": ["allocator_rng"]},
            "outputs": {"allocate": ["allocate"], "bulk": ["bulk"],
                        "draw_sum": ["draw_sum"]},
            "instance": cp.HandleAllocator({}),
        },
        "evolver": {
            "_type": "step",
            "address": EVO_ADDR,
            "config": {},
            "inputs": {"process": ["process", "p0"],
                       "allocate": ["allocate"], "bulk": ["bulk"]},
            "outputs": {"bulk": ["bulk"], "process": ["process", "p0"]},
            "instance": cp.HandleEvolver({}),
        },
    }
    return sl, obj, rng


def _observable(state):
    """The bit-identity-comparable observable: bulk counts + the draw_sum."""
    bulk = state["bulk"]
    count = bulk["count"]
    return {
        "bulk_count": list(count.tolist() if hasattr(count, "tolist") else count),
        "draw_sum": int(state["draw_sum"]),
    }


@pytest.fixture(scope="module")
def results():
    core = build_core()
    set_current_core(core)

    # Reference: pb Composite on one build. pb's `run(interval)` is duration-based
    # while the Rust engine's `run(n_ticks)` counts scheduler iterations; with this
    # clock (global_time advances 1.0/iteration) the two coincide exactly and
    # bit-identically at `run(n_ticks + 1.0)` (an iteration-vs-duration unit
    # offset, deterministic — verified across N).
    cp.HANDLE_LOG.clear()
    ref_slice, ref_obj, ref_rng = _build_slice(SEED)
    ref_sim = Composite({"state": ref_slice}, core=core)
    ref_sim.run(float(N_TICKS + 1))
    ref_log = list(cp.HANDLE_LOG)
    ref_obs = _observable(ref_sim.state)

    # Candidate: the Python->Rust handoff on a second, identical build. Fresh
    # objects (same seed) so pb's run above did not advance them.
    cp.HANDLE_LOG.clear()
    got_slice, got_obj, got_rng = _build_slice(SEED)
    got_state = prebuilt.run_from_state(got_slice, n_ticks=N_TICKS)
    got_log = list(cp.HANDLE_LOG)
    got_obs = _observable(got_state)

    return {
        "ref_log": ref_log, "ref_obs": ref_obs,
        "got_log": got_log, "got_obs": got_obs,
        "got_state": got_state,
        "got_obj": got_obj, "got_rng": got_rng,
        "got_slice": got_slice,
    }


def _draws(log):
    return [e["draw"] for e in log if e["step"] == "allocator"]


def _counters(log, step):
    return [e["counter"] for e in log if e["step"] == step]


# (a) observable state bit-identical to pb --------------------------------------

def test_handle_slice_bit_identical(results):
    assert results["got_log"], "handle steps never ran"
    # Same number of cascades under both engines (scheduling preserved).
    assert len(results["got_log"]) == len(results["ref_log"]) > 3
    assert results["got_obs"] == results["ref_obs"]
    # Non-vacuous: the RNG/handle-driven bulk actually changed from the start.
    assert results["got_obs"]["bulk_count"] != [100, 200, 300]


# (b) is-identity: the view got the EXACT object passed in -----------------------

def test_handle_is_identity(results):
    got = results["got_slice"]
    want_obj_id = id(got["process"]["p0"][0])
    want_rng_id = id(got["allocator_rng"])

    req_ids = {e["obj_id"] for e in results["got_log"] if e["step"] == "requester"}
    evo_ids = {e["obj_id"] for e in results["got_log"] if e["step"] == "evolver"}
    rng_ids = {e["rng_id"] for e in results["got_log"] if e["step"] == "allocator"}

    # Requester and Evolver were both handed the SAME, passed-in live object.
    assert req_ids == {want_obj_id}, f"requester obj ids {req_ids} != {want_obj_id}"
    assert evo_ids == {want_obj_id}, f"evolver obj ids {evo_ids} != {want_obj_id}"
    # The allocator was handed the passed-in RandomState itself.
    assert rng_ids == {want_rng_id}, f"allocator rng ids {rng_ids} != {want_rng_id}"
    # It was carried as the (instance,) tuple so `proc_state[0]` works.
    assert all(e["is_tuple"] for e in results["got_log"]
               if e["step"] in ("requester", "evolver"))


# (c) a write to a handle path is a no-op (same object, in-place mutation kept) ---

def test_handle_write_is_noop(results):
    got = results["got_slice"]
    obj = got["process"]["p0"][0]
    n_req = len(_counters(results["got_log"], "requester"))
    # The live object survived the whole run with its in-place increments
    # accumulated — it was never replaced by a decoded `[None]`/placeholder.
    assert obj.counter == n_req > 0, f"counter {obj.counter} != requester calls {n_req}"
    assert obj.request_set is True
    # The returned final state holds the SAME live objects (state_to_py re-inserts
    # the handle), not a serialized copy.
    assert results["got_state"]["process"]["p0"][0] is obj
    assert results["got_state"]["allocator_rng"] is got["allocator_rng"]
    # The shared counter the Evolver read equals what the Requester set that tick.
    assert _counters(results["got_log"], "evolver") == \
        _counters(results["got_log"], "requester")


# (d) RNG stream unchanged: allocator draws match pb exactly ---------------------

def test_handle_rng_stream_matches_pb(results):
    got_draws = _draws(results["got_log"])
    ref_draws = _draws(results["ref_log"])
    assert got_draws, "allocator never drew"
    assert got_draws == ref_draws, "RNG stream diverged from pb (copy, not shared?)"
