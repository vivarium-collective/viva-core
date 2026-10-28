"""M4 Task 3 — whole-model conformance on the REAL v2ecoli baseline.

This is the honest M4 gate: how much of the *real* baseline composite (built by
``ecoli_baseline.baseline`` with on-disk sim_data, one generation, baseline media,
single seed) runs on the Rust engine via ``viva_core.run_prebuilt`` bit-identical
to process-bigraph 1.8.3 — and exactly where it does not.

What runs bit-identical (and is gated here):
  * the real, full-scale ``ProteinDegradation`` process (16321-wide real bulk,
    real sim_data degradation rates + per-process seeded RNG), driven for N ticks,
    under run_prebuilt native-swap OFF (Python instance on the Rust engine) AND ON
    (the M3 native Rust twin, config sourced from the live instance).

What does NOT yet run (recorded precisely, not faked):
  * the *whole* 46-edge agent cannot be handed over through the JSON state channel.
    Two hard blockers, asserted below so a future fix trips this test:
      (1) the partition "process-as-data" store holds LIVE process instances
          (the 3 PartitionedProcesses' Requester/Evolver read them from state) —
          live Python objects can't round-trip through the JSON state codec;
      (2) pint/Unum ``Quantity`` values live in the ``listeners`` (29/46 edges
          touch it) and ``environment`` (~11/46) stores — the Rust codec has no
          unit type and the view/update boundary can't re-wrap units.
    These are the M4 forward unknowns; they gate the whole-model engine number.

See docs/milestone-4-results.md for the measured numbers and the honest scope.
"""

import copy
import json
import os

import numpy as np
import pytest

import viva_core
from viva_core import prebuilt
from viva_core.prebuilt import _serialize

from process_bigraph import Composite
from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core
from viva_core.conformance_processes import TickTrigger

PD_ADDR = "local:v2ecoli.processes.protein_degradation.ProteinDegradation"
CACHE = os.environ.get("V2ECOLI_CACHE", "/Users/eranagmon/code/v2ecoli/out/cache")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE, "sim_data_cache.dill")),
    reason=f"v2ecoli sim_data cache not found under {CACHE}",
)


def _strip(w):
    if isinstance(w, dict):
        return {k: _strip(v) for k, v in w.items() if not str(k).startswith("_layer")}
    return w


@pytest.fixture(scope="module")
def ctx():
    core = build_core()
    set_current_core(core)
    bundle = load_cache_bundle(CACHE)
    return core, bundle


def _agent(core, bundle, seed=0):
    return baseline(core, seed=seed, cache_dir=CACHE, bundle=bundle)["state"]["agents"]["0"]


def _pd_doc(core, bundle, seed=0):
    """A fixed-timestep driver + the REAL pre-built ProteinDegradation instance
    (full-scale bulk + live sim_data). TickTrigger advances global_time and
    re-writes ``timestep`` each tick so the PD step fires every scheduler iteration."""
    agent = _agent(core, bundle, seed)
    pd = dict(agent["ecoli-protein-degradation"])
    pd["inputs"] = _strip(pd["inputs"])
    pd["outputs"] = _strip(pd["outputs"])
    tick = TickTrigger(config={"dt": 1.0, "value": 1}, core=core)
    return {
        "global_time": 0.0,
        "timestep": 1,
        "bulk": agent["bulk"],
        "tick": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.TickTrigger",
            "config": {"dt": 1.0, "value": 1},
            "instance": tick,
            "inputs": {},
            "outputs": {"timestep": ["timestep"]},
        },
        "ecoli-protein-degradation": pd,
    }


def _bulk(state):
    b = state["bulk"]
    if hasattr(b, "dtype") and getattr(b.dtype, "names", None):
        return np.asarray(b["count"]).tolist()
    return list(b["count"])


# --------------------------------------------------------------------------- #
# What RUNS: the real, full-scale ProteinDegradation, bit-identical to pb.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pd_results(ctx):
    core, bundle = ctx
    n = 15
    from process_bigraph.emitter import gather_emitter_results  # noqa: F401
    ref_sim = Composite({"state": copy.deepcopy(_pd_doc(core, bundle))}, core=core)
    ref_sim.run(float(n))
    ref = _bulk(ref_sim.state)
    init = _bulk(_pd_doc(core, bundle))

    off = _bulk(prebuilt.run_from_state(_pd_doc(core, bundle), n_ticks=n))
    on = _bulk(_run_native(_pd_doc(core, bundle), n))
    return {"ref": ref, "init": init, "off": off, "on": on, "n": n}


def _run_native(doc, n):
    os.environ["VIVA_NATIVE_PROCESSES"] = "ProteinDegradation"
    try:
        return prebuilt.run_from_state(doc, n_ticks=n, native_addresses=[PD_ADDR])
    finally:
        os.environ.pop("VIVA_NATIVE_PROCESSES", None)


def test_real_pd_actually_degrades(pd_results):
    """Non-vacuous: the real sim_data + RNG drove hundreds of bulk counts down."""
    ref, init = np.array(pd_results["ref"]), np.array(pd_results["init"])
    changed = int(np.count_nonzero(ref - init))
    assert changed > 100, f"expected real degradation, only {changed} bulk entries changed"


def test_native_off_bit_identical(pd_results):
    """Python ProteinDegradation on the Rust engine == pb, bit-identical, N ticks."""
    assert pd_results["off"] == pd_results["ref"]


def test_native_on_bit_identical(pd_results):
    """The M3 native Rust ProteinDegradation (config sourced from the live
    instance) == pb, bit-identical — RNG-exact across N ticks."""
    assert pd_results["on"] == pd_results["ref"]


def test_native_path_is_actually_native(ctx):
    """Native-ON must not call back into the Python process."""
    core, bundle = ctx
    from v2ecoli.processes.protein_degradation import ProteinDegradation as PD

    def _poison(self, *a, **k):
        raise AssertionError("Python ProteinDegradation.update called on native path")

    orig = PD.update
    PD.update = _poison
    try:
        got = _bulk(_run_native(_pd_doc(core, bundle), 5))
    finally:
        PD.update = orig
    ref_sim = Composite({"state": copy.deepcopy(_pd_doc(core, bundle))}, core=core)
    ref_sim.run(5.0)
    assert got == _bulk(ref_sim.state)


# --------------------------------------------------------------------------- #
# What does NOT run yet: the whole agent through the JSON handoff. Record the
# precise blockers so a future fix (unit bridge / live-store objects) trips this.
# --------------------------------------------------------------------------- #

def test_wholemodel_handoff_blocked_process_store(ctx):
    """BLOCKER 1: the partition ``process`` store holds live process instances
    (processes-as-data); they cannot be JSON-serialized into the state channel."""
    core, bundle = ctx
    agent = _agent(core, bundle)
    assert "process" in agent and isinstance(agent["process"], dict) and agent["process"], \
        "expected a populated partition 'process' store"
    with pytest.raises(TypeError):
        json.dumps(_serialize(agent["process"]))


def test_wholemodel_handoff_pint_units_resolved(ctx):
    """BLOCKER 2 — RESOLVED by M5 Task 1 (the pint unit bridge, Option A).

    pint Quantities in ``listeners``/``environment`` now lower to the ``__pint__``
    tagged JSON the Rust codec decodes (and rebuild as real Quantities on the
    view / return boundary). Both stores serialize cleanly and the lowered blob
    carries ``__pint__`` tags — the Quantities were lowered, not dropped. (Was
    previously asserted to FAIL as the M4 unit blocker; this test now trips if the
    bridge regresses.)"""
    core, bundle = ctx
    agent = _agent(core, bundle)
    for store in ("listeners", "environment"):
        blob = json.dumps(_serialize(agent[store]))  # no longer raises
        assert '"__pint__"' in blob, f"{store}: expected lowered pint tags, got none"


def test_wholemodel_full_agent_not_serializable(ctx):
    """The whole baseline agent still cannot be handed to run_prebuilt as one
    payload — the honest whole-model gate. With M5 Task 1 the pint-unit blocker is
    gone, so it now refuses for one of the two STRUCTURAL reasons that remain: the
    as-built agent isn't fully pre-realized (not every edge carries a live
    ``instance``, e.g. ``shape_step`` -> ValueError), or the live process-as-data
    store can't JSON-serialize (TypeError). Removing this failure needs
    pre-realizing the agent AND the live-store-object passing (M5 Tasks 2-3), per
    docs/milestone-4-results.md."""
    core, bundle = ctx
    agent = _agent(core, bundle)
    with pytest.raises((ValueError, TypeError)):
        prebuilt.build_payload(agent)
