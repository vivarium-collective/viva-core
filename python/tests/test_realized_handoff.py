"""M5 Task 3 — the realized-`Composite` handoff.

The raw ``baseline()`` agent leaves some edges unrealized (``shape_step`` has no
``instance`` and its address is a ``{'protocol','data'}`` dict), so ``build_payload``
raises. Handing it a REALIZED ``process_bigraph.Composite`` (every edge carries a
live instance) removes that blocker; combined with the unit bridge (Task 1) and
handle table (Task 2) the WHOLE baseline agent builds a payload.
"""

import json
import os

import pytest
from process_bigraph import Composite

from viva_core import prebuilt
from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core

CACHE = os.environ.get("V2ECOLI_CACHE", "/Users/eranagmon/code/v2ecoli/out/cache")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE, "sim_data_cache.dill")),
    reason=f"v2ecoli sim_data cache not found under {CACHE}",
)


@pytest.fixture(scope="module")
def built():
    core = build_core()
    set_current_core(core)
    doc = baseline(core, seed=0, cache_dir=CACHE, bundle=load_cache_bundle(CACHE))
    raw = doc["state"]["agents"]["0"]
    comp = Composite(doc, core=core)
    return raw, comp


def test_raw_agent_still_unrealized(built):
    raw, _ = built
    with pytest.raises(ValueError, match="no realized 'instance'"):
        prebuilt.build_payload(raw)


def test_realized_composite_builds_whole_agent_payload(built):
    _, comp = built
    # Composite accepted directly; the agent is addressed by path.
    processes, topo_json, state_json, handles = prebuilt.build_payload(
        comp, path=("agents", "0")
    )
    assert "shape_step" in processes and processes["shape_step"] is not None
    topo = json.loads(topo_json)
    assert topo["shape_step"]["address"] == "local:ShapeStep"  # dict address normalized
    # Task 1 (pint) + Task 2 (live stores) compose with the realized handoff.
    assert '"__pint__"' in state_json
    assert "allocator_rng" in handles and any(p.startswith("process/") for p in handles)
    # Same result from the realized sub-state dict.
    p2 = prebuilt.build_payload(comp.state["agents"]["0"])
    assert set(p2[0]) == set(processes)
