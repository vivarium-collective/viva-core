"""Shared composite builders for the conformance suite + the upstream battery.

Centralized so the test and `upstream_suite` drive the SAME real v2ecoli
composites. All composites are run at `(n_ticks=1, until=6.0)`, where the Rust
engine's scheduler and pb's `_run_inner` invoke every process/step the same
number of times (the GlobalClock-locked schedule), so the decoded state is
directly comparable.
"""

import numpy as np

from v2ecoli.library.schema import MetadataArray
from v2ecoli.steps.derivers.replication_data import ReplicationData

UNTIL = 6.0
N_TICKS = 1

_SUBMASS = [
    (f"massDiff_{x}", "float64")
    for x in ["rRNA", "tRNA", "mRNA", "miscRNA", "nonspecific_RNA",
              "protein", "metabolite", "water", "DNA"]
]
_TAIL = _SUBMASS + [("_entryState", "int8"), ("unique_index", "int64")]


def _mk_unique(head, rows):
    dt = np.dtype(head + _TAIL)
    arr = np.zeros(len(rows), dtype=dt)
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[k][i] = v
        arr["_entryState"][i] = 1
    meta = int(arr["unique_index"].max()) + 1 if len(rows) else 0
    return MetadataArray(arr, metadata=meta)


def unique_stores():
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


def replication_defaults():
    rd = ReplicationData({}).outputs()["listeners"]["replication_data"]
    return {field: spec.get("_default", 0) for field, spec in rd.items()}


def bulk_store(n=12):
    dt = np.dtype([("id", "<U8"), ("count", "int64")])
    arr = np.zeros(n, dtype=dt)
    arr["id"] = [f"MOL-{i}" for i in range(n)]
    arr["count"] = [1000 + 37 * i for i in range(n)]
    return arr


def _clock_edge():
    return {
        "_type": "process",
        "address": "local:v2ecoli.steps.global_clock.GlobalClock",
        "config": {},
        "inputs": {"global_time": ["global_time"],
                   "next_update_time": ["next_update_time"]},
        "outputs": {"global_time": ["global_time"]},
    }


def _emitter_edge():
    return {
        "_type": "step",
        "address": "local:process_bigraph.emitter.RAMEmitter",
        "config": {"emit": {"global_time": "float"}},
        "inputs": {"global_time": ["global_time"]},
        "outputs": {},
    }


def _bulk_edge(shape):
    return {
        "_type": "process",
        "address": "local:viva_core.conformance_processes.BulkProcess",
        "config": {"modulus": 7, "shape": shape},
        "inputs": {"bulk": ["bulk"], "global_time": ["global_time"],
                   "next_update_time": ["next_update_time"]},
        "outputs": {"bulk": ["bulk"]},
    }


def clock_scaffold(timestep=4.0):
    return {
        "global_time": 0.0,
        "timestep": float(timestep),
        "next_update_time": {"p": 3.0},
        "global_clock": _clock_edge(),
    }


# ---- individual battery composites ----------------------------------------- #

def clock_doc():
    """Clock only: apply(float) global_time add + calculate_timestep."""
    return clock_scaffold(), ["global_time", "timestep", "next_update_time"]


def emitter_doc():
    doc = clock_scaffold()
    doc["emitter"] = _emitter_edge()
    return doc, ["global_time"]


def bulk_doc(shape="vectorized"):
    doc = clock_scaffold()
    doc["bulk"] = bulk_store()
    doc["bulk_process"] = _bulk_edge(shape)
    return doc, ["global_time", "bulk"]


def unique_gating_doc():
    """Unique set/add/delete + flush, step gating (update_condition), and an
    overwrite tree (listeners: int/float/bool leaves)."""
    topo = {"oriC": "", "DnaA_box": "", "active_replisome": ""}
    doc = clock_scaffold()
    doc["unique"] = unique_stores()
    doc["listeners"] = {"replication_data": replication_defaults()}
    doc["replication_data_listener"] = {
        "_type": "step",
        "address": "local:v2ecoli.steps.derivers.replication_data.ReplicationData",
        "config": {},
        "inputs": {"oriCs": ["unique", "oriC"],
                   "DnaA_boxes": ["unique", "DnaA_box"],
                   "active_replisomes": ["unique", "active_replisome"],
                   "global_time": ["global_time"], "timestep": ["timestep"]},
        "outputs": {"listeners": ["listeners"]},
    }
    doc["unique_update"] = {
        "_type": "step",
        "address": "local:v2ecoli.steps.unique_update.UniqueUpdate",
        "config": {"unique_topo": topo},
        "inputs": {"oriC": ["unique", "oriC"],
                   "DnaA_box": ["unique", "DnaA_box"],
                   "active_replisome": ["unique", "active_replisome"]},
        "outputs": {"oriC": ["unique", "oriC"],
                    "DnaA_box": ["unique", "DnaA_box"],
                    "active_replisome": ["unique", "active_replisome"]},
    }
    return doc, ["global_time", "unique", "listeners"]


def full_model_doc(shape="vectorized"):
    """The largest composite: clock + bulk PROCESS + unique stores + gated
    deriver + unique flush + emitter. Everything M1 implements, at once."""
    doc, _ = unique_gating_doc()
    doc["bulk"] = bulk_store()
    doc["bulk_process"] = _bulk_edge(shape)
    doc["emitter"] = _emitter_edge()
    keys = ["global_time", "timestep", "next_update_time", "unique",
            "listeners", "bulk"]
    return doc, keys
