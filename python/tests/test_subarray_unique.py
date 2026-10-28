"""M4 Task 2 — per-row SUB-ARRAY dtype support in the zero-copy unique store.

`run_prebuilt` (Task 1) runs the real v2ecoli baseline on the Rust engine
bit-identically to pb 1.8.3 for every unique store whose fields are fixed-width
scalars, but HARD-ERRORS on the two stores that carry per-row sub-array fields:

  * ``promoter.bound_TF``          — ``array[23,boolean]``  (dtype ``('?', (23,))``)
  * ``chromosome_domain.child_domains`` — ``array[2,integer]`` (dtype ``('<i4', (2,))``)

A sub-array field IS fixed-width (``elem_size * prod(shape)`` contiguous bytes
per record); the interleaved record buffer already supports arbitrary
fixed-width fields — this task adds the layout/codec arithmetic for a field that
is an array of scalars rather than one scalar.

Two properties, both bit-identical to pb:

  * ``test_subarray_unique_bit_identical`` — a composite driving BOTH sub-array
    stores with a ``set``/``add``/``delete``/``update`` unique update every tick
    (the :class:`SubArrayUpdater` process) runs bit-identical under pb
    ``Composite`` and the Rust engine, INCLUDING every element of every row of
    the sub-array fields (the Rust engine must decode the view, let the process
    read it, apply the update, and read the state back — all exactly as pb).

  * ``test_subarray_view_survives_mutation`` — a zero-copy unique view RETAINED
    across a mutating tick (``set`` on existing rows + ``add`` forcing a grow)
    still reads its PRE-mutation sub-array bytes element-for-element (the
    copy-on-write pin), while the live view reflects the mutation.
"""

import numpy as np

import viva_core
from viva_core import conformance as C
from viva_core import conformance_processes as cp


# --------------------------------------------------------------------------- #
# bit-identical to pb: sub-array decode + view + mutate + read-back
# --------------------------------------------------------------------------- #

_SUBMASS = [
    (f"massDiff_{x}", "float64")
    for x in ["rRNA", "tRNA", "mRNA", "miscRNA", "nonspecific_RNA",
              "protein", "metabolite", "water", "DNA"]
]
_TAIL = _SUBMASS + [("_entryState", "int8"), ("unique_index", "int64")]


def _mk_unique(head, rows):
    """A small real-shaped unique store (the tail matches the v2ecoli schema)."""
    from v2ecoli.library.schema import MetadataArray
    dt = np.dtype(head + _TAIL)
    arr = np.zeros(len(rows), dtype=dt)
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[k][i] = v
        arr["_entryState"][i] = 1
    meta = int(arr["unique_index"].max()) + 1 if len(rows) else 0
    return MetadataArray(arr, metadata=meta)


def _chromosome_domain():
    return _mk_unique(
        [("domain_index", "int32"), ("child_domains", "<i4", (2,))],
        [{"domain_index": 0, "child_domains": [10, 20], "unique_index": 0},
         {"domain_index": 1, "child_domains": [11, 21], "unique_index": 1},
         {"domain_index": 2, "child_domains": [12, 22], "unique_index": 2}],
    )


def _promoter():
    return _mk_unique(
        [("TU_index", "int64"), ("coordinates", "int64"),
         ("domain_index", "int32"), ("bound_TF", "?", (23,))],
        [{"TU_index": 0, "coordinates": 100, "domain_index": 0,
          "bound_TF": [j % 2 == 0 for j in range(23)], "unique_index": 0},
         {"TU_index": 1, "coordinates": 200, "domain_index": 1,
          "bound_TF": [j % 3 == 0 for j in range(23)], "unique_index": 1}],
    )


def _clock_edge():
    return {
        "_type": "process",
        "address": "local:v2ecoli.steps.global_clock.GlobalClock",
        "config": {},
        "inputs": {"global_time": ["global_time"],
                   "next_update_time": ["next_update_time"]},
        "outputs": {"global_time": ["global_time"]},
    }


def _subarray_doc():
    doc = {
        "global_time": 0.0,
        "timestep": 4.0,
        "next_update_time": {"p": 3.0},
        "global_clock": _clock_edge(),
        "unique": {
            "chromosome_domain": _chromosome_domain(),
            "promoter": _promoter(),
        },
        "subarray_updater": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.SubArrayUpdater",
            "config": {},
            "inputs": {"chromosome_domain": ["unique", "chromosome_domain"],
                       "promoter": ["unique", "promoter"],
                       "global_time": ["global_time"],
                       "next_update_time": ["next_update_time"]},
            "outputs": {"chromosome_domain": ["unique", "chromosome_domain"],
                        "promoter": ["unique", "promoter"]},
        },
    }
    return doc, ["global_time", "unique"]


def test_subarray_unique_bit_identical():
    """The two sub-array unique stores, driven set/add/delete/flush across the
    PyO3 boundary, are bit-identical under pb 1.8.3 and the Rust engine —
    including every element of every row of ``child_domains`` and ``bound_TF``."""
    # pb's UniqueNumpyUpdater keeps a module-global per-path registry; clear it
    # so a prior test's accumulator can't leak into this reference run.
    import v2ecoli.types.unique_numpy as un
    un.clear_updater_registry()

    doc, keys = _subarray_doc()
    rep = C.compare_composite(doc, until=6.0, n_ticks=1, store_keys=keys)
    assert rep.bit_identical, rep.summary()

    # Non-vacuous: the sub-array fields actually changed (set + add + delete ran)
    # and are present with their full per-row width.
    cd = rep.got_state["unique"]["chromosome_domain"]
    pr = rep.got_state["unique"]["promoter"]
    assert all(len(row) == 2 for row in cd["child_domains"]), cd["child_domains"]
    assert all(len(row) == 23 for row in pr["bound_TF"]), "bound_TF row width"
    # molecules were added (buffer grew past the initial 3 / 2 rows)
    assert len(cd["child_domains"]) > 3
    assert len(pr["bound_TF"]) > 2
    # and the added child_domains values round-tripped element-exact vs pb
    assert cd["child_domains"] == rep.ref_state["unique"]["chromosome_domain"]["child_domains"]
    assert pr["bound_TF"] == rep.ref_state["unique"]["promoter"]["bound_TF"]


# --------------------------------------------------------------------------- #
# realloc-safety: a view taken before a mutating tick keeps its sub-array bytes
# --------------------------------------------------------------------------- #

def _view_doc(n=2, n_add=5):
    # All `n` rows active, zero free -> the add forces a grow/realloc on flush,
    # and the set mutates the existing rows' sub-array field.
    dt = np.dtype([("domain_index", "int32"), ("child_domains", "<i4", (2,)),
                   ("_entryState", "int8"), ("unique_index", "int64")])
    arr = np.zeros(n, dtype=dt)
    arr["domain_index"] = list(range(n))
    arr["child_domains"] = [[10 + i, 20 + i] for i in range(n)]
    arr["_entryState"] = 1
    arr["unique_index"] = list(range(n))
    return {
        "global_time": 0.0,
        "active": arr,
        "probe": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.SubArrayViewProbe",
            "config": {"dt": 1.0, "n_add": n_add},
            "inputs": {"active": ["active"]},
            "outputs": {"active": ["active"]},
        },
    }


def test_subarray_view_survives_mutation():
    """A zero-copy unique view retained across a mutating tick (set on existing
    rows + add forcing a grow) still reads its PRE-mutation sub-array bytes."""
    cp.SUBARRAY_VIEW_LOG.clear()
    cp.SubArrayViewProbe._stash = None
    cp.SubArrayViewProbe._snapshot = None

    out = viva_core.run(C.spec_json(_view_doc(n=2, n_add=5)), 3)
    log = cp.SUBARRAY_VIEW_LOG
    assert len(log) >= 1
    for rec in log:
        assert rec["grew"] is True
        assert rec["new_nrows"] > rec["old_nrows"]
        # the retained pre-mutation view's sub-array bytes are intact ...
        assert rec["stash_child_domains_ok"] is True
        assert rec["stash_unique_index_ok"] is True
        # ... while the live view reflects the set mutation.
        assert rec["live_changed"] is True

    # Final state: 2 original - 0 deleted + 5 added = 7 active (probe deletes none).
    active = out["active"]
    es = active["_entryState"]
    assert sum(int(x) != 0 for x in es) == 7
    # Every child_domains row still has its full 2-element width.
    assert all(len(row) == 2 for row in active["child_domains"])
