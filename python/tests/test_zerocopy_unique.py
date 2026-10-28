"""M2 Task 2: the UNIQUE read view crosses the Python<->Rust boundary zero-copy.

The Rust `UniqueStore` is backed by ONE interleaved numpy-record-layout byte
buffer, so a Python process is handed a genuine numpy structured array that
VIEWS that buffer (read-only). Writes still go back on the unique Op path.

Two properties are asserted:
  * `test_unique_view_is_zero_copy` — the view is a true no-copy view
    (`np.shares_memory` holds tick-over-tick, non-vacuous: it would FAIL under
    the old `np.zeros + per-column set_item` copy, whose fresh array each tick
    shares no memory with the previous one), reads its fields via the `attrs`
    `_entryState.view(bool)` idiom, is `writeable=False`, and its field reads are
    bit-identical to the known store values.
  * `test_unique_view_survives_realloc` — a view RETAINED across a buffer GROW
    (add -> `get_free_indices` grows/reallocs) stays valid and equal to its
    pre-grow snapshot (no use-after-free: the pin kept the old allocation alive).

The M1 unique-parity + quadrant-3 tests (test_quadrant3_parity.py) are the
interleaved-layout regression gate; they must still pass bit-identically.
"""

import numpy as np

import viva_core
from viva_core import conformance as C
from viva_core import conformance_processes as cp

TICKS = 5


def _unique_doc(n=8):
    dt = np.dtype([("data0", "int64"), ("data1", "float64"),
                   ("_entryState", "int8"), ("unique_index", "int64")])
    arr = np.zeros(n, dtype=dt)
    arr["data0"] = [10 * i for i in range(n)]
    arr["data1"] = [0.5 * i for i in range(n)]
    arr["_entryState"] = 1
    arr["unique_index"] = list(range(n))
    return {
        "global_time": 0.0,
        "active": arr,
        "probe": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.UniqueViewProbe",
            "config": {"dt": 1.0},
            "inputs": {"active": ["active"]},
            "outputs": {},
        },
    }


def test_unique_view_is_zero_copy():
    cp.UNIQUE_VIEW_LOG.clear()
    cp.UniqueViewProbe._prev = None
    n = 8
    viva_core.run(C.spec_json(_unique_doc(n)), TICKS)
    log = cp.UNIQUE_VIEW_LOG
    assert len(log) == TICKS
    for rec in log:
        # A genuine numpy structured array: fields + read-only + attrs idiom.
        assert rec["names"] == ("data0", "data1", "_entryState", "unique_index")
        assert rec["writeable"] is False
        assert rec["field_shares_base"] is True  # field is a view into the record
        assert rec["unique_index"] == list(range(n))
        assert rec["active_data0"] == [10 * i for i in range(n)]
    # No copy: a later tick's view is the SAME memory as the prior tick's (the
    # store buffer is never mutated here). Fails under the np.zeros copy path.
    for rec in log[1:]:
        assert rec["shares_prev"] is True
        assert rec["prev_live"] is True


def _realloc_doc(n=2, n_add=5):
    # All `n` rows active, zero free -> the add forces a grow/realloc on flush.
    dt = np.dtype([("data0", "int64"), ("data1", "float64"),
                   ("_entryState", "int8"), ("unique_index", "int64")])
    arr = np.zeros(n, dtype=dt)
    arr["data0"] = [10 * i for i in range(n)]
    arr["data1"] = [0.5 * i for i in range(n)]
    arr["_entryState"] = 1
    arr["unique_index"] = list(range(n))
    return {
        "global_time": 0.0,
        "active": arr,
        "probe": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.UniqueReallocProbe",
            "config": {"dt": 1.0, "n_add": n_add},
            "inputs": {"active": ["active"]},
            "outputs": {"active": ["active"]},
        },
    }


def test_unique_view_survives_realloc():
    cp.UNIQUE_REALLOC_LOG.clear()
    cp.UniqueReallocProbe._stash = None
    cp.UniqueReallocProbe._snapshot = None
    out = viva_core.run(C.spec_json(_realloc_doc(n=2, n_add=5)), 3)
    log = cp.UNIQUE_REALLOC_LOG
    assert len(log) >= 1
    for rec in log:
        # The buffer actually grew (realloc happened) ...
        assert rec["grew"] is True
        assert rec["new_nrows"] > rec["old_nrows"]
        # ... and the RETAINED pre-grow view is still valid + unchanged (no UAF).
        assert rec["stash_len_ok"] is True
        assert rec["stash_unique_index_ok"] is True
        assert rec["stash_data0_ok"] is True
        assert rec["stash_active_sum"] == rec["snap_active_sum"]
    # Final state carries the 2 original + 5 added active molecules.
    active = out["active"]
    es = active["_entryState"]
    assert sum(int(x) != 0 for x in es) == 7
