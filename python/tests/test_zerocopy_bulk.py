"""M2 Task 1: the BULK read view crosses the Python<->Rust boundary zero-copy.

The Python process is handed a numpy structured array `[("count", "<i8")]` that
VIEWS the Rust `BulkStore.count` buffer (read-only). Writes still go back on the
`Op::BulkDelta` path. These assertions would FAIL under the old
`np.zeros + per-column copy` path: a fresh array each tick shares no memory with
the previous tick's, and a previous view would be a stale snapshot.
"""

import numpy as np

import viva_core
from viva_core import conformance as C
from viva_core import conformance_processes as cp

N = 16
TICKS = 5


def _doc():
    dt = np.dtype([("id", "<U12"), ("count", "int64")])
    bulk = np.zeros(N, dtype=dt)
    bulk["id"] = [f"MOL-{i}" for i in range(N)]
    bulk["count"] = [1000 + 37 * i for i in range(N)]
    return {
        "global_time": 0.0,
        "bulk": bulk,
        "probe": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.ViewProbeProcess",
            "config": {"dt": 1.0},
            "inputs": {"bulk": ["bulk"]},
            "outputs": {"bulk": ["bulk"]},
        },
    }


def test_bulk_view_is_zero_copy():
    cp.VIEW_LOG.clear()
    cp.ViewProbeProcess._prev = None
    out = viva_core.run(C.spec_json(_doc()), TICKS)
    log = cp.VIEW_LOG
    assert len(log) == TICKS
    for rec in log:
        assert rec["names"] == ("count",)
        assert rec["writeable"] is False  # writes go via Op::BulkDelta only
    # No copy: every later tick's view is the same memory as the prior tick's,
    # and the prior view tracks the Rust-side updates live.
    for rec in log[1:]:
        assert rec["shares_prev"] is True
        assert rec["prev_live"] is True
    # Bit-identical to the hand-rolled reference of the process's own update.
    ref = np.array([1000 + 37 * i for i in range(N)], dtype=np.int64)
    for _ in range(TICKS):
        ref = ref + ((ref % 7) - 3)
    assert list(out["bulk"]["count"]) == ref.tolist()
    # ... and the per-tick counts the process saw are the evolving state.
    assert log[0]["counts"].tolist() == [1000 + 37 * i for i in range(N)]
