"""Task 8 capstone: the cross-language conformance suite.

`test_conformance_full_model` runs the largest composite Task 5/6 reached — the
real v2ecoli unique-store + gating + emitter model — PLUS the carry-forward the
earlier tasks never exercised: a Python PROCESS that READS and WRITES a **bulk**
store (`bulk["count"]`, v2ecoli's hot path) across the PyO3 boundary. The
decoded final state must be bit-identical under pb 1.8.3 and the Rust engine
(ints exact, floats at epsilon), including the bulk store.

The upstream-suite passing fraction + (a)/(b)/(c) breakdown is produced by
`viva_core.upstream_suite` and asserted here as a non-regression floor.
"""

import pytest

from viva_core import conformance as C
from viva_core import _conformance_models as M

EPS = 1e-12


@pytest.fixture(scope="module")
def full_report():
    doc, keys = M.full_model_doc(shape="vectorized")
    from v2ecoli.steps.derivers.replication_data import ReplicationData
    return C.compare_composite(
        doc, until=M.UNTIL, n_ticks=M.N_TICKS, store_keys=keys, eps=EPS,
        gate_classes={"replication_data_listener": ReplicationData},
        gate_edge="replication_data_listener",
    )


def test_conformance_full_model(full_report):
    """Largest composite runs bit-identical under both engines, INCLUDING the
    bulk-store-driving process."""
    assert full_report.bit_identical, full_report.summary()
    assert full_report.got_state["global_time"] == 6.0
    rd = full_report.got_state["listeners"]["replication_data"]
    assert rd["number_of_oric"] == 2
    assert rd["total_DnaA_boxes"] == 3


def test_bulk_store_crosses_boundary(full_report):
    """The BULK carry-forward: the bulk `count` column, read AND written by a
    Python process across the boundary, is bit-identical to pb."""
    ref_bulk = full_report.ref_state["bulk"]
    got_bulk = full_report.got_state["bulk"]
    assert C.compare(ref_bulk, got_bulk, EPS) == [], "bulk diverged across boundary"
    # The process must actually have changed the counts (both read + write
    # crossed the boundary).
    initial = [1000 + 37 * i for i in range(12)]
    assert got_bulk["count"] != initial
    assert ref_bulk["count"] == got_bulk["count"]


def test_bulk_scalar_pair_shape():
    """The scalar `(index, delta)`-list bulk update shape is also bit-identical
    (covers the scalar branch of the Rust `bulk_op`)."""
    doc, keys = M.full_model_doc(shape="scalar")
    rep = C.compare_composite(doc, until=M.UNTIL, n_ticks=M.N_TICKS,
                              store_keys=keys, eps=EPS)
    assert rep.bit_identical, rep.summary()


def test_differential_gating_and_emitters(full_report):
    """Gating decisions and emitter row counts match pb."""
    assert full_report.got_gate_on == full_report.ref_gate_on == {0.0}
    assert full_report.got_emitter_counts.get("emitter") == full_report.ref_emitter_rows == 2


def test_upstream_suite_report():
    from viva_core import upstream_suite

    rep = upstream_suite.run_all()
    b = rep["battery"]
    # Any category-(c) real divergence is a conformance bug and fails here.
    assert b["real_divergence"] == 0, b["divergence_detail"]
    assert b["passed"] == b["total_cases"], b
    assert b["pass_fraction"] == 1.0, b
    assert b["passed"] >= 6, b
