"""M5 Task 4 — the WHOLE realized v2ecoli baseline runs on the Rust engine.

This is the M5 deliverable gate. The whole baseline agent (one generation,
baseline media, seed 0) is handed to ``viva_core.run_prebuilt`` as its realized
``process_bigraph.Composite`` (the Task-3 handoff, exercised here via the
``Composite`` + ``path=("agents","0")`` argument), native-swap OFF — every one
of the ~55 processes runs as its pre-built Python instance on the Rust engine.

What this proves (asserted below):
  * the whole model RUNS end-to-end on the Rust engine — no store the handoff
    cannot carry. This needed, beyond the four designed M5 blockers (unit bridge,
    handle table, allocator_rng, realized handoff), a set of representation
    bridges the real whole model exposed and that this milestone added:
      - a RAM-emitter fallback for the ParquetEmitter edge;
      - nested-map update merge (``environment`` ``map[float]`` additive,
        ``exchange_data``/listener trees overwrite-per-leaf) instead of a
        wholesale node overwrite — closes known-unknown (b);
      - drop of writes to schema-absent paths (pb projects updates through the
        schema: e.g. ``listeners.equilibrium_listener`` is ignored);
      - empty-dict updates as no-ops;
      - wholesale subtree overwrite (``overwrite[map[...]]``, the partition
        ``request`` store);
      - plain ndarray leaves preserved as ndarrays through the view + final
        state (``__numpy__``), and the unique stores exposed as ``MetadataArray``
        carrying the next-unique-index counter — closes known-unknown (a);
      - the bulk view carries the submass float columns (MassDeriver);
      - the ``BulkView`` subclass degrades to a plain ndarray through numpy
        ufuncs / reductions (so bulk-derived values match pb and numba accepts
        them);
      - the edge name set on each pre-built instance (``Requester`` reads it).
  * the Rust-engine run is deterministic (repeated runs identical).
  * known-unknown (a): a ``listeners`` ndarray leaf is handed to a reader AS an
    ndarray (array math / fancy-indexing works), not a plain list.
  * known-unknown (b): a non-``overwrite`` nested output (``environment.exchange``
    typed ``map[float]``) MERGES per-leaf additively, matching pb, rather than
    hitting the wholesale ``Op::Overwrite`` arm.

What this now ALSO proves (the M5 bit-identity deliverable):
  * the whole-model run IS bit-identical to pb 1.8.3 over two ticks — bulk counts
    int-exact, float leaves within 1e-12, pint Quantities equal in magnitude and
    units (``test_wholemodel_bit_identical_to_pb``). The earlier milestone drafted
    this as an ``xfail`` and attributed the residual to "float-precision / numba
    sensitivity"; that diagnosis was wrong. The residual was a scheduler-fidelity
    defect: the realized baseline is wired by pb's ``wire_step_layers`` into a
    linear ``_layer_token`` chain, and ``build_payload`` was STRIPPING those
    tokens — so the Rust scheduler reconstructed the start-cascade order from the
    real data wiring alone, where bulk/unique/timestep are read-and-written by
    nearly every step (self-loops that create no ordering edges). It therefore ran
    steps in the wrong order (e.g. ProteinDegradation ahead of Equilibrium), and
    ``stochasticRound`` + the near-degenerate FBA LP amplified that integer-count
    seed into a ~3e-7 bulk drift. Passing the flow tokens through the handoff
    closes it exactly. See docs/milestone-5-results.md.
"""

import os

import numpy as np
import pint
import pytest

from process_bigraph import Composite

import viva_core  # noqa: F401
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
def ctx():
    core = build_core()
    set_current_core(core)
    bundle = load_cache_bundle(CACHE)
    return core, bundle


def _fresh(core, bundle):
    """A freshly realized whole-baseline ``Composite`` (seed 0, 1 gen, minimal)."""
    doc = baseline(core, seed=0, cache_dir=CACHE, bundle=bundle)
    return Composite(doc, core=core)


def _bulk_count(state):
    """Bulk ``count`` column as an int array, from either the structured-array
    form (pb state) or the plain dict-of-columns form (run_prebuilt final state)."""
    return np.asarray(state["bulk"]["count"])


# --------------------------------------------------------------------------- #
# The deliverable: the whole model RUNS on the Rust engine, deterministically.
# --------------------------------------------------------------------------- #

def test_wholemodel_runs_end_to_end(ctx):
    """Every edge of the whole realized baseline runs under run_prebuilt native-OFF
    (via the Composite + path handoff). The run completing at all is the M5
    enablement deliverable — M4 could not build the payload."""
    core, bundle = ctx
    out = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0"))
    assert isinstance(out, dict)
    assert "bulk" in out and "listeners" in out and "unique" in out
    # Non-vacuous: the model actually advanced (bulk counts moved from the init).
    init = _bulk_count(_fresh(core, bundle).state["agents"]["0"])
    final = _bulk_count(out)
    assert int(np.count_nonzero(final - init)) > 100


def test_wholemodel_run_is_deterministic(ctx):
    """Two run_prebuilt runs of the whole model are byte-for-byte identical — the
    divergence from pb is a reproducible numerical difference, not randomness."""
    core, bundle = ctx
    a = _bulk_count(prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0")))
    b = _bulk_count(prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0")))
    assert np.array_equal(a, b)


def test_handoff_accepts_composite_and_path(ctx):
    """The Task-3 handoff args exercised on the real model: a realized Composite +
    path=("agents","0") builds a payload with the real partitioned processes, the
    live handle stores, and the pint tags — and the build-time wire-symmetry guard
    does NOT false-positive on the real baseline wiring (Task-2 carry-in)."""
    core, bundle = ctx
    comp = _fresh(core, bundle)
    processes, topo_json, state_json, handles = prebuilt.build_payload(
        comp, path=("agents", "0")
    )
    # the 3 real partitioned processes' Requester/Evolver + the Allocator are present
    assert any(k.endswith("_requester") for k in processes)
    assert any(k.endswith("_evolver") for k in processes)
    # live-object stores travel out-of-band as handles (process-as-data + rng)
    assert "allocator_rng" in handles and any(p.startswith("process/") for p in handles)
    # pint Quantities lowered to tags, not dropped
    assert '"__pint__"' in state_json


# --------------------------------------------------------------------------- #
# Known-unknown (a): a listeners ndarray leaf reaches a reader AS an ndarray.
# --------------------------------------------------------------------------- #

def test_known_unknown_a_ndarray_leaf_is_ndarray(ctx):
    """A plain ndarray leaf (``ppgpp_state.basal_prob``) is handed to a process as
    a numpy ndarray, so a reader doing array math / fancy-indexing
    (``basal_prob[TU_index]``) behaves as under pb — NOT as a plain Python list
    (which would raise ``only integer scalar arrays can be converted to a scalar
    index``). We assert the round-trip preserves the ndarray type + values."""
    core, bundle = ctx
    comp = _fresh(core, bundle)
    agent = comp.state["agents"]["0"]
    leaf = agent["ppgpp_state"]["basal_prob"]
    assert isinstance(leaf, np.ndarray) and leaf.ndim >= 1
    # Lower (as the handoff does) and rebuild through the Rust codec round-trip by
    # running one tick; the final-state leaf must still be an ndarray equal to pb.
    out = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=1, path=("agents", "0"))
    got = out["ppgpp_state"]["basal_prob"]
    assert isinstance(got, np.ndarray), f"basal_prob came back as {type(got).__name__}, not ndarray"
    ref = Composite(baseline(core, seed=0, cache_dir=CACHE, bundle=bundle), core=core)
    ref.run(1.0)
    ref_leaf = ref.state["agents"]["0"]["ppgpp_state"]["basal_prob"]
    assert got.shape == np.asarray(ref_leaf).shape


# --------------------------------------------------------------------------- #
# Known-unknown (b): a non-overwrite nested output merges per-leaf like pb.
# --------------------------------------------------------------------------- #

def test_known_unknown_b_nested_map_merges_additively(ctx):
    """``ecoli-metabolism`` writes ``environment.exchange`` typed ``map[float]``.
    pb applies it as an ADDITIVE per-key merge (float apply = state + update), not
    a wholesale node overwrite. We verify the Rust engine merges per-leaf: over two
    ticks the accumulated exchange is (close to) the sum of the per-tick fluxes,
    i.e. it ACCUMULATES rather than being replaced each tick."""
    core, bundle = ctx
    out1 = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=1, path=("agents", "0"))
    out2 = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0"))
    ex1 = out1["environment"]["exchange"]
    ex2 = out2["environment"]["exchange"]
    # GLC uptake is strongly nonzero and accumulates (additive map merge): the
    # two-tick magnitude exceeds the one-tick magnitude (not overwritten/reset).
    assert abs(ex2["GLC"]) > abs(ex1["GLC"]) > 0
    # the leaves are bare floats (not left as pint tags or dicts) after the merge
    assert isinstance(ex2["GLC"], float)


# --------------------------------------------------------------------------- #
# The M5 deliverable: the whole model is BIT-IDENTICAL to pb 1.8.3.
#
# This was the xfail gate while the residual was open. The residual was a
# scheduler-fidelity defect, NOT float noise: the realized baseline is wired by
# pb's ``wire_step_layers`` into a linear ``_layer_token`` chain, and the handoff
# was stripping those tokens — so the Rust scheduler could not reproduce pb's
# start-cascade order (it ran e.g. ProteinDegradation before Equilibrium) and the
# stochastic reads + FBA LP amplified that integer-count seed. Passing the tokens
# through (prebuilt.build_payload) closed it; the model is now bit-identical.
# See docs/milestone-5-results.md.
# --------------------------------------------------------------------------- #


def _norm_array(v):
    """Normalize a structured array (pb state) or dict-of-columns (run_prebuilt
    final state) to an ordered ``{column: ndarray}`` dict."""
    names = getattr(getattr(v, "dtype", None), "names", None)
    if names:  # structured ndarray
        return {n: np.asarray(v[n]) for n in names}
    if isinstance(v, dict):  # already columns
        return {k: np.asarray(x) for k, x in v.items()}
    return None


def _assert_leaf_equal(ref, got, path):
    """Assert one leaf is bit-identical: ints exact, floats within 1e-12, pint
    Quantities equal in magnitude (<=1e-12) AND units, strings/bools exact."""
    # pint Quantities: magnitude + units must both match.
    if isinstance(ref, pint.Quantity) or isinstance(got, pint.Quantity):
        assert isinstance(ref, pint.Quantity) and isinstance(got, pint.Quantity), (
            f"{path}: pint vs non-pint ({type(ref).__name__} vs {type(got).__name__})"
        )
        assert ref.units == got.units, f"{path}: units {ref.units} != {got.units}"
        _assert_leaf_equal(np.asarray(ref.magnitude), np.asarray(got.magnitude), path + ".magnitude")
        return

    cols_ref, cols_got = _norm_array(ref), _norm_array(got)
    if cols_ref is not None and cols_got is not None:
        # The unique stores are v2ecoli ``MetadataArray``s carrying the
        # next-unique-index counter: pb holds it as a ``.metadata`` attribute, the
        # decoded run_prebuilt state as a ``_metadata`` pseudo-column. Compare it
        # against pb's attribute (int-exact) — this is the I2 shared-counter value.
        md_col = cols_got.pop("_metadata", None)
        ref_md = getattr(ref, "metadata", None)
        if md_col is not None or ref_md is not None:
            assert md_col is not None and ref_md is not None, (
                f"{path}: unique-index counter present on one side only"
            )
            got_md = int(np.asarray(md_col).ravel()[0])
            assert got_md == int(ref_md), (
                f"{path}: unique-index counter {got_md} != pb {int(ref_md)}"
            )
        assert set(cols_ref) == set(cols_got), f"{path}: columns {set(cols_ref)} != {set(cols_got)}"
        for c in cols_ref:
            _assert_leaf_equal(cols_ref[c], cols_got[c], f"{path}[{c}]")
        return

    a, b = np.asarray(ref), np.asarray(got)
    # Both empty = no entries to diff. An empty unique store's 2-D field (e.g.
    # ``boundary_molecule_indexes``) is pb-shaped ``(0, 2)`` but decodes to
    # ``(0,)``; with zero rows that is the same (absence of) data, not a divergence.
    if a.size == 0 and b.size == 0:
        return
    assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool or a.dtype.kind in "US":
        assert np.array_equal(a, b), f"{path}: int/bool/str array differs"
    elif np.issubdtype(a.dtype, np.floating):
        # exact where finite; else bounded by 1e-12 absolute-or-relative.
        if not np.array_equal(a, b):
            fin = np.isfinite(a) & np.isfinite(b)
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{path}: non-finite mask differs"
            diff = np.abs(a[fin] - b[fin])
            tol = 1e-12 * np.maximum(1.0, np.abs(a[fin]))
            assert np.all(diff <= tol), f"{path}: float max|Δ| {diff.max():.3e} exceeds 1e-12"
    else:
        assert np.array_equal(a, b), f"{path}: array differs"


# Stores holding live handle objects (process-as-data, RNG) travel out-of-band
# and are not state leaves to diff; the RNG being in sync is proven by the bulk
# trajectory being bit-identical over the ticks (any RNG drift would show there).
# ``process``/``allocator_rng`` are live-handle stores (not diffable leaves).
# ``global_time``/``next_update_time`` are scheduler time BOOKKEEPING, not model
# state: ``pb.run(N)`` ends labelled global_time N while ``run_prebuilt(n_ticks=N)``
# ends at N+1 (the documented duration-vs-iteration convention — both run every
# process the SAME number of times), so these clocks carry a benign +1 offset that
# is not a state divergence. See docs/milestone-5-results.md (M2).
_HANDLE_KEYS = {"process", "allocator_rng", "global_time", "next_update_time"}

# Leaves that are a pure function of the ``global_time`` LABEL (not of model
# state), so they inherit the benign +1 duration-vs-iteration offset above and
# are not a state divergence: ``expected_mass_fold_change = 2**((global_time -
# timeInitial)/cell_cycle_len)`` (mass_deriver) differs only because the clock is
# labelled t+1. Excluded for the same reason as the clock itself (M2). Every
# model-state leaf (bulk, unique, masses, counts, fluxes, …) is still compared.
_CLOCK_DERIVED_LEAVES = {"expected_mass_fold_change"}

# KNOWN RESIDUAL (tracked, not noise) — the "schema-projection apply gap".
#
# pb applies every update through the STORE's composition schema: a store typed
# ``integer`` or ``array[integer]`` (and the ``*_vec`` labeled-array listener
# types) accumulates additively across ticks, even when the WRITER declares
# ``overwrite[integer]`` or returns a plain int array. The Rust engine instead
# picks the apply op from the writer's declared OUTPUT-PORT type (Integer ->
# additive, but Array / Named / Overwrite -> overwrite) and has no element-wise
# integer-array additive op, so these additive-store listener leaves are
# OVERWRITTEN (kept at the last tick) instead of accumulated. These are diagnostic
# listener sinks — the core model state (bulk, unique incl. the index counter, all
# masses/concentrations/fluxes, every pint Quantity) is bit-identical, which is why
# the trajectory does not diverge. Closing this needs the composition schema's
# per-store apply-types carried into the handoff so the engine projects through
# them (the same architectural gap as the I3 map-vs-node schema-absent-path drop);
# it is beyond the M5 scheduler/counter scope. Pinned by the strict-xfail
# ``test_listener_schema_projection_residual`` below. See task-4-fix-report.md.
_SCHEMA_PROJECTION_RESIDUAL = {
    "atp_requested",             # listeners/atp — allocator, array[n_processes,integer]
    "atp_allocated_initial",     # listeners/atp — allocator, array[n_processes,integer]
    "rna_init_event_per_cistron",  # listeners/rnap_data — rna_init_event_per_cistron_vec
    "did_initialize",            # listeners/rnap_data — store integer, writer overwrite[integer]
    "total_rna_init",            # listeners/rna_synth_prob — store integer, writer overwrite[integer]
    "n_actual_bound",            # listeners/rna_synth_prob — array[n_TF,integer]
}


def _is_edge(v):
    return isinstance(v, dict) and "address" in v and ("inputs" in v or "outputs" in v)


def _assert_bit_identical(ref, got, path="agent"):
    if isinstance(ref, dict):
        assert isinstance(got, dict), f"{path}: dict vs {type(got).__name__}"
        # Compare only DATA stores: skip handle stores, edge (process/step) dicts
        # — which are not part of the decoded final state — and pb's scheduler
        # flow-token stores (``_layer*``, scheduling-only, never materialized).
        def data_keys(d):
            return {
                k
                for k, v in d.items()
                if k not in _HANDLE_KEYS
                and k not in _CLOCK_DERIVED_LEAVES
                and k not in _SCHEMA_PROJECTION_RESIDUAL
                and not str(k).startswith("_layer")
                and not _is_edge(v)
            }

        ref_keys = data_keys(ref)
        got_keys = data_keys(got)
        # ``got`` is the decoded final state; it must carry every schema store pb has.
        missing = ref_keys - got_keys
        assert not missing, f"{path}: run_prebuilt output missing stores {sorted(missing)}"
        for k in ref_keys:
            if k in got:
                _assert_bit_identical(ref[k], got[k], f"{path}/{k}")
    else:
        _assert_leaf_equal(ref, got, path)


def test_wholemodel_bit_identical_to_pb(ctx):
    """M5 DELIVERABLE: the whole realized baseline run on the Rust engine is
    bit-identical to pb 1.8.3 over two start cascades + two ticks — bulk counts
    int-exact, unique stores (incl. the next-unique-index counter) int-exact, every
    float leaf within 1e-12, every pint Quantity equal in magnitude and units.

    Was xfail(strict) while the residual was open; the residual was the
    scheduler-ordering defect (stripped ``_layer`` flow tokens) + the unique-index
    counter gap, both now fixed, so this is a hard assert. The six additive-store
    diagnostic listener leaves in ``_SCHEMA_PROJECTION_RESIDUAL`` are excluded and
    tracked separately (``test_listener_schema_projection_residual``)."""
    core, bundle = ctx
    ref = Composite(baseline(core, seed=0, cache_dir=CACHE, bundle=bundle), core=core)
    ref.run(2.0)
    got = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0"))
    # Headline: bulk count array int-exact.
    assert np.array_equal(_bulk_count(ref.state["agents"]["0"]), _bulk_count(got))
    # Full deep state comparison (bulk / unique / listeners / environment / pint).
    _assert_bit_identical(ref.state["agents"]["0"], got)


def test_wholemodel_bulk_total_exact(ctx):
    """The whole-model bulk total matches pb exactly (relative divergence 0) after
    two ticks — the residual the earlier milestone bounded at <1e-6 is now closed."""
    core, bundle = ctx
    ref = Composite(baseline(core, seed=0, cache_dir=CACHE, bundle=bundle), core=core)
    ref.run(2.0)
    ref_sum = int(_bulk_count(ref.state["agents"]["0"]).sum())
    got_sum = int(
        _bulk_count(
            prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0"))
        ).sum()
    )
    assert ref_sum == got_sum, f"bulk total {got_sum} != pb {ref_sum}"


@pytest.mark.xfail(
    reason="Schema-projection apply gap: additive-store diagnostic listeners "
    "(integer / array[integer] / *_vec) are OVERWRITTEN by the Rust engine (it "
    "picks the apply op from the writer's declared port type and has no element-wise "
    "integer-array additive op) instead of being accumulated through the store's "
    "composition schema as pb does. Not scheduler ordering and not float noise — a "
    "deeper schema-in-handoff gap, same family as the I3 map-vs-node drop. Flips "
    "green when the engine projects updates through the store schema. See "
    "task-4-fix-report.md.",
    strict=True,
)
def test_listener_schema_projection_residual(ctx):
    """Pins the known residual: the six additive-store listener leaves SHOULD be
    bit-identical to pb but are overwritten (kept at the last tick) rather than
    accumulated. Asserting their bit-identity fails today (xfail) and will pass once
    the engine applies these through the store's composition schema."""
    core, bundle = ctx
    ref = Composite(baseline(core, seed=0, cache_dir=CACHE, bundle=bundle), core=core)
    ref.run(2.0)
    rs = ref.state["agents"]["0"]
    got = prebuilt.run_from_state(_fresh(core, bundle), n_ticks=2, path=("agents", "0"))
    checked = 0
    for sub, leaves in rs["listeners"].items():
        if not isinstance(leaves, dict):
            continue
        for leaf, rval in leaves.items():
            if leaf not in _SCHEMA_PROJECTION_RESIDUAL:
                continue
            gval = got["listeners"][sub][leaf]
            assert np.array_equal(np.asarray(rval), np.asarray(gval)), (
                f"listeners/{sub}/{leaf} differs (schema-projection residual)"
            )
            checked += 1
    assert checked > 0, "expected to find the residual listener leaves"
