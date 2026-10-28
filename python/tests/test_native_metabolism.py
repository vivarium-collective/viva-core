"""M3 Task 3: v2ecoli `Metabolism` (FBA) — the NATIVE Rust GLPK solve core.

Metabolism is the single biggest per-tick compute (the FBA `glp_simplex` solve).
This suite proves the native Rust port of that compute is **bit-identical** to the
Python path:

  1. ``fba_solve`` — the native ``extern "C"`` GLPK solve with swiglpk-matched
     simplex parameters — returns the SAME objective, reaction fluxes (col
     primals), reduced costs (col duals), and shadow prices (row duals) as
     swiglpk, on LPs assembled by the EXACT solver wrapper metabolism uses
     (``wholecell.utils._netflow.nf_glpk.NetworkFlowGLPK``).

  2. ``flux_to_bulk_delta`` — the flux→integer-count conversion
     (``Metabolism._fba_output_to_deltas``: ``fmax(stochasticRound(n₀ +
     Δc/counts_to_molar), 0) − n₀``) — returns the SAME integer bulk delta as
     numpy's ``stochasticRound`` on the same ``RandomState(seed)``.

Both GLPK paths call the same GLPK 5.0 C library with identical parameters on
identical cold-basis problems, so equality is **exact** (``==`` on the floats),
not within an epsilon — the arch-01 "metabolism 0.0-divergent bit-identical"
result, at the solver level.

Deferred (reported, not silently wrong): a full native drop-in `Metabolism`
engine process. Its per-tick LP *assembly* reads live sim_data methods that are
not in the serialized edge config (see ``metabolism.rs`` module docs), so the
engine-level port is out of scope; the SOLVE — the dominant compute — is ported
here and proven bit-identical.
"""

import numpy as np
import pytest

# The native GLPK primitives exist only when the wheel is built `--features glpk`.
fba = pytest.importorskip(
    "viva_core._native",
    reason="viva_core._native unavailable",
)
if not hasattr(fba, "fba_solve"):
    pytest.skip(
        "viva_core built without the `glpk` feature (no fba_solve); "
        "build with `maturin develop --release --features glpk`",
        allow_module_level=True,
    )

from v2ecoli.processes.parca.wholecell.utils._netflow.nf_glpk import (  # noqa: E402
    NetworkFlowGLPK,
)
from wholecell.utils.random import stochasticRound  # noqa: E402


# --------------------------------------------------------------------------- #
# LP assembly via the REAL netflow GLPK wrapper (what metabolism actually uses)
# --------------------------------------------------------------------------- #
def _dump_lp(nf):
    """Extract (LP arrays, swiglpk solution) from a built NetworkFlowGLPK, in the
    exact row/column order GLPK holds them (flow insertion order = columns,
    sorted materials = rows)."""
    flows = nf.getFlowNames()          # column order (GLPK col j -> flows[j])
    materials = nf.getMaterialNames()  # row order    (GLPK row i -> materials[i])
    A = nf.getSMatrix()                # (n_materials, n_flows), same order
    lb, ub, obj = nf.getLowerBounds(), nf.getUpperBounds(), nf.getObjective()

    mat_rows, mat_cols, mat_vals = [], [], []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                mat_rows.append(i)
                mat_cols.append(j)
                mat_vals.append(float(A[i, j]))

    lp = {
        "n_rows": len(materials),
        "n_cols": len(flows),
        "mat_rows": mat_rows,
        "mat_cols": mat_cols,
        "mat_vals": mat_vals,
        "col_lb": [float(lb[f]) for f in flows],
        "col_ub": [float(ub[f]) for f in flows],
        "obj": [float(obj.get(f, 0.0)) for f in flows],
        "maximize": bool(nf._maximize),
    }
    sol = {
        "objective": float(nf.getObjectiveValue()),
        "col_primals": [float(x) for x in nf.getFlowRates(flows)],
        "col_duals": [float(x) for x in nf.getReducedCosts(flows)],
        "row_duals": [float(x) for x in nf.getShadowPrices(materials)],
    }
    return lp, sol, flows


def _tiny_network():
    """A minimal mass-balanced FBA: uptake -> conversion -> biomass.
        R_in: -> A   (uptake, <= 10)
        R1 : A -> B
        R_bio: B ->  (objective, maximize)
    Optimum: flux 10 through all, objective 10."""
    nf = NetworkFlowGLPK()
    nf.setFlowMaterialCoeff("R_in", "A", 1.0)
    nf.setFlowMaterialCoeff("R1", "A", -1.0)
    nf.setFlowMaterialCoeff("R1", "B", 1.0)
    nf.setFlowMaterialCoeff("R_bio", "B", -1.0)
    nf.setFlowBounds("R_in", lowerBound=0.0, upperBound=10.0)
    nf.setFlowObjectiveCoeff("R_bio", 1.0)
    nf.maximizeObjective(True)
    nf.buildEqConst()
    return nf


def _random_network(seed, n_internal=40, n_rxn=120, uptake=25.0):
    """A larger random but feasible/degenerate FBA-shaped network, to stress the
    simplex pivot path (alternative optima). Internal metabolites are mass
    balanced; a pool of random reactions plus guaranteed uptake/biomass paths."""
    rng = np.random.RandomState(seed)
    mets = [f"M{i}" for i in range(n_internal)]
    nf = NetworkFlowGLPK()

    # Guaranteed feasible backbone: uptake into M0, a chain M0->M1->...->M_last,
    # then biomass drains the last metabolite.
    nf.setFlowMaterialCoeff("R_uptake", mets[0], 1.0)
    nf.setFlowBounds("R_uptake", lowerBound=0.0, upperBound=uptake)
    for i in range(n_internal - 1):
        r = f"R_chain{i}"
        nf.setFlowMaterialCoeff(r, mets[i], -1.0)
        nf.setFlowMaterialCoeff(r, mets[i + 1], 1.0)
    nf.setFlowMaterialCoeff("R_bio", mets[-1], -1.0)
    nf.setFlowObjectiveCoeff("R_bio", 1.0)

    # Random extra reactions (consume one met, produce another) -> alternative
    # optima / degeneracy. Reversible via [0, ub] bounds.
    for k in range(n_rxn):
        a, b = rng.randint(0, n_internal, size=2)
        if a == b:
            continue
        r = f"R_rand{k}"
        nf.setFlowMaterialCoeff(r, mets[a], -1.0)
        nf.setFlowMaterialCoeff(r, mets[b], 1.0)
        nf.setFlowBounds(r, lowerBound=0.0, upperBound=float(rng.randint(1, 50)))

    nf.maximizeObjective(True)
    nf.buildEqConst()
    return nf


@pytest.mark.parametrize(
    "nf_factory",
    [
        _tiny_network,
        lambda: _random_network(0),
        lambda: _random_network(7),
        lambda: _random_network(42, n_internal=60, n_rxn=240),
        lambda: _random_network(123, n_internal=30, n_rxn=200, uptake=13.0),
    ],
)
def test_native_fba_solve_bit_identical(nf_factory):
    nf = nf_factory()
    lp, py_sol, flows = _dump_lp(nf)
    rs = fba.fba_solve(lp)

    assert rs["status_optimal"], "native GLPK did not reach GLP_OPT"
    # Non-vacuous: the LP actually did something (biomass flux > 0).
    assert py_sol["objective"] > 0.0

    # Bit-identical — EXACT equality (same C library, same params, cold basis).
    assert rs["objective"] == py_sol["objective"], "objective diverged"
    assert rs["col_primals"] == py_sol["col_primals"], "reaction fluxes diverged"
    assert rs["col_duals"] == py_sol["col_duals"], "reduced costs diverged"
    assert rs["row_duals"] == py_sol["row_duals"], "shadow prices diverged"


@pytest.mark.parametrize("seed", [0, 1, 7, 2**31 - 1])
def test_flux_to_bulk_delta_bit_identical(seed):
    rng = np.random.RandomState(99 + (seed % 1000))
    n = 250
    counts_init = rng.randint(0, 5_000_000, size=n).astype(np.int64)
    # Concentration changes that span up/down and whole regimes of fractional part.
    occ = (rng.randn(n) * 1e-4).astype(np.float64)
    counts_to_molar = 3.7e-9  # representative mM-per-count magnitude

    # Python reference: exactly Metabolism._fba_output_to_deltas (metabolite branch).
    delta_mag = occ / counts_to_molar
    vals = counts_init.astype(np.float64) + delta_mag
    counts_final_ref = np.fmax(
        stochasticRound(np.random.RandomState(seed), vals.copy()), 0
    ).astype(np.int64)
    delta_ref = counts_final_ref - counts_init

    out = fba.flux_to_bulk_delta(
        occ.tolist(), counts_init.tolist(), counts_to_molar, seed
    )
    # Integer bulk delta is int-EXACT (the arch-01 bit-identical criterion).
    assert out["delta"] == delta_ref.tolist(), "integer bulk delta diverged"
    assert out["counts_final"] == counts_final_ref.tolist()
    assert delta_ref.tolist() != [0] * n  # non-vacuous: real movement occurred


def test_native_metabolism_bit_identical():
    """Headline parity: the native FBA solve + flux→count conversion together
    reproduce a metabolism tick's compute bit-identically against the Python
    (swiglpk + numpy stochasticRound) path."""
    nf = _random_network(2024, n_internal=50, n_rxn=220)
    lp, py_sol, flows = _dump_lp(nf)
    rs = fba.fba_solve(lp)

    # (a) FBA objective + flux vector: bit-identical.
    assert rs["objective"] == py_sol["objective"]
    assert rs["col_primals"] == py_sol["col_primals"]

    # (b) Feed the solved fluxes through the count conversion; integer bulk delta
    #     must match numpy exactly. (output_molecule_changes stands in for the FBA
    #     per-metabolite concentration change; here we use the solved primals.)
    seed = 0
    occ = np.array(rs["col_primals"][: min(64, len(flows))], dtype=np.float64) * 1e-6
    counts_init = np.arange(1_000_000, 1_000_000 + occ.size, dtype=np.int64)
    ctm = 3.7e-9
    vals = counts_init.astype(np.float64) + occ / ctm
    ref = (
        np.fmax(stochasticRound(np.random.RandomState(seed), vals.copy()), 0)
        .astype(np.int64)
        - counts_init
    )
    out = fba.flux_to_bulk_delta(occ.tolist(), counts_init.tolist(), ctm, seed)
    assert out["delta"] == ref.tolist()
