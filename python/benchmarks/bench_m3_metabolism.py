"""M3 Task 3 benchmark: quadrant-2 FBA solve — the metabolism hot compute.

Metabolism's dominant per-tick cost is the GLPK `glp_simplex` solve. This times
the SAME flux-balance LP solved two ways, on identical cold-basis problems:

  * Q2 native : the Rust `extern "C"` GLPK solve (`viva_core._native.fba_solve`)
  * Py swiglpk: the identical GLPK C work driven from Python via swiglpk — a
    faithful re-expression of `glpk::solve` / `NetworkFlowGLPK._solve` (same
    init_smcp + msg_lev=ERR + it_lim=10000, same bound types, same load_matrix,
    same simplex, same result reads)

Both do identical GLPK C work; the ratio is the FFI/dispatch win of driving the
solver from native Rust vs from the Python swiglpk layer. The two solutions are
asserted bit-identical before any timing is reported.

The LP is assembled once by the REAL solver wrapper metabolism uses
(`NetworkFlowGLPK`) at metabolism-ish scale; a full real-sim LP capture (~9k
reactions) needs ParCa sim_data and is out of this task's scope.

Run (release wheel, glpk feature):
  PYTHONPATH=python <venv>/bin/python python/benchmarks/bench_m3_metabolism.py
"""

import time

import numpy as np
import swiglpk as glp

from viva_core import _native
from v2ecoli.processes.parca.wholecell.utils._netflow.nf_glpk import NetworkFlowGLPK


def build_lp(seed=2024, n_internal=400, n_rxn=2000, uptake=30.0):
    """A metabolism-shaped random mass-balanced FBA LP, built by the real
    NetworkFlowGLPK wrapper, then dumped to flat arrays (GLPK row/col order)."""
    rng = np.random.RandomState(seed)
    mets = [f"M{i}" for i in range(n_internal)]
    nf = NetworkFlowGLPK()
    nf.setFlowMaterialCoeff("R_uptake", mets[0], 1.0)
    nf.setFlowBounds("R_uptake", lowerBound=0.0, upperBound=uptake)
    for i in range(n_internal - 1):
        nf.setFlowMaterialCoeff(f"R_chain{i}", mets[i], -1.0)
        nf.setFlowMaterialCoeff(f"R_chain{i}", mets[i + 1], 1.0)
    nf.setFlowMaterialCoeff("R_bio", mets[-1], -1.0)
    nf.setFlowObjectiveCoeff("R_bio", 1.0)
    for k in range(n_rxn):
        a, b = rng.randint(0, n_internal, size=2)
        if a == b:
            continue
        nf.setFlowMaterialCoeff(f"R_rand{k}", mets[a], -1.0)
        nf.setFlowMaterialCoeff(f"R_rand{k}", mets[b], 1.0)
        nf.setFlowBounds(f"R_rand{k}", lowerBound=0.0, upperBound=float(rng.randint(1, 50)))
    nf.maximizeObjective(True)
    nf.buildEqConst()

    flows, materials = nf.getFlowNames(), nf.getMaterialNames()
    A = nf.getSMatrix()
    lb, ub, obj = nf.getLowerBounds(), nf.getUpperBounds(), nf.getObjective()
    rows, cols, vals = np.nonzero(A)[0], np.nonzero(A)[1], A[np.nonzero(A)]
    return {
        "n_rows": len(materials),
        "n_cols": len(flows),
        "mat_rows": rows.tolist(),
        "mat_cols": cols.tolist(),
        "mat_vals": [float(v) for v in vals],
        "col_lb": [float(lb[f]) for f in flows],
        "col_ub": [float(ub[f]) for f in flows],
        "obj": [float(obj.get(f, 0.0)) for f in flows],
        "maximize": True,
    }


def py_fba_solve(lp):
    """Faithful Python/swiglpk re-expression of `glpk::solve` (identical GLPK
    work): fresh problem, matched params, cold simplex."""
    p = glp.glp_create_prob()
    if lp["n_rows"]:
        glp.glp_add_rows(p, lp["n_rows"])
        for i in range(1, lp["n_rows"] + 1):
            glp.glp_set_row_bnds(p, i, glp.GLP_FX, 0.0, 0.0)
    if lp["n_cols"]:
        glp.glp_add_cols(p, lp["n_cols"])
        for j in range(lp["n_cols"]):
            lo, hi = lp["col_lb"][j], lp["col_ub"][j]
            if np.isinf(lo) and np.isinf(hi):
                t = glp.GLP_FR
            elif lo == hi:
                t = glp.GLP_FX
            elif not np.isinf(lo) and not np.isinf(hi):
                t = glp.GLP_DB
            elif np.isinf(hi):
                t = glp.GLP_LO
            else:
                t = glp.GLP_UP
            glp.glp_set_col_bnds(p, j + 1, t, lo, hi)
            glp.glp_set_obj_coef(p, j + 1, lp["obj"][j])
    ne = len(lp["mat_vals"])
    ia, ja, ar = glp.intArray(ne + 1), glp.intArray(ne + 1), glp.doubleArray(ne + 1)
    for k in range(ne):
        ia[k + 1] = lp["mat_rows"][k] + 1
        ja[k + 1] = lp["mat_cols"][k] + 1
        ar[k + 1] = lp["mat_vals"][k]
    if ne:
        glp.glp_load_matrix(p, ne, ia, ja, ar)
    smcp = glp.glp_smcp()
    glp.glp_init_smcp(smcp)
    smcp.msg_lev = glp.GLP_MSG_ERR
    smcp.it_lim = 10000
    glp.glp_set_obj_dir(p, glp.GLP_MAX if lp["maximize"] else glp.GLP_MIN)
    glp.glp_simplex(p, smcp)
    obj = glp.glp_get_obj_val(p)
    primals = [glp.glp_get_col_prim(p, j) for j in range(1, lp["n_cols"] + 1)]
    glp.glp_delete_prob(p)
    return obj, primals


def main(repeats=7, iters=200):
    lp = build_lp()
    print(f"=== M3 Task 3: FBA GLPK solve, LP {lp['n_rows']} materials x "
          f"{lp['n_cols']} flows, {len(lp['mat_vals'])} nonzeros ===")

    nat = _native.fba_solve(lp)
    py_obj, py_primals = py_fba_solve(lp)
    assert nat["status_optimal"] and nat["objective"] > 0
    assert nat["objective"] == py_obj, "objective diverged"
    assert nat["col_primals"] == py_primals, "flux vector diverged"
    print("native vs swiglpk solve: BIT-IDENTICAL (objective + flux vector)")

    _native.fba_solve(lp); py_fba_solve(lp)  # warm-up

    def bench(fn):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            times.append((time.perf_counter() - t0) / iters)
        return min(times)

    q2 = bench(lambda: _native.fba_solve(lp))
    pysw = bench(lambda: py_fba_solve(lp))
    print(f"  Py swiglpk solve   {1/pysw:>9.1f} solves/s  ({pysw*1e3:7.3f} ms/solve)")
    print(f"  Q2 native  solve   {1/q2:>9.1f} solves/s  ({q2*1e3:7.3f} ms/solve)")
    print(f"  Q2 native vs Py swiglpk : {pysw / q2:.2f}x")


if __name__ == "__main__":
    main()
