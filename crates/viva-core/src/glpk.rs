//! Native Rust FFI to the GNU Linear Programming Kit (GLPK) C API, plus a safe
//! FBA LP solver that mirrors v2ecoli's GLPK solver wrapper
//! (`wholecell/utils/_netflow/nf_glpk.py::NetworkFlowGLPK`) **bit-identically**.
//!
//! The metabolism process solves, each tick, a flux-balance LP
//!
//! ```text
//! max/min  cᵀv   s.t.  S·v = 0 (mass balance),   v_lb ≤ v ≤ v_ub
//! ```
//!
//! via swiglpk → `glp_simplex`. This module links GLPK's C library directly
//! (`extern "C"`, the pattern proven in the M1 spike `glpk-smoke`) and reproduces
//! the EXACT solver configuration swiglpk's wrapper uses so the LP solution
//! (objective + primal fluxes + duals) is bit-identical:
//!
//!   * `glp_init_smcp` populates GLPK's own defaults (version-independent),
//!   * then only two overrides, exactly as `NetworkFlowGLPK.__init__`
//!     (`msg_lev = GLP_MSG_ERR`, `it_lim = 10000`),
//!   * rows (materials) are fixed to 0 (`GLP_FX 0,0`),
//!   * column (flow) bound types are derived with the same branch ladder as
//!     `_set_col_bounds` (FR/FX/DB/LO/UP),
//!   * the constraint matrix is loaded via `glp_load_matrix` (1-indexed COO),
//!   * the objective direction is `GLP_MAX`/`GLP_MIN` per `_maximize`.
//!
//! A fresh solve (cold basis) is deterministic given identical problem + params,
//! so a fresh Rust solve equals a fresh swiglpk solve bit-for-bit. (Across ticks
//! the Python process reuses one warm-started `_lp`; this module exposes the
//! single-solve core — the hot compute — which the parity oracle drives cold on
//! both sides. See `python/tests/test_native_metabolism.py`.)
#![allow(non_camel_case_types)]

use std::os::raw::{c_double, c_int};

// --- opaque problem handle --------------------------------------------------

#[repr(C)]
pub struct glp_prob {
    _private: [u8; 0],
}

// --- GLPK constants (verified against swiglpk 5.0) --------------------------

const GLP_MIN: c_int = 1;
const GLP_MAX: c_int = 2;
const GLP_FR: c_int = 1; // free (unbounded)
const GLP_LO: c_int = 2; // lower bound only
const GLP_UP: c_int = 3; // upper bound only
const GLP_DB: c_int = 4; // double bounded
const GLP_FX: c_int = 5; // fixed
const GLP_MSG_ERR: c_int = 1; // warning + error only (netflow default)
const GLP_OPT: c_int = 5; // solution is optimal

/// Simplex iteration limit set by `NetworkFlowGLPK.__init__`.
const SIMPLEX_ITERATION_LIMIT: c_int = 10000;

/// `glp_smcp` — simplex solver control parameters. Field order and the trailing
/// `foo_bar[33]` reserved padding match `/opt/homebrew/include/glpk.h` (GLPK 5.0)
/// exactly, so `glp_init_smcp` writes GLPK's own defaults into it.
#[repr(C)]
struct glp_smcp {
    msg_lev: c_int,
    meth: c_int,
    pricing: c_int,
    r_test: c_int,
    tol_bnd: c_double,
    tol_dj: c_double,
    tol_piv: c_double,
    obj_ll: c_double,
    obj_ul: c_double,
    it_lim: c_int,
    tm_lim: c_int,
    out_frq: c_int,
    out_dly: c_int,
    presolve: c_int,
    excl: c_int,
    shift: c_int,
    aorn: c_int,
    foo_bar: [c_double; 33],
}

extern "C" {
    fn glp_create_prob() -> *mut glp_prob;
    fn glp_delete_prob(p: *mut glp_prob);
    fn glp_set_obj_dir(p: *mut glp_prob, dir: c_int);
    fn glp_add_rows(p: *mut glp_prob, n: c_int) -> c_int;
    fn glp_add_cols(p: *mut glp_prob, n: c_int) -> c_int;
    fn glp_set_row_bnds(p: *mut glp_prob, i: c_int, t: c_int, lb: c_double, ub: c_double);
    fn glp_set_col_bnds(p: *mut glp_prob, j: c_int, t: c_int, lb: c_double, ub: c_double);
    fn glp_set_obj_coef(p: *mut glp_prob, j: c_int, c: c_double);
    fn glp_load_matrix(
        p: *mut glp_prob,
        ne: c_int,
        ia: *const c_int,
        ja: *const c_int,
        ar: *const c_double,
    );
    fn glp_init_smcp(parm: *mut glp_smcp);
    fn glp_simplex(p: *mut glp_prob, parm: *const glp_smcp) -> c_int;
    fn glp_get_status(p: *const glp_prob) -> c_int;
    fn glp_get_obj_val(p: *const glp_prob) -> c_double;
    fn glp_get_col_prim(p: *const glp_prob, j: c_int) -> c_double;
    fn glp_get_col_dual(p: *const glp_prob, j: c_int) -> c_double;
    fn glp_get_row_dual(p: *const glp_prob, i: c_int) -> c_double;
}

/// A flux-balance LP in the exact shape `NetworkFlowGLPK` builds:
/// `n_rows` material (mass-balance) constraints × `n_cols` flow variables.
pub struct Lp {
    /// Number of material rows (constraints), all fixed to 0.
    pub n_rows: usize,
    /// Number of flow columns (structural variables).
    pub n_cols: usize,
    /// Sparse matrix entries `(row, col, coeff)`, 0-indexed. No duplicate
    /// `(row, col)` pairs (GLPK forbids them), matching netflow's COO load.
    pub entries: Vec<(usize, usize, f64)>,
    /// Per-column lower bound (may be `-inf`).
    pub col_lb: Vec<f64>,
    /// Per-column upper bound (may be `+inf`).
    pub col_ub: Vec<f64>,
    /// Per-column objective coefficient.
    pub obj: Vec<f64>,
    /// `true` → maximize (`GLP_MAX`), `false` → minimize (`GLP_MIN`).
    pub maximize: bool,
}

/// The LP solution read back the same way netflow does.
pub struct LpSolution {
    pub status_optimal: bool,
    pub objective: f64,
    /// Primal flow values (reaction fluxes), one per column.
    pub col_primals: Vec<f64>,
    /// Column dual values (reduced costs), one per column.
    pub col_duals: Vec<f64>,
    /// Row dual values (shadow prices), one per row.
    pub row_duals: Vec<f64>,
}

/// `_set_col_bounds`: derive the GLPK bound-type code from a `(lower, upper)`
/// pair, with the exact branch order of the Python wrapper.
fn col_bound_type(lower: f64, upper: f64) -> c_int {
    if lower.is_infinite() && upper.is_infinite() {
        GLP_FR
    } else if lower == upper {
        GLP_FX
    } else if !lower.is_infinite() && !upper.is_infinite() {
        GLP_DB
    } else if upper.is_infinite() {
        GLP_LO
    } else {
        GLP_UP
    }
}

/// Solve one FBA LP via GLPK's simplex, with swiglpk-matched parameters, on a
/// fresh (cold-basis) problem. Mirrors `NetworkFlowGLPK.buildEqConst` +
/// `_solve` for the common (first, optimal) path: no presolve, no scaling
/// fallback, primal simplex. Returns the same quantities netflow caches
/// (`glp_get_obj_val`, `get_col_primals`, `get_col_duals`, `get_row_duals`).
///
/// # Safety
/// Calls into the GLPK C library; all pointers are owned and freed here.
pub fn solve(lp: &Lp) -> LpSolution {
    assert_eq!(lp.col_lb.len(), lp.n_cols, "col_lb length");
    assert_eq!(lp.col_ub.len(), lp.n_cols, "col_ub length");
    assert_eq!(lp.obj.len(), lp.n_cols, "obj length");

    unsafe {
        let p = glp_create_prob();

        if lp.n_rows > 0 {
            glp_add_rows(p, lp.n_rows as c_int);
            for i in 1..=lp.n_rows as c_int {
                // Mass balance: S·v = 0 → every material row fixed at 0.
                glp_set_row_bnds(p, i, GLP_FX, 0.0, 0.0);
            }
        }

        if lp.n_cols > 0 {
            glp_add_cols(p, lp.n_cols as c_int);
            for j in 0..lp.n_cols {
                let (lo, hi) = (lp.col_lb[j], lp.col_ub[j]);
                let t = col_bound_type(lo, hi);
                glp_set_col_bnds(p, (j + 1) as c_int, t, lo, hi);
                glp_set_obj_coef(p, (j + 1) as c_int, lp.obj[j]);
            }
        }

        // 1-indexed COO arrays (element 0 unused), exactly as glp_load_matrix wants.
        let ne = lp.entries.len();
        let mut ia = Vec::with_capacity(ne + 1);
        let mut ja = Vec::with_capacity(ne + 1);
        let mut ar = Vec::with_capacity(ne + 1);
        ia.push(0);
        ja.push(0);
        ar.push(0.0);
        for &(r, c, v) in &lp.entries {
            ia.push((r + 1) as c_int);
            ja.push((c + 1) as c_int);
            ar.push(v);
        }
        if ne > 0 {
            glp_load_matrix(p, ne as c_int, ia.as_ptr(), ja.as_ptr(), ar.as_ptr());
        }

        let mut parm: glp_smcp = std::mem::zeroed();
        glp_init_smcp(&mut parm); // GLPK's own defaults (version-independent)
        parm.msg_lev = GLP_MSG_ERR;
        parm.it_lim = SIMPLEX_ITERATION_LIMIT;

        glp_set_obj_dir(p, if lp.maximize { GLP_MAX } else { GLP_MIN });
        glp_simplex(p, &parm);

        let status_optimal = glp_get_status(p) == GLP_OPT;
        let objective = glp_get_obj_val(p);
        let col_primals = (1..=lp.n_cols as c_int)
            .map(|j| glp_get_col_prim(p, j))
            .collect();
        let col_duals = (1..=lp.n_cols as c_int)
            .map(|j| glp_get_col_dual(p, j))
            .collect();
        let row_duals = (1..=lp.n_rows as c_int)
            .map(|i| glp_get_row_dual(p, i))
            .collect();

        glp_delete_prob(p);

        LpSolution {
            status_optimal,
            objective,
            col_primals,
            col_duals,
            row_duals,
        }
    }
}
