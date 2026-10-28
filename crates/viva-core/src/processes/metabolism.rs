//! Native Rust port of the v2ecoli `Metabolism` (FBA) process — the **core
//! flux-balance solve path**, bit-identical to the Python twin
//! (`v2ecoli/processes/metabolism.py` + `wholecell/utils/modular_fba.py` +
//! `_netflow/nf_glpk.py`).
//!
//! # What is ported (bit-identical, tested)
//!
//! The two pieces that are the actual per-tick *compute*:
//!
//!  1. **The GLPK LP solve** ([`crate::glpk::solve`]) — the single biggest
//!     per-tick cost. Native `extern "C"` GLPK with swiglpk-matched simplex
//!     parameters, proven bit-identical to swiglpk on the real netflow-assembled
//!     LP (objective + primal fluxes + duals).
//!  2. **The flux → integer bulk-count conversion** ([`flux_to_bulk_delta`]) —
//!     `metabolite_counts_final = fmax(stochasticRound(n₀ + Δc/counts_to_molar), 0)`
//!     then `Δn = counts_final − n₀`, using the shared numpy-parity
//!     `stochastic_round` (one batched `rand` draw, C order) so the integer bulk
//!     delta is int-exact.
//!
//! # What is deferred (and why) — honest scope
//!
//! A full native drop-in `Metabolism` engine `Process` is **not** achievable
//! bit-identically in this task, because — unlike protein degradation (M3 Task 2)
//! — metabolism's hot path is **not** fed by serialized config. Its
//! `config_schema` carries live Python handles (`metabolism` is `'node'`;
//! `exchange_data_from_media`, `get_biomass_as_concentrations`,
//! `get_import_constraints`, `get_masses`, `get_ppGpp_conc` are `'method'`), and
//! `_do_update` calls them **every tick** through the pint/Unum unit bridge to
//! (re)assemble the LP bounds and objective. Reproducing that assembly natively
//! means porting `modular_fba.py` (1.6k LOC), the netflow builder, several
//! sim_data methods, and the unit bridge — far beyond one task, and impossible to
//! drive purely from the JSON edge `config`. (This contradicts the M3 design's
//! Crux-2 "fully serializable config" assumption, which holds for protein
//! degradation but NOT for metabolism.)
//!
//! So this port isolates and proves the **solve** — the dominant compute — is
//! portable and bit-identical, and registers the native factory plumbing so the
//! address is claimed under the flag. The engine-level [`Process::update`] is
//! **fail-loud**: flag on → a clear error that LP assembly is deferred (never a
//! silent divergence); flag off → the Python path is unchanged.

use bigraph_schema_rs::doc::{Map, Value};
use bigraph_schema_rs::rng::RandomState;
use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};

fn err(msg: impl Into<String>) -> ProcessError {
    ProcessError(msg.into())
}

/// Convert an FBA concentration-change solution to an integer bulk-count delta,
/// bit-identically to `Metabolism._fba_output_to_deltas` (the metabolite branch):
///
/// ```text
/// delta_mag       = output_molecule_changes / counts_to_molar     # CONC_UNITS cancels
/// counts_final    = fmax(stochasticRound(rs, counts_init + delta_mag), 0).astype(int64)
/// delta_metabolites = counts_final - counts_init
/// ```
///
/// `stochastic_round` draws **one** batched `rand(n)` in C order (stream order
/// matters), so the returned integers match numpy exactly for a matched
/// `RandomState`. Returns `(counts_final, delta_metabolites)`.
pub fn flux_to_bulk_delta(
    output_molecule_changes: &[f64],
    counts_init: &[i64],
    counts_to_molar: f64,
    rng: &mut RandomState,
) -> (Vec<i64>, Vec<i64>) {
    let n = counts_init.len();
    assert_eq!(output_molecule_changes.len(), n, "length mismatch");
    let values: Vec<f64> = (0..n)
        .map(|i| counts_init[i] as f64 + output_molecule_changes[i] / counts_to_molar)
        .collect();
    let rounded = rng.stochastic_round(&values);
    let counts_final: Vec<i64> = rounded
        .iter()
        .map(|&r| r.max(0.0) as i64) // np.fmax(., 0) then .astype(int64) (truncation)
        .collect();
    let delta: Vec<i64> = (0..n).map(|i| counts_final[i] - counts_init[i]).collect();
    (counts_final, delta)
}

/// Native metabolism process. Holds the numpy-parity RNG (seeded as the Python
/// twin's `np.random.RandomState(seed)`); the LP solve is [`crate::glpk::solve`].
///
/// The per-tick LP *assembly* is deferred (see module docs), so [`Process::update`]
/// is fail-loud under the flag rather than silently divergent.
pub struct MetabolismCore {
    #[allow(dead_code)]
    rng: RandomState,
}

impl MetabolismCore {
    /// Engine-registry constructor (the serialized metabolism config).
    pub fn create(config: Option<&Value>) -> Result<Box<dyn Process>, ProcessError> {
        Ok(Box::new(Self::from_config(config)?))
    }

    pub fn from_config(config: Option<&Value>) -> Result<Self, ProcessError> {
        let empty = Map::new();
        let cfg = match config {
            Some(Value::Map(m)) => m,
            None | Some(Value::Null) => &empty,
            Some(_) => return Err(err("Metabolism config must be a map")),
        };
        let seed = match cfg.get("seed") {
            Some(Value::Int(i)) => u32::try_from(*i)
                .map_err(|_| err(format!("seed {i} out of numpy RandomState range")))?,
            None => 0, // config_schema default
            Some(_) => return Err(err("seed must be an integer")),
        };
        Ok(Self {
            rng: RandomState::new(seed),
        })
    }
}

impl Process for MetabolismCore {
    fn inputs(&self) -> Ports {
        vec![
            ("bulk".into(), "bulk_array".into()),
            ("timestep".into(), "integer".into()),
        ]
    }

    fn outputs(&self) -> Ports {
        vec![("bulk".into(), "bulk_array".into())]
    }

    fn update(
        &mut self,
        _view: &PortView,
        _interval: f64,
    ) -> Result<Vec<PortUpdate>, ProcessError> {
        // Fail loud: the FBA solve + flux→count conversion are ported and tested
        // (see glpk::solve and flux_to_bulk_delta), but the per-tick LP assembly
        // reads live sim_data methods (not in the serialized config), so a
        // bit-identical engine drop-in is deferred. Never silently diverge.
        Err(err(
            "native Metabolism: per-tick LP assembly is deferred (needs live \
             sim_data methods not in the serialized config); the FBA solve core \
             (glpk::solve) and flux→count conversion are ported and bit-identical. \
             Run metabolism on the Python path (VIVA_NATIVE_PROCESSES off for it).",
        ))
    }
}
