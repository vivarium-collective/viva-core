//! Native (all-Rust) ports of v2ecoli processes (quadrant 2).
//!
//! Each port implements the engine [`Process`] trait directly, so the Rust
//! engine runs it with NO Python callback. A port is registered at the SAME
//! `local:<module>.<Class>` address as its Python twin, ahead of the
//! [`crate::pyproc::PyProcess`] fallback, but only when enabled by the
//! `VIVA_NATIVE_PROCESSES` environment variable (so one `.pbg` runs either path).

use bigraph_schema_rs::doc::Value;
use process_bigraph_rs::process::{Process, ProcessError};

pub mod protein_degradation;

// Metabolism's native FBA solve links GLPK (a hard C dependency), so the whole
// module — and its registration entry — is gated behind the `glpk` cargo feature.
// The default wheel build does not require GLPK; build `--features glpk` to enable
// the native metabolism factory.
#[cfg(feature = "glpk")]
pub mod metabolism;

/// Constructor of a native process from the edge config.
pub type NativeCtor = fn(Option<&Value>) -> Result<Box<dyn Process>, ProcessError>;

/// `(address, class name, constructor)` for every native port.
#[cfg(not(feature = "glpk"))]
pub const NATIVE_PROCESSES: &[(&str, &str, NativeCtor)] = &[(
    "local:v2ecoli.processes.protein_degradation.ProteinDegradation",
    "ProteinDegradation",
    protein_degradation::ProteinDegradationCore::create,
)];

/// `(address, class name, constructor)` for every native port (GLPK enabled:
/// adds the metabolism FBA factory at the metabolism `local:` address).
#[cfg(feature = "glpk")]
pub const NATIVE_PROCESSES: &[(&str, &str, NativeCtor)] = &[
    (
        "local:v2ecoli.processes.protein_degradation.ProteinDegradation",
        "ProteinDegradation",
        protein_degradation::ProteinDegradationCore::create,
    ),
    (
        "local:v2ecoli.processes.metabolism.Metabolism",
        "Metabolism",
        metabolism::MetabolismCore::create,
    ),
];

/// Whether `VIVA_NATIVE_PROCESSES` enables `class_name`. The variable is a
/// comma-separated list of class names, or `all` / `1`. Unset/empty = off.
pub fn native_enabled(class_name: &str) -> bool {
    match std::env::var("VIVA_NATIVE_PROCESSES") {
        Ok(v) => v
            .split(',')
            .map(str::trim)
            .any(|t| t == class_name || t == "all" || t == "1"),
        Err(_) => false,
    }
}

/// The native constructor for `address`, if one exists and is enabled.
pub fn native_ctor(address: &str) -> Option<NativeCtor> {
    NATIVE_PROCESSES
        .iter()
        .find(|(a, name, _)| *a == address && native_enabled(name))
        .map(|(_, _, c)| *c)
}
