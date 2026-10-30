//! General native-process registry.
//!
//! A native process implements the engine [`Process`] trait directly, so the
//! Rust engine runs it with NO Python callback. A domain process registers in
//! [`NATIVE_PROCESSES`] at the SAME `local:<module>.<Class>` address as its
//! Python twin, ahead of the [`crate::pyproc::PyProcess`] fallback, and is used
//! only when enabled by the `VIVA_NATIVE_PROCESSES` environment variable (so one
//! `.pbg` runs either path). The registry ships empty: the engine is
//! domain-agnostic and runs Python processes unless something is registered.

use bigraph_schema_rs::doc::Value;
use process_bigraph_rs::process::{Process, ProcessError};

/// Constructor of a native process from the edge config.
pub type NativeCtor = fn(Option<&Value>) -> Result<Box<dyn Process>, ProcessError>;

/// `(address, class name, constructor)` for every registered native process.
/// Empty in the general engine: domain crates/processes add their entries here.
pub const NATIVE_PROCESSES: &[(&str, &str, NativeCtor)] = &[];

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
