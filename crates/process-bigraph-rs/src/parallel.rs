//! Rayon ensemble runner. Engines hold non-`Send` boxed processes, so each
//! member's engine is built *and* run inside one rayon worker via `build`;
//! only the final (`Send`) `State` crosses threads. Members share nothing, so
//! results are identical to a serial run and independent of thread count.

use crate::engine::{Engine, EngineError};
use bigraph_schema_rs::state::State;
use rayon::prelude::*;

/// Build + run one engine per member for `n_ticks`, in parallel. Output order
/// matches `members`.
pub fn run_ensemble<T, F>(
    members: &[T],
    n_ticks: usize,
    build: F,
) -> Result<Vec<State>, EngineError>
where
    T: Sync,
    F: Fn(&T) -> Result<Engine, EngineError> + Sync,
{
    members
        .par_iter()
        .map(|m| {
            let mut e = build(m)?;
            e.run(n_ticks)?;
            Ok(e.state)
        })
        .collect()
}
