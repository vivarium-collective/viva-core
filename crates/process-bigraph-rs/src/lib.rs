//! Scheduler, tick loop and emitters mirroring process-bigraph 1.8.3's
//! execution order (`composite.py` run loop + `scheduling.py` step layering).

pub mod emitter;
pub mod engine;
pub mod parallel;
pub mod process;
pub mod registry;
pub mod schedule;
