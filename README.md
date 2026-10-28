# viva-core

A general-purpose, high-performance **Rust** reimplementation of **bigraph-schema + process-bigraph**, byte-compatible with the existing bigraph-schema JSON contract so the Python viva-ecosystem plugs in unchanged.

- **Why:** the Python stack is dynamic-dispatch-bound (~300 `plum` dispatches/tick); a static-dispatch Rust engine ran the same work ~160× faster in a spike.
- **Interchangeable:** Python↔Rust in both directions — run existing Python processes on the Rust engine, and run Rust-ported processes under the Python engine (both spike-proven, bit-identical).
- **Parallel + distributed:** `rayon` threads, `rsmpi`→system MPI (Slurm, ensemble-level).
- **Packaging:** one pip-installable `viva-core` wheel (PyO3/maturin) from a Cargo workspace (`bigraph-schema-rs` + `process-bigraph-rs` + binding crate).

Design: `docs/superpowers/specs/2026-10-27-rust-core-design.md`.

Status: Milestone 1 (engine MVP) in progress.
