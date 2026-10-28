# viva-core — Milestone 2 results (zero-copy boundary)

**Date:** 2026-10-28
**Branch:** `m2-zerocopy`
**Scope:** make the bulk and unique structured-array read views zero-copy
(Task 1: bulk `count` view; Task 2: unique store as an interleaved record
buffer exposed as a read-only numpy view). Writes still go through the Op path.
Benchmark: `python/benchmarks/bench_m1.py`, 2000 ticks, best-of-5, warm-up
excluded, this Mac. Ratio = Rust engine + Python processes / Python engine +
Python processes (same composite, same processes).

## 1. Conformance (no regression): all bit-identical

| Gate | Result |
|---|---|
| `cargo test --workspace` | all green (oracle-backed tests incl.) |
| `pytest python/tests/` (quadrant 3/4, conformance incl. bulk + unique zero-copy) | 12 passed |
| `python -m viva_core.upstream_suite` run()-level battery | 6/6, 0 real divergences |
| `cargo fmt --check`, `cargo clippy --workspace` | clean |

Pinned target unchanged: bigraph-schema 1.6.0 / process-bigraph 1.8.3.

## 2. Benchmark, before -> after

Build mode matters. M1's published numbers were measured on a **debug**
build. The M2 numbers below are both **release** (headline) and **debug**
(like-for-like with M1), measured on the same M2 code.

| Composite | M1 (debug) | M2, debug build | M2, release build |
|---|---|---|---|
| A. Bulk, n=200 | 0.066x | 0.13x | **~1.16-1.20x** (110k vs 93-95k tps) |
| A2. Unique, n=2000 | 0.011x (n=2000/13000 per M2 Task 2) | ~0.68x | **~1.35-1.42x** (150k vs 106-111k tps) |
| B. Scalar | 1.27x | ~1.32x | **~6.0-6.6x** (840-905k vs ~138k tps) |

Release numbers are from three consecutive runs; the ranges above are the
spread. Debug numbers are one run.

## 3. Honest reading

* The copy is gone. Unique went from 0.011x to ~0.68x on the like-for-like
  debug build (about 60x for the engine side), and the bulk view no longer
  copies either.
* On the **release** build, every composite measured here is faster than the
  Python engine: bulk ~1.2x, unique ~1.4x, scalar ~6x. So the M2 goal (flip
  the array-heavy hot path from net slowdown) is achieved on a release build.
* The earlier "unique ~0.66x" figure was a debug-build number. It should not be
  quoted as the release result. Part of the M1->M2 improvement in the table is
  the debug->release change, not only zero-copy; the like-for-like debug
  column isolates the zero-copy effect (bulk 0.066x -> 0.13x, unique
  0.011x -> ~0.68x, scalar ~unchanged).
* The scalar 6x is much larger than M1's 1.27x. That is almost entirely a
  release-build effect on the Rust engine/PyO3 callback path (debug scalar is
  1.3x). It is not a zero-copy result, and no scalar code changed in M2.
* Caveats: these are micro-composites with one trivial Python process, so the
  Python process body is cheap and the engine overhead dominates the ratio. A
  real v2ecoli tick spends most of its time inside process bodies, so
  whole-model speedups will be far smaller than these ratios. Not measured yet.
* Residual in debug builds (unique ~0.68x) is view-construction overhead, not a
  copy; pooling/reusing views is a possible further lever but is not needed to
  beat the Python engine on release.

## 4. Status

* Conformance: bit-identical, no regressions.
* Day-one numbers (release): bulk ~1.2x, unique ~1.4x, scalar ~6x.
* Next: process-port milestone (elongation + RNG parity; GLPK FFI), and a
  whole-model benchmark on a real v2ecoli composite.
