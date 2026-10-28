"""Honest day-one benchmark for viva-core Milestone 1 (Task 8).

Measures the SAME composite under:
  * the Rust engine + Python processes (quadrant 3: `viva_core.run`), and
  * the Python pb 1.8.3 engine + the same Python processes (`Composite.run`),
warm-up excluded, and reports ticks/sec for each plus the ratio.

CRITICAL HONESTY (carried from the Task-5 review, spec S7):
  * The engine-swap day-one number is a *modest* win whose whole point is
    interchangeability + parallelism + a migration path, NOT dramatic speed.
  * Zero-copy is NOT achieved for structured arrays: `build_structured_array`
    rebuilds a fresh numpy structured array for every bulk/unique view every
    tick. So this number is a PESSIMISTIC UPPER BOUND on the Rust-engine
    overhead (equivalently a LOWER BOUND on the day-one speedup) — the true
    engine-swap value improves once the array boundary is made zero-copy.
  * The full-native ceiling (Rust engine + Rust processes) is NOT measurable
    through the M1 `run()` surface: viva_core.run drives Python processes only;
    there is no Python entry point that registers native Rust processes into a
    run() composite in M1. So only the day-one (Q3) number is reported here.

Run:
  cd /Users/eranagmon/code/viva-core && \
    VIRTUAL_ENV=$PWD/.venv-m1 PATH=$PWD/.venv-m1/bin:$PATH \
    .venv-m1/bin/python python/benchmarks/bench_m1.py
"""

from __future__ import annotations

import time

import numpy as np

import viva_core
from viva_core import conformance as C


def _bench_doc(n_bulk: int) -> dict:
    dt = np.dtype([("id", "<U12"), ("count", "int64")])
    bulk = np.zeros(n_bulk, dtype=dt)
    bulk["id"] = [f"MOL-{i}" for i in range(n_bulk)]
    bulk["count"] = [1000 + i for i in range(n_bulk)]
    return {
        "global_time": 0.0,
        "bulk": bulk,
        "bench": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.BenchBulkProcess",
            "config": {"dt": 1.0},
            "inputs": {"bulk": ["bulk"]},
            "outputs": {"bulk": ["bulk"]},
        },
    }


def _bench_doc_scalar() -> dict:
    """Scalar-only composite: NO structured array crosses the boundary, so it
    isolates engine + PyO3-callback overhead without the per-tick array copy."""
    return {
        "global_time": 0.0,
        "value": 0.0,
        "bench": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.BenchScalarProcess",
            "config": {"dt": 1.0},
            "inputs": {"value": ["value"]},
            "outputs": {"value": ["value"]},
        },
    }


def _time_viva(spec: str, n_ticks: int, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        viva_core.run(spec, n_ticks)
        best = min(best, time.perf_counter() - t0)
    return best


def _time_pb(doc_fn, until: float, repeats: int, register) -> float:
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.library.ecoli_step import set_current_core
    best = float("inf")
    for _ in range(repeats):
        core = build_core()
        set_current_core(core)
        register(core)
        doc = doc_fn()
        sim = Composite({"state": doc}, core=core)
        t0 = time.perf_counter()
        sim.run(until)
        best = min(best, time.perf_counter() - t0)
    return best


def _measure(doc_fn, address_cls, n_ticks: int, warmup: int, repeats: int) -> dict:
    from viva_core import conformance_processes as cp
    name, cls = address_cls

    def register(core):
        core.register_link(f"viva_core.conformance_processes.{name}", getattr(cp, name))

    spec = C.spec_json(doc_fn())
    for _ in range(warmup):
        viva_core.run(spec, 50)
    _time_pb(doc_fn, 50.0, warmup, register)

    rust_t = _time_viva(spec, n_ticks, repeats)
    pb_t = _time_pb(doc_fn, float(n_ticks), repeats, register)
    rust_tps, pb_tps = n_ticks / rust_t, n_ticks / pb_t
    return {
        "rust_tps": rust_tps,
        "pb_tps": pb_tps,
        "ratio": rust_tps / pb_tps,
    }


def run(n_ticks: int = 2000, n_bulk: int = 200, warmup: int = 2, repeats: int = 5) -> dict:
    bulk = _measure(lambda: _bench_doc(n_bulk), ("BenchBulkProcess", None),
                    n_ticks, warmup, repeats)
    scalar = _measure(_bench_doc_scalar, ("BenchScalarProcess", None),
                      n_ticks, warmup, repeats)
    return {"n_ticks": n_ticks, "n_bulk": n_bulk, "repeats": repeats,
            "bulk": bulk, "scalar": scalar}


def main() -> None:
    r = run()
    print("=== viva-core Milestone 1 - day-one benchmark (HONEST) ===")
    print(f"{r['n_ticks']} ticks, best-of-{r['repeats']}, warm-up excluded")
    print()
    print("A. Structured-array composite (1 Python bulk-evolver, "
          f"bulk n={r['n_bulk']}) - the per-tick build_structured_array copy:")
    print(f"     Rust engine   + Python procs : {r['bulk']['rust_tps']:>12.1f} ticks/s")
    print(f"     Python engine + Python procs : {r['bulk']['pb_tps']:>12.1f} ticks/s")
    print(f"     day-one ratio (Rust / Py)    : {r['bulk']['ratio']:>12.3f}x "
          "(UPPER BOUND on overhead / LOWER BOUND on speedup)")
    print()
    print("B. Scalar-only composite (no array crosses the boundary) - "
          "pure engine + PyO3-callback delta:")
    print(f"     Rust engine   + Python procs : {r['scalar']['rust_tps']:>12.1f} ticks/s")
    print(f"     Python engine + Python procs : {r['scalar']['pb_tps']:>12.1f} ticks/s")
    print(f"     ratio (Rust / Py)            : {r['scalar']['ratio']:>12.3f}x")
    print()
    print("  Zero-copy for structured arrays is NOT done; A is dominated by that")
    print("  copy and is the pessimistic bound. The full-native ceiling (Rust")
    print("  engine + Rust processes) is NOT measurable via the M1 run() surface.")
    print("  See docs/milestone-1-results.md.")


if __name__ == "__main__":
    main()
