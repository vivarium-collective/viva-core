"""M5 Task 4 — the honest WHOLE-MODEL benchmark.

Unlike M4 (where the whole model could NOT run on the Rust engine, so there was
no engine number — only pb), M5 makes the whole realized baseline RUN on the Rust
engine via ``run_prebuilt`` native-swap OFF. So both engines are measurable here:

  (A) pb 1.8.3            — the full ~55-process baseline agent under pb.
  (B) run_prebuilt OFF    — the SAME agent, every process as its pre-built Python
                            instance on the Rust engine (via the realized-Composite
                            handoff, path=("agents","0")).

Per-tick cost is isolated by the differential method (time at N1 and N2 ticks;
per-tick = (t(N2)-t(N1))/(N2-N1)), which cancels the one-time realize / engine-
build cost. Warm-up (the first-tick FBA/ODE/JIT compile) is excluded.

Honest expectation (and the M5 reading): run_prebuilt OFF is SLOWER than pb
(<1x). M5 is an ENABLEMENT milestone — it makes the whole model run on the Rust
engine; it does not make it faster. Every Python process still pays a PyO3
view/update crossing per tick, on top of pb's in-process scheduling. The speedup
is the M6+ native-port program (porting the hot processes to Rust twins).

Also note (printed): the run is NOT yet bit-identical to pb — a ~1e-7 relative
bulk-total divergence after two ticks, deterministic, rooted in float-precision
sensitivity in the early metabolic/stochastic processes (see
docs/milestone-5-results.md). The speed number stands regardless; this is the
honest enablement measurement, not a correctness claim.

Run (release wheel):
  V2ECOLI_CACHE=<cache> PYTHONPATH=<v2ecoli@main>:<v2ecoli venv site-packages> \
    <venv>/bin/python python/benchmarks/bench_m5_wholemodel.py
"""

import os
import sys
import time

_ROOT = os.environ.get("V2ECOLI_ROOT", "/Users/eranagmon/code/v2ecoli")
for _p in (os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"), _ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import numpy as np  # noqa: E402

from viva_core import prebuilt  # noqa: E402
from process_bigraph import Composite  # noqa: E402
from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle  # noqa: E402
from v2ecoli.core import build_core  # noqa: E402
from v2ecoli.library.ecoli_step import set_current_core  # noqa: E402

CACHE = os.environ.get("V2ECOLI_CACHE", "/Users/eranagmon/code/v2ecoli/out/cache")

core = build_core()
set_current_core(core)
bundle = load_cache_bundle(CACHE)


def _fresh():
    return Composite(baseline(core, seed=0, cache_dir=CACHE, bundle=bundle), core=core)


def _bulk_count(state):
    return np.asarray(state["bulk"]["count"])


# ---- (A) pb whole-model per-tick ----------------------------------------- #
def _pb_per_tick(n1=1, n2=3):
    def run_to(n):
        comp = _fresh()  # realize (excluded by the differential)
        t = time.time()
        comp.run(float(n))
        return time.time() - t

    run_to(n2)  # warm (first tick JITs FBA/ODE/numba)
    t1 = min(run_to(n1) for _ in range(2))
    t2 = min(run_to(n2) for _ in range(2))
    return (t2 - t1) / (n2 - n1)


# ---- (B) run_prebuilt OFF whole-model per-tick --------------------------- #
def _rust_per_tick(n1=1, n2=3):
    def run_to(n):
        comp = _fresh()
        procs, topo, state_json, handles = prebuilt.build_payload(comp, path=("agents", "0"))
        t = time.time()
        import viva_core
        viva_core.run_prebuilt(procs, topo, state_json, n, None, handles)
        return time.time() - t

    run_to(n2)
    t1 = min(run_to(n1) for _ in range(2))
    t2 = min(run_to(n2) for _ in range(2))
    return (t2 - t1) / (n2 - n1)


def _divergence(n=2):
    ref = _fresh()
    ref.run(float(n))
    ref_sum = int(_bulk_count(ref.state["agents"]["0"]).sum())
    got_sum = int(_bulk_count(prebuilt.run_from_state(_fresh(), n_ticks=n, path=("agents", "0"))).sum())
    return abs(ref_sum - got_sum) / max(1, abs(ref_sum))


def main():
    print("=" * 72)
    print("M5 whole-model benchmark — REAL v2ecoli baseline (1 gen, minimal, seed 0)")
    print("release wheel; warm-up excluded; per-tick via differential method")
    print("=" * 72)

    pb = _pb_per_tick()
    rust = _rust_per_tick()
    print("\nWHOLE MODEL (~55 processes, native-swap OFF)")
    print(f"  pb 1.8.3          : {pb*1000:9.2f} ms/tick   {1/pb:8.2f} ticks/s")
    print(f"  run_prebuilt OFF  : {rust*1000:9.2f} ms/tick   {1/rust:8.2f} ticks/s   "
          f"x{pb/rust:.2f} vs pb")
    print("\nHonest reading:")
    print(f"  * the whole model now RUNS on the Rust engine (M5 enablement); it is")
    print(f"    x{pb/rust:.2f} vs pb — {'slower' if rust > pb else 'faster'}, as expected: every")
    print("    process pays a PyO3 crossing per tick. The speedup is the M6+ ports.")
    rel = _divergence()
    print(f"  * NOT bit-identical: bulk-total relative divergence {rel:.2e} after 2 ticks,")
    print("    deterministic (float-precision residual in metabolism/elongation).")


if __name__ == "__main__":
    main()
