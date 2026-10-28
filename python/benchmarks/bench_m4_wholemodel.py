"""M4 Task 3 — the honest whole-model benchmark.

This measures what can actually be measured on the REAL v2ecoli baseline composite
(one generation, baseline media, single seed), NOT an optimistic micro-benchmark.

Three numbers, all RELEASE wheel, warm-up excluded, per-tick cost isolated by the
differential method (time at N1 and N2 ticks; per-tick = (t(N2)-t(N1))/(N2-N1),
which cancels the one-time engine-build / realize cost):

  (A) pb WHOLE MODEL  — the full 46-edge baseline agent under pb 1.8.3. This is
      the reference the Rust engine must eventually beat. (The whole model cannot
      run on run_prebuilt yet — see docs/milestone-4-results.md: pint units in
      listeners/environment + the partition process-as-data store. So there is NO
      whole-model engine number; this is the honest status.)

  (B) real ProteinDegradation at FULL baseline scale (16321-wide real bulk, live
      sim_data), the one ported process, under:
        pb                 — Python pb engine + Python process (reference)
        run_prebuilt OFF   — Rust engine + the PYTHON instance (quadrant 3)
        run_prebuilt ON    — Rust engine + the NATIVE Rust twin (quadrant 2)
      Bit-identity is asserted before timing.

The headline honest findings (see the results doc): running a Python process on
the Rust engine (OFF) is SLOWER than pb — the PyO3 view/update boundary costs more
than pb's in-process scheduling; the speedup comes only from the NATIVE port (ON).
With 1 of ~55 processes ported, the projected whole-model speedup today is < 1x.

Run (release wheel):
  <venv>/bin/python python/benchmarks/bench_m4_wholemodel.py
"""

import copy
import json
import os
import sys
import time

# Bridge the read-only v2ecoli venv + checkout (benchmarks run outside pytest).
_ROOT = os.environ.get("V2ECOLI_ROOT", "/Users/eranagmon/code/v2ecoli")
for _p in (os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"), _ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import numpy as np  # noqa: E402

import viva_core  # noqa: E402
from viva_core import prebuilt  # noqa: E402
from viva_core.conformance_processes import TickTrigger  # noqa: E402
from process_bigraph import Composite  # noqa: E402
from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle  # noqa: E402
from v2ecoli.core import build_core  # noqa: E402
from v2ecoli.library.ecoli_step import set_current_core  # noqa: E402

PD_ADDR = "local:v2ecoli.processes.protein_degradation.ProteinDegradation"
CACHE = os.environ.get("V2ECOLI_CACHE", "/Users/eranagmon/code/v2ecoli/out/cache")

core = build_core()
set_current_core(core)
bundle = load_cache_bundle(CACHE)


def _strip(w):
    if isinstance(w, dict):
        return {k: _strip(v) for k, v in w.items() if not str(k).startswith("_layer")}
    return w


def _full_agent_doc(seed=0):
    return baseline(core, seed=seed, cache_dir=CACHE, bundle=bundle)


def _pd_doc(seed=0):
    agent = baseline(core, seed=seed, cache_dir=CACHE, bundle=bundle)["state"]["agents"]["0"]
    pd = dict(agent["ecoli-protein-degradation"])
    pd["inputs"] = _strip(pd["inputs"])
    pd["outputs"] = _strip(pd["outputs"])
    tick = TickTrigger(config={"dt": 1.0, "value": 1}, core=core)
    return {
        "global_time": 0.0, "timestep": 1, "bulk": agent["bulk"],
        "tick": {"_type": "process",
                 "address": "local:viva_core.conformance_processes.TickTrigger",
                 "config": {"dt": 1.0, "value": 1}, "instance": tick,
                 "inputs": {}, "outputs": {"timestep": ["timestep"]}},
        "ecoli-protein-degradation": pd,
    }


def _bulk(state):
    b = state["bulk"]
    if hasattr(b, "dtype") and getattr(b.dtype, "names", None):
        return np.asarray(b["count"]).tolist()
    return list(b["count"])


def _best(fn, reps=5):
    return min(fn() for _ in range(reps))


# ---- bit-identity gate (OFF + ON vs pb) before timing --------------------- #
def _check_bit_identical(n=15):
    ref_sim = Composite({"state": copy.deepcopy(_pd_doc())}, core=core)
    ref_sim.run(float(n))
    ref = _bulk(ref_sim.state)
    off = _bulk(prebuilt.run_from_state(_pd_doc(), n_ticks=n))
    os.environ["VIVA_NATIVE_PROCESSES"] = "ProteinDegradation"
    try:
        on = _bulk(prebuilt.run_from_state(_pd_doc(), n_ticks=n, native_addresses=[PD_ADDR]))
    finally:
        os.environ.pop("VIVA_NATIVE_PROCESSES", None)
    assert off == ref, "run_prebuilt native-OFF diverged from pb"
    assert on == ref, "run_prebuilt native-ON diverged from pb"
    changed = int(np.count_nonzero(np.array(ref) - np.array(_bulk(_pd_doc()))))
    return changed


# ---- (A) pb whole-model reference ---------------------------------------- #
def _pb_wholemodel_per_tick(n1=3, n2=8):
    def run_to(n):
        comp = Composite(_full_agent_doc(), core=core)  # realize (excluded by diff)
        t = time.time(); comp.run(float(n)); return time.time() - t
    run_to(n2)  # warm (first tick JITs FBA etc.)
    t1 = _best(lambda: run_to(n1), reps=3)
    t2 = _best(lambda: run_to(n2), reps=3)
    return (t2 - t1) / (n2 - n1)


# ---- (B) ProteinDegradation at full scale -------------------------------- #
def _pb_pd_per_tick(n1=100, n2=600):
    def run_to(n):
        comp = Composite({"state": _pd_doc()}, core=core)
        t = time.time(); comp.run(float(n)); return time.time() - t
    run_to(n2)
    t1 = _best(lambda: run_to(n1)); t2 = _best(lambda: run_to(n2))
    return (t2 - t1) / (n2 - n1), t1 - (t2 - t1) / (n2 - n1) * n1


def _hand_pd_per_tick(native, n1=100, n2=600):
    nat = [PD_ADDR] if native else None

    def run_to(n):
        doc = _pd_doc()
        if native:
            os.environ["VIVA_NATIVE_PROCESSES"] = "ProteinDegradation"
        try:
            procs, tj, sj, handles = prebuilt.build_payload(doc, nat)
            t = time.time(); viva_core.run_prebuilt(procs, tj, sj, n, nat, handles); return time.time() - t
        finally:
            os.environ.pop("VIVA_NATIVE_PROCESSES", None)
    run_to(n2)
    t1 = _best(lambda: run_to(n1)); t2 = _best(lambda: run_to(n2))
    per = (t2 - t1) / (n2 - n1)
    return per, t1 - per * n1  # (per-tick, one-time engine-build)


def main():
    print("=" * 72)
    print("M4 whole-model benchmark — REAL v2ecoli baseline (1 gen, minimal, seed 0)")
    print("release wheel; warm-up excluded; per-tick via differential method")
    print("=" * 72)

    changed = _check_bit_identical()
    print(f"\n[gate] ProteinDegradation OFF & ON bit-identical to pb "
          f"({changed} bulk counts moved) — PASS\n")

    pb_wm = _pb_wholemodel_per_tick()
    print("(A) WHOLE MODEL (46 edges, ~55 processes)")
    print(f"    pb 1.8.3          : {pb_wm*1000:8.3f} ms/tick   {1/pb_wm:8.1f} ticks/s")
    print("    run_prebuilt      : BLOCKED (not measurable) — pint units in")
    print("                        listeners/environment + partition process-as-data")
    print("                        store cannot cross the JSON state channel.")
    print("                        See docs/milestone-4-results.md.\n")

    pb_pd, pb_realize = _pb_pd_per_tick()
    off, off_build = _hand_pd_per_tick(False)
    on, on_build = _hand_pd_per_tick(True)
    print("(B) ProteinDegradation @ full baseline scale (16321-wide real bulk)")
    print(f"    pb 1.8.3          : {pb_pd*1000:8.4f} ms/tick   {1/pb_pd:9.1f} ticks/s")
    print(f"    run_prebuilt OFF  : {off*1000:8.4f} ms/tick   {1/off:9.1f} ticks/s   "
          f"x{pb_pd/off:.2f} vs pb   (engine-build {off_build*1000:.0f}ms once)")
    print(f"    run_prebuilt ON   : {on*1000:8.4f} ms/tick   {1/on:9.1f} ticks/s   "
          f"x{pb_pd/on:.2f} vs pb   (engine-build {on_build*1000:.0f}ms once)")
    print(f"    native ON vs OFF  : x{off/on:.2f} per-tick\n")

    print("Honest reading:")
    print(f"  * Python-on-Rust (OFF) is {'slower' if off > pb_pd else 'faster'} than pb "
          f"(x{pb_pd/off:.2f}); the engine boundary costs more than in-process Python.")
    print(f"  * the NATIVE port (ON) is x{pb_pd/on:.2f} vs pb — the real win is the port.")
    print("  * with 1 of ~55 processes ported, a whole-model run today would be")
    print("    dominated by OFF-class Python processes -> projected < 1x vs pb.")


if __name__ == "__main__":
    main()
