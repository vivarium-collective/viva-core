"""M3 Task 2 benchmark: the first real quadrant-2 (all-native) per-process number.

Same real-size ProteinDegradation composite (4300 proteins x 21 amino acids,
bulk ~16k molecules, the real v2ecoli address + config shape) under:
  * Q3: Rust engine + the PYTHON process (VIVA_NATIVE_PROCESSES unset)
  * Q2: Rust engine + the NATIVE Rust process (VIVA_NATIVE_PROCESSES=ProteinDegradation)
  * pb: the Python pb 1.8.3 engine + the Python process (reference)
Final bulk counts are asserted bit-identical across all three before timing.

Run (release wheel): PYTHONPATH=python <venv>/bin/python python/benchmarks/bench_m3_protein_degradation.py
"""

import os
import time

import numpy as np

import viva_core
from viva_core import conformance as C

ADDR = "local:v2ecoli.processes.protein_degradation.ProteinDegradation"


def build_doc(n_prot=4300, n_aa=21, n_other=12000, seed=7):
    rng = np.random.RandomState(1)
    prot_ids = [f"PROT-{i}[c]" for i in range(n_prot)]
    aa_ids = [f"AA-{j}[c]" for j in range(n_aa)]
    aa_counts = rng.randint(0, 40, size=(n_prot, n_aa)).astype(np.int64)
    counts = rng.randint(0, 6000, size=n_prot).astype(np.int64)
    rates = rng.uniform(1e-5, 8e-4, size=n_prot)
    ids = prot_ids + aa_ids + ["h2o[c]"] + [f"OTHER-{i}[c]" for i in range(n_other)]
    dt = np.dtype([("id", "<U16"), ("count", "int64")])
    bulk = np.zeros(len(ids), dtype=dt)
    bulk["id"] = ids
    bulk["count"] = np.concatenate(
        [counts, np.full(n_aa, 10**8), [10**10], rng.randint(0, 1000, size=n_other)]
    )
    cfg = {"time_step": 1, "raw_degradation_rate": rates.tolist(),
           "water_id": "h2o[c]", "amino_acid_ids": aa_ids,
           "amino_acid_counts": aa_counts.tolist(), "protein_ids": prot_ids,
           "protein_lengths": aa_counts.sum(axis=1).tolist(), "seed": seed}
    return {
        "global_time": 0.0, "timestep": 1, "bulk": bulk,
        "tick": {"_type": "process",
                 "address": "local:viva_core.conformance_processes.TickTrigger",
                 "config": {"dt": 1.0, "value": 1}, "inputs": {},
                 "outputs": {"timestep": ["timestep"]}},
        "protein_degradation": {"_type": "step", "address": ADDR, "config": cfg,
                                "inputs": {"bulk": ["bulk"], "timestep": ["timestep"]},
                                "outputs": {"bulk": ["bulk"]}},
    }


def _run(spec, n, native):
    if native:
        os.environ["VIVA_NATIVE_PROCESSES"] = "ProteinDegradation"
    else:
        os.environ.pop("VIVA_NATIVE_PROCESSES", None)
    try:
        t0 = time.perf_counter()
        out = viva_core.run(spec, n)
        return time.perf_counter() - t0, list(out["bulk"]["count"])
    finally:
        os.environ.pop("VIVA_NATIVE_PROCESSES", None)


def _run_pb(doc, n):
    t0 = time.perf_counter()
    state, _, _ = C.run_reference(doc, float(n))
    return time.perf_counter() - t0, np.asarray(state["bulk"]["count"]).tolist()


def main(n_ticks=300, repeats=5):
    spec = C.spec_json(build_doc())
    _run(spec, 10, False), _run(spec, 10, True)  # warm-up
    _, ref = _run_pb(build_doc(), n_ticks)
    _, py = _run(spec, n_ticks, False)
    _, nat = _run(spec, n_ticks, True)
    assert py == ref and nat == ref, "NOT bit-identical; refusing to report timings"
    # Per-tick cost = (T(n_ticks) - T(1)) / (n_ticks - 1): engine build + spec
    # parse + final-state decode are one-off and cancel out.
    def per_tick(fn):
        tn = min(fn(n_ticks) for _ in range(repeats))
        t1 = min(fn(1) for _ in range(repeats))
        return (tn - t1) * n_ticks / (n_ticks - 1)

    q3 = per_tick(lambda n: _run(spec, n, False)[0])
    q2 = per_tick(lambda n: _run(spec, n, True)[0])
    pb = per_tick(lambda n: _run_pb(build_doc(), n)[0])
    print(f"=== M3 Task 2: ProteinDegradation, {n_ticks} ticks, 4300 proteins, best-of-{repeats} ===")
    print("final bulk bit-identical across pb / Rust+Python / Rust+native: yes")
    for name, t in (("pb engine + Python proc", pb), ("Rust engine + Python proc (Q3)", q3),
                    ("Rust engine + NATIVE proc (Q2)", q2)):
        print(f"  {name:34s} {n_ticks / t:>10.1f} ticks/s  ({t / n_ticks * 1e3:8.3f} ms/tick, setup excluded)")
    print(f"  Q2 native vs Q3 Python-on-Rust : {q3 / q2:.2f}x")
    print(f"  Q2 native vs pb all-Python     : {pb / q2:.2f}x")


if __name__ == "__main__":
    main()
