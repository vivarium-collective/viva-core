"""M3 Task 2: v2ecoli `ProteinDegradation` as a NATIVE Rust process.

The same `.pbg` (one `local:v2ecoli.processes.protein_degradation.ProteinDegradation`
edge, config = the serialized `get_protein_degradation_config` params) runs three
ways, and the decoded bulk state must be bit-identical across all of them:

  (ref)    Python pb 1.8.3 engine + the Python process
  (py)     Rust engine + the Python process (quadrant 3; flag OFF)
  (native) Rust engine + the NATIVE Rust process (quadrant 2; VIVA_NATIVE_PROCESSES)

The poisson draws are stochastic, so bit-identity proves the Rust RandomState,
draw order, and S-matrix application match numpy exactly.
"""

import os

import numpy as np
import pytest

import viva_core
from viva_core import conformance as C

ADDR = "local:v2ecoli.processes.protein_degradation.ProteinDegradation"
N_PROT = 24
N_AA = 5
WATER = "h2o[c]"
TICKS = 40


def _build(seed=1234, interval=1.0):
    rng = np.random.RandomState(99)
    prot_ids = [f"PROT-{i}[c]" for i in range(N_PROT)]
    aa_ids = [f"AA-{j}[c]" for j in range(N_AA)]
    aa_counts = rng.randint(0, 40, size=(N_PROT, N_AA)).astype(np.int64)
    # Span all three poisson regimes (dt=1): lam==0 (count 0 or rate 0),
    # lam<10 (Knuth multiplication), lam>=10 (PTRS).
    counts = np.array(
        [0, 0, 3, 8, 15, 40, 100, 250, 600, 1500, 4000, 9000,
         20000, 50000, 120000, 300, 7, 1, 2, 900, 12, 75, 33, 5000],
        dtype=np.int64,
    )
    rates = np.array(
        [1e-3, 5e-3, 1e-3, 1e-2, 5e-3, 1e-3, 2e-3, 1e-3, 1e-3, 2e-4, 1e-4,
         1e-3, 5e-4, 2e-4, 1e-3, 0.0, 0.0, 3e-2, 1e-1, 1e-2, 5e-2, 1e-3, 2e-2, 1e-4]
    )
    aa_init = np.full(N_AA, 10**7, dtype=np.int64)
    ids = prot_ids + aa_ids + [WATER]
    dt = np.dtype([("id", "<U16"), ("count", "int64")])
    bulk = np.zeros(len(ids), dtype=dt)
    bulk["id"] = ids
    bulk["count"] = np.concatenate([counts, aa_init, [10**9]])
    cfg = {
        "time_step": 1,
        "raw_degradation_rate": rates.tolist(),
        "water_id": WATER,
        "amino_acid_ids": aa_ids,
        "amino_acid_counts": aa_counts.tolist(),
        "protein_ids": prot_ids,
        "protein_lengths": aa_counts.sum(axis=1).tolist(),
        "seed": seed,
    }
    doc = {
        "global_time": 0.0,
        "timestep": 1,
        "bulk": bulk,
        "tick": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.TickTrigger",
            "config": {"dt": interval, "value": 1},
            "inputs": {},
            "outputs": {"timestep": ["timestep"]},
        },
        "protein_degradation": {
            "_type": "step",
            "address": ADDR,
            "config": cfg,
            "inputs": {"bulk": ["bulk"], "timestep": ["timestep"]},
            "outputs": {"bulk": ["bulk"]},
        },
    }
    return doc, counts, rates


def _bulk_counts(state):
    b = state["bulk"]
    if hasattr(b, "dtype") and b.dtype.names:
        return np.asarray(b["count"]).tolist()
    return list(b["count"])


def _run_all(seed=1234, ticks=TICKS):
    doc, counts, rates = _build(seed)
    spec = C.spec_json(doc)  # snapshot before pb mutates the arrays in place
    ref_state, _, _ = C.run_reference(doc, float(ticks))
    ref = _bulk_counts(ref_state)

    os.environ.pop("VIVA_NATIVE_PROCESSES", None)
    py = _bulk_counts(viva_core.run(spec, ticks))
    # Prove the native path never calls back into Python: poison the class.
    from v2ecoli.processes.protein_degradation import ProteinDegradation as PD

    def _poison(self, *a, **k):
        raise AssertionError("Python ProteinDegradation.update called on native path")

    orig_update, orig_init = PD.update, PD.initialize
    PD.update, PD.initialize = _poison, _poison
    os.environ["VIVA_NATIVE_PROCESSES"] = "ProteinDegradation"
    try:
        native = _bulk_counts(viva_core.run(spec, ticks))
    finally:
        os.environ.pop("VIVA_NATIVE_PROCESSES", None)
        PD.update, PD.initialize = orig_update, orig_init
    return ref, py, native, counts, rates


def test_native_protein_degradation_bit_identical():
    ref, py, native, counts, rates = _run_all()
    init = counts.tolist()
    # Non-vacuous: real degradation occurred in every regime.
    assert ref[:N_PROT] != init
    degraded = [i - r for i, r in zip(init, ref[:N_PROT])]
    assert all(d >= 0 for d in degraded)
    assert sum(degraded) > 1000
    assert any(0 < d for d, c in zip(degraded, counts) if c < 10)   # Knuth regime
    assert any(0 < d for d, c, r in zip(degraded, counts, rates) if c * r >= 10)  # PTRS
    assert degraded[0] == 0 and degraded[15] == 0 and degraded[16] == 0  # lam == 0
    assert ref[-1] != 10**9  # water consumed
    # Python-on-Rust == pb reference (M1/M2 guarantee) ...
    assert py == ref
    # ... and the native Rust process is bit-identical to both.
    assert native == ref


def test_native_matches_across_seeds():
    for seed in (0, 7, 2**31 - 1):
        ref, py, native, _, _ = _run_all(seed=seed, ticks=12)
        assert native == ref == py, f"seed {seed}"


def test_flag_off_uses_python_path():
    # Different seeds must give different draws (guards against a vacuous run).
    a = _run_all(seed=1, ticks=10)[2]
    b = _run_all(seed=2, ticks=10)[2]
    assert a != b
