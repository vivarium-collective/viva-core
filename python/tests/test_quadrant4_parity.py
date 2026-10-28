"""Quadrant-4 parity: the REAL Python process-bigraph 1.8.3 engine drives a
RUST-ported process (via the thin Python shell), bit-identical to a pure-Python
twin that does the same arithmetic, run the same way under `Composite`.
"""

import time

from process_bigraph import Composite, Process, allocate_core

from viva_core.rust_process import RustGrow


class PyGrow(Process):
    """Genuine pure-Python twin of the Rust `Grow` process."""

    config_schema = {"rate": {"_type": "float", "_default": 0.1}}

    def inputs(self):
        return {"mass": "float"}

    def outputs(self):
        return {"mass": "float"}

    def update(self, state, interval=1.0):
        return {"mass": self.config["rate"] * state.get("mass", 0.0) * interval}


def _run(address, n_ticks, rate=0.037, mass0=1.3):
    core = allocate_core()
    core.register_link("RustGrow", RustGrow)
    core.register_link("PyGrow", PyGrow)
    doc = {
        "mass": mass0,
        "grow": {
            "_type": "process",
            "address": address,
            "config": {"rate": rate},
            "interval": 0.5,
            "inputs": {"mass": ["mass"]},
            "outputs": {"mass": ["mass"]},
        },
    }
    sim = Composite({"state": doc}, core=core)
    t0 = time.perf_counter()
    sim.run(0.5 * n_ticks)
    return sim.state, time.perf_counter() - t0


def test_quadrant4_bit_identical():
    n = 200
    rs, _ = _run("local:RustGrow", n)
    ps, _ = _run("local:PyGrow", n)
    assert rs["mass"] != 1.3  # the process actually ran
    assert rs["mass"] == ps["mass"]  # bit-identical (exact float equality)
