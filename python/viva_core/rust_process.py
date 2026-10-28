"""Quadrant 4: a Rust-ported process addressable as ``local:RustGrow`` by pb."""

from process_bigraph import Process

from viva_core._native import RustGrowCore


class RustGrow(Process):
    config_schema = {"rate": {"_type": "float", "_default": 0.1}}

    def initialize(self, config):
        self._rust = RustGrowCore({"rate": float(config["rate"])})

    def inputs(self):
        return self._rust.inputs()

    def outputs(self):
        return self._rust.outputs()

    def update(self, state, interval=1.0):
        return self._rust.update(state, interval)
