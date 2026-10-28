"""Oracle for Task 4 `layer_order_matches_pb`.

Runs process-bigraph 1.8.3's REAL scheduling functions (`build_step_network`,
`build_trigger_state`, `determine_steps`) over tests/fixtures/layers_spec.json
-- the exact walk `Composite.cycle_step_state` performs -- and prints the
resulting layer list (paths sorted inside a layer, since pb iterates a Python
set there). Run inside /Users/eranagmon/code/v2ecoli/.venv:

  v2ecoli/.venv/bin/python tests/oracle/gen_layers.py > tests/fixtures/layers_expected.json
"""
import json, os, sys
from process_bigraph.scheduling import (
    build_step_network, build_trigger_state, determine_steps)

here = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(here, "..", "fixtures", "layers_spec.json")))


class Fake:
    def __init__(self, e):
        self.e = e

    def interface(self):
        return {"inputs": {}, "outputs": {}}


steps = {}


def walk(node, path):
    for k, v in node.items():
        if not isinstance(v, dict):
            continue
        if "address" in v and ("inputs" in v or "outputs" in v):
            if v.get("_type") == "step":
                steps[path + (k,)] = dict(v, instance=Fake(v))
        else:
            walk(v, path + (k,))


walk(spec, ())
ancestors, nodes = build_step_network(steps)
remaining = set(steps)
fulfilled = build_trigger_state(nodes, steps)
layers = []
while remaining:
    to_run, remaining, fulfilled = determine_steps(ancestors, remaining, fulfilled)
    if not to_run:
        break
    layers.append(sorted("/".join(p) for p in to_run))
print(json.dumps(layers))
