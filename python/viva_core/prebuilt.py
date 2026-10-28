"""Python->Rust composite handoff for Milestone 4.

The whole v2ecoli baseline *cannot* be constructed from a JSON ``.pbg``: 11 of
its process types carry live, non-JSON-serializable ``sim_data`` handles that are
baked into each instance at construction. So instead of re-building processes
from a JSON ``address``+``cfg`` (what :func:`viva_core.run` does), this helper
takes an ALREADY-REALIZED composite state — v2ecoli's own builder
(``ecoli_baseline.baseline`` -> ``process_bigraph.Composite``) constructs the
instances with their ``sim_data`` — and hands the *instances* to the Rust engine
via :func:`viva_core.run_prebuilt`, which schedules each as an instance-bound
``PyProcess`` (quadrant 3), swapping a native twin for any allow-listed address.

Given a realized composite state dict (``{store_or_edge_name: value}``) it:

* splits edges (``{...,'address','inputs'/'outputs','instance'}``) from stores;
* builds the ``topology`` spec (edge wiring only — configs dropped, the live
  ``_layer_*`` scheduler tokens stripped so the Rust engine re-derives the step
  layers from the data wiring, exactly as the JSON ``run`` path does);
* serializes the stores to the same ``__structured_array__``-tagged JSON the Rust
  codec already decodes (bulk/unique structured arrays, scalars, nested maps);
* collects the pre-built instances keyed by edge path and calls ``run_prebuilt``.
"""

import json

import numpy as np

import viva_core


def _is_edge(v):
    return isinstance(v, dict) and "address" in v and ("inputs" in v or "outputs" in v)


def _is_emitter(address):
    return (
        "Emitter" in address
        or address.endswith("ram-emitter")
        or address.endswith("json-emitter")
    )


def _strip_layer_ports(wires):
    """Drop the ``_layer_in_*`` / ``_layer_out_*`` scheduler tokens pb injects
    into realized edge wiring (the Rust engine computes step layers itself)."""
    if isinstance(wires, dict):
        return {
            k: _strip_layer_ports(v)
            for k, v in wires.items()
            if not str(k).startswith("_layer")
        }
    return wires


def _serialize(value):
    """A store value -> the tagged JSON the Rust codec decodes.

    numpy structured arrays (bulk: ``id``+``count``; unique: ``_entryState``+
    ``unique_index``) become ``__structured_array__`` dicts; plain arrays/scalars
    lower via ``tolist``/``item``; dicts/lists recurse.
    """
    names = getattr(getattr(value, "dtype", None), "names", None)
    if names:  # structured array (ndarray or v2ecoli MetadataArray)
        return {
            "__structured_array__": True,
            "dtype": [[n, str(value.dtype[n])] for n in names],
            "columns": {n: value[n].tolist() for n in names},
            "length": int(len(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):  # plain ndarray
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _native_config_from_instance(edge):
    """Serialize a pre-built instance's ``.parameters`` into the JSON config its
    NATIVE twin's constructor reads.

    The baseline's edges carry an *empty* JSON ``config`` (the sim_data-derived
    params are baked into the live instance, not serialized). The native twins
    (M3) are, by contrast, built from the serialized config dict. So to swap a
    native twin for a pre-built instance the config must be reconstructed from
    the live instance — this closes the M4 "native-swap empty-config sourcing"
    gap for ported processes (e.g. ProteinDegradation), whose ``.parameters`` is
    exactly the serializable config dict. Only called for allow-listed native
    addresses, so no unserializable handles are reached.
    """
    inst = edge.get("instance")
    params = getattr(inst, "parameters", None)
    if not isinstance(params, dict):
        return {}
    return {k: _serialize(v) for k, v in params.items()}


def build_payload(state, native_addresses=None):
    """Split a realized composite state into the ``run_prebuilt`` arguments.

    Returns ``(processes, topology_json, state_json)``:
      * ``processes``: ``{edge_name: instance}`` for every non-emitter edge;
      * ``topology_json``: JSON spec of the edges (wiring only, configs dropped —
        emitter edges keep their ``config`` since they are built natively; an
        allow-listed ``native_addresses`` edge keeps a config sourced from its
        live instance so its native twin can be constructed);
      * ``state_json``: JSON spec of the data stores (tagged structured arrays).
    """
    native = set(native_addresses or ())
    edges = {k: v for k, v in state.items() if _is_edge(v)}
    stores = {k: v for k, v in state.items() if not _is_edge(v)}

    processes = {}
    topology = {}
    for name, edge in edges.items():
        address = edge["address"]
        node = {
            "_type": edge.get("_type", "step"),
            "address": address,
            "inputs": _strip_layer_ports(edge.get("inputs", {})),
            "outputs": _strip_layer_ports(edge.get("outputs", {})),
        }
        if edge.get("interval") is not None:
            node["interval"] = edge["interval"]
        if _is_emitter(address):
            node["config"] = edge.get("config", {})
        else:
            if "instance" not in edge:
                raise ValueError(
                    f"edge {name!r} ({address}) has no realized 'instance' to hand "
                    "over; realize the composite (process_bigraph.Composite) first"
                )
            processes[name] = edge["instance"]
            # Native-twin config is sourced from the live instance (empty on the
            # baseline edge) so the swap can build correct sim_data-derived params.
            # The instance is still handed over, so if the native path is disabled
            # (VIVA_NATIVE_PROCESSES off) the engine falls back to it bit-identically.
            if address in native:
                node["config"] = _native_config_from_instance(edge)
        topology[name] = node

    return processes, json.dumps(topology), json.dumps({k: _serialize(v) for k, v in stores.items()})


def run_from_state(state, n_ticks, native_addresses=None):
    """Run a realized composite state on the Rust engine via the handoff.

    ``state`` is a realized composite state dict (edges carry ``instance``);
    ``n_ticks`` scheduler iterations; ``native_addresses`` optionally swaps those
    addresses for their native twin (also gated by ``VIVA_NATIVE_PROCESSES``) —
    each such edge's native config is sourced from its live instance.
    Returns the decoded final state dict.
    """
    processes, topology_json, state_json = build_payload(state, native_addresses)
    return viva_core.run_prebuilt(
        processes, topology_json, state_json, n_ticks, native_addresses
    )
