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
import pint

import viva_core


def _serialize_quantity(q):
    """A ``pint.Quantity`` -> the ``__pint__`` tagged dict the Rust codec decodes.

    Mirrors ``convert.rs::quantity_to_value`` so a Quantity in the initial state
    and a Quantity written by a process lower to the SAME shape: a numpy-scalar
    magnitude records the viva-core-private ``__viva_scalar_dtype__`` key (so the
    ``np.float64`` vs ``float`` distinction survives the round-trip — the
    ``1/Quantity(0.0)`` inf-vs-ZeroDivisionError hazard), while an ndarray
    magnitude uses the upstream ``__magnitude_dtype__``. The private key is
    ignored by ``bigraph_json_hook`` (which keys scalars on the *absence* of
    ``__magnitude_dtype__``).
    """
    mag = q.magnitude
    units = str(q.units)
    if isinstance(mag, np.ndarray):
        return {"__pint__": True, "magnitude": mag.tolist(),
                "__magnitude_dtype__": str(mag.dtype), "units": units}
    if isinstance(mag, np.generic):
        return {"__pint__": True, "magnitude": mag.item(),
                "units": units, "__viva_scalar_dtype__": str(mag.dtype)}
    return {"__pint__": True, "magnitude": mag, "units": units}


_JSON_LEAF = (int, float, bool, str, bytes, type(None), list, dict)


def _is_live_element(x):
    """True for an object the JSON state channel cannot carry — not a JSON
    primitive/container, not a numpy array/scalar."""
    return not isinstance(x, _JSON_LEAF + (np.ndarray, np.generic))


def _is_handle(value):
    """A store value that is a LIVE Python object passed by reference, not
    serialized (Option 1): a numpy ``RandomState``/``Generator``
    (``allocator_rng``), or a tuple holding a live instance (the ``(instance,)``
    ``process`` store)."""
    if isinstance(value, (np.random.RandomState, np.random.Generator)):
        return True
    if isinstance(value, tuple):
        return any(_is_live_element(x) for x in value)
    return False


def _split_stores(stores, handles):
    """Serialize the data stores to the tagged JSON the Rust codec decodes,
    routing every live-object leaf into ``handles`` (keyed by its ``/``-joined
    absolute store path) and leaving a ``null`` placeholder in the JSON at that
    path — so the store still exists, the wiring + the Requester→Allocator→Evolver
    step ordering are unchanged, and ``_serialize`` never meets a live object."""
    def walk(value, path):
        if _is_handle(value):
            handles["/".join(path)] = value
            return None  # placeholder: the handle travels out-of-band
        if isinstance(value, dict):
            return {k: walk(v, path + [k]) for k, v in value.items()}
        return _serialize(value)

    return {k: walk(v, [k]) for k, v in stores.items()}


def _is_edge(v):
    return isinstance(v, dict) and "address" in v and ("inputs" in v or "outputs" in v)


def _address_str(address):
    """Normalize an edge address to its ``protocol:data`` string. A realized
    ``Composite`` leaves some edges (e.g. ``shape_step``) with the parsed form
    ``{'protocol': 'local', 'data': 'ShapeStep'}``."""
    if isinstance(address, dict):
        return f"{address['protocol']}:{address['data']}"
    return address


def _unwrap_state(state, path=()):
    """Accept a realized ``process_bigraph.Composite`` (its ``.state``) or a state
    dict, then descend ``path`` (e.g. ``("agents", "0")``)."""
    if hasattr(state, "state") and not isinstance(state, dict):
        state = state.state
    for key in path:
        state = state[key]
    return state


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


def _serialize(value, numpy_tag=True):
    """A store value -> the tagged JSON the Rust codec decodes.

    numpy structured arrays (bulk: ``id``+``count``; unique: ``_entryState``+
    ``unique_index``) become ``__structured_array__`` dicts; plain arrays/scalars
    lower via ``tolist``/``item``; dicts/lists recurse.

    ``numpy_tag`` controls how a plain (non-structured) ndarray leaf lowers: as a
    ``__numpy__`` tag (the default, for STATE leaves — so they round-trip as real
    ndarrays through the view and compare equal to pb) or as a bare list
    (``numpy_tag=False``, for a native twin's CONFIG, whose constructor reads
    ``protein_ids`` etc. as plain lists, not tagged dicts).
    """
    if isinstance(value, pint.Quantity):  # must precede the `tolist` branch
        return _serialize_quantity(value)
    names = getattr(getattr(value, "dtype", None), "names", None)
    if names:  # structured array (ndarray or v2ecoli MetadataArray)
        out = {
            "__structured_array__": True,
            "dtype": [[n, str(value.dtype[n])] for n in names],
            "columns": {n: value[n].tolist() for n in names},
            "length": int(len(value)),
        }
        # Preserve the v2ecoli ``MetadataArray.metadata`` next-unique-index counter
        # verbatim: it is molecule-type-namespaced (high-bit prefix), so it is NOT
        # ``max(unique_index)+1`` for an empty unique store (e.g.
        # ``chromosomal_segment`` ships empty with a large base). Without it the
        # Rust side would hand out colliding indices for that molecule type.
        md = getattr(value, "metadata", None)
        if md is not None:
            out["metadata"] = int(md)
        return out
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # A 0-d array is a scalar leaf; pb serves it (and its schema coerces it)
        # as a plain scalar, not an ndarray — keep it a scalar so numba kernels
        # that do `min(x, scalar)` typecheck as they do under pb.
        return value.item()
    if isinstance(value, np.ndarray):  # plain (non-structured) ndarray, ndim >= 1
        if numpy_tag:
            # Preserve it as a ``__numpy__`` tag (dtype + shape + data) so the Rust
            # codec decodes an ndarray leaf that the view rebuilds as a real
            # ndarray, not a Python list — a reader doing array math / fancy-
            # indexing (e.g. ``basal_prob[TU_index]``) then behaves as under pb,
            # and the final state compares equal to pb's ndarray. (known-unknown a)
            return {"__numpy__": True, "dtype": str(value.dtype),
                    "shape": list(value.shape), "data": value.tolist()}
        return value.tolist()
    if hasattr(value, "tolist"):  # other array-likes
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serialize(v, numpy_tag) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v, numpy_tag) for v in value]
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
    # Native-twin config: plain lists for ndarray params (the twin's constructor
    # reads e.g. ``protein_ids`` as a list), NOT the ``__numpy__`` state tag.
    return {k: _serialize(v, numpy_tag=False) for k, v in params.items()}


def build_payload(state, native_addresses=None, path=()):
    """Split a realized composite state into the ``run_prebuilt`` arguments.

    ``state`` may be a realized ``process_bigraph.Composite`` (preferred: every edge
    carries a live ``instance``) or a state dict; ``path`` descends to the agent
    sub-state (e.g. ``("agents", "0")``). A raw ``baseline()`` doc is NOT realized
    and raises ``ValueError`` on the first edge lacking an instance.

    Returns ``(processes, topology_json, state_json, handles)``:
      * ``processes``: ``{edge_name: instance}`` for every non-emitter edge;
      * ``topology_json``: JSON spec of the edges (wiring only, configs dropped —
        emitter edges keep their ``config`` since they are built natively; an
        allow-listed ``native_addresses`` edge keeps a config sourced from its
        live instance so its native twin can be constructed);
      * ``state_json``: JSON spec of the data stores (tagged structured arrays;
        live-object stores carry a ``null`` placeholder);
      * ``handles``: ``{path: live_object}`` (path ``/``-joined) for the live
        process-as-data store (``["process", <name>]`` → the ``(instance,)``
        tuple) and ``["allocator_rng"]`` (the ``RandomState``) — passed by
        reference so the Rust engine never serializes them (Option 1).
    """
    state = _unwrap_state(state, path)
    native = set(native_addresses or ())
    edges = {k: v for k, v in state.items() if _is_edge(v)}
    stores = {k: v for k, v in state.items() if not _is_edge(v)}

    processes = {}
    topology = {}
    for name, edge in edges.items():
        address = _address_str(edge["address"])
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
            inst = edge["instance"]
            # pb sets the edge name on each realized instance; some processes read
            # it (v2ecoli `Requester.update_condition` logs `self.name`). The
            # pre-built instances carry it for EcoliProcesses but not for the
            # partition Requester/Evolver, so set it to the edge name to match pb.
            try:
                if getattr(inst, "name", None) != name:
                    inst.name = name
            except (AttributeError, TypeError):
                pass
            processes[name] = inst
            # Native-twin config is sourced from the live instance (empty on the
            # baseline edge) so the swap can build correct sim_data-derived params.
            # The instance is still handed over, so if the native path is disabled
            # (VIVA_NATIVE_PROCESSES off) the engine falls back to it bit-identically.
            if address in native:
                node["config"] = _native_config_from_instance(edge)
        topology[name] = node

    handles = {}
    state_json = json.dumps(_split_stores(stores, handles))
    return processes, json.dumps(topology), state_json, handles


def run_from_state(state, n_ticks, native_addresses=None, path=()):
    """Run a realized composite state on the Rust engine via the handoff.

    ``state`` is a realized composite state dict (edges carry ``instance``);
    ``n_ticks`` scheduler iterations; ``native_addresses`` optionally swaps those
    addresses for their native twin (also gated by ``VIVA_NATIVE_PROCESSES``) —
    each such edge's native config is sourced from its live instance.
    Returns the decoded final state dict.
    """
    processes, topology_json, state_json, handles = build_payload(state, native_addresses, path)
    return viva_core.run_prebuilt(
        processes, topology_json, state_json, n_ticks, native_addresses, handles
    )
