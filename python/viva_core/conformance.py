"""Cross-language conformance harness for viva-core (Milestone 1, Task 8).

Mirrors the v2ecoli `distilled_compare.py` methodology across the language
boundary: run the SAME composite under BOTH engines and report per-store /
per-observable divergence on the **decoded state** (integer counts exact,
floats within epsilon) rather than on serialized bytes (serde reformats floats).

Two engines:
  * reference  — Python `process_bigraph` 1.8.3 `Composite` on the v2ecoli core;
  * viva_core  — the Rust engine (`viva_core.run` / `run_report`), Python
    processes driven across the PyO3 boundary (quadrant 3).

Public API:
  * :func:`spec_json`        — serialize a Python composite doc to the `.pbg`
                               JSON shape the Rust engine ingests;
  * :func:`run_reference`    — run the pb reference, return state + gate log;
  * :func:`project_state`    — normalize either engine's state to a comparable
                               tree of plain Python values;
  * :func:`compare`          — list every divergence (non-raising);
  * :func:`compare_composite`— run both engines and return a full report dict.

This module imports v2ecoli lazily so it can be imported without the heavy
dependency when only the serialization helpers are needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import viva_core

EPS = 1e-12


# --------------------------------------------------------------------------- #
# serialize a Python composite doc -> the .pbg JSON shape
# --------------------------------------------------------------------------- #

def _serialize(value: Any) -> Any:
    """Serialize one value to the JSON spec shape.

    Structured arrays (bulk/unique) become `__structured_array__` tagged dicts
    with per-field dtype strings, so the Rust codec rebuilds the exact numpy
    dtypes (e.g. `_entryState` as int8). Plain ndarrays lower via `tolist()`.
    """
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        return {
            "__structured_array__": True,
            "dtype": [[name, str(value.dtype[name])] for name in value.dtype.names],
            "columns": {name: value[name].tolist() for name in value.dtype.names},
            "length": int(len(value)),
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def serialize_doc(doc: dict) -> dict:
    return {k: _serialize(v) for k, v in doc.items()}


def spec_json(doc: dict) -> str:
    return json.dumps(serialize_doc(doc))


# --------------------------------------------------------------------------- #
# normalize a (pb OR viva_core) state tree to comparable plain Python values
# --------------------------------------------------------------------------- #

def _tolist(x: Any) -> Any:
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _is_pb_structured(node: Any) -> bool:
    return hasattr(node, "dtype") and getattr(node.dtype, "names", None) is not None


def _normalize_store(node: Any) -> Any:
    """Normalize a single node from either engine.

    Handles: pb structured arrays (bulk ndarray / unique MetadataArray),
    viva_core bulk dicts (`id`+`count` columns), viva_core unique dicts
    (`_metadata` + columns), plain branches (recurse), and scalars.
    """
    # pb bulk/unique: a numpy structured array (MetadataArray carries .metadata).
    if _is_pb_structured(node):
        cols = {f: node[f].tolist() for f in node.dtype.names}
        meta = getattr(node, "metadata", None)
        if meta is not None:
            cols["_metadata"] = int(meta)
        return cols

    if isinstance(node, dict):
        # viva_core unique store: dict-of-columns + stringified _metadata.
        if "_metadata" in node:
            out = {}
            for k, v in node.items():
                out[k] = int(v) if k == "_metadata" else _tolist(v)
            return out
        # viva_core bulk store: columnar `id` + `count` (+ float submasses),
        # no nested dicts.
        if "id" in node and "count" in node and not any(
            isinstance(v, dict) for v in node.values()
        ):
            return {k: _tolist(v) for k, v in node.items()}
        # otherwise a plain branch: recurse.
        return {k: _normalize_store(v) for k, v in node.items()}

    return _scalar(node)


def _scalar(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def project_state(state: dict, keys: list[str] | None = None) -> dict:
    """Project a full engine state (pb `Composite.state` or the dict from
    `viva_core.run`) to a comparable tree. Edge nodes (processes/steps) are
    dropped; only data stores are kept.

    `keys`, if given, restricts projection to those top-level stores (skips the
    process/step edges, which the two engines represent differently).
    """
    out = {}
    items = state.items() if keys is None else ((k, state[k]) for k in keys if k in state)
    for k, v in items:
        if keys is None and _looks_like_edge(v):
            continue
        out[k] = _normalize_store(v)
    return out


def _looks_like_edge(v: Any) -> bool:
    if isinstance(v, dict):
        return v.get("_type") in ("process", "step") or "instance" in v
    # pb stores the instantiated edge object under the edge key.
    return hasattr(v, "update") and hasattr(v, "inputs") and not isinstance(v, dict)


# --------------------------------------------------------------------------- #
# compare (non-raising): list every divergence
# --------------------------------------------------------------------------- #

@dataclass
class Divergence:
    path: str
    kind: str  # "missing" | "type" | "length" | "int" | "float" | "value"
    ref: Any
    got: Any

    def __str__(self) -> str:
        return f"{self.path}: {self.kind} ref={self.ref!r} got={self.got!r}"


def compare(ref: Any, got: Any, eps: float = EPS, path: str = "") -> list[Divergence]:
    """Return the list of divergences between two normalized trees (empty ==
    bit-identical, ints exact + floats within `eps`)."""
    out: list[Divergence] = []
    ref, got = _tolist(ref), _tolist(got)

    if isinstance(ref, dict):
        if not isinstance(got, dict):
            return [Divergence(path, "type", type(ref).__name__, type(got).__name__)]
        for k, v in ref.items():
            if k not in got:
                out.append(Divergence(f"{path}/{k}", "missing", v, None))
            else:
                out += compare(v, got[k], eps, f"{path}/{k}")
        return out

    if isinstance(ref, (list, tuple)):
        if not isinstance(got, (list, tuple)):
            return [Divergence(path, "type", type(ref).__name__, type(got).__name__)]
        if len(ref) != len(got):
            return [Divergence(path, "length", len(ref), len(got))]
        for i, (a, b) in enumerate(zip(ref, got)):
            out += compare(a, b, eps, f"{path}[{i}]")
        return out

    if isinstance(ref, bool) or isinstance(got, bool):
        if bool(ref) != bool(got):
            out.append(Divergence(path, "value", ref, got))
        return out

    if isinstance(ref, int) and isinstance(got, int):
        if ref != got:
            out.append(Divergence(path, "int", ref, got))
        return out

    if isinstance(ref, (int, float)) and isinstance(got, (int, float)):
        if abs(float(ref) - float(got)) > eps:
            out.append(Divergence(path, "float", ref, got))
        return out

    if ref != got:
        out.append(Divergence(path, "value", ref, got))
    return out


# --------------------------------------------------------------------------- #
# run the two engines
# --------------------------------------------------------------------------- #

def _ensure_core():
    """Build the v2ecoli core and register the conformance process classes.

    pb's `local:` protocol resolves addresses through `core.link_registry`
    (NOT importlib), so our processes must be registered under the SAME dotted
    key the Rust engine imports by — `viva_core.conformance_processes.<Class>` —
    so one address drives both engines.
    """
    from v2ecoli.core import build_core
    from v2ecoli.library.ecoli_step import set_current_core
    from viva_core import conformance_processes as cp

    core = build_core()
    set_current_core(core)
    for name in ("BulkProcess", "BulkStepper", "BulkScalarStepper",
                 "IntAccumulator", "FloatAccumulator", "TickTrigger"):
        core.register_link(f"viva_core.conformance_processes.{name}",
                           getattr(cp, name))
    return core


def run_reference(doc: dict, until: float, gate_classes: dict | None = None):
    """Run the pb 1.8.3 `Composite` reference.

    `gate_classes` maps edge-name -> process class whose `update_condition`
    should be instrumented, so the gate log can be compared against viva_core's
    `step_runs`. Returns `(state, gate_log, emitter_rows)`.
    """
    from process_bigraph import Composite
    from process_bigraph.emitter import gather_emitter_results

    core = _ensure_core()
    gate_log: list[tuple[str, float, bool]] = []
    patches = []
    if gate_classes:
        for name, cls in gate_classes.items():
            orig = cls.update_condition

            def make(orig, name):
                def patched(self, ts, st):
                    r = orig(self, ts, st)
                    gate_log.append((name, float(st.get("global_time")), bool(r)))
                    return r
                return patched

            cls.update_condition = make(orig, name)
            patches.append((cls, orig))
    try:
        sim = Composite({"state": doc}, core=core)
        sim.run(until)
    finally:
        for cls, orig in patches:
            cls.update_condition = orig

    rows = sum(len(v) for v in gather_emitter_results(sim).values())
    return sim.state, gate_log, rows


def run_viva(doc: dict, n_ticks: int) -> dict:
    """Run the Rust engine on the serialized doc; return the `run_report` dict
    (`state`, `step_runs`, `emitter_counts`)."""
    return viva_core.run_report(spec_json(doc), n_ticks)


@dataclass
class ConformanceReport:
    divergences: list[Divergence]
    ref_state: dict
    got_state: dict
    ref_gate_on: set
    got_gate_on: set
    ref_emitter_rows: int
    got_emitter_counts: dict
    store_keys: list[str] = field(default_factory=list)

    @property
    def bit_identical(self) -> bool:
        return not self.divergences

    def summary(self) -> str:
        if self.bit_identical:
            return f"bit-identical across {len(self.store_keys)} stores"
        head = "; ".join(str(d) for d in self.divergences[:5])
        return f"{len(self.divergences)} divergence(s): {head}"


def compare_composite(
    doc: dict,
    until: float,
    n_ticks: int,
    store_keys: list[str],
    eps: float = EPS,
    gate_classes: dict | None = None,
    gate_edge: str | None = None,
) -> ConformanceReport:
    """Run both engines on `doc` and return a full divergence report over the
    named `store_keys` (the data stores — scalars + bulk/unique)."""
    # Snapshot the spec from the PRISTINE doc first: the pb reference mutates
    # the doc's numpy arrays in place (bulk_numpy_updater writes `count` on the
    # same buffer), so serializing after the reference run would hand viva_core
    # the post-pb state and double-count the updates.
    spec = spec_json(doc)
    ref_state, gate_log, ref_rows = run_reference(doc, until, gate_classes)
    report = viva_core.run_report(spec, n_ticks)

    ref_proj = project_state(ref_state, store_keys)
    got_proj = project_state(report["state"], store_keys)
    divergences = compare(ref_proj, got_proj, eps)

    ref_on = {gt for (_n, gt, ran) in gate_log if ran}
    got_on = set()
    if gate_edge:
        got_on = {
            gt for (name, gt, ran) in report["step_runs"] if name == gate_edge and ran
        }

    return ConformanceReport(
        divergences=divergences,
        ref_state=ref_proj,
        got_state=got_proj,
        ref_gate_on=ref_on,
        got_gate_on=got_on,
        ref_emitter_rows=ref_rows,
        got_emitter_counts=dict(report["emitter_counts"]),
        store_keys=store_keys,
    )
