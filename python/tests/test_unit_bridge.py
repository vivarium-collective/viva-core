"""M5 Task 1 — the pint unit bridge at the PyO3 boundary (Option A).

The Rust codec already carries a ``__pint__`` tagged dict opaquely through state
and ``Op::Overwrite`` (the hot path never inspects a unit). The loss was purely
in ``crates/viva-core/src/convert.rs``: ``value_to_py`` handed a pint leaf back
as a plain ``dict`` and ``py_to_value`` turned a ``pint.Quantity`` into ``Null``
(Python-float magnitude) or infinite-recursed (``np.float64`` magnitude, because
``q.tolist()`` returns another Quantity). This test drives real pint Quantities
through ``py_to_value -> state -> value_to_py`` with SYNTHETIC quantities (no
sim_data cache needed, same approach as ``test_subarray_unique``) and asserts:

  (a) a scalar ``np.float64`` femtogram mass, a ``1 / second`` rate, and an
      ``amino_acid / second`` rate round-trip to a real ``pint.Quantity`` with the
      unit preserved, the magnitude bit-exact, AND the magnitude dtype preserved
      (``type(q.magnitude) is np.float64`` — not demoted to a Python ``float``);
  (b) a zero-valued ``np.float64`` mass round-trips so that ``1 / q`` yields
      ``inf`` rather than raising ``ZeroDivisionError`` (the demotion hazard —
      mass divisors hit this);
  (c) the strict ``effective_elongation_rate`` read survives write->read across
      ticks as a real Quantity that ``.to(units.aa/units.s)`` accepts;
  (d) a Python-float magnitude stays a Python ``float`` (the distinction matters);
  (e) ``prebuilt._serialize`` on a Quantity emits the json-dumpable tagged dict.

Pure Rust-boundary mechanics; no v2ecoli sim_data cache is touched.
"""

import json

import numpy as np
import pint
import pytest

import viva_core
from viva_core import conformance as C
from viva_core import conformance_processes as cp
from viva_core import prebuilt

from bigraph_schema.units import get_quantity_registry

EPS = 1e-12


def _doc():
    """A one-process composite: QuantityCycler reads and writes four quantity
    stores (inputs and outputs wired to the SAME stores, so tick t+1 reads what
    tick t wrote). Stores start as a bare-float sentinel so a Quantity seen on a
    later tick must have come from a write (py_to_value), not the initial state."""
    return {
        "global_time": 0.0,
        "mass": -1.0,
        "rate": -1.0,
        "eff_rate": -1.0,
        "zero_mass": -1.0,
        "py_mass": -1.0,
        "cycler": {
            "_type": "process",
            "address": "local:viva_core.conformance_processes.QuantityCycler",
            "config": {"dt": 1.0},
            "inputs": {p: [p] for p in ("mass", "rate", "eff_rate", "zero_mass", "py_mass")},
            "outputs": {p: [p] for p in ("mass", "rate", "eff_rate", "zero_mass", "py_mass")},
        },
    }


@pytest.fixture(scope="module")
def ureg():
    # Build + register the v2ecoli core (defines amino_acid etc. in the shared
    # registry and sets the current core EcoliProcess construction needs).
    from v2ecoli.core import build_core
    from v2ecoli.library.ecoli_step import set_current_core

    core = build_core()
    set_current_core(core)
    return get_quantity_registry()


@pytest.fixture(scope="module")
def final_state(ureg):
    cp.QUANTITY_READ_LOG.clear()
    return viva_core.run(C.spec_json(_doc()), 3)


# --------------------------------------------------------------------------- #
# (a) scalar np.float64 round-trip: type, unit, magnitude, AND dtype
# --------------------------------------------------------------------------- #

def test_scalar_quantity_roundtrip_type_unit_magnitude_dtype(final_state, ureg):
    expected = {"mass": (5.0, "femtogram"),
                "rate": (2.0, "1 / second"),
                "eff_rate": (1.2345, "amino_acid / second")}
    for port, (mag, unit) in expected.items():
        q = final_state[port]
        assert isinstance(q, pint.Quantity), f"{port}: {type(q)} is not a Quantity"
        # unit preserved (re-parses to the same registry unit)
        assert q.units == ureg.Quantity(1.0, unit).units, f"{port}: units {q.units}"
        # magnitude bit-exact
        assert abs(float(q.to(unit).magnitude) - mag) <= EPS, f"{port}: mag {q.magnitude}"
        # dtype preserved: np.float64, NOT demoted to a Python float
        assert type(q.magnitude) is np.float64, \
            f"{port}: magnitude dtype {type(q.magnitude)} (demoted?)"


# --------------------------------------------------------------------------- #
# (b) zero-valued np.float64 mass: 1/q is inf, not ZeroDivisionError
# --------------------------------------------------------------------------- #

def test_zero_mass_reciprocal_is_inf_not_zerodiv(final_state):
    q = final_state["zero_mass"]
    assert isinstance(q, pint.Quantity)
    assert type(q.magnitude) is np.float64
    # Would raise ZeroDivisionError if the magnitude had been demoted to float.
    inv = 1.0 / q
    assert np.isinf(inv.magnitude), f"expected inf, got {inv.magnitude!r}"


# --------------------------------------------------------------------------- #
# (c) the strict effective_elongation_rate read survives write->read across ticks
# --------------------------------------------------------------------------- #

def test_effective_elongation_rate_strict_read_across_ticks(final_state):
    log = cp.QUANTITY_READ_LOG
    assert log, "QuantityCycler never ran"
    # Tick 0 read the bare-float sentinel, so it was NOT yet a Quantity.
    assert log[0]["eff_rate"]["is_quantity"] is False
    # A later tick read back a REAL Quantity written the previous tick, and the
    # strict `.to(amino_acid/second)` conversion succeeded on it.
    last = log[-1]
    assert last["eff_rate"]["is_quantity"] is True
    assert last["eff_rate"]["to_ok"] is True
    assert abs(last["eff_rate"]["mag"] - 1.2345) <= EPS
    assert last["eff_rate"]["mag_dtype"] == "float64"
    # mass + rate likewise round-tripped as Quantities with .to() working.
    for port, mag in (("mass", 5.0), ("rate", 2.0)):
        assert last[port]["is_quantity"] is True
        assert last[port]["to_ok"] is True
        assert abs(last[port]["mag"] - mag) <= EPS
        assert last[port]["mag_dtype"] == "float64"


# --------------------------------------------------------------------------- #
# (d) a Python-float magnitude stays a Python float (distinction preserved)
# --------------------------------------------------------------------------- #

def test_python_float_magnitude_not_promoted(final_state):
    q = final_state["py_mass"]
    assert isinstance(q, pint.Quantity)
    assert abs(float(q.to("femtogram").magnitude) - 3.0) <= EPS
    assert type(q.magnitude) is float, f"py_mass magnitude {type(q.magnitude)}"
    # The read-side log agrees: it was a Quantity but with a plain-float magnitude.
    last = cp.QUANTITY_READ_LOG[-1]
    assert last["py_mass"]["is_quantity"] is True
    assert last["py_mass"]["mag_dtype"] == "float"


# --------------------------------------------------------------------------- #
# (e) prebuilt._serialize emits the json-dumpable tagged dict
# --------------------------------------------------------------------------- #

def test_prebuilt_serialize_quantity_tagged_dict(ureg):
    # scalar np.float64 -> tagged dict with the private scalar-dtype key
    d = prebuilt._serialize(ureg.Quantity(np.float64(7.5), "femtogram"))
    assert d["__pint__"] is True
    assert d["units"] == "femtogram"
    assert abs(d["magnitude"] - 7.5) <= EPS
    assert d["__viva_scalar_dtype__"] == "float64"
    json.dumps(d)  # must be json-dumpable
    # the private key is harmless to bigraph_schema's decode hook (it keys on
    # __magnitude_dtype__, which a scalar does NOT carry) -> stays a scalar.
    from bigraph_schema.json_codec import bigraph_json_hook
    back = bigraph_json_hook(dict(d))
    assert isinstance(back, pint.Quantity)
    assert not isinstance(back.magnitude, np.ndarray)

    # Python-float scalar -> NO scalar-dtype key (rebuilds as a Python float)
    d2 = prebuilt._serialize(ureg.Quantity(3.0, "femtogram"))
    assert "__viva_scalar_dtype__" not in d2
    json.dumps(d2)

    # array magnitude -> upstream __magnitude_dtype__ shape
    d3 = prebuilt._serialize(ureg.Quantity(np.array([1, 2, 3], dtype="int64"), "count"))
    assert d3["__magnitude_dtype__"] == "int64"
    assert d3["magnitude"] == [1, 2, 3]
    json.dumps(d3)
