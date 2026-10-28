//! Conversions across the Python boundary: [`Value`]/[`StoreNode`] <-> Python,
//! numpy structured-array view building, and update-dict -> typed [`Op`]
//! translation.

use std::sync::Arc;

use ndarray::{aview1, Array1};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{IntoPyDict, PyCapsule};
use pyo3::types::{PyAnyMethods, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use bigraph_schema_rs::apply::{Op, UniqueOp};
use bigraph_schema_rs::doc::{Map, TagKind, Value};
use bigraph_schema_rs::grammar::parse_type;
use bigraph_schema_rs::state::{
    parse_field_dtype, parse_subarray_dtype, read_value, FieldLayout, RecordLayout, State,
    StoreNode,
};
use bigraph_schema_rs::types::Type;
use process_bigraph_rs::process::{PortUpdate, PortView};

use crate::DtypeMap;

fn perr(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---- Value <-> Python ------------------------------------------------------

/// Convert a decoded [`Value`] into a Python object.
pub fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::UInt(u) => u.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::Array(a) => {
            let list = PyList::empty(py);
            for x in a {
                list.append(value_to_py(py, x)?)?;
            }
            list.into_any()
        }
        // A pint Quantity leaf: rebuild a real `pint.Quantity` (Option A) rather
        // than hand back a plain dict, restoring the magnitude dtype so a reader
        // gets `np.float64` not a demoted Python `float`.
        Value::Tagged(TagKind::Pint, m) => pint_to_py(py, m)?,
        Value::Map(m) | Value::Tagged(_, m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, value_to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
}

// ---- pint Quantity bridge (Option A: opaque tag in Rust, wrap/unwrap here) --

/// Cached `bigraph_schema.units.get_quantity_registry()` handle — the registry
/// v2ecoli installs as the pint application registry, so decoded Quantities are
/// registry-coherent with v2ecoli's (incl. units like `amino_acid`).
fn quantity_registry<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    static REG: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    REG.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import("bigraph_schema.units")?
            .getattr("get_quantity_registry")?
            .call0()?
            .unbind())
    })
    .map(|r| r.bind(py).clone())
}

/// Cached `pint.Quantity` base class for the `isinstance` check in `py_to_value`.
fn pint_quantity_cls<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    static CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    CLS.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("pint")?.getattr("Quantity")?.unbind())
    })
    .map(|c| c.bind(py).clone())
}

/// Cached `numpy.generic` — the base of numpy scalar types (e.g. `np.float64`),
/// used to tell a numpy-scalar magnitude from a Python scalar / an ndarray.
fn numpy_generic_cls<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    static CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    CLS.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("numpy")?.getattr("generic")?.unbind())
    })
    .map(|c| c.bind(py).clone())
}

/// `Some(obj)` if `obj` is a `pint.Quantity`, else `None` (never errors — a
/// failed pint import just means "not a Quantity").
fn as_pint_quantity<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyAny>> {
    let cls = pint_quantity_cls(obj.py()).ok()?;
    if obj.is_instance(&cls).unwrap_or(false) {
        Some(obj)
    } else {
        None
    }
}

/// Lower a `pint.Quantity` to `Value::Tagged(Pint, {..})`, preserving the
/// magnitude dtype so [`pint_to_py`] can rebuild `np.float64` vs a Python
/// `float` (the `1/Quantity(0.0)` inf-vs-`ZeroDivisionError` hazard). A numpy
/// scalar magnitude records a viva-core-private `__viva_scalar_dtype__` key —
/// `bigraph_json_hook` keys scalars on the *absence* of `__magnitude_dtype__`,
/// so the private key leaves that decode path (and the scalar vs 0-d-array
/// distinction) unchanged. An ndarray magnitude uses the upstream
/// `__magnitude_dtype__` shape.
fn quantity_to_value(q: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = q.py();
    let mag = q.getattr("magnitude")?;
    let units = q.getattr("units")?.str()?.to_str()?.to_string();
    let mut m = Map::new();
    m.insert("__pint__".to_string(), Value::Bool(true));

    let generic = numpy_generic_cls(py)?;
    if mag.is_instance(&generic).unwrap_or(false) {
        // numpy scalar (np.float64, ...): record the private scalar dtype.
        let dtype = mag.getattr("dtype")?.str()?.to_str()?.to_string();
        let item = mag.call_method0("item")?;
        m.insert("magnitude".to_string(), py_to_value(&item)?);
        m.insert("units".to_string(), Value::Str(units));
        m.insert("__viva_scalar_dtype__".to_string(), Value::Str(dtype));
    } else if mag.hasattr("dtype").unwrap_or(false) && mag.hasattr("tolist").unwrap_or(false) {
        // numpy ndarray magnitude: upstream array shape.
        let dtype = mag.getattr("dtype")?.str()?.to_str()?.to_string();
        let lowered = mag.call_method0("tolist")?;
        m.insert("magnitude".to_string(), py_to_value(&lowered)?);
        m.insert("units".to_string(), Value::Str(units));
        m.insert("__magnitude_dtype__".to_string(), Value::Str(dtype));
    } else {
        // Python scalar magnitude (float/int): no dtype key -> rebuilds as a
        // Python scalar.
        m.insert("magnitude".to_string(), py_to_value(&mag)?);
        m.insert("units".to_string(), Value::Str(units));
    }
    Ok(Value::Tagged(TagKind::Pint, m))
}

/// Rebuild a `pint.Quantity` from a `Value::Tagged(Pint, {..})`, restoring the
/// magnitude dtype (numpy scalar / ndarray / Python scalar) recorded by
/// [`quantity_to_value`] (or the upstream `__magnitude_dtype__` for arrays).
fn pint_to_py<'py>(py: Python<'py>, m: &Map<Value>) -> PyResult<Bound<'py, PyAny>> {
    let ureg = quantity_registry(py)?;
    let units = match m.get("units") {
        Some(Value::Str(s)) => s.as_str(),
        _ => return Err(perr("pint tag missing a `units` string")),
    };
    let unit_obj = parsed_unit(py, &ureg, units)?;
    let mag_val = m
        .get("magnitude")
        .or_else(|| m.get("value"))
        .ok_or_else(|| perr("pint tag missing a `magnitude`"))?;

    let np = py.import("numpy")?;
    let mag_py = if let Some(Value::Str(dt)) = m.get("__magnitude_dtype__") {
        // array magnitude: np.array(mag, dtype=dt).
        let lst = value_to_py(py, mag_val)?;
        np.getattr("array")?
            .call((lst,), Some(&[("dtype", dt.as_str())].into_py_dict(py)?))?
    } else if let Some(Value::Str(dt)) = m.get("__viva_scalar_dtype__") {
        // numpy scalar magnitude: np.dtype(dt).type(mag) -> e.g. np.float64(x).
        let scalar = value_to_py(py, mag_val)?;
        np.getattr("dtype")?
            .call1((dt.as_str(),))?
            .getattr("type")?
            .call1((scalar,))?
    } else {
        // Python scalar magnitude.
        value_to_py(py, mag_val)?
    };

    ureg.getattr("Quantity")?.call1((mag_py, unit_obj))
}

/// Parse (and cache) a pint unit string -> a registry Unit object. The same
/// ~23 unit strings recur every tick, so parsing them once removes the per-tick
/// `parse_units` cost.
fn parsed_unit<'py>(
    py: Python<'py>,
    ureg: &Bound<'py, PyAny>,
    units: &str,
) -> PyResult<Bound<'py, PyAny>> {
    static CACHE: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    let cache = CACHE
        .get_or_try_init(py, || -> PyResult<Py<PyDict>> {
            Ok(PyDict::new(py).unbind())
        })?
        .bind(py);
    if let Some(u) = cache.get_item(units)? {
        return Ok(u);
    }
    let unit = ureg.call_method1("parse_units", (units,))?;
    cache.set_item(units, &unit)?;
    Ok(unit)
}

/// Convert an arbitrary Python object (including numpy scalars/arrays) into a
/// [`Value`]. numpy objects are lowered via `.tolist()`.
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_exact_instance_of::<PyInt>() {
        return Ok(int_value(obj));
    }
    if obj.is_exact_instance_of::<PyFloat>() {
        return Ok(Value::Float(obj.extract::<f64>()?));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::Str(s.to_str()?.to_string()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in dict.iter() {
            m.insert(k.str()?.to_str()?.to_string(), py_to_value(&v)?);
        }
        return Ok(Value::Map(m));
    }
    // pint.Quantity: lower to a Pint-tagged map BEFORE the numpy `tolist`
    // fallback. An `np.float64`-magnitude Quantity's `tolist()` returns another
    // Quantity (the fallback would recurse); a Python-float-magnitude Quantity's
    // `tolist()` raises (it would then drop to `Null`, losing value + units).
    if let Some(q) = as_pint_quantity(obj) {
        return quantity_to_value(q);
    }
    // numpy scalar / ndarray: lower to a Python list/scalar, then recurse.
    if let Ok(tolist) = obj.getattr("tolist") {
        if tolist.is_callable() {
            let lowered = tolist.call0()?;
            return py_to_value(&lowered);
        }
    }
    // Last resort: numeric coercions.
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    Ok(Value::Null)
}

fn int_value(obj: &Bound<'_, PyAny>) -> Value {
    if let Ok(i) = obj.extract::<i64>() {
        Value::Int(i)
    } else if let Ok(u) = obj.extract::<u64>() {
        Value::UInt(u)
    } else {
        Value::Int(0)
    }
}

// ---- StoreNode -> Python (views + return state) ----------------------------

/// Convert a [`StoreNode`] to Python. `path`/`dtypes` give bulk/unique stores
/// their exact numpy dtypes (so `_entryState` stays `int8`, etc.). When
/// `structured` is false, bulk/unique are returned as plain dict-of-columns
/// (used for the returned final state, which the parity test compares).
pub fn storenode_to_py<'py>(
    py: Python<'py>,
    node: &StoreNode,
    path: &[String],
    dtypes: &DtypeMap,
    structured: bool,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match node {
        StoreNode::Leaf(v) => value_to_py(py, v)?,
        StoreNode::Branch(m) => {
            let d = PyDict::new(py);
            let mut sub = path.to_vec();
            for (k, n) in m {
                sub.push(k.clone());
                d.set_item(k, storenode_to_py(py, n, &sub, dtypes, structured)?)?;
                sub.pop();
            }
            d.into_any()
        }
        StoreNode::Bulk(b) => {
            if structured {
                // M2 zero-copy: a read-only view over `BulkStore.count`.
                let view = bulk_count_view(py, &b.count)?;
                let ids = bulk_ids(py, path, &b.ids, dtypes)?;
                py.import("viva_core._bulkview")?
                    .getattr("wrap")?
                    .call1((view, ids))?
            } else {
                let d = PyDict::new(py);
                d.set_item("id", &b.ids)?;
                d.set_item("count", b.count.to_vec())?;
                for (k, a) in &b.floats {
                    d.set_item(k, a.to_vec())?;
                }
                d.into_any()
            }
        }
        StoreNode::Unique(u) => {
            if structured {
                // M2 Task 2: a read-only zero-copy view over the interleaved buffer.
                unique_view(py, &u.buf, &u.layout, dtypes.get(path))?
            } else {
                let d = PyDict::new(py);
                let n = u.len();
                let itemsize = u.layout.itemsize;
                for fl in &u.layout.fields {
                    let list = PyList::empty(py);
                    for row in 0..n {
                        list.append(value_to_py(py, &read_value(&u.buf, itemsize, row, fl))?)?;
                    }
                    d.set_item(&fl.name, list)?;
                }
                d.set_item("_metadata", u.metadata.to_string())?;
                d.into_any()
            }
        }
    })
}

/// numpy `id` array for a bulk store. Built ONCE per store at engine build
/// (`DtypeMap::ids`, ids are immutable for a run); per tick this is an O(1)
/// refcount clone. Falls back to building (uncached) if the path has no handle.
fn bulk_ids<'py>(
    py: Python<'py>,
    path: &[String],
    ids: &[String],
    dtypes: &DtypeMap,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(h) = dtypes.ids.get(path) {
        return Ok(h.bind(py).clone());
    }
    build_bulk_ids(py, ids)
}

/// Build the numpy unicode `id` array for a bulk store's ids.
pub fn build_bulk_ids<'py>(py: Python<'py>, ids: &[String]) -> PyResult<Bound<'py, PyAny>> {
    py.import("viva_core._bulkview")?
        .getattr("make_ids")?
        .call1((PyList::new(py, ids)?,))
}

/// Cached numpy dtype `[("count", "<i8")]` (built once, not per tick).
fn count_dtype<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    static DT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    DT.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let descr = PyList::new(py, [PyTuple::new(py, ["count", "<i8"])?])?;
        Ok(py
            .import("numpy")?
            .getattr("dtype")?
            .call1((descr,))?
            .unbind())
    })
    .map(|d| d.bind(py).clone())
}

/// Zero-copy, READ-ONLY numpy structured-array view (`[("count","<i8")]`) over
/// the Rust-owned `BulkStore.count` buffer. No allocation or copy: the array's
/// data pointer is the `Array1<i64>`'s own storage.
///
/// Ownership/lifetime: Rust owns the buffer; the view is handed out for the
/// duration of one process call. The buffer is only ever written in place by
/// `Op::BulkDelta` (never reallocated mid-run) and only after the process has
/// returned, so a view used inside the call can never dangle. The view is
/// `writeable=False`: a process cannot mutate Rust state through it (writes go
/// back as update dicts -> `Op::BulkDelta`). A process must not retain the view
/// past the run (v2ecoli processes return updates; they do not stash views).
fn bulk_count_view<'py>(py: Python<'py>, count: &Array1<i64>) -> PyResult<Bound<'py, PyAny>> {
    if count.as_slice().is_none() {
        // Non-contiguous (never produced by M1 loaders): fall back to a copy.
        let owned = PyArray1::from_vec(py, count.to_vec());
        return owned.call_method1("view", (count_dtype(py)?,));
    }
    let pin = PyCapsule::new_with_value(py, (), c"viva_core.bulk_pin")?;
    // SAFETY: `count` is contiguous and outlives this call (see doc above); the
    // array is made read-only before it is exposed.
    let base = unsafe { PyArray1::borrow_from_array(count, pin.into_any()) };
    base.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
    base.call_method1("view", (count_dtype(py)?,))
}

/// The `(name, dtype_str)` descriptor pairs for a unique store's record dtype,
/// in `layout` field order, preferring the exact `.pbg` dtype string for each
/// field (so numpy idioms like `arr["_entryState"].view(bool)` behave exactly as
/// in pb), else the field kind's canonical numpy name.
fn dtype_descr(
    layout: &RecordLayout,
    dtypes: Option<&Vec<(String, String)>>,
) -> Vec<(String, String)> {
    layout
        .fields
        .iter()
        .map(|fl| {
            let dt = dtypes
                .and_then(|ds| {
                    ds.iter()
                        .find(|(n, _)| *n == fl.name)
                        .map(|(_, d)| d.clone())
                })
                .unwrap_or_else(|| {
                    // No explicit `.pbg` dtype: a scalar is its canonical numpy
                    // name; a sub-array reconstructs the tuple form so its width
                    // survives (sub-arrays normally carry an explicit dtype).
                    if fl.elems > 1 {
                        format!("('{}', ({},))", fl.kind.numpy_dtype(), fl.elems)
                    } else {
                        fl.kind.numpy_dtype().to_string()
                    }
                });
            (fl.name.clone(), dt)
        })
        .collect()
}

/// Append one numpy dtype `descr` entry for a field to `list`. A scalar field is
/// a `(name, dtype)` 2-tuple; a 1-D sub-array field (`('<i4', (2,))`) becomes
/// numpy's `(name, base_dtype, (elems,))` 3-tuple — the 2-tuple string form is
/// rejected by `np.dtype`, and the 3-tuple is what makes numpy's own
/// itemsize/offsets match the Rust sub-array layout (cross-checked below).
fn push_descr_entry(
    py: Python<'_>,
    list: &Bound<'_, PyList>,
    name: &str,
    dstr: &str,
) -> PyResult<()> {
    if let Some((base, elems)) = parse_subarray_dtype(dstr) {
        list.append((name, base, (elems,)).into_pyobject(py)?)?;
    } else {
        list.append(PyTuple::new(py, [name, dstr])?)?;
    }
    Ok(())
}

/// Process-wide cache of built record `np.dtype`s, keyed by descriptor
/// signature. A unique store hands out a view every tick; rebuilding (and
/// re-validating) the `PyArrayDescr` each time is pure Python-call overhead, so
/// the dtype is built + offset-gated ONCE per distinct layout and reused.
fn cached_record_dtype<'py>(
    py: Python<'py>,
    layout: &RecordLayout,
    dtypes: Option<&Vec<(String, String)>>,
) -> PyResult<Bound<'py, PyAny>> {
    static CACHE: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    let cache = CACHE.get_or_try_init(py, || -> PyResult<Py<PyDict>> {
        Ok(PyDict::new(py).unbind())
    })?;
    let cache = cache.bind(py);

    let descr = dtype_descr(layout, dtypes);
    let key: String = descr
        .iter()
        .map(|(n, d)| format!("{n}={d}"))
        .collect::<Vec<_>>()
        .join(";");

    if let Some(d) = cache.get_item(&key)? {
        return Ok(d);
    }

    let np = py.import("numpy")?;
    let descr_list = PyList::empty(py);
    for (name, dt) in &descr {
        push_descr_entry(py, &descr_list, name, dt)?;
    }
    let dtype = np.getattr("dtype")?.call1((descr_list,))?;

    // Gate ONCE per distinct dtype: the record stride and every field offset,
    // derived from the built PyArrayDescr, must match the Rust buffer layout, or
    // field reads would silently corrupt.
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    if itemsize != layout.itemsize {
        return Err(perr(format!(
            "unique dtype itemsize {itemsize} != Rust record stride {}",
            layout.itemsize
        )));
    }
    let fields_obj = dtype.getattr("fields")?;
    for fl in &layout.fields {
        let entry = fields_obj.get_item(&fl.name)?;
        let offset: usize = entry.get_item(1)?.extract()?;
        if offset != fl.offset {
            return Err(perr(format!(
                "unique field {:?} numpy offset {offset} != Rust offset {}",
                fl.name, fl.offset
            )));
        }
    }

    cache.set_item(&key, &dtype)?;
    Ok(dtype)
}

/// READ-ONLY, zero-copy numpy structured-array view over a [`UniqueStore`]'s
/// interleaved byte buffer.
///
/// # Lifetime / safety model (the realloc-safe pin)
///
/// The buffer is an `Arc<Vec<u8>>`. This view's numpy `base` is a `PyCapsule`
/// that OWNS an `Arc` clone of that buffer, so the exact allocation this view's
/// data pointer references stays alive for as long as the view (or any Python
/// object derived from it) is alive — its refcount cannot reach zero first.
///
/// The buffer is mutated only by the unique apply path, and only through
/// `Arc::make_mut`: once this view exists the Arc refcount is >= 2, so the next
/// `set`/`add`/`delete`/grow **copies-on-write into a fresh allocation** and
/// leaves this view's allocation untouched. Therefore a view handed to Python
/// can never observe a reallocation of its own bytes and can never dangle, even
/// if the process RETAINS it across the add/grow that reallocates the store
/// (exercised by `test_unique_view_survives_realloc`). The retained view simply
/// keeps showing the pre-mutation snapshot, which is sound because the view is
/// read-only (`writeable=False`): writes go back as update dicts on the Op path.
///
/// Offsets are NOT hand-trusted: the record stride and every field offset are
/// derived from a built `PyArrayDescr` and checked against the Rust `layout`
/// (once per distinct dtype, in [`cached_record_dtype`]), so a layout/numpy
/// disagreement fails loud instead of silently corrupting a field read.
fn unique_view<'py>(
    py: Python<'py>,
    buf: &Arc<Vec<u8>>,
    layout: &RecordLayout,
    dtypes: Option<&Vec<(String, String)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = cached_record_dtype(py, layout, dtypes)?;

    let np = py.import("numpy")?;
    if buf.is_empty() {
        // Zero rows: no bytes to borrow; hand back an empty structured array.
        return np.getattr("zeros")?.call1((0usize, dtype));
    }

    // Pin the CURRENT allocation by giving the view a base object that owns an
    // Arc clone of the buffer (see the safety note above).
    let pin = PyCapsule::new_with_value(py, Arc::clone(buf), c"viva_core.unique_pin")?;
    let bytes: &[u8] = buf.as_slice();
    let av = aview1(bytes);
    // SAFETY: `av`'s data pointer is the pinned Arc allocation's bytes; the
    // capsule keeps that allocation alive as long as the numpy array (and its
    // views) live, and `Arc::make_mut` in apply copies-on-write rather than
    // reallocating in place while this base is alive. The array is made
    // read-only before exposure.
    let base = unsafe { PyArray1::<u8>::borrow_from_array(&av, pin.into_any()) };
    base.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
    base.call_method1("view", (dtype,))
}

/// Derive a [`RecordLayout`] from a built `PyArrayDescr` (rust-numpy): field
/// offsets + record itemsize come straight from numpy, not hand arithmetic.
/// Installed on each unique store at engine-build time so the runtime
/// byte-offset arithmetic uses numpy-authoritative offsets.
pub fn derive_layout(py: Python<'_>, dtypes: &[(String, String)]) -> PyResult<RecordLayout> {
    let np = py.import("numpy")?;
    let descr = PyList::empty(py);
    for (name, dstr) in dtypes {
        push_descr_entry(py, &descr, name, dstr)?;
    }
    let dtype = np.getattr("dtype")?.call1((descr,))?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    let fields_obj = dtype.getattr("fields")?;
    let mut fields = Vec::with_capacity(dtypes.len());
    for (name, dstr) in dtypes {
        let Some((kind, elems)) = parse_field_dtype(dstr) else {
            // Non-fixed-width field (string / N-D sub-array): not a zero-copy
            // unique store; fail loud (the store kept its decoded layout).
            return Err(perr(format!(
                "unique field {name:?} dtype {dstr:?} is not fixed-width"
            )));
        };
        let entry = fields_obj.get_item(name)?;
        let offset: usize = entry.get_item(1)?.extract()?;
        fields.push(FieldLayout {
            name: name.clone(),
            kind,
            offset,
            elems,
            size: kind.size() * elems,
        });
    }
    Ok(RecordLayout { fields, itemsize })
}

/// Build the Python view dict handed to a process: each wired input port
/// resolved to its store, scalars/maps as native Python, bulk/unique as numpy
/// structured arrays.
pub fn build_view<'py>(
    py: Python<'py>,
    view: &PortView<'_>,
    dtypes: &DtypeMap,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for port in view.port_names() {
        let node = view.get(&[port]);
        let path = view.path(&[port]).unwrap_or_default();
        if let Some(node) = node {
            out.set_item(port, storenode_to_py(py, node, &path, dtypes, true)?)?;
        } else {
            out.set_item(port, py.None())?;
        }
    }
    Ok(out)
}

/// Convert the whole final [`State`] to a Python dict (plain columns; the parity
/// test compares this against the pb `Composite.state`).
pub fn state_to_py<'py>(py: Python<'py>, state: &State) -> PyResult<Bound<'py, PyDict>> {
    let empty = DtypeMap::new();
    let d = PyDict::new(py);
    let mut path = Vec::new();
    for (k, n) in &state.root {
        path.push(k.clone());
        d.set_item(k, storenode_to_py(py, n, &path, &empty, false)?)?;
        path.pop();
    }
    Ok(d)
}

// ---- update dict -> typed Op ----------------------------------------------

/// A process output-port schema: a leaf type, or a nested sub-schema tree.
pub enum OutSchema {
    Leaf(Type),
    Nested(Vec<(String, OutSchema)>),
}

/// Parse a Python `outputs()` schema value into an [`OutSchema`].
pub fn parse_out_schema(obj: &Bound<'_, PyAny>) -> PyResult<OutSchema> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(OutSchema::Leaf(parse_leaf_type(s.to_str()?)));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        if let Some(t) = dict.get_item("_type")? {
            if let Ok(s) = t.cast::<PyString>() {
                return Ok(OutSchema::Leaf(parse_leaf_type(s.to_str()?)));
            }
        }
        let mut entries = Vec::new();
        for (k, v) in dict.iter() {
            let key = k.str()?.to_str()?.to_string();
            if key.starts_with('_') {
                continue; // _default / _emit / etc.
            }
            entries.push((key, parse_out_schema(&v)?));
        }
        return Ok(OutSchema::Nested(entries));
    }
    Ok(OutSchema::Leaf(Type::Node))
}

fn parse_leaf_type(s: &str) -> Type {
    parse_type(s).unwrap_or(Type::Named(s.to_string()))
}

/// Walk a returned update value against the output schema, appending typed
/// [`PortUpdate`]s. `port_path` accumulates the (sub-)port keys.
pub fn translate_update(
    update: &Bound<'_, PyAny>,
    schema: &OutSchema,
    port_path: &mut Vec<String>,
    out: &mut Vec<PortUpdate>,
) -> PyResult<()> {
    match schema {
        OutSchema::Leaf(ty) => {
            let op = op_from_type(ty, update)?;
            if let Some(op) = op {
                out.push(PortUpdate {
                    port: port_path.clone(),
                    op,
                });
            }
        }
        OutSchema::Nested(entries) => {
            let Ok(dict) = update.cast::<PyDict>() else {
                return Ok(());
            };
            for (key, sub) in entries {
                if let Some(v) = dict.get_item(key)? {
                    port_path.push(key.clone());
                    translate_update(&v, sub, port_path, out)?;
                    port_path.pop();
                }
            }
        }
    }
    Ok(())
}

/// Map a (type, Python value) to the Rust [`Op`]. Returns `None` for a no-op.
fn op_from_type(ty: &Type, value: &Bound<'_, PyAny>) -> PyResult<Option<Op>> {
    let op = match ty {
        Type::Float { .. } => Op::FloatDelta(value.extract::<f64>()?),
        Type::Integer { .. } => Op::IntDelta(value.extract::<i64>()? as i128),
        Type::String => Op::SetStr(value.str()?.to_str()?.to_string()),
        Type::Boolean => Op::SetBool(value.is_truthy()?),
        Type::Overwrite(_) => Op::Overwrite(py_to_value(value)?),
        Type::BulkArray { .. } => bulk_op(value)?,
        Type::UniqueArray { .. } => unique_op(value)?,
        // Named/other: detect a unique-update dict by shape, else overwrite.
        _ => {
            if is_unique_update(value)? {
                unique_op(value)?
            } else {
                Op::Overwrite(py_to_value(value)?)
            }
        }
    };
    Ok(Some(op))
}

fn is_unique_update(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(dict) = value.cast::<PyDict>() {
        for key in ["set", "add", "delete", "update"] {
            if dict.contains(key)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Parse a v2ecoli `BulkNumpyUpdate` into [`Op::BulkDelta`]. The update is a
/// list of `(index, delta)` pairs; faithful to `bulk_numpy_updater`
/// (`state["count"][idx] += value`), **each element of a pair may be a scalar
/// OR a numpy array** — the real v2ecoli hot path returns one vectorized pair
/// `(np.array(indices), np.array(deltas))`. Array/array pairs are zipped
/// element-wise; a scalar broadcasts against the array side.
fn bulk_op(value: &Bound<'_, PyAny>) -> PyResult<Op> {
    let mut deltas: Vec<(usize, i128)> = Vec::new();
    let v = py_to_value(value)?;
    if let Value::Array(pairs) = v {
        for pair in pairs {
            if let Value::Array(p) = pair {
                if p.len() >= 2 {
                    push_bulk_pair(&p[0], &p[1], &mut deltas);
                }
            }
        }
    }
    Ok(Op::BulkDelta(deltas))
}

/// Expand one `(index, delta)` pair into flat `(usize, i128)` deltas, matching
/// numpy's `count[idx] += value` broadcasting for scalar/array operands.
fn push_bulk_pair(idx: &Value, delta: &Value, out: &mut Vec<(usize, i128)>) {
    let to_idx = |v: &Value| as_i128(v).max(0) as usize;
    match (idx, delta) {
        (Value::Array(ix), Value::Array(dv)) => {
            for (i, d) in ix.iter().zip(dv.iter()) {
                out.push((to_idx(i), as_i128(d)));
            }
        }
        (Value::Array(ix), d) => {
            let dd = as_i128(d);
            for i in ix {
                out.push((to_idx(i), dd));
            }
        }
        (i, Value::Array(dv)) => {
            let ii = to_idx(i);
            for d in dv {
                out.push((ii, as_i128(d)));
            }
        }
        (i, d) => out.push((to_idx(i), as_i128(d))),
    }
}

fn as_i128(v: &Value) -> i128 {
    match v {
        Value::Int(i) => *i as i128,
        Value::UInt(u) => *u as i128,
        Value::Float(f) => *f as i128,
        Value::Bool(b) => *b as i128,
        _ => 0,
    }
}

/// Parse a v2ecoli unique-store update dict (`set`/`add`/`delete`/`update`)
/// into [`Op::Unique`].
fn unique_op(value: &Bound<'_, PyAny>) -> PyResult<Op> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| perr("unique update must be a dict"))?;
    let mut u = UniqueOp::default();
    if let Some(s) = dict.get_item("set")? {
        u.set = parse_field_dicts(&s)?;
    }
    if let Some(a) = dict.get_item("add")? {
        u.add = parse_field_dicts(&a)?;
    }
    if let Some(d) = dict.get_item("delete")? {
        u.delete = parse_delete(&d)?;
    }
    if let Some(up) = dict.get_item("update")? {
        u.flush = up.is_truthy()?;
    }
    Ok(Op::Unique(u))
}

/// Parse `set`/`add`: a dict `{field: array}` or a list of such dicts, into a
/// `Vec<Map<field, Vec<Value>>>`.
fn parse_field_dicts(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Map<Vec<Value>>>> {
    let mut out = Vec::new();
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            out.push(parse_one_field_dict(&item)?);
        }
    } else if obj.cast::<PyDict>().is_ok() {
        out.push(parse_one_field_dict(obj)?);
    }
    Ok(out)
}

fn parse_one_field_dict(obj: &Bound<'_, PyAny>) -> PyResult<Map<Vec<Value>>> {
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| perr("unique set/add entry must be a dict"))?;
    let mut m = Map::new();
    for (k, v) in dict.iter() {
        let field = k.str()?.to_str()?.to_string();
        let vals = match py_to_value(&v)? {
            Value::Array(a) => a,
            other => vec![other],
        };
        m.insert(field, vals);
    }
    Ok(m)
}

/// Parse `delete`: a list of ints, or a list of lists of ints.
fn parse_delete(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let v = py_to_value(obj)?;
    let as_usize = |x: &Value| -> usize { as_i128(x).max(0) as usize };
    match v {
        Value::Array(items) => {
            if items.iter().all(|x| matches!(x, Value::Array(_))) {
                Ok(items
                    .into_iter()
                    .map(|inner| match inner {
                        Value::Array(a) => a.iter().map(as_usize).collect(),
                        _ => Vec::new(),
                    })
                    .collect())
            } else {
                Ok(vec![items.iter().map(as_usize).collect()])
            }
        }
        _ => Ok(Vec::new()),
    }
}
