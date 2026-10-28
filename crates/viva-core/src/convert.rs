//! Conversions across the Python boundary: [`Value`]/[`StoreNode`] <-> Python,
//! numpy structured-array view building, and update-dict -> typed [`Op`]
//! translation.

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use bigraph_schema_rs::apply::{Op, UniqueOp};
use bigraph_schema_rs::doc::{Map, Value};
use bigraph_schema_rs::grammar::parse_type;
use bigraph_schema_rs::state::{Column, State, StoreNode};
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
        Value::Map(m) | Value::Tagged(_, m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, value_to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
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
                let mut cols: Vec<(String, &Column)> = Vec::new();
                let count = Column::Int(b.count.clone());
                let ids = Column::Str(b.ids.clone());
                cols.push(("id".into(), &ids));
                cols.push(("count".into(), &count));
                let float_cols: Vec<(String, Column)> = b
                    .floats
                    .iter()
                    .map(|(k, a)| (k.clone(), Column::Float(a.clone())))
                    .collect();
                for (k, c) in &float_cols {
                    cols.push((k.clone(), c));
                }
                build_structured_array(py, &cols, dtypes.get(path))?
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
                let cols: Vec<(String, &Column)> =
                    u.columns.iter().map(|(k, c)| (k.clone(), c)).collect();
                build_structured_array(py, &cols, dtypes.get(path))?
            } else {
                let d = PyDict::new(py);
                for (k, c) in &u.columns {
                    d.set_item(k, column_to_py(py, c)?)?;
                }
                d.set_item("_metadata", u.metadata.to_string())?;
                d.into_any()
            }
        }
    })
}

fn column_to_py<'py>(py: Python<'py>, c: &Column) -> PyResult<Bound<'py, PyAny>> {
    Ok(match c {
        Column::Int(a) => a.to_vec().into_pyobject(py)?.into_any(),
        Column::Float(a) => a.to_vec().into_pyobject(py)?.into_any(),
        Column::Bool(v) => v.clone().into_pyobject(py)?.into_any(),
        Column::Str(v) => v.clone().into_pyobject(py)?.into_any(),
        Column::Nested(v) => {
            let list = PyList::empty(py);
            for x in v {
                list.append(value_to_py(py, x)?)?;
            }
            list.into_any()
        }
    })
}

/// Build a numpy structured array for a bulk/unique store, with the field
/// dtypes from the original `.pbg` (so numpy views like
/// `arr["_entryState"].view(bool)` behave exactly as in pb).
fn build_structured_array<'py>(
    py: Python<'py>,
    columns: &[(String, &Column)],
    dtypes: Option<&Vec<(String, String)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let length = columns.first().map_or(0, |(_, c)| c.len());

    // Build the dtype descriptor: [(name, dtype_str), ...]. Prefer the real
    // `.pbg` dtype for each field; otherwise infer from the column kind.
    let descr = PyList::empty(py);
    for (name, col) in columns {
        let dt = dtypes
            .and_then(|ds| ds.iter().find(|(n, _)| n == name).map(|(_, d)| d.clone()))
            .unwrap_or_else(|| infer_dtype(col, length));
        if dt.starts_with('(') {
            return Err(perr(format!(
                "sub-array field dtype {dt:?} on {name:?} not yet supported in the \
                 Python view (Task-5 scope: no model used needs it)"
            )));
        }
        descr.append(PyTuple::new(py, [name.as_str(), dt.as_str()])?)?;
    }
    let dtype = np.getattr("dtype")?.call1((descr,))?;
    let arr = np.getattr("zeros")?.call1((length, dtype))?;

    for (name, col) in columns {
        let values = column_to_py(py, col)?;
        arr.set_item(name.as_str(), values)?;
    }
    Ok(arr)
}

fn infer_dtype(col: &Column, length: usize) -> String {
    match col {
        Column::Int(_) => "int64".into(),
        Column::Float(_) => "float64".into(),
        Column::Bool(_) => "bool".into(),
        Column::Str(_) => format!("<U{}", length.max(1)),
        Column::Nested(_) => "O".into(),
    }
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
