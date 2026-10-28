//! `viva_core`: the PyO3 wheel where the **Rust** process-bigraph engine runs a
//! **real v2ecoli `.pbg`** with all processes still in **Python** (quadrant 3),
//! bit-identical to the Python pb 1.8.3 engine.
//!
//! Each tick, for every process/step edge the engine resolves the edge's
//! `local:<Name>` address, imports the Python class, instantiates it with the
//! edge config, and calls it through a [`PyProcess`] shim: it builds a view of
//! the wired input stores (scalars/maps as native Python; bulk & unique as
//! numpy structured arrays, with the hot `count` column shared zero-copy where
//! the dtype allows), honours the Python gate (`perform_update` /
//! `calculate_timestep`), then translates the returned update dict into the
//! typed Rust [`Op`]s that Task-3 `apply`/`reconcile` applies.
//!
//! The JSON contract is the single source of truth: the same `.pbg` runs under
//! `process_bigraph.Composite` and under [`run`], and the decoded states are
//! compared bit-identically.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};

use bigraph_schema_rs::codec;
use bigraph_schema_rs::doc::{CompositeSpec, Map, Node, TagKind, Value};
use bigraph_schema_rs::state::{from_initial, StoreNode};
use process_bigraph_rs::engine::Engine;
use process_bigraph_rs::process::Process;
use process_bigraph_rs::registry::ProcessRegistry;

mod convert;
mod processes;
mod pyproc;
mod rust_process;

use pyproc::PyProcess;

/// Numpy dtypes of every tagged structured store, keyed by absolute store path.
/// Used to rebuild bulk/unique views with the exact field dtypes the real `.pbg`
/// carried (e.g. `_entryState` as `int8`, so `attrs`' `.view(bool)` works).
#[derive(Default)]
pub struct DtypeMap {
    dtypes: HashMap<Vec<String>, Vec<(String, String)>>,
    /// Prebuilt numpy `id` array per bulk store (immutable for a run).
    pub ids: HashMap<Vec<String>, Py<PyAny>>,
}

impl DtypeMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, path: &[String]) -> Option<&Vec<(String, String)>> {
        self.dtypes.get(path)
    }
    pub fn insert(&mut self, path: Vec<String>, d: Vec<(String, String)>) {
        self.dtypes.insert(path, d);
    }
}

fn perr(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Parse the spec argument: a path to a `.pbg` file, or an inline JSON string.
fn load_spec(arg: &str) -> PyResult<CompositeSpec> {
    let text = if arg.trim_start().starts_with('{') {
        arg.to_string()
    } else {
        std::fs::read_to_string(arg).map_err(|e| perr(format!("reading {arg:?}: {e}")))?
    };
    codec::from_json(&text).map_err(|e| perr(format!("parsing composite spec: {e}")))
}

/// Collect the numpy dtypes of every tagged structured store, by absolute path.
fn collect_dtypes(spec: &CompositeSpec) -> DtypeMap {
    fn walk(m: &Map<Node>, path: &mut Vec<String>, out: &mut DtypeMap) {
        for (k, n) in m {
            path.push(k.clone());
            match n {
                Node::Store(s) => walk(s, path, out),
                Node::Leaf(Value::Tagged(
                    TagKind::StructuredArray | TagKind::NumpyStructured,
                    tag,
                )) => {
                    if let Some(Value::Array(entries)) = tag.get("dtype") {
                        let dtypes = entries
                            .iter()
                            .filter_map(|e| match e {
                                Value::Array(pair) if pair.len() >= 2 => {
                                    match (&pair[0], &pair[1]) {
                                        (Value::Str(n), Value::Str(d)) => {
                                            Some((n.clone(), d.clone()))
                                        }
                                        _ => None,
                                    }
                                }
                                _ => None,
                            })
                            .collect();
                        out.insert(path.clone(), dtypes);
                    }
                }
                _ => {}
            }
            path.pop();
        }
    }
    let mut out = DtypeMap::new();
    walk(&spec.root, &mut Vec::new(), &mut out);
    out
}

/// Build the engine for a spec: decode the initial state (incl. tagged
/// structured arrays) and register a [`PyProcess`] factory for every `local:`
/// Python edge (emitters stay native Rust).
/// Walk the state tree and install each unique store's PyArrayDescr-derived
/// record layout (field offsets + itemsize straight from numpy), so the runtime
/// byte-offset arithmetic uses numpy-authoritative offsets rather than any
/// hand-computed ones. A no-op re-pack when the decoded layout already matches
/// numpy's packed layout (the common case).
fn install_unique_layouts(
    py: Python<'_>,
    m: &mut Map<StoreNode>,
    path: &mut Vec<String>,
    dtypes: &DtypeMap,
) -> PyResult<()> {
    for (k, n) in m.iter_mut() {
        path.push(k.clone());
        match n {
            StoreNode::Branch(b) => install_unique_layouts(py, b, path, dtypes)?,
            StoreNode::Unique(u) => {
                if let Some(ds) = dtypes.get(path.as_slice()) {
                    let layout = convert::derive_layout(py, ds)?;
                    u.relayout(layout);
                }
            }
            _ => {}
        }
        path.pop();
    }
    Ok(())
}

/// Build each bulk store's numpy `id` array once (ids never change in a run).
fn collect_bulk_ids(
    py: Python<'_>,
    m: &Map<StoreNode>,
    path: &mut Vec<String>,
    out: &mut DtypeMap,
) -> PyResult<()> {
    for (k, n) in m {
        path.push(k.clone());
        match n {
            StoreNode::Branch(b) => collect_bulk_ids(py, b, path, out)?,
            StoreNode::Bulk(b) => {
                let arr = convert::build_bulk_ids(py, &b.ids)?;
                out.ids.insert(path.clone(), arr.unbind());
            }
            _ => {}
        }
        path.pop();
    }
    Ok(())
}

fn build_engine(spec: &CompositeSpec) -> PyResult<Engine> {
    let mut dtype_map = collect_dtypes(spec);
    let mut state = from_initial(spec);
    Python::attach(|py| -> PyResult<()> {
        collect_bulk_ids(py, &state.root, &mut Vec::new(), &mut dtype_map)?;
        install_unique_layouts(py, &mut state.root, &mut Vec::new(), &dtype_map)
    })?;
    let dtypes = Arc::new(dtype_map);

    let mut registry = ProcessRegistry::with_builtin_emitters();
    let mut addresses: Vec<String> = Vec::new();
    for (_path, edge) in spec.edges() {
        if !is_emitter(&edge.address) && !addresses.contains(&edge.address) {
            addresses.push(edge.address.clone());
        }
    }
    for address in addresses {
        // Native (all-Rust) twin, when enabled by VIVA_NATIVE_PROCESSES: same
        // address, registered INSTEAD of the PyProcess fallback.
        if let Some(ctor) = processes::native_ctor(&address) {
            registry.register(&address, ctor);
            continue;
        }
        let addr = address.clone();
        let dmap = Arc::clone(&dtypes);
        registry.register(&address, move |cfg| {
            Python::attach(|py| PyProcess::create(py, &addr, cfg, Arc::clone(&dmap)))
                .map(|p| Box::new(p) as Box<dyn Process>)
        });
    }

    Engine::with_state(spec, state, &registry).map_err(|e| perr(format!("building engine: {e}")))
}

fn is_emitter(address: &str) -> bool {
    address.contains("Emitter")
        || address.ends_with("ram-emitter")
        || address.ends_with("json-emitter")
}

/// Run a real `.pbg` (path or inline JSON) for `n_ticks` scheduler iterations
/// and return the decoded final state as a Python dict.
#[pyfunction]
#[pyo3(signature = (spec, n_ticks))]
fn run(py: Python<'_>, spec: &str, n_ticks: usize) -> PyResult<Py<PyDict>> {
    let parsed = load_spec(spec)?;
    let mut engine = build_engine(&parsed)?;
    engine
        .run(n_ticks)
        .map_err(|e| perr(format!("running engine: {e}")))?;
    convert::state_to_py(py, &engine.state).map(|d| d.unbind())
}

/// Like [`run`], but also returns the differential-gating diagnostics: the
/// per-cascade step-run log and per-emitter row counts. Returns
/// `{"state", "step_runs", "emitter_counts"}`.
#[pyfunction]
#[pyo3(signature = (spec, n_ticks))]
fn run_report(py: Python<'_>, spec: &str, n_ticks: usize) -> PyResult<Py<PyDict>> {
    let parsed = load_spec(spec)?;
    let mut engine = build_engine(&parsed)?;
    engine
        .run(n_ticks)
        .map_err(|e| perr(format!("running engine: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("state", convert::state_to_py(py, &engine.state)?)?;

    let step_runs = PyList::empty(py);
    for sr in &engine.step_run_log {
        let name = sr.path.last().cloned().unwrap_or_default();
        let row = PyTuple::new(
            py,
            [
                name.into_pyobject(py)?.into_any(),
                sr.global_time.into_pyobject(py)?.into_any(),
                PyBool::new(py, sr.ran).to_owned().into_any(),
            ],
        )?;
        step_runs.append(row)?;
    }
    out.set_item("step_runs", step_runs)?;

    // Emitter row counts, by edge name.
    let emitters = PyDict::new(py);
    for (path, edge) in parsed.edges() {
        if is_emitter(&edge.address) {
            let abs: Vec<&str> = path.iter().map(String::as_str).collect();
            if let Some(hist) = engine.emitter_history(&abs) {
                let name = path.last().cloned().unwrap_or_default();
                emitters.set_item(name, hist.len())?;
            }
        }
    }
    out.set_item("emitter_counts", emitters)?;
    Ok(out.unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_report, m)?)?;
    m.add_class::<rust_process::RustGrowCore>()?;
    Ok(())
}
