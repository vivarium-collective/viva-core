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
use process_bigraph_rs::process::{wire_path, Process};
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
    /// Python-owned live objects per store path (M5 Task 2, same pattern/lifetime
    /// as [`ids`]): the `(instance,)` tuple for a `["process", <name>]` store and
    /// the `numpy.random.RandomState` for `["allocator_rng"]`. Consulted at the
    /// view boundary ([`convert::storenode_to_py`]) — Rust NEVER serializes these;
    /// the state JSON carries only a placeholder at these paths.
    pub handles: HashMap<Vec<String>, Py<PyAny>>,
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

/// Edge wire map (`inputs`/`outputs`) as a `Map<Value>`, or empty.
fn edge_wires(w: &Option<Value>) -> Map<Value> {
    match w {
        Some(Value::Map(m)) => m.clone(),
        _ => Map::new(),
    }
}

/// Collect every leaf port path of a wire tree (the nested port-key sequences a
/// write/read addresses).
fn collect_leaf_ports(wires: &Map<Value>, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    for (k, v) in wires {
        prefix.push(k.clone());
        match v {
            Value::Map(m) => collect_leaf_ports(m, prefix, out),
            _ => out.push(prefix.clone()),
        }
        prefix.pop();
    }
}

/// Fail loud if any edge writes a handle path through wiring that is not
/// symmetric between inputs and outputs.
///
/// The update-boundary DROP ([`convert::translate_update`]) decides whether a
/// write targets a handle path by resolving the output port through the INPUT
/// wires ([`PortView::path`]), whereas the engine routes the actual write through
/// the OUTPUT wires ([`Engine::map_updates`]). That is sound ONLY because a handle
/// port (e.g. `process`, read AND written) wires identically on both sides. This
/// checks the invariant at build time, so an asymmetric wiring — an output port
/// landing on a handle path whose same-port input wiring is absent or different —
/// fails here instead of silently misrouting (and corrupting) a handle write.
fn check_handle_wire_symmetry(
    spec: &CompositeSpec,
    handles: &HashMap<Vec<String>, Py<PyAny>>,
) -> PyResult<()> {
    for (path, edge) in spec.edges() {
        let base = &path[..path.len().saturating_sub(1)];
        let inputs = edge_wires(&edge.inputs);
        let outputs = edge_wires(&edge.outputs);
        let mut ports = Vec::new();
        collect_leaf_ports(&outputs, &mut Vec::new(), &mut ports);
        for port in &ports {
            let Some(abs) = wire_path(&outputs, base, port) else {
                continue;
            };
            if !handles.contains_key(&abs) {
                continue;
            }
            let in_abs = wire_path(&inputs, base, port);
            if in_abs.as_ref() != Some(&abs) {
                return Err(perr(format!(
                    "handle port {port:?} on edge {path:?} writes the handle path {abs:?}, \
                     but its INPUT wiring resolves to {in_abs:?}; the update-boundary drop \
                     keys off input wires and would miss this write (asymmetric handle wiring)"
                )));
            }
        }
    }
    Ok(())
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
    // The JSON `run` path has no live-object handles.
    convert::state_to_py(py, &engine.state, &DtypeMap::new()).map(|d| d.unbind())
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
    out.set_item(
        "state",
        convert::state_to_py(py, &engine.state, &DtypeMap::new())?,
    )?;

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

/// Build an [`Engine`] from a Python->Rust composite handoff (Milestone 4): the
/// state + topology come as two JSON specs (the state stores, and the
/// process/step EDGES without their non-serializable configs), while the process
/// instances themselves are handed over ALREADY CONSTRUCTED in `processes`
/// (keyed by the edge's absolute path joined with `/`), carrying their live
/// `sim_data`. Each edge becomes: a native twin (if its address is in
/// `native_addresses` and `VIVA_NATIVE_PROCESSES` enables it), a builtin emitter
/// (emitter addresses), or an instance-bound [`PyProcess`] driving the given
/// instance. See [`run_prebuilt`].
fn build_prebuilt_engine(
    py: Python<'_>,
    processes: &Bound<'_, PyDict>,
    topology: &str,
    state: &str,
    native_addresses: &[String],
    handles: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Engine, Arc<DtypeMap>)> {
    // Merge the two specs into one composite: state stores first, then the edges.
    let mut combined = load_spec(state)?;
    let topo = load_spec(topology)?;
    combined.root.extend(topo.root);

    let mut dtype_map = collect_dtypes(&combined);
    let mut st = from_initial(&combined);
    collect_bulk_ids(py, &st.root, &mut Vec::new(), &mut dtype_map)?;
    install_unique_layouts(py, &mut st.root, &mut Vec::new(), &dtype_map)?;
    // M5 Task 2: install the Python-owned handle objects, keyed by absolute store
    // path (the `/`-joined key from `build_payload`). Each is held as a `Py<PyAny>`
    // and handed back verbatim at the view boundary — never serialized.
    if let Some(handles) = handles {
        for (key, obj) in handles.iter() {
            let path: Vec<String> = key
                .str()?
                .to_str()?
                .split('/')
                .map(str::to_string)
                .collect();
            dtype_map.handles.insert(path, obj.unbind());
        }
    }
    // Guard the drop's input-vs-output wire-symmetry assumption (see the function
    // docs); a no-op when there are no handles.
    if !dtype_map.handles.is_empty() {
        check_handle_wire_symmetry(&combined, &dtype_map.handles)?;
    }
    let dtypes = Arc::new(dtype_map);

    let registry = ProcessRegistry::with_builtin_emitters();
    let mut prebuilt: HashMap<Vec<String>, Box<dyn Process>> = HashMap::new();
    for (path, edge) in combined.edges() {
        let address = edge.address.clone();
        // Emitters stay native Rust (handled by the registry fallback), exactly
        // as on the JSON `run` path.
        if is_emitter(&address) {
            continue;
        }
        // Native-twin swap: only for an allow-listed address whose native ctor is
        // also enabled by VIVA_NATIVE_PROCESSES (so one handoff runs either path).
        if native_addresses.iter().any(|a| a == &address) {
            if let Some(ctor) = processes::native_ctor(&address) {
                let p = ctor(edge.config.as_ref())
                    .map_err(|e| perr(format!("native {address}: {e}")))?;
                prebuilt.insert(path, p);
                continue;
            }
        }
        // Otherwise: drive the pre-built Python instance bound to this edge.
        let key = path.join("/");
        let inst = processes.get_item(&key)?.ok_or_else(|| {
            perr(format!(
                "run_prebuilt: no pre-built instance for edge {key:?} (address {address})"
            ))
        })?;
        let name = inst
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| key.clone());
        let pyproc = PyProcess::from_instance(inst, Arc::clone(&dtypes), name)
            .map_err(|e| perr(format!("binding {key}: {e}")))?;
        prebuilt.insert(path, Box::new(pyproc) as Box<dyn Process>);
    }

    let engine = Engine::with_prebuilt(&combined, st, &registry, prebuilt)
        .map_err(|e| perr(format!("building prebuilt engine: {e}")))?;
    Ok((engine, dtypes))
}

/// Run a composite whose processes are handed over ALREADY CONSTRUCTED (the M4
/// Python->Rust handoff), for `n_ticks` scheduler iterations, and return the
/// decoded final state as a Python dict (same shape as [`run`]).
///
/// `processes`: `{edge_path: instance}` (edge path joined with `/`), each an
/// already-built v2ecoli process/step carrying its live `sim_data` — NOT
/// reconstructed from JSON (the whole point: 11 baseline process types carry
/// non-serializable handles). `topology`: a JSON spec of ONLY the edges
/// (`_type`/`address`/`inputs`/`outputs`/`interval`), configs omitted.
/// `state`: a JSON spec of ONLY the data stores (bulk/unique as
/// `__structured_array__`, scalars/maps inline). `native_addresses`: addresses
/// to swap for their native twin (also gated by `VIVA_NATIVE_PROCESSES`).
#[pyfunction]
#[pyo3(signature = (processes, topology, state, n_ticks, native_addresses=None, handles=None))]
fn run_prebuilt(
    py: Python<'_>,
    processes: &Bound<'_, PyDict>,
    topology: &str,
    state: &str,
    n_ticks: usize,
    native_addresses: Option<Vec<String>>,
    handles: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDict>> {
    let native = native_addresses.unwrap_or_default();
    let (mut engine, dtypes) =
        build_prebuilt_engine(py, processes, topology, state, &native, handles)?;
    engine
        .run(n_ticks)
        .map_err(|e| perr(format!("running prebuilt engine: {e}")))?;
    // Pass `dtypes` so the final state re-inserts the live handle objects at their
    // store paths (the conformance diff then sees the same objects pb carries).
    convert::state_to_py(py, &engine.state, &dtypes).map(|d| d.unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_report, m)?)?;
    m.add_function(wrap_pyfunction!(run_prebuilt, m)?)?;
    m.add_class::<rust_process::RustGrowCore>()?;
    Ok(())
}
