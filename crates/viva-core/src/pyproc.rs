//! [`PyProcess`]: a Rust [`Process`] that drives a Python process/step instance
//! across the PyO3 boundary (quadrant 3).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};

use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};

use crate::convert::{build_view, parse_out_schema, translate_update, value_to_py, OutSchema};
use crate::DtypeMap;

/// Wraps a Python process/step instance. Each tick it builds a numpy/native
/// view of the wired stores, honours the Python gate + timestep, and translates
/// the returned update dict into typed Rust ops.
pub struct PyProcess {
    instance: Py<PyAny>,
    out_schema: OutSchema,
    has_perform_update: bool,
    has_calculate_timestep: bool,
    dtypes: Arc<DtypeMap>,
    name: String,
}

fn proc_err(e: impl std::fmt::Display) -> ProcessError {
    ProcessError(e.to_string())
}

impl PyProcess {
    /// Resolve a `local:module.path.Class` address, import the class, and
    /// instantiate it with the edge config (passed as the vEcoli `parameters`
    /// positional). Requires the caller to have set the v2ecoli core via
    /// `set_current_core` (as any v2ecoli runner does).
    pub fn create(
        py: Python<'_>,
        address: &str,
        config: Option<&bigraph_schema_rs::doc::Value>,
        dtypes: Arc<DtypeMap>,
    ) -> Result<Self, ProcessError> {
        let target = address.strip_prefix("local:").unwrap_or(address);
        let (module, cls) = target
            .rsplit_once('.')
            .ok_or_else(|| proc_err(format!("address {address:?} is not module.Class")))?;

        let module_obj = py
            .import(module)
            .map_err(|e| proc_err(format!("import {module:?}: {e}")))?;
        let class = module_obj
            .getattr(cls)
            .map_err(|e| proc_err(format!("getattr {cls:?}: {e}")))?;

        let cfg_py = match config {
            Some(v) => value_to_py(py, v).map_err(proc_err)?,
            None => PyDict::new(py).into_any(),
        };
        let instance = class
            .call1((cfg_py,))
            .map_err(|e| proc_err(format!("instantiating {cls}: {e}")))?;

        Self::from_instance(instance, dtypes, cls.to_string())
    }

    /// Bind an ALREADY-CONSTRUCTED Python process/step instance (M4 `run_prebuilt`).
    ///
    /// Unlike [`create`], this does NOT import a class or re-run `__init__` from a
    /// JSON `cfg`: it drives the exact instance the caller hands over. That is the
    /// whole point of the Python->Rust composite handoff — the baseline's 11
    /// process types carry live, non-JSON-serializable `sim_data` handles baked in
    /// at construction, so the engine must schedule the pre-built instance itself.
    /// The per-tick driving (view build, gate, `update`, delta translation) is
    /// identical to [`create`]'s — only the construction source differs.
    pub fn from_instance(
        instance: Bound<'_, PyAny>,
        dtypes: Arc<DtypeMap>,
        name: String,
    ) -> Result<Self, ProcessError> {
        let outputs = instance
            .call_method0("outputs")
            .map_err(|e| proc_err(format!("{name}.outputs(): {e}")))?;
        let out_schema = parse_out_schema(&outputs).map_err(proc_err)?;

        let has_perform_update = attr_is_callable(&instance, "perform_update");
        let has_calculate_timestep = attr_is_callable(&instance, "calculate_timestep");

        Ok(PyProcess {
            instance: instance.unbind(),
            out_schema,
            has_perform_update,
            has_calculate_timestep,
            dtypes,
            name,
        })
    }
}

fn attr_is_callable(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.getattr(name).map(|a| a.is_callable()).unwrap_or(false)
}

impl Process for PyProcess {
    fn inputs(&self) -> Ports {
        Vec::new() // Wiring comes from the edge; the engine never reads these.
    }
    fn outputs(&self) -> Ports {
        Vec::new()
    }

    fn update(&mut self, view: &PortView, interval: f64) -> Result<Vec<PortUpdate>, ProcessError> {
        Python::attach(|py| {
            let view_dict = build_view(py, view, &self.dtypes).map_err(proc_err)?;
            let instance = self.instance.bind(py);
            let result = instance
                .call_method1("update", (view_dict, interval))
                .map_err(|e| proc_err(format!("{}.update: {e}", self.name)))?;
            let mut out = Vec::new();
            let mut port_path = Vec::new();
            translate_update(
                &result,
                &self.out_schema,
                &mut port_path,
                &mut out,
                view,
                &self.dtypes,
            )
            .map_err(proc_err)?;
            Ok(out)
        })
    }

    fn calculate_timestep(
        &mut self,
        view: &PortView,
        configured: f64,
    ) -> Result<f64, ProcessError> {
        if !self.has_calculate_timestep {
            return Ok(configured);
        }
        Python::attach(|py| {
            let view_dict = build_view(py, view, &self.dtypes).map_err(proc_err)?;
            let instance = self.instance.bind(py);
            // pb: instance.calculate_timestep(process['interval'], state).
            let result = instance
                .call_method1("calculate_timestep", (configured, view_dict))
                .map_err(|e| proc_err(format!("{}.calculate_timestep: {e}", self.name)))?;
            result
                .extract::<f64>()
                .map_err(|e| proc_err(format!("{}.calculate_timestep -> float: {e}", self.name)))
        })
    }

    fn perform_update(&mut self, view: &PortView) -> Result<bool, ProcessError> {
        if !self.has_perform_update {
            return Ok(true);
        }
        Python::attach(|py| {
            let view_dict = build_view(py, view, &self.dtypes).map_err(proc_err)?;
            let instance = self.instance.bind(py);
            let result = instance
                .call_method1("perform_update", (view_dict,))
                .map_err(|e| proc_err(format!("{}.perform_update: {e}", self.name)))?;
            result
                .is_truthy()
                .map_err(|e| proc_err(format!("{}.perform_update -> bool: {e}", self.name)))
        })
    }
}
