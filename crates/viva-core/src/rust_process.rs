//! A small, real Rust-native process for quadrant 4: the **Python**
//! process-bigraph engine driving a **Rust** process.
//!
//! A `#[pyclass]` cannot subclass a Python class, so this exposes only the pb
//! `Process` surface (`inputs`/`outputs`/`update`); the thin Python shell in
//! `python/viva_core/rust_process.py` subclasses pb's `Process` and delegates.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Exponential-growth process: `delta mass = rate * mass * interval`.
#[pyclass]
pub struct RustGrowCore {
    rate: f64,
}

#[pymethods]
impl RustGrowCore {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let rate = match config.as_ref().map(|c| c.get_item("rate")).transpose()? {
            Some(Some(v)) => v.extract::<f64>()?,
            _ => 0.1,
        };
        Ok(Self { rate })
    }

    fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("mass", "float")?;
        Ok(d)
    }

    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("mass", "float")?;
        Ok(d)
    }

    #[pyo3(signature = (state, interval=1.0))]
    fn update<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyDict>,
        interval: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mass: f64 = match state.get_item("mass")? {
            Some(v) => v.extract()?,
            None => 0.0,
        };
        let out = PyDict::new(py);
        out.set_item("mass", self.rate * mass * interval)?;
        Ok(out)
    }
}
