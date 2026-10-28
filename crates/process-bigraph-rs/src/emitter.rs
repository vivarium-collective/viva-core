//! RAM/JSON emitter step (pb `RAMEmitter` analog).
//!
//! In pb the emitter is an ordinary *step* wired to the state it records; it
//! has no outputs, so it depends on every producer of its inputs and lands in
//! the last layer, running every tick (and once before the first tick). It is
//! modelled the same way: a `Process` whose `update` snapshots its input view
//! and returns no updates. `JSONEmitter` shares the capture (pb only differs
//! in serialising rows to a file, which is a sink concern, not state).

use crate::process::{PortUpdate, PortView, Ports, Process, ProcessError, Row};
use bigraph_schema_rs::doc::Value;

pub struct RamEmitter {
    history: Vec<Row>,
    step: u64,
    subsample: u64,
    max_len: Option<usize>,
}

impl RamEmitter {
    pub fn new() -> Self {
        RamEmitter {
            history: Vec::new(),
            step: 0,
            subsample: 1,
            max_len: None,
        }
    }

    /// Honours pb's `subsample` / `max_len` config.
    pub fn from_config(cfg: Option<&Value>) -> Self {
        let mut e = Self::new();
        if let Some(Value::Map(m)) = cfg {
            if let Some(Value::Int(n)) = m.get("subsample") {
                e.subsample = (*n).max(1) as u64;
            }
            if let Some(Value::Int(n)) = m.get("max_len") {
                e.max_len = Some((*n).max(1) as usize);
            }
        }
        e
    }
}

impl Default for RamEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Process for RamEmitter {
    fn inputs(&self) -> Ports {
        // Dynamic: whatever is wired.
        vec![]
    }
    fn outputs(&self) -> Ports {
        vec![]
    }
    fn update(&mut self, view: &PortView, _dt: f64) -> Result<Vec<PortUpdate>, ProcessError> {
        let step = self.step;
        self.step += 1;
        if !step.is_multiple_of(self.subsample) {
            return Ok(vec![]);
        }
        self.history.push(view.snapshot());
        if let Some(max) = self.max_len {
            if self.history.len() > max {
                let drop = self.history.len() - max;
                self.history.drain(..drop);
            }
        }
        Ok(vec![])
    }
    fn history(&self) -> Option<Vec<Row>> {
        Some(self.history.clone())
    }
}
