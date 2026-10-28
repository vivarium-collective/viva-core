//! Tick loop mirroring process-bigraph 1.8.3 `Composite._run_inner`.
//!
//! Order per scheduler iteration (`tick`):
//!  1. every process whose front time <= `global_time` computes its update from
//!     the *current* state (`run_process`), scheduling itself at `time +
//!     interval`; `full_step` = the smallest time to the next event;
//!  2. `global_time += full_step`;
//!  3. updates of all processes due by the new time are applied in ONE
//!     `reconcile` + `apply_update` (process document order);
//!  4. the step layers run in order; within a layer every step views the same
//!     state, their updates are reconciled together and applied once, then the
//!     next layer starts (`run_steps` -> `cycle_step_state`).
//!
//! Before the first iteration the step layers run once (pb's initial
//! `to_run`), so emitters record a t=0 row.
//!
//! Simplification: pb re-fires only steps downstream of changed trigger
//! paths; because layer-0 steps are fired by `global_time` every tick (the
//! `_triggers` rule in `schedule`), every step is downstream and fires every
//! tick, which is what this engine does.

use crate::process::{wire_path, PortUpdate, PortView, Process, ProcessError, Row};
use crate::registry::ProcessRegistry;
use crate::schedule::{wire_step_layers, Layer};
use bigraph_schema_rs::apply::{apply_update, reconcile, ApplyError, Update};
use bigraph_schema_rs::doc::{CompositeSpec, Map, Value};
use bigraph_schema_rs::state::{from_initial, State, StoreNode};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Process(#[from] ProcessError),
    #[error(transparent)]
    Apply(#[from] ApplyError),
    #[error("process at {path:?} has non-advancing interval {interval}")]
    BadInterval { path: Vec<String>, interval: f64 },
    #[error("process at {path:?} updated unwired output port {port:?}")]
    UnwiredPort {
        path: Vec<String>,
        port: Vec<String>,
    },
    #[error("no `global_time` float in state")]
    NoClock,
}

struct Slot {
    path: Vec<String>,
    base: Vec<String>,
    process: Box<dyn Process>,
    inputs: Map<Value>,
    outputs: Map<Value>,
    interval: f64,
    /// pb `front[path]['time']`.
    front_time: f64,
    /// pb `front[path]['update']`.
    pending: Vec<Update>,
}

pub struct Engine {
    pub state: State,
    pub layers: Vec<Layer>,
    processes: Vec<Slot>,
    steps: Vec<Slot>,
    step_index: HashMap<Vec<String>, usize>,
    started: bool,
}

fn wires(v: &Option<Value>) -> Map<Value> {
    match v {
        Some(Value::Map(m)) => m.clone(),
        _ => Map::new(),
    }
}

impl Engine {
    /// Build state from the spec's initial values and instantiate every edge
    /// through `registry`.
    pub fn new(spec: &CompositeSpec, registry: &ProcessRegistry) -> Result<Self, EngineError> {
        Self::with_state(spec, from_initial(spec), registry)
    }

    /// Seam for Task 5: supply a pre-decoded `State` (e.g. real-.pbg tagged
    /// structured-array initial state) instead of `from_initial`.
    pub fn with_state(
        spec: &CompositeSpec,
        state: State,
        registry: &ProcessRegistry,
    ) -> Result<Self, EngineError> {
        let layers = wire_step_layers(spec);
        let (mut processes, mut steps) = (Vec::new(), Vec::new());
        for (path, e) in spec.edges() {
            let interval = match &e.interval {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => *i as f64,
                _ => 1.0, // pb `process` schema default
            };
            let slot = Slot {
                base: path[..path.len() - 1].to_vec(),
                process: registry.create(&e.address, e.config.as_ref())?,
                inputs: wires(&e.inputs),
                outputs: wires(&e.outputs),
                interval,
                front_time: 0.0,
                pending: Vec::new(),
                path,
            };
            if crate::schedule::is_step(e) {
                steps.push(slot);
            } else {
                processes.push(slot);
            }
        }
        let step_index = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.path.clone(), i))
            .collect();
        let mut engine = Engine {
            state,
            layers,
            processes,
            steps,
            step_index,
            started: false,
        };
        let t = engine.global_time()?;
        for p in &mut engine.processes {
            p.front_time = t;
        }
        Ok(engine)
    }

    pub fn global_time(&self) -> Result<f64, EngineError> {
        match self.state.get(&["global_time"]) {
            Some(StoreNode::Leaf(Value::Float(f))) => Ok(*f),
            Some(StoreNode::Leaf(Value::Int(i))) => Ok(*i as f64),
            _ => Err(EngineError::NoClock),
        }
    }

    fn set_global_time(&mut self, t: f64) {
        self.state
            .insert(&["global_time"], StoreNode::Leaf(Value::Float(t)));
    }

    /// Captured rows of the emitter step at `path`.
    pub fn emitter_history(&self, path: &[&str]) -> Option<Vec<Row>> {
        self.steps
            .iter()
            .chain(self.processes.iter())
            .find(|s| s.path.iter().map(String::as_str).eq(path.iter().copied()))
            .and_then(|s| s.process.history())
    }

    /// Advance by `n_ticks` scheduler iterations (each moves the clock by the
    /// smallest process interval).
    pub fn run(&mut self, n_ticks: usize) -> Result<(), EngineError> {
        self.start()?;
        for _ in 0..n_ticks {
            self.tick(f64::INFINITY)?;
        }
        Ok(())
    }

    /// pb `Composite.run(interval)` to an absolute end time.
    pub fn run_until(&mut self, end_time: f64) -> Result<(), EngineError> {
        self.start()?;
        while self.global_time()? < end_time {
            self.tick(end_time)?;
        }
        Ok(())
    }

    /// pb's initial `to_run`: the step cascade fires once before any tick.
    fn start(&mut self) -> Result<(), EngineError> {
        if !self.started {
            self.started = true;
            self.run_step_layers()?;
        }
        Ok(())
    }

    fn map_updates(slot: &Slot, ups: Vec<PortUpdate>) -> Result<Vec<Update>, EngineError> {
        ups.into_iter()
            .map(|u| {
                wire_path(&slot.outputs, &slot.base, &u.port)
                    .map(|path| Update { path, op: u.op })
                    .ok_or_else(|| EngineError::UnwiredPort {
                        path: slot.path.clone(),
                        port: u.port.clone(),
                    })
            })
            .collect()
    }

    fn invoke(state: &State, slot: &mut Slot, dt: f64) -> Result<Vec<Update>, EngineError> {
        let view = PortView {
            state,
            wires: &slot.inputs,
            base: &slot.base,
        };
        let ups = slot.process.update(&view, dt)?;
        Self::map_updates(slot, ups)
    }

    fn apply_all(&mut self, updates: Vec<Update>) -> Result<(), EngineError> {
        if updates.is_empty() {
            return Ok(());
        }
        let combined = reconcile(&updates);
        apply_update(&mut self.state, &combined)?;
        Ok(())
    }

    /// pb `run_steps` over every layer in order.
    fn run_step_layers(&mut self) -> Result<(), EngineError> {
        for li in 0..self.layers.len() {
            let mut updates = Vec::new();
            for si in 0..self.layers[li].steps.len() {
                let idx = self.step_index[&self.layers[li].steps[si].path];
                // Steps are invoked with interval = -1.0.
                updates.extend(Self::invoke(&self.state, &mut self.steps[idx], -1.0)?);
            }
            self.apply_all(updates)?;
        }
        Ok(())
    }

    /// One iteration of pb's `_run_inner` while-loop body.
    fn tick(&mut self, end_time: f64) -> Result<(), EngineError> {
        let mut gt = self.global_time()?;
        let mut full_step = f64::INFINITY;

        for i in 0..self.processes.len() {
            let slot = &mut self.processes[i];
            if slot.front_time <= gt {
                if slot.interval.is_nan() || slot.interval <= 0.0 {
                    return Err(EngineError::BadInterval {
                        path: slot.path.clone(),
                        interval: slot.interval,
                    });
                }
                let future = slot.front_time + slot.interval;
                let interval = future - gt;
                if interval < full_step {
                    full_step = interval;
                }
                if future <= end_time {
                    let dt = slot.interval;
                    let update = Self::invoke(&self.state, &mut self.processes[i], dt)?;
                    let slot = &mut self.processes[i];
                    slot.front_time = future;
                    slot.pending = update;
                }
            } else {
                let delay = slot.front_time - gt;
                if delay < full_step {
                    full_step = delay;
                }
            }
        }

        if full_step == f64::INFINITY {
            let mut next_event = end_time;
            for p in &self.processes {
                if p.front_time < next_event {
                    next_event = p.front_time;
                }
            }
            if next_event.is_finite() {
                self.set_global_time(next_event);
            }
        } else if gt + full_step <= end_time {
            gt += full_step;
            self.set_global_time(gt);
            let mut updates = Vec::new();
            for p in &mut self.processes {
                if p.front_time <= gt && !p.pending.is_empty() {
                    updates.extend(std::mem::take(&mut p.pending));
                }
            }
            self.apply_all(updates)?;
            self.run_step_layers()?;
        } else {
            self.set_global_time(end_time);
        }
        Ok(())
    }
}
