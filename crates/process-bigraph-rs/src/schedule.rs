//! Step-layer scheduling, mirroring process-bigraph 1.8.3 `scheduling.py`.
//!
//! pb has no standalone "compute layers" call: `wire_step_layers` writes layer
//! tokens into the state from a caller-supplied dependency graph, and the
//! composite then *executes* steps by walking `build_step_network` +
//! `determine_steps` over the data wiring. This module reproduces that walk
//! directly from the spec, so the layers returned are exactly the successive
//! `to_run` batches `Composite.cycle_step_state` yields:
//!
//! * `find_leaves` / `resolve_path` -> input/output paths per step (wires are
//!   relative to the edge's parent; `..` ascends);
//! * a step with `_triggers` only derives its *dependency inputs* from the
//!   trigger ports (find_step_triggers / build_step_network);
//! * `build_step_network`: producer/consumer `nodes` incl. the two ancestor/
//!   descendant propagation passes;
//! * `determine_steps`: a step runs once every input node has no unfulfilled
//!   producer; if nothing is runnable (cycle) the highest `priority` step runs.
//!
//! The `wire_step_layers` rule: layer-0 steps get `_triggers = {global_time:
//! 'float'}` (`root_trigger`), so even steps with no `global_time` wire are
//! treated as fired by the clock every tick. Later layers keep their own
//! `_triggers` ports if declared, else all input ports.
//!
//! Only edges with `_type == "step"` are layered; everything else is a
//! process scheduled by interval (see `engine`).

use bigraph_schema_rs::doc::{CompositeSpec, Map, ProcessEdge, Value};
use std::collections::{BTreeSet, HashMap};

use crate::process::resolve_path;

pub type Path = Vec<String>;

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledStep {
    pub path: Path,
    /// Input ports that fire this step (see module docs).
    pub triggers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub index: usize,
    pub steps: Vec<ScheduledStep>,
}

pub const ROOT_TRIGGER: &str = "global_time";

pub fn is_step(e: &ProcessEdge) -> bool {
    e.edge_type.as_deref() == Some("step")
}

/// pb `find_leaves`: every leaf path of a wire tree, resolved against `prefix`.
fn find_leaves(wires: &Map<Value>, prefix: &[String], out: &mut Vec<Path>) {
    for v in wires.values() {
        match v {
            Value::Map(m) => find_leaves(m, prefix, out),
            Value::Array(a) => {
                let mut p = prefix.to_vec();
                p.extend(a.iter().filter_map(|x| match x {
                    Value::Str(s) => Some(s.clone()),
                    _ => None,
                }));
                if let Some(r) = resolve_path(&p) {
                    out.push(r);
                }
            }
            Value::Str(s) => {
                let mut p = prefix.to_vec();
                p.push(s.clone());
                if let Some(r) = resolve_path(&p) {
                    out.push(r);
                }
            }
            _ => {}
        }
    }
}

fn wires_of(v: &Option<Value>) -> Map<Value> {
    match v {
        Some(Value::Map(m)) => m.clone(),
        _ => Map::new(),
    }
}

fn explicit_triggers(e: &ProcessEdge) -> Option<Vec<String>> {
    match e.extra.get("_triggers") {
        Some(Value::Map(m)) => Some(m.keys().cloned().collect()),
        _ => None,
    }
}

fn priority(e: &ProcessEdge) -> f64 {
    match e.extra.get("priority") {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        _ => 0.0,
    }
}

fn explode(path: &[String]) -> Vec<Path> {
    (0..=path.len()).map(|i| path[..i].to_vec()).collect()
}

#[derive(Default)]
struct Node {
    before: BTreeSet<usize>,
    after: BTreeSet<usize>,
}

struct StepInfo {
    path: Path,
    input_paths: Vec<Path>,
    output_paths: Vec<Path>,
    priority: f64,
    triggers: Option<Vec<String>>,
    input_ports: Vec<String>,
}

/// pb `build_step_network` + the initial `build_trigger_state`/`determine_steps`
/// walk. Returns the layer batches (step indices, in document order).
fn walk_layers(steps: &[StepInfo]) -> Vec<Vec<usize>> {
    let mut nodes: HashMap<Path, Node> = HashMap::new();
    // Insertion-ordered node keys keep the propagation passes deterministic.
    let mut node_keys: Vec<Path> = Vec::new();
    let mut node = |nodes: &mut HashMap<Path, Node>, p: &Path| {
        if !nodes.contains_key(p) {
            nodes.insert(p.clone(), Node::default());
            node_keys.push(p.clone());
        }
    };

    for (i, s) in steps.iter().enumerate() {
        for p in &s.input_paths {
            node(&mut nodes, p);
            nodes.get_mut(p).unwrap().after.insert(i);
        }
        for out in &s.output_paths {
            for ex in explode(out).into_iter().skip(1) {
                // skip self-loops: don't register as producer of a consumed path
                if !s.input_paths.contains(&ex) {
                    node(&mut nodes, &ex);
                    nodes.get_mut(&ex).unwrap().before.insert(i);
                }
            }
        }
    }

    let mut all_outputs: Vec<(Path, BTreeSet<usize>)> = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        for out in &s.output_paths {
            match all_outputs.iter_mut().find(|(p, _)| p == out) {
                Some((_, set)) => {
                    set.insert(i);
                }
                None => all_outputs.push((out.clone(), BTreeSet::from([i]))),
            }
        }
    }
    let all_inputs: BTreeSet<&Path> = steps.iter().flat_map(|s| s.input_paths.iter()).collect();

    // Pass 2: producer of an ancestor path feeds consumers of descendants.
    for key in &node_keys {
        if !nodes[key].before.is_empty() {
            continue;
        }
        let mut add = BTreeSet::new();
        for (out, producers) in &all_outputs {
            if key.len() > out.len() && key[..out.len()] == out[..] {
                add.extend(producers.iter().copied());
            }
        }
        nodes.get_mut(key).unwrap().before.extend(add);
    }
    // Pass 3: producer of a child path feeds consumers of the parent.
    for (out, producers) in &all_outputs {
        for inp in &all_inputs {
            if out.len() > inp.len() && out[..inp.len()] == inp[..] {
                if let Some(n) = nodes.get_mut(*inp) {
                    for p in producers {
                        if !n.after.contains(p) {
                            n.before.insert(*p);
                        }
                    }
                }
            }
        }
    }

    // build_trigger_state: before ∩ all steps (= before).
    let mut fulfilled: HashMap<Path, BTreeSet<usize>> = nodes
        .iter()
        .map(|(k, n)| (k.clone(), n.before.clone()))
        .collect();

    let mut remaining: Vec<usize> = (0..steps.len()).collect();
    let mut layers = Vec::new();
    while !remaining.is_empty() {
        let mut to_run: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| {
                steps[i]
                    .input_paths
                    .iter()
                    .all(|p| fulfilled.get(p).is_none_or(|s| s.is_empty()))
            })
            .collect();
        if to_run.is_empty() {
            // cycle: highest priority (first in document order on ties)
            let best = remaining
                .iter()
                .copied()
                .reduce(|a, b| {
                    if steps[b].priority > steps[a].priority {
                        b
                    } else {
                        a
                    }
                })
                .unwrap();
            to_run = vec![best];
        }
        for &i in &to_run {
            remaining.retain(|&r| r != i);
            for out in &steps[i].output_paths {
                for ex in explode(out).into_iter().skip(1) {
                    if let Some(f) = fulfilled.get_mut(&ex) {
                        f.remove(&i);
                    }
                }
            }
        }
        layers.push(to_run);
    }
    layers
}

/// Dependency-order the spec's step edges into layers (see module docs).
pub fn wire_step_layers(spec: &CompositeSpec) -> Vec<Layer> {
    let mut steps: Vec<StepInfo> = Vec::new();
    for (path, e) in spec.edges() {
        if !is_step(e) {
            continue;
        }
        let prefix = &path[..path.len() - 1];
        let inputs = wires_of(&e.inputs);
        let triggers = explicit_triggers(e);
        let dep_wires: Map<Value> = match &triggers {
            Some(t) => inputs
                .iter()
                .filter(|(k, _)| t.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            None => inputs.clone(),
        };
        let (mut ip, mut op) = (Vec::new(), Vec::new());
        find_leaves(&dep_wires, prefix, &mut ip);
        find_leaves(&wires_of(&e.outputs), prefix, &mut op);
        steps.push(StepInfo {
            path: path.clone(),
            input_paths: ip,
            output_paths: op,
            priority: priority(e),
            triggers,
            input_ports: inputs.keys().cloned().collect(),
        });
    }

    walk_layers(&steps)
        .into_iter()
        .enumerate()
        .map(|(index, batch)| Layer {
            index,
            steps: batch
                .into_iter()
                .map(|i| {
                    let s = &steps[i];
                    let triggers = if index == 0 {
                        vec![ROOT_TRIGGER.to_string()]
                    } else {
                        s.triggers.clone().unwrap_or_else(|| s.input_ports.clone())
                    };
                    ScheduledStep {
                        path: s.path.clone(),
                        triggers,
                    }
                })
                .collect(),
        })
        .collect()
}
