//! The `Process` trait and the port view/update types it works with.
//!
//! The engine owns wiring: a process only sees *port names*. `PortView`
//! resolves a port to the wired store path; `PortUpdate` ports are mapped back
//! through the output wires. Task 5's Python-process path implements this same
//! trait (one PyO3 callback per edge).

use bigraph_schema_rs::apply::Op;
use bigraph_schema_rs::doc::{Map, Value};
use bigraph_schema_rs::state::{State, StoreNode};

/// Port name -> type string (`"float"`, `"overwrite[float]"`, ...).
pub type Ports = Vec<(String, String)>;

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("process error: {0}")]
pub struct ProcessError(pub String);

/// An update addressed to a port (optionally a sub-path within the port).
#[derive(Debug, Clone, PartialEq)]
pub struct PortUpdate {
    pub port: Vec<String>,
    pub op: Op,
}

impl PortUpdate {
    pub fn new(port: &[&str], op: Op) -> Self {
        PortUpdate {
            port: port.iter().map(|s| s.to_string()).collect(),
            op,
        }
    }
}

pub trait Process {
    fn inputs(&self) -> Ports;
    fn outputs(&self) -> Ports;
    /// Compute this process' update from its wired view. `interval` is the
    /// process interval, or `-1.0` for steps (as pb invokes steps).
    fn update(&mut self, view: &PortView, interval: f64) -> Result<Vec<PortUpdate>, ProcessError>;
    /// Emitters expose their captured rows here.
    fn history(&self) -> Option<Vec<Row>> {
        None
    }
}

/// One captured emitter row: port name -> value tree.
pub type Row = Map<StoreNode>;

/// Resolve `..` segments (bigraph-schema `resolve_path`). `None` if it
/// would ascend above the root.
pub fn resolve_path(path: &[String]) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for s in path {
        if s == ".." {
            out.pop()?;
        } else {
            out.push(s.clone());
        }
    }
    Some(out)
}

/// Resolve a port path (`[port, sub...]`) through a wire tree to an absolute
/// store path. `base` is the edge's parent path (wires are relative to it).
pub fn wire_path(wires: &Map<Value>, base: &[String], port: &[String]) -> Option<Vec<String>> {
    let mut wire = wires.get(port.first()?)?;
    let mut rest = &port[1..];
    while let Value::Map(m) = wire {
        let (k, tail) = rest.split_first()?;
        wire = m.get(k)?;
        rest = tail;
    }
    let mut p: Vec<String> = base.to_vec();
    match wire {
        Value::Array(a) => p.extend(a.iter().filter_map(|v| match v {
            Value::Str(s) => Some(s.clone()),
            _ => None,
        })),
        Value::Str(s) => p.push(s.clone()),
        _ => return None,
    }
    p.extend(rest.iter().cloned());
    resolve_path(&p)
}

/// A process' read view of the state through its input wires.
pub struct PortView<'a> {
    pub(crate) state: &'a State,
    pub(crate) wires: &'a Map<Value>,
    pub(crate) base: &'a [String],
}

impl<'a> PortView<'a> {
    pub fn get(&self, port: &[&str]) -> Option<&'a StoreNode> {
        let port: Vec<String> = port.iter().map(|s| s.to_string()).collect();
        let abs = wire_path(self.wires, self.base, &port)?;
        let abs: Vec<&str> = abs.iter().map(String::as_str).collect();
        self.state.get(&abs)
    }

    pub fn port_names(&self) -> Vec<&'a str> {
        self.wires.keys().map(String::as_str).collect()
    }

    /// Clone the whole viewed tree (shape follows the wire tree), as
    /// `tree_copy(state)` does for pb's RAMEmitter.
    pub fn snapshot(&self) -> Row {
        fn go(v: &PortView, wires: &Map<Value>, prefix: &mut Vec<String>) -> Row {
            let mut out = Row::new();
            for (k, w) in wires {
                prefix.push(k.clone());
                match w {
                    Value::Map(m) => {
                        out.insert(k.clone(), StoreNode::Branch(go(v, m, prefix)));
                    }
                    _ => {
                        if let Some(abs) = wire_path_full(v, prefix) {
                            let abs: Vec<&str> = abs.iter().map(String::as_str).collect();
                            if let Some(n) = v.state.get(&abs) {
                                out.insert(k.clone(), n.clone());
                            }
                        }
                    }
                }
                prefix.pop();
            }
            out
        }
        fn wire_path_full(v: &PortView, port: &[String]) -> Option<Vec<String>> {
            wire_path(v.wires, v.base, port)
        }
        go(self, self.wires, &mut Vec::new())
    }
}
