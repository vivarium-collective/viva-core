//! Data model for the bigraph-schema composite-spec JSON contract.
//!
//! A spec is a JSON object tree. Nodes that look like process/step edges
//! (an `address` string plus `inputs`/`outputs`) are lifted into
//! [`ProcessEdge`]; everything else is a [`Store`] (nested object) or a
//! [`Value`] leaf. Tagged dicts from `bigraph_schema.json_codec` are kept
//! opaquely as [`Value::Tagged`] so they round-trip exactly.

use indexmap::IndexMap;

/// Insertion-ordered string-keyed map (JSON object).
pub type Map<V> = IndexMap<String, V>;

/// The `json_codec` tagged-dict forms (`{"__tag__": true, ...}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    /// `__pint__` (also the legacy `__pint_array__`).
    Pint,
    /// `__numpy__`.
    Numpy,
    /// `__numpy_structured__`.
    NumpyStructured,
    /// `__structured_array__`: legacy v2ecoli columnar structured array.
    StructuredArray,
    /// `__set__`.
    Set,
    /// `__bytes__` (hex payload).
    Bytes,
}

impl TagKind {
    /// Tag keys, in detection order. A dict is tagged when the key maps to `true`.
    pub const TAGS: [(&'static str, TagKind); 7] = [
        ("__pint__", TagKind::Pint),
        ("__pint_array__", TagKind::Pint),
        ("__numpy__", TagKind::Numpy),
        ("__numpy_structured__", TagKind::NumpyStructured),
        ("__structured_array__", TagKind::StructuredArray),
        ("__set__", TagKind::Set),
        ("__bytes__", TagKind::Bytes),
    ];

    pub fn detect(map: &Map<Value>) -> Option<TagKind> {
        Self::TAGS
            .iter()
            .find(|(k, _)| matches!(map.get(*k), Some(Value::Bool(true))))
            .map(|(_, kind)| *kind)
    }
}

/// A JSON value. `Float` may be non-finite (`Infinity`, `-Infinity`, `NaN`).
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    /// Integers above `i64::MAX` (kept exact).
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Map(Map<Value>),
    /// A `json_codec` tagged dict, preserved opaquely (all keys, in order).
    Tagged(TagKind, Map<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (Null, Null) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (UInt(a), UInt(b)) => a == b,
            // Bit equality so NaN == NaN (this is decoded-state equality).
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Str(a), Str(b)) => a == b,
            (Array(a), Array(b)) => a == b,
            (Map(a), Map(b)) => a == b,
            (Tagged(ka, a), Tagged(kb, b)) => ka == kb && a == b,
            _ => false,
        }
    }
}

/// A process or step edge.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessEdge {
    /// `_type` (`"process"` / `"step"`), when present.
    pub edge_type: Option<String>,
    pub address: String,
    pub config: Option<Value>,
    pub inputs: Option<Value>,
    pub outputs: Option<Value>,
    pub interval: Option<Value>,
    /// `_contract`.
    pub contract: Option<Value>,
    /// Any other keys on the edge, preserved in order.
    pub extra: Map<Value>,
}

/// A node in the spec tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Store(Map<Node>),
    Edge(Box<ProcessEdge>),
    Leaf(Value),
}

/// A parsed composite spec: the root store tree (edges live inside it).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompositeSpec {
    pub root: Map<Node>,
}

impl CompositeSpec {
    /// All edges with their paths from the root, in document order.
    pub fn edges(&self) -> Vec<(Vec<String>, &ProcessEdge)> {
        fn walk<'a>(
            m: &'a Map<Node>,
            path: &mut Vec<String>,
            out: &mut Vec<(Vec<String>, &'a ProcessEdge)>,
        ) {
            for (k, n) in m {
                path.push(k.clone());
                match n {
                    Node::Edge(e) => out.push((path.clone(), e)),
                    Node::Store(s) => walk(s, path, out),
                    Node::Leaf(_) => {}
                }
                path.pop();
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &mut Vec::new(), &mut out);
        out
    }
}

impl Node {
    /// Classify a parsed JSON object as store / edge, recursing into children.
    pub(crate) fn from_map(map: Map<Value>) -> Node {
        let is_edge = matches!(map.get("address"), Some(Value::Str(_)))
            && (map.contains_key("inputs") || map.contains_key("outputs"))
            && matches!(map.get("_type"), None | Some(Value::Str(_)));
        if is_edge {
            let mut map = map;
            let Some(Value::Str(address)) = map.shift_remove("address") else {
                unreachable!()
            };
            let edge_type = match map.shift_remove("_type") {
                Some(Value::Str(s)) => Some(s),
                _ => None,
            };
            let mut take = |k: &str| map.shift_remove(k);
            let (config, inputs, outputs, interval, contract) = (
                take("config"),
                take("inputs"),
                take("outputs"),
                take("interval"),
                take("_contract"),
            );
            return Node::Edge(Box::new(ProcessEdge {
                edge_type,
                address,
                config,
                inputs,
                outputs,
                interval,
                contract,
                extra: map,
            }));
        }
        Node::Store(
            map.into_iter()
                .map(|(k, v)| (k, Node::from_value(v)))
                .collect(),
        )
    }

    pub(crate) fn from_value(v: Value) -> Node {
        match v {
            Value::Map(m) => Node::from_map(m),
            other => Node::Leaf(other),
        }
    }
}
