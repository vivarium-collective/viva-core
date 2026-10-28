//! Runtime state tree for the engine hot path.
//!
//! The store tree mirrors the composite spec: nested [`StoreNode::Branch`]es
//! hold scalar [`StoreNode::Leaf`]s and the two array stores that dominate the
//! E. coli model —
//!
//! * [`BulkStore`]: per-molecule counts as a flat `ndarray` column (`count`),
//!   plus the string `id` column and any float submass columns. Updates are
//!   additive `(index, delta)` pairs.
//! * [`UniqueStore`]: a structured record array (named, mixed-dtype columns)
//!   with the per-store updater state (`add`/`set`/`delete` accumulators and
//!   the `unique_index` metadata counter). This is the Rust analogue of
//!   vEcoli's path-keyed `UniqueNumpyUpdater`: because each unique store lives
//!   at a distinct path in the tree, holding the accumulator *inside* the store
//!   is exactly a per-path updater registry (see `apply`).
//!
//! Counts are Python unbounded ints upstream; see `apply` for the i128/checked
//! arithmetic that keeps the i64 columns from silently wrapping.

use crate::doc::{Map, TagKind, Value};
use crate::types::Field;
use ndarray::Array1;

/// A typed column of a structured (unique) store. Integer / float columns use
/// `ndarray` flat arrays (zero-copy-ready for the future numpy boundary);
/// bool / string / nested-array columns use plain vectors.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// Any `integer` field, including `_entryState` and `unique_index`.
    Int(Array1<i64>),
    Float(Array1<f64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
    /// An `array[...]` sub-field held per row, opaquely (one [`Value`] per row).
    Nested(Vec<Value>),
}

impl Column {
    /// Number of rows in the column.
    pub fn len(&self) -> usize {
        match self {
            Column::Int(a) => a.len(),
            Column::Float(a) => a.len(),
            Column::Bool(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append `n` zero rows (numpy `np.append(.., np.zeros(n, dtype))`).
    fn grow_zeros(&mut self, n: usize) {
        match self {
            Column::Int(a) => {
                let mut v = std::mem::take(a).into_raw_vec_and_offset().0;
                v.extend(std::iter::repeat_n(0, n));
                *a = Array1::from_vec(v);
            }
            Column::Float(a) => {
                let mut v = std::mem::take(a).into_raw_vec_and_offset().0;
                v.extend(std::iter::repeat_n(0.0, n));
                *a = Array1::from_vec(v);
            }
            Column::Bool(v) => v.extend(std::iter::repeat_n(false, n)),
            Column::Str(v) => v.extend(std::iter::repeat_n(String::new(), n)),
            Column::Nested(v) => v.extend(std::iter::repeat_n(Value::Null, n)),
        }
    }

    /// Zero a single row (numpy `result[row] = np.zeros(1, dtype)`).
    fn zero_row(&mut self, row: usize) {
        match self {
            Column::Int(a) => a[row] = 0,
            Column::Float(a) => a[row] = 0.0,
            Column::Bool(v) => v[row] = false,
            Column::Str(v) => v[row] = String::new(),
            Column::Nested(v) => v[row] = Value::Null,
        }
    }

    /// Write one coerced cell at `row` from a JSON [`Value`].
    fn set_cell(&mut self, row: usize, v: &Value) {
        match self {
            Column::Int(a) => a[row] = value_to_i64(v),
            Column::Float(a) => a[row] = value_to_f64(v),
            Column::Bool(b) => b[row] = matches!(v, Value::Bool(true)),
            Column::Str(s) => {
                s[row] = match v {
                    Value::Str(x) => x.clone(),
                    _ => String::new(),
                }
            }
            Column::Nested(n) => n[row] = v.clone(),
        }
    }
}

fn value_to_i64(v: &Value) -> i64 {
    match v {
        Value::Int(i) => *i,
        Value::UInt(u) => *u as i64,
        Value::Bool(b) => *b as i64,
        Value::Float(f) => *f as i64,
        _ => 0,
    }
}

fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        Value::UInt(u) => *u as f64,
        _ => 0.0,
    }
}

/// Per-molecule bulk counts: `count` as a flat `ndarray` column, plus the `id`
/// strings and any float submass columns (static w.r.t. the additive apply).
#[derive(Debug, Clone, PartialEq)]
pub struct BulkStore {
    pub ids: Vec<String>,
    pub count: Array1<i64>,
    pub floats: Map<Array1<f64>>,
}

impl BulkStore {
    pub fn new(ids: Vec<String>, count: Array1<i64>) -> Self {
        Self {
            ids,
            count,
            floats: Map::new(),
        }
    }
}

/// Accumulated unique-store updates awaiting the flush signal, in the exact
/// shape vEcoli's `UniqueNumpyUpdater` keeps them.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct UpdaterState {
    pub set: Vec<Map<Vec<Value>>>,
    pub add: Vec<Map<Vec<Value>>>,
    pub delete: Vec<Vec<usize>>,
}

/// A structured record array for one unique-molecule type, plus its path-keyed
/// updater state (accumulators + the `unique_index` metadata counter).
#[derive(Debug, Clone, PartialEq)]
pub struct UniqueStore {
    /// Resolved field schema (optional; drives zero-values for nested fields).
    pub fields: Vec<Field>,
    /// Named columns, in field order.
    pub columns: Map<Column>,
    /// Next `unique_index` to hand out (Python `MetadataArray.metadata`; an
    /// unbounded counter — kept as i128).
    pub metadata: i128,
    pub(crate) updater: UpdaterState,
}

impl UniqueStore {
    /// Build directly from columns + the next `unique_index` counter.
    pub fn from_columns(columns: Map<Column>, metadata: i128) -> Self {
        Self {
            fields: Vec::new(),
            columns,
            metadata,
            updater: UpdaterState::default(),
        }
    }

    /// Read an integer column as a slice (panics if absent / wrong kind).
    pub fn int_col(&self, name: &str) -> &[i64] {
        match self.columns.get(name) {
            Some(Column::Int(a)) => a.as_slice().expect("contiguous"),
            _ => panic!("no integer column {name:?}"),
        }
    }

    /// Number of rows (columns are kept equal-length).
    pub fn len(&self) -> usize {
        self.columns.values().next().map_or(0, Column::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indices of active molecules (`_entryState == 1`), ascending.
    pub(crate) fn active_indices(
        &self,
        path: &[String],
    ) -> Result<Vec<usize>, crate::apply::ApplyError> {
        match self.columns.get("_entryState") {
            Some(Column::Int(es)) => Ok(es
                .iter()
                .enumerate()
                .filter(|(_, &s)| s != 0)
                .map(|(i, _)| i)
                .collect()),
            _ => Err(crate::apply::ApplyError::TypeMismatch(path.to_vec())),
        }
    }

    /// vEcoli `get_free_indices`: inactive rows (`_entryState == 0`) in
    /// ascending order, growing every column by at least 10% when there are too
    /// few, then returning the first `n` free rows.
    pub(crate) fn get_free_indices(&mut self, n: usize) -> Vec<usize> {
        let mut free: Vec<usize> = match self.columns.get("_entryState") {
            Some(Column::Int(es)) => es
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == 0)
                .map(|(i, _)| i)
                .collect(),
            _ => Vec::new(),
        };
        if free.len() < n {
            let old = self.len();
            // Python: n_new_entries = max(int(old_size * 0.1), n - n_free).
            let growth = (old as f64 * 0.1).trunc() as usize;
            let n_new_entries = growth.max(n - free.len());
            for col in self.columns.values_mut() {
                col.grow(n_new_entries);
            }
            free.extend(old..old + n_new_entries);
        }
        free.truncate(n);
        free
    }
}

/// A node in the runtime state tree.
#[derive(Debug, Clone, PartialEq)]
pub enum StoreNode {
    Branch(Map<StoreNode>),
    Leaf(Value),
    Bulk(BulkStore),
    Unique(UniqueStore),
}

/// The runtime state: the root store tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct State {
    pub root: Map<StoreNode>,
}

/// Build the runtime [`State`] from a parsed composite spec.
///
/// Walks the spec's store tree (edges ignored — they are process wiring, not
/// state): object stores become [`StoreNode::Branch`]es, scalar values become
/// [`StoreNode::Leaf`]s, and a *columnar* object leaf is lifted into a
/// [`BulkStore`] (has `id` + `count` columns) or a [`UniqueStore`] (has
/// `_entryState` + `unique_index` columns).
///
/// Subset note: initial state is carried inside the `CompositeSpec` (Task-1
/// data model), so no separate `initial_json` argument is needed. Decoding the
/// numpy-/pint-tagged binary structured-array encodings is a codec concern
/// deferred to a later task; this handles the plain columnar JSON shape.
pub fn from_initial(spec: &crate::doc::CompositeSpec) -> State {
    State {
        root: conv_store(&spec.root),
    }
}

fn conv_store(m: &Map<crate::doc::Node>) -> Map<StoreNode> {
    use crate::doc::Node;
    let mut out = Map::new();
    for (k, n) in m {
        match n {
            Node::Edge(_) => {}
            Node::Store(s) => {
                // A columnar sub-store (all children are arrays) may be a
                // bulk/unique molecule store; otherwise it is a plain branch.
                out.insert(
                    k.clone(),
                    classify_store(s).unwrap_or_else(|| StoreNode::Branch(conv_store(s))),
                );
            }
            Node::Leaf(v) => {
                // A real `.pbg` delivers bulk/unique stores as tagged
                // structured arrays (numpy/pint `json_codec` forms), not plain
                // columnar objects — decode those into Bulk/Unique here.
                let node = match v {
                    Value::Tagged(TagKind::StructuredArray | TagKind::NumpyStructured, m) => {
                        decode_structured(m).unwrap_or_else(|| StoreNode::Leaf(v.clone()))
                    }
                    _ => StoreNode::Leaf(v.clone()),
                };
                out.insert(k.clone(), node);
            }
        }
    }
    out
}

/// Lift a store whose children are columnar arrays into a [`BulkStore`] (has
/// `id` + `count`) or a [`UniqueStore`] (has `_entryState` + `unique_index`).
fn classify_store(s: &Map<crate::doc::Node>) -> Option<StoreNode> {
    use crate::doc::Node;
    let col = |key: &str| matches!(s.get(key), Some(Node::Leaf(Value::Array(_))));
    if col("id") && col("count") {
        let (Some(Node::Leaf(Value::Array(ids))), Some(Node::Leaf(Value::Array(counts)))) =
            (s.get("id"), s.get("count"))
        else {
            return None;
        };
        let mut floats = Map::new();
        for (k, n) in s {
            if k == "id" || k == "count" {
                continue;
            }
            if let Node::Leaf(Value::Array(a)) = n {
                floats.insert(k.clone(), to_f64_array(a));
            }
        }
        return Some(StoreNode::Bulk(BulkStore {
            ids: ids.iter().map(value_to_string).collect(),
            count: to_i64_array(counts),
            floats,
        }));
    }
    if col("_entryState") && col("unique_index") {
        let mut columns = Map::new();
        for (k, n) in s {
            if let Node::Leaf(Value::Array(a)) = n {
                columns.insert(k.clone(), to_column(a));
            }
        }
        return Some(StoreNode::Unique(UniqueStore::from_columns(columns, 0)));
    }
    None
}

/// Decode a `json_codec` tagged structured array (`__structured_array__` /
/// `__numpy_structured__`: `{dtype:[[name,dtype],...], columns:{name:[...]},
/// length:n}`) into a [`BulkStore`] (`id`+`count`) or a [`UniqueStore`]
/// (`_entryState`+`unique_index`).
///
/// This is the real-`.pbg` initial-state owner (Task-5 requirement 1): the
/// engine can only load a real model once bulk/unique binary structured arrays
/// lift into the columnar [`State`] representation. Column kinds come from the
/// numpy dtype string (`int*`/`uint*`/`bool`→Int/Bool, `float*`→Float, `<U`/
/// `str`→Str, tuple sub-array dtypes like `('<i4', (2,))`→Nested).
///
/// For a unique store, the `metadata` (`unique_index`) counter is initialised
/// to `max(unique_index) + 1` (Task-5 requirement 2): a real store already
/// holds molecules, so the first `add` must number new rows past the existing
/// maximum or it would collide `unique_index` (matches
/// `UniqueNumpyUpdater`/`create_unique_indices` semantics).
pub fn decode_structured(m: &Map<Value>) -> Option<StoreNode> {
    let dtypes = structured_dtypes(m)?;
    let columns = match m.get("columns") {
        Some(Value::Map(c)) => c,
        _ => return None,
    };
    let col_kind = |name: &str| -> Option<&str> {
        dtypes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_str())
    };

    let has = |k: &str| columns.contains_key(k);
    if has("id") && has("count") {
        let ids = match columns.get("id") {
            Some(Value::Array(a)) => a.iter().map(value_to_string).collect(),
            _ => Vec::new(),
        };
        let count = match columns.get("count") {
            Some(Value::Array(a)) => to_i64_array(a),
            _ => Array1::from_vec(Vec::new()),
        };
        let mut floats = Map::new();
        for (k, v) in columns {
            if k == "id" || k == "count" {
                continue;
            }
            if let Value::Array(a) = v {
                floats.insert(k.clone(), to_f64_array(a));
            }
        }
        return Some(StoreNode::Bulk(BulkStore { ids, count, floats }));
    }
    if has("_entryState") && has("unique_index") {
        let mut cols = Map::new();
        for (k, v) in columns {
            if let Value::Array(a) = v {
                cols.insert(k.clone(), typed_column(a, col_kind(k)));
            }
        }
        let metadata = match cols.get("unique_index") {
            Some(Column::Int(a)) => a.iter().copied().max().map_or(0, |mx| mx as i128 + 1),
            _ => 0,
        };
        return Some(StoreNode::Unique(UniqueStore::from_columns(cols, metadata)));
    }
    None
}

/// Parse the `dtype` field (`[[name, dtype_str], ...]`) into `(name, dtype)` pairs.
fn structured_dtypes(m: &Map<Value>) -> Option<Vec<(String, String)>> {
    match m.get("dtype") {
        Some(Value::Array(entries)) => Some(
            entries
                .iter()
                .filter_map(|e| match e {
                    Value::Array(pair) if pair.len() >= 2 => match (&pair[0], &pair[1]) {
                        (Value::Str(n), Value::Str(d)) => Some((n.clone(), d.clone())),
                        _ => None,
                    },
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Choose a [`Column`] kind from a numpy dtype string, falling back to
/// first-element inference when the dtype is unknown.
fn typed_column(a: &[Value], dtype: Option<&str>) -> Column {
    let Some(d) = dtype else {
        return to_column(a);
    };
    // A tuple dtype (`('<i4', (2,))`) marks a per-row sub-array field.
    if d.starts_with('(') {
        return Column::Nested(a.to_vec());
    }
    let d = d.trim_start_matches(['<', '>', '=', '|']);
    if d.starts_with("int") || d.starts_with("uint") || d.starts_with('i') || d.starts_with('u') {
        Column::Int(to_i64_array(a))
    } else if d.starts_with("float") || d.starts_with('f') {
        Column::Float(to_f64_array(a))
    } else if d.starts_with("bool") || d == "b1" || d.starts_with('?') {
        Column::Bool(a.iter().map(|v| matches!(v, Value::Bool(true))).collect())
    } else if d.starts_with('U') || d.starts_with('S') || d.starts_with("str") {
        Column::Str(a.iter().map(value_to_string).collect())
    } else {
        to_column(a)
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        _ => String::new(),
    }
}

fn to_i64_array(a: &[Value]) -> Array1<i64> {
    Array1::from_vec(a.iter().map(value_to_i64).collect())
}

fn to_f64_array(a: &[Value]) -> Array1<f64> {
    Array1::from_vec(a.iter().map(value_to_f64).collect())
}

/// Infer a column kind from the first element.
fn to_column(a: &[Value]) -> Column {
    match a.first() {
        Some(Value::Float(_)) => Column::Float(to_f64_array(a)),
        Some(Value::Int(_) | Value::UInt(_)) => Column::Int(to_i64_array(a)),
        Some(Value::Bool(_)) => {
            Column::Bool(a.iter().map(|v| matches!(v, Value::Bool(true))).collect())
        }
        Some(Value::Str(_)) => Column::Str(a.iter().map(value_to_string).collect()),
        Some(Value::Array(_) | Value::Map(_) | Value::Tagged(..)) => Column::Nested(a.to_vec()),
        Some(Value::Null) | None => Column::Int(to_i64_array(a)),
    }
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `node` at `path`, creating intermediate [`StoreNode::Branch`]es.
    pub fn insert(&mut self, path: &[&str], node: StoreNode) {
        assert!(!path.is_empty(), "path must be non-empty");
        let mut cur = &mut self.root;
        for key in &path[..path.len() - 1] {
            let entry = cur
                .entry((*key).to_string())
                .or_insert_with(|| StoreNode::Branch(Map::new()));
            match entry {
                StoreNode::Branch(b) => cur = b,
                _ => panic!("path segment {key:?} is not a branch"),
            }
        }
        cur.insert(path[path.len() - 1].to_string(), node);
    }

    pub fn get(&self, path: &[&str]) -> Option<&StoreNode> {
        let mut node = self.root.get(*path.first()?)?;
        for key in &path[1..] {
            match node {
                StoreNode::Branch(b) => node = b.get(*key)?,
                _ => return None,
            }
        }
        Some(node)
    }

    pub(crate) fn get_mut_path(&mut self, path: &[String]) -> Option<&mut StoreNode> {
        let mut node = self.root.get_mut(path.first()?.as_str())?;
        for key in &path[1..] {
            match node {
                StoreNode::Branch(b) => node = b.get_mut(key.as_str())?,
                _ => return None,
            }
        }
        Some(node)
    }
}

// Re-export the grow/zero/set helpers to the apply module.
impl Column {
    pub(crate) fn grow(&mut self, n: usize) {
        self.grow_zeros(n);
    }
    pub(crate) fn zero_at(&mut self, row: usize) {
        self.zero_row(row);
    }
    pub(crate) fn write(&mut self, row: usize, v: &Value) {
        self.set_cell(row, v);
    }
}
