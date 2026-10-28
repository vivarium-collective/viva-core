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
use std::sync::Arc;

// ---- interleaved numpy-record layout --------------------------------------
//
// A [`UniqueStore`] is backed by ONE contiguous byte buffer in numpy's
// (non-aligned / packed) record layout, so Python can view it zero-copy as a
// genuine structured array. Every in-scope unique field is fixed-width
// (int8/16/32/64, uint*, float32/64, bool) — unique stores carry no string or
// sub-array fields (the Python view path has always rejected those).
//
// Endianness is little-endian: the real `.pbg` dtypes are native (`<`), and all
// supported hosts are little-endian. Reads/writes below pack LE accordingly.

/// A fixed-width numpy scalar kind for one field of an interleaved record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl FieldKind {
    /// Byte width of one cell of this kind.
    pub fn size(self) -> usize {
        match self {
            FieldKind::I8 | FieldKind::U8 | FieldKind::Bool => 1,
            FieldKind::I16 | FieldKind::U16 => 2,
            FieldKind::I32 | FieldKind::U32 | FieldKind::F32 => 4,
            FieldKind::I64 | FieldKind::U64 | FieldKind::F64 => 8,
        }
    }

    /// Canonical numpy dtype string (used when the spec dtype is unavailable).
    pub fn numpy_dtype(self) -> &'static str {
        match self {
            FieldKind::I8 => "int8",
            FieldKind::I16 => "int16",
            FieldKind::I32 => "int32",
            FieldKind::I64 => "int64",
            FieldKind::U8 => "uint8",
            FieldKind::U16 => "uint16",
            FieldKind::U32 => "uint32",
            FieldKind::U64 => "uint64",
            FieldKind::F32 => "float32",
            FieldKind::F64 => "float64",
            FieldKind::Bool => "bool",
        }
    }

    /// Parse a numpy dtype string (`int8`/`<i8`/`float64`/`bool`/…) to a kind.
    /// Returns `None` for non-fixed-width fields (strings, sub-arrays): those
    /// are out of the zero-copy unique scope.
    pub fn from_dtype(d: &str) -> Option<FieldKind> {
        let d = d.trim_start_matches(['<', '>', '=', '|']);
        Some(match d {
            "int8" | "i1" => FieldKind::I8,
            "int16" | "i2" => FieldKind::I16,
            "int32" | "i4" => FieldKind::I32,
            "int64" | "i8" => FieldKind::I64,
            "uint8" | "u1" => FieldKind::U8,
            "uint16" | "u2" => FieldKind::U16,
            "uint32" | "u4" => FieldKind::U32,
            "uint64" | "u8" => FieldKind::U64,
            "float32" | "f4" => FieldKind::F32,
            "float64" | "f8" => FieldKind::F64,
            "bool" | "b1" | "?" => FieldKind::Bool,
            _ => return None,
        })
    }
}

/// One field's placement inside an interleaved record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: String,
    pub kind: FieldKind,
    pub offset: usize,
    pub size: usize,
}

/// The interleaved record layout: ordered fields + the record stride
/// (`itemsize`). Field order is the numpy dtype field order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordLayout {
    pub fields: Vec<FieldLayout>,
    pub itemsize: usize,
}

impl RecordLayout {
    /// Pack fields in order with numpy's non-aligned rule: cumulative offsets,
    /// no padding, `itemsize` = sum of sizes. (Verified against a built
    /// `PyArrayDescr` on every view build; see `viva-core::convert`.)
    pub fn packed(fields: Vec<(String, FieldKind)>) -> Self {
        let mut offset = 0;
        let mut out = Vec::with_capacity(fields.len());
        for (name, kind) in fields {
            let size = kind.size();
            out.push(FieldLayout {
                name,
                kind,
                offset,
                size,
            });
            offset += size;
        }
        RecordLayout {
            fields: out,
            itemsize: offset,
        }
    }

    pub fn field(&self, name: &str) -> Option<&FieldLayout> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Write one coerced cell at `row` into the interleaved `buf` at `fl`'s offset,
/// matching numpy's per-field cast of a scalar assignment.
pub fn write_cell(buf: &mut [u8], itemsize: usize, row: usize, fl: &FieldLayout, v: &Value) {
    let base = row * itemsize + fl.offset;
    match fl.kind {
        FieldKind::I8
        | FieldKind::I16
        | FieldKind::I32
        | FieldKind::I64
        | FieldKind::U8
        | FieldKind::U16
        | FieldKind::U32
        | FieldKind::U64 => {
            // LE two's-complement low bytes == numpy's int cast (mod 2^8k).
            let le = value_to_i64(v).to_le_bytes();
            buf[base..base + fl.size].copy_from_slice(&le[..fl.size]);
        }
        FieldKind::F32 => {
            let f = value_to_f64(v) as f32;
            buf[base..base + 4].copy_from_slice(&f.to_le_bytes());
        }
        FieldKind::F64 => {
            let f = value_to_f64(v);
            buf[base..base + 8].copy_from_slice(&f.to_le_bytes());
        }
        FieldKind::Bool => {
            buf[base] = matches!(v, Value::Bool(true)) as u8;
        }
    }
}

/// Read a cell as i64 (sign/zero-extended per kind). Bools read 0/1.
pub fn read_i64(buf: &[u8], itemsize: usize, row: usize, fl: &FieldLayout) -> i64 {
    let base = row * itemsize + fl.offset;
    let b = &buf[base..base + fl.size];
    match fl.kind {
        FieldKind::I8 => b[0] as i8 as i64,
        FieldKind::I16 => i16::from_le_bytes([b[0], b[1]]) as i64,
        FieldKind::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        FieldKind::I64 => i64::from_le_bytes(b.try_into().unwrap()),
        FieldKind::U8 => b[0] as i64,
        FieldKind::U16 => u16::from_le_bytes([b[0], b[1]]) as i64,
        FieldKind::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        FieldKind::U64 => u64::from_le_bytes(b.try_into().unwrap()) as i64,
        FieldKind::Bool => (b[0] != 0) as i64,
        FieldKind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        FieldKind::F64 => f64::from_le_bytes(b.try_into().unwrap()) as i64,
    }
}

/// Read a cell as a decoded [`Value`] (for the non-structured state readback).
pub fn read_value(buf: &[u8], itemsize: usize, row: usize, fl: &FieldLayout) -> Value {
    let base = row * itemsize + fl.offset;
    let b = &buf[base..base + fl.size];
    match fl.kind {
        FieldKind::F32 => Value::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        FieldKind::F64 => Value::Float(f64::from_le_bytes(b.try_into().unwrap())),
        FieldKind::Bool => Value::Bool(b[0] != 0),
        _ => Value::Int(read_i64(buf, itemsize, row, fl)),
    }
}

/// Zero one whole record (numpy `result[row] = np.zeros(1, dtype)`).
pub fn zero_record(buf: &mut [u8], itemsize: usize, row: usize) {
    let base = row * itemsize;
    for byte in &mut buf[base..base + itemsize] {
        *byte = 0;
    }
}

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

    /// Read one cell at `row` as a [`Value`] (used when packing to bytes).
    fn get(&self, row: usize) -> Value {
        match self {
            Column::Int(a) => Value::Int(a[row]),
            Column::Float(a) => Value::Float(a[row]),
            Column::Bool(v) => Value::Bool(v[row]),
            Column::Str(v) => Value::Str(v[row].clone()),
            Column::Nested(v) => v[row].clone(),
        }
    }

    /// Default fixed-width kind for a column with no explicit numpy dtype
    /// (integers -> int64, floats -> float64, bools -> bool). String / nested
    /// columns are not representable in the interleaved unique buffer.
    fn default_kind(&self) -> FieldKind {
        match self {
            Column::Int(_) => FieldKind::I64,
            Column::Float(_) => FieldKind::F64,
            Column::Bool(_) => FieldKind::Bool,
            Column::Str(_) | Column::Nested(_) => {
                panic!("unique store field is string/nested; not supported in the zero-copy buffer")
            }
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

/// A structured record array for one unique-molecule type, backed by ONE
/// interleaved byte buffer in numpy record layout, plus its path-keyed updater
/// state (accumulators + the `unique_index` metadata counter).
///
/// The buffer is held behind an [`Arc`] so a Python zero-copy view can pin the
/// *current* allocation (the numpy array's `base` owns an `Arc` clone). The
/// apply path mutates via [`Arc::make_mut`]: if any view still references the
/// buffer (refcount > 1) the mutation copies-on-write into a fresh allocation
/// and the old one stays alive for the retained view — so a view handed to
/// Python can never dangle across an add/grow realloc (see
/// `viva-core::convert::unique_view`).
#[derive(Debug, Clone, PartialEq)]
pub struct UniqueStore {
    /// Resolved field schema (optional; drives zero-values for nested fields).
    pub fields: Vec<Field>,
    /// Interleaved record bytes (`nrows * layout.itemsize`), numpy record order.
    pub buf: Arc<Vec<u8>>,
    /// Field placement (offsets/widths/itemsize) for the interleaved buffer.
    pub layout: RecordLayout,
    /// Next `unique_index` to hand out (Python `MetadataArray.metadata`; an
    /// unbounded counter — kept as i128).
    pub metadata: i128,
    pub(crate) updater: UpdaterState,
}

/// Pack ordered columns into an interleaved byte buffer under `layout`.
fn pack_columns(layout: &RecordLayout, columns: &Map<Column>, nrows: usize) -> Vec<u8> {
    let mut buf = vec![0u8; nrows * layout.itemsize];
    for fl in &layout.fields {
        if let Some(col) = columns.get(&fl.name) {
            for row in 0..nrows.min(col.len()) {
                write_cell(&mut buf, layout.itemsize, row, fl, &col.get(row));
            }
        }
    }
    buf
}

impl UniqueStore {
    /// Build from columns with an INFERRED packed layout (int64/float64/bool),
    /// in column insertion order. Used by the plain-JSON classifier and the
    /// Rust oracle tests, where no explicit numpy dtype is carried.
    pub fn from_columns(columns: Map<Column>, metadata: i128) -> Self {
        let fields: Vec<(String, FieldKind)> = columns
            .iter()
            .map(|(k, c)| (k.clone(), c.default_kind()))
            .collect();
        let layout = RecordLayout::packed(fields);
        Self::from_columns_with_layout(layout, columns, metadata)
    }

    /// Build from columns and an explicit layout (field order = `layout` order;
    /// used by the tagged-structured-array decode, which carries exact dtypes).
    pub fn from_columns_with_layout(
        layout: RecordLayout,
        columns: Map<Column>,
        metadata: i128,
    ) -> Self {
        let nrows = columns.values().next().map_or(0, Column::len);
        let buf = Arc::new(pack_columns(&layout, &columns, nrows));
        Self {
            fields: Vec::new(),
            buf,
            layout,
            metadata,
            updater: UpdaterState::default(),
        }
    }

    /// Install a (PyArrayDescr-derived) layout, re-packing the buffer if the
    /// offsets differ. A no-op when the layout already matches (the common case:
    /// numpy's packed layout == the inferred/decoded one).
    pub fn relayout(&mut self, new: RecordLayout) {
        if new == self.layout {
            return;
        }
        let nrows = self.len();
        let mut nb = vec![0u8; nrows * new.itemsize];
        for nf in &new.fields {
            if let Some(of) = self.layout.field(&nf.name) {
                for row in 0..nrows {
                    let v = read_value(&self.buf, self.layout.itemsize, row, of);
                    write_cell(&mut nb, new.itemsize, row, nf, &v);
                }
            }
        }
        self.buf = Arc::new(nb);
        self.layout = new;
    }

    /// Read an integer field as a `Vec<i64>` (all rows).
    pub fn int_col_vec(&self, name: &str) -> Vec<i64> {
        let fl = self
            .layout
            .field(name)
            .unwrap_or_else(|| panic!("no field {name:?}"));
        (0..self.len())
            .map(|r| read_i64(&self.buf, self.layout.itemsize, r, fl))
            .collect()
    }

    /// Number of rows (`buf.len() / itemsize`).
    pub fn len(&self) -> usize {
        self.buf
            .len()
            .checked_div(self.layout.itemsize)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indices of active molecules (`_entryState != 0`), ascending.
    pub(crate) fn active_indices(
        &self,
        path: &[String],
    ) -> Result<Vec<usize>, crate::apply::ApplyError> {
        let fl = self
            .layout
            .field("_entryState")
            .ok_or_else(|| crate::apply::ApplyError::TypeMismatch(path.to_vec()))?;
        let itemsize = self.layout.itemsize;
        Ok((0..self.len())
            .filter(|&r| read_i64(&self.buf, itemsize, r, fl) != 0)
            .collect())
    }

    /// vEcoli `get_free_indices`: inactive rows (`_entryState == 0`) in
    /// ascending order, growing the buffer by at least 10% when there are too
    /// few, then returning the first `n` free rows. The grow is the only place
    /// the buffer can reallocate — via [`Arc::make_mut`], copy-on-write when a
    /// view is live.
    pub(crate) fn get_free_indices(&mut self, n: usize) -> Vec<usize> {
        let itemsize = self.layout.itemsize;
        let es = self.layout.field("_entryState").cloned();
        let mut free: Vec<usize> = match &es {
            Some(fl) => (0..self.len())
                .filter(|&r| read_i64(&self.buf, itemsize, r, fl) == 0)
                .collect(),
            None => Vec::new(),
        };
        if free.len() < n {
            let old = self.len();
            // Python: n_new_entries = max(int(old_size * 0.1), n - n_free).
            let growth = (old as f64 * 0.1).trunc() as usize;
            let n_new_entries = growth.max(n - free.len());
            let buf = Arc::make_mut(&mut self.buf);
            buf.resize(buf.len() + n_new_entries * itemsize, 0);
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
        // Build the interleaved layout from the EXACT numpy dtype list (field
        // order + fixed widths). Any non-fixed-width field (string / sub-array)
        // is out of the zero-copy unique scope -> bail (treated as unsupported).
        let mut layout_fields: Vec<(String, FieldKind)> = Vec::new();
        for (name, dstr) in &dtypes {
            if !columns.contains_key(name) {
                continue;
            }
            let kind = FieldKind::from_dtype(dstr)?;
            layout_fields.push((name.clone(), kind));
        }
        let layout = RecordLayout::packed(layout_fields);

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
        return Some(StoreNode::Unique(UniqueStore::from_columns_with_layout(
            layout, cols, metadata,
        )));
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
