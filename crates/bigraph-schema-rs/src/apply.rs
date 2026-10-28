//! `apply_update` + `reconcile` for the subset of ops the E. coli model uses.
//!
//! These mirror, bit-for-bit on the decoded state, the Python reference:
//!   * scalar additive / replace / overwrite — bigraph-schema 1.6.0
//!     `methods/apply.py` (`Integer`/`Float`: `state + update`; `String`/
//!     `Boolean`: `return update`; `Overwrite`: replace) and `methods/
//!     reconcile.py` (`Integer`/`Float`: sum, `0 -> None`; `String`/`Boolean`/
//!     `Overwrite`: last non-None wins).
//!   * bulk — v2ecoli `types/bulk_numpy.py` (`apply`: `count[idx] += value`;
//!     `reconcile`: concatenate the `(idx, delta)` lists of a layer).
//!   * unique — v2ecoli `library/schema.py::UniqueNumpyUpdater` (accumulate
//!     set/add/delete per store path; flush in the order **set -> add ->
//!     delete**, with `initially_active_idx` snapshotted *before* adds and new
//!     rows numbered from the per-store `unique_index` counter).
//!
//! ## i64 / bigint audit
//! Upstream molecule counts are Python unbounded ints; numpy stores them as
//! int64 and would wrap silently on overflow. Every count/accumulator path here
//! accumulates in **i128** and writes back through a **checked** i64 conversion,
//! so an out-of-range value is a loud [`ApplyError::Overflow`], never a wrapped
//! negative. The `unique_index` metadata counter is likewise i128.

use crate::doc::{Map, Value};
use crate::state::{State, StoreNode, UniqueStore};

/// A single update targeting a store path.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    pub path: Vec<String>,
    pub op: Op,
}

/// The accumulate-until-flush unique update (one "layer" of set/add/delete).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UniqueOp {
    /// `set` dicts: field -> new value per currently-active molecule.
    pub set: Vec<Map<Vec<Value>>>,
    /// `add` dicts: field -> value per new molecule.
    pub add: Vec<Map<Vec<Value>>>,
    /// `delete` lists: indices into the pre-add active molecules.
    pub delete: Vec<Vec<usize>>,
    /// The `update: True` flush signal.
    pub flush: bool,
}

/// The subset of apply/reconcile ops the model exercises.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `apply(Float)`: `state + update`; reconcile = sum (0 -> drop).
    FloatDelta(f64),
    /// `apply(Integer)`: `state + update`; reconcile = sum (0 -> drop).
    IntDelta(i128),
    /// `apply(String)`: replace; reconcile = last non-None wins.
    SetStr(String),
    /// `apply(Boolean)`: replace; reconcile = last non-None wins.
    SetBool(bool),
    /// `apply(Overwrite)`: replace; reconcile = last non-None wins.
    Overwrite(Value),
    /// `apply(BulkNumpyUpdate)`: additive `(index, delta)` pairs;
    /// reconcile = concatenate.
    BulkDelta(Vec<(usize, i128)>),
    /// `apply(UniqueNumpyUpdate)` via the path-keyed updater.
    Unique(UniqueOp),
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ApplyError {
    #[error("no store at path {0:?}")]
    PathNotFound(Vec<String>),
    #[error("op/store type mismatch at path {0:?}")]
    TypeMismatch(Vec<String>),
    #[error("integer overflow at path {path:?} (would exceed i64)")]
    Overflow { path: Vec<String> },
}

/// Add an i128 delta to an i64 cell with overflow checking (no silent wrap).
fn checked_add_i64(current: i64, delta: i128, path: &[String]) -> Result<i64, ApplyError> {
    let sum = current as i128 + delta;
    i64::try_from(sum).map_err(|_| ApplyError::Overflow {
        path: path.to_vec(),
    })
}

/// Apply a reconciled update list to the state, in order.
pub fn apply_update(state: &mut State, updates: &[Update]) -> Result<(), ApplyError> {
    for update in updates {
        let node = state
            .get_mut_path(&update.path)
            .ok_or_else(|| ApplyError::PathNotFound(update.path.clone()))?;
        apply_one(node, &update.op, &update.path)?;
    }
    Ok(())
}

fn apply_one(node: &mut StoreNode, op: &Op, path: &[String]) -> Result<(), ApplyError> {
    match op {
        Op::BulkDelta(deltas) => {
            let StoreNode::Bulk(b) = node else {
                return Err(ApplyError::TypeMismatch(path.to_vec()));
            };
            // apply(BulkNumpyUpdate): count[idx] += value, per pair.
            for (idx, delta) in deltas {
                b.count[*idx] = checked_add_i64(b.count[*idx], *delta, path)?;
            }
        }
        Op::IntDelta(delta) => {
            let StoreNode::Leaf(v) = node else {
                return Err(ApplyError::TypeMismatch(path.to_vec()));
            };
            // apply(Integer): update == 0 -> no-op (reference fast path).
            if *delta != 0 {
                let cur = match v {
                    Value::Int(i) => *i,
                    Value::UInt(u) => *u as i64,
                    _ => return Err(ApplyError::TypeMismatch(path.to_vec())),
                };
                *v = Value::Int(checked_add_i64(cur, *delta, path)?);
            }
        }
        Op::FloatDelta(delta) => {
            let StoreNode::Leaf(v) = node else {
                return Err(ApplyError::TypeMismatch(path.to_vec()));
            };
            if *delta != 0.0 {
                let cur = match v {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Err(ApplyError::TypeMismatch(path.to_vec())),
                };
                *v = Value::Float(cur + delta);
            }
        }
        Op::SetStr(s) => replace_leaf(node, Value::Str(s.clone()), path)?,
        Op::SetBool(b) => replace_leaf(node, Value::Bool(*b), path)?,
        Op::Overwrite(val) => replace_leaf(node, val.clone(), path)?,
        Op::Unique(u) => {
            let StoreNode::Unique(store) = node else {
                return Err(ApplyError::TypeMismatch(path.to_vec()));
            };
            accumulate_unique(store, u);
            if u.flush {
                flush_unique(store, path)?;
            }
        }
    }
    Ok(())
}

fn replace_leaf(node: &mut StoreNode, val: Value, path: &[String]) -> Result<(), ApplyError> {
    let StoreNode::Leaf(v) = node else {
        return Err(ApplyError::TypeMismatch(path.to_vec()));
    };
    *v = val;
    Ok(())
}

// ---- unique updater -------------------------------------------------------

fn accumulate_unique(store: &mut UniqueStore, u: &UniqueOp) {
    store.updater.set.extend(u.set.iter().cloned());
    store.updater.add.extend(u.add.iter().cloned());
    store
        .updater
        .delete
        .extend(u.delete.iter().filter(|d| !d.is_empty()).cloned());
}

/// Flush the accumulated set/add/delete in the exact vEcoli order, operating on
/// the interleaved record buffer via field-offset + record-stride arithmetic.
///
/// Every buffer mutation goes through [`Arc::make_mut`]: if a Python view still
/// pins the buffer, the first mutation of the flush copies-on-write into a fresh
/// allocation (the retained view keeps the old one), so the layout change is
/// bit-identical to the former columnar path AND realloc-safe.
fn flush_unique(store: &mut UniqueStore, path: &[String]) -> Result<(), ApplyError> {
    use crate::state::write_cell;
    use std::sync::Arc;

    let st = std::mem::take(&mut store.updater);

    // Active rows at the start of the flush; snapshotted *before* any add so
    // deletes index into the pre-add active molecules (the critical ordering).
    let active_idx = store.active_indices(path)?;
    let initially_active = active_idx.clone();

    // 1. set — write to the rows active at the start of the layer.
    for set in &st.set {
        let layout = &store.layout;
        let itemsize = layout.itemsize;
        let buf = Arc::make_mut(&mut store.buf);
        for (field, vals) in set {
            if let Some(fl) = layout.field(field) {
                for (k, &row) in active_idx.iter().enumerate() {
                    if let Some(v) = vals.get(k) {
                        write_cell(buf, itemsize, row, fl, v);
                    }
                }
            }
        }
    }

    // 2. add — fill free rows, assign unique_index from the metadata counter.
    for add in &st.add {
        let n_new = add.values().next().map_or(0, Vec::len);
        if n_new == 0 {
            continue;
        }
        let free = store.get_free_indices(n_new);
        let itemsize = store.layout.itemsize;

        if !add.contains_key("unique_index") {
            let base = store.metadata;
            if let Some(fl) = store.layout.field("unique_index").cloned() {
                let buf = Arc::make_mut(&mut store.buf);
                for (k, &row) in free.iter().enumerate() {
                    let idx = base + k as i128;
                    let x = i64::try_from(idx).map_err(|_| ApplyError::Overflow {
                        path: path.to_vec(),
                    })?;
                    write_cell(buf, itemsize, row, &fl, &Value::Int(x));
                }
            }
            store.metadata = base + n_new as i128;
        }

        {
            let layout = &store.layout;
            let buf = Arc::make_mut(&mut store.buf);
            for (field, vals) in add {
                if let Some(fl) = layout.field(field) {
                    for (k, &row) in free.iter().enumerate() {
                        if let Some(v) = vals.get(k) {
                            write_cell(buf, itemsize, row, fl, v);
                        }
                    }
                }
            }
        }

        if let Some(fl) = store.layout.field("_entryState").cloned() {
            let buf = Arc::make_mut(&mut store.buf);
            for &row in &free {
                write_cell(buf, itemsize, row, &fl, &Value::Int(1));
            }
        }
    }

    // 3. delete — zero the whole record of each row selected from the pre-add
    // active snapshot (numpy `result[row] = np.zeros(1, dtype)`).
    if !st.delete.is_empty() {
        let itemsize = store.layout.itemsize;
        let buf = Arc::make_mut(&mut store.buf);
        for del in &st.delete {
            for &d in del {
                crate::state::zero_record(buf, itemsize, initially_active[d]);
            }
        }
    }

    Ok(())
}

// ---- reconcile ------------------------------------------------------------

/// Merge the concurrent updates of one execution layer, per the Python
/// `reconcile` dispatch for each op kind. Updates to the same path are combined;
/// additive deltas that cancel to zero drop out (Python returns `None`).
pub fn reconcile(updates: &[Update]) -> Vec<Update> {
    let mut order: Vec<Vec<String>> = Vec::new();
    let mut groups: Map<Vec<&Op>> = Map::new();
    for u in updates {
        let key = u.path.join("\u{1f}");
        if !groups.contains_key(&key) {
            order.push(u.path.clone());
        }
        groups.entry(key).or_default().push(&u.op);
    }

    let mut out = Vec::new();
    for path in order {
        let key = path.join("\u{1f}");
        let ops = &groups[&key];
        if let Some(op) = merge_ops(ops) {
            out.push(Update { path, op });
        }
    }
    out
}

fn merge_ops(ops: &[&Op]) -> Option<Op> {
    match ops.first()? {
        Op::BulkDelta(_) => {
            // reconcile(BulkNumpyUpdate): concatenate the delta lists.
            let mut combined = Vec::new();
            for op in ops {
                if let Op::BulkDelta(d) = op {
                    combined.extend(d.iter().copied());
                }
            }
            (!combined.is_empty()).then_some(Op::BulkDelta(combined))
        }
        Op::IntDelta(_) => {
            // reconcile(Integer): sum; 0 -> None.
            let total: i128 = ops
                .iter()
                .filter_map(|op| match op {
                    Op::IntDelta(d) => Some(*d),
                    _ => None,
                })
                .sum();
            (total != 0).then_some(Op::IntDelta(total))
        }
        Op::FloatDelta(_) => {
            // reconcile(Float): sum; 0.0 -> None.
            let total: f64 = ops
                .iter()
                .filter_map(|op| match op {
                    Op::FloatDelta(d) => Some(*d),
                    _ => None,
                })
                .sum();
            (total != 0.0).then_some(Op::FloatDelta(total))
        }
        Op::Unique(_) => {
            // Node-default merge: concatenate set/add/delete, OR the flush flags
            // (equivalent to applying each unique update in sequence, since the
            // updater accumulates).
            let mut merged = UniqueOp::default();
            for op in ops {
                if let Op::Unique(u) = op {
                    merged.set.extend(u.set.iter().cloned());
                    merged.add.extend(u.add.iter().cloned());
                    merged.delete.extend(u.delete.iter().cloned());
                    merged.flush |= u.flush;
                }
            }
            Some(Op::Unique(merged))
        }
        // String / Boolean / Overwrite: last non-None wins.
        _ => ops.last().map(|op| (*op).clone()),
    }
}
