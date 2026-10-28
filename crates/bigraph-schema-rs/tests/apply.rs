//! Parity tests for `state` + `apply`/`reconcile` against the Python oracle.
//!
//! The expected values are produced by the AUTHORITATIVE Python apply paths the
//! E. coli model runs (bigraph-schema 1.6.0 + v2ecoli):
//!
//! - bulk: `v2ecoli.types.bulk_numpy` apply (`state["count"][idx] += value`),
//!   reconciled by `reconcile(BulkNumpyUpdate)` = concatenate delta lists.
//! - unique: `v2ecoli.library.schema.UniqueNumpyUpdater.updater` — accumulate,
//!   then flush in the order set -> add -> delete.
//!
//! Oracle generator + how to regenerate:
//!   crates/bigraph-schema-rs/tests/oracle/gen_oracle.py
//!   cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -I <that file>
//! The printed values are hard-coded below.

use bigraph_schema_rs::apply::{apply_update, reconcile, Op, UniqueOp, Update};
use bigraph_schema_rs::doc::{Map, TagKind, Value};
use bigraph_schema_rs::grammar::parse_type;
use bigraph_schema_rs::registry::{Registry, ResolveError};
use bigraph_schema_rs::state::{from_initial, BulkStore, Column, State, StoreNode, UniqueStore};
use ndarray::array;

fn col_patch(pairs: &[(&str, &[i64])]) -> Map<Vec<Value>> {
    let mut m = Map::new();
    for (name, vals) in pairs {
        m.insert(
            (*name).to_string(),
            vals.iter().map(|v| Value::Int(*v)).collect(),
        );
    }
    m
}

/// bulk apply, reconciled over a layer, matches the Python `apply` output.
#[test]
fn apply_bulk_matches_python() {
    let mut state = State::new();
    state.insert(
        &["bulk"],
        StoreNode::Bulk(BulkStore::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            array![10, 20, 30, 40, 50],
        )),
    );

    // One execution layer: three concurrent bulk writers at the same path.
    let layer = vec![
        Update {
            path: vec!["bulk".into()],
            op: Op::BulkDelta(vec![(0, 5), (2, -3)]),
        },
        Update {
            path: vec!["bulk".into()],
            op: Op::BulkDelta(vec![(0, 100), (4, 7)]),
        },
        Update {
            path: vec!["bulk".into()],
            op: Op::BulkDelta(vec![(2, 3), (1, 1)]),
        },
    ];
    let reconciled = reconcile(&layer);
    apply_update(&mut state, &reconciled).unwrap();

    let StoreNode::Bulk(b) = state.get(&["bulk"]).unwrap() else {
        panic!("not a bulk store")
    };
    // Oracle: bulk.count = [115, 21, 30, 40, 57]
    assert_eq!(b.count.to_vec(), vec![115, 21, 30, 40, 57]);
}

/// unique set/add/delete through the path-keyed updater matches the Python
/// `UniqueNumpyUpdater.updater` output (set->add->delete; initially_active_idx
/// captured before adds; auto unique_index from the metadata counter).
#[test]
fn apply_unique_set_add_delete() {
    let mut cols = Map::new();
    cols.insert(
        "TU_index".into(),
        Column::Int(array![100, 101, 0, 103, 0, 105]),
    );
    cols.insert("_entryState".into(), Column::Int(array![1, 1, 0, 1, 0, 1]));
    cols.insert("unique_index".into(), Column::Int(array![0, 1, 0, 3, 0, 5]));

    let mut state = State::new();
    state.insert(
        &["unique"],
        StoreNode::Unique(UniqueStore::from_columns(cols, 6)),
    );

    let op = Op::Unique(UniqueOp {
        set: vec![col_patch(&[("TU_index", &[200, 201, 203, 205])])],
        add: vec![col_patch(&[("TU_index", &[900, 901, 902])])],
        delete: vec![vec![0, 2]],
        flush: true,
    });
    apply_update(
        &mut state,
        &[Update {
            path: vec!["unique".into()],
            op,
        }],
    )
    .unwrap();

    let StoreNode::Unique(u) = state.get(&["unique"]).unwrap() else {
        panic!("not a unique store")
    };
    // Oracle (size 7):
    //   TU_index    = [0, 201, 900, 0, 901, 205, 902]
    //   _entryState = [0, 1,   1,   0, 1,   1,   1]
    //   unique_index= [0, 1,   6,   0, 7,   5,   8]
    //   metadata    = 9
    assert_eq!(u.int_col("TU_index"), &[0, 201, 900, 0, 901, 205, 902]);
    assert_eq!(u.int_col("_entryState"), &[0, 1, 1, 0, 1, 1, 1]);
    assert_eq!(u.int_col("unique_index"), &[0, 1, 6, 0, 7, 5, 8]);
    assert_eq!(u.metadata, 9);
}

/// Large counts must not silently wrap. The accumulator is i128 and the store
/// write is a checked i64 conversion: a value that would exceed i64 is a loud
/// error, never a wrapped-around negative count.
#[test]
fn no_i64_overflow_on_large_counts() {
    // (a) large-but-valid counts beyond i32 accumulate exactly.
    let mut state = State::new();
    state.insert(
        &["bulk"],
        StoreNode::Bulk(BulkStore::new(vec!["X".into()], array![5_000_000_000])),
    );
    apply_update(
        &mut state,
        &[Update {
            path: vec!["bulk".into()],
            op: Op::BulkDelta(vec![(0, 4_000_000_000)]),
        }],
    )
    .unwrap();
    let StoreNode::Bulk(b) = state.get(&["bulk"]).unwrap() else {
        panic!()
    };
    assert_eq!(b.count[0], 9_000_000_000);

    // (b) a delta that would push past i64::MAX is caught, not wrapped.
    let mut state = State::new();
    state.insert(
        &["bulk"],
        StoreNode::Bulk(BulkStore::new(vec!["X".into()], array![i64::MAX])),
    );
    let err = apply_update(
        &mut state,
        &[Update {
            path: vec!["bulk".into()],
            op: Op::BulkDelta(vec![(0, 1)]),
        }],
    );
    assert!(
        err.is_err(),
        "overflow must be a loud error, not a silent wrap"
    );
    // And the store is unchanged / never shows a wrapped negative value.
    let StoreNode::Bulk(b) = state.get(&["bulk"]).unwrap() else {
        panic!()
    };
    assert!(b.count[0] > 0, "count must not have wrapped negative");
}

/// Quantity apply: bigraph-schema 1.6.0 has no `apply(Quantity)` overload, so a
/// quantity leaf inherits `apply(Node)` whose scalar branch is `result = update`
/// (replace). Confirm it against a real tagged-pint value.
#[test]
fn apply_quantity_replace() {
    // A real pint quantity as the json_codec tagged dict preserves on the leaf.
    let mut q0 = Map::new();
    q0.insert("__pint__".into(), Value::Bool(true));
    q0.insert("value".into(), Value::Float(1.5));
    q0.insert("units".into(), Value::Str("millimolar".into()));
    let mut q1 = q0.clone();
    q1.insert("value".into(), Value::Float(9.0));

    let mut state = State::new();
    state.insert(
        &["c", "glucose"],
        StoreNode::Leaf(Value::Tagged(TagKind::Pint, q0)),
    );

    let new = Value::Tagged(TagKind::Pint, q1.clone());
    apply_update(
        &mut state,
        &[Update {
            path: vec!["c".into(), "glucose".into()],
            op: Op::Overwrite(new.clone()),
        }],
    )
    .unwrap();

    let StoreNode::Leaf(v) = state.get(&["c", "glucose"]).unwrap() else {
        panic!()
    };
    assert_eq!(v, &new);
}

/// `from_initial` lifts columnar bulk/unique object leaves into stores and keeps
/// scalar leaves, round-tripping through the Task-1 codec.
#[test]
fn from_initial_builds_stores() {
    let json = r#"{
        "t": 0.0,
        "bulk": {"id": ["A", "B"], "count": [10, 20]},
        "unique": {"TU_index": [5, 0], "_entryState": [1, 0], "unique_index": [5, 0]}
    }"#;
    let spec = bigraph_schema_rs::codec::from_json(json).unwrap();
    let state = from_initial(&spec);

    assert!(matches!(
        state.get(&["t"]),
        Some(StoreNode::Leaf(Value::Float(_)))
    ));
    let StoreNode::Bulk(b) = state.get(&["bulk"]).unwrap() else {
        panic!("bulk")
    };
    assert_eq!(b.count.to_vec(), vec![10, 20]);
    assert_eq!(b.ids, vec!["A".to_string(), "B".to_string()]);
    let StoreNode::Unique(u) = state.get(&["unique"]).unwrap() else {
        panic!("unique")
    };
    assert_eq!(u.int_col("TU_index"), &[5, 0]);
}

/// An unresolved type name fails loud (never silently defaults): `resolve`
/// returns `ResolveError::Unknown`, and `dispatch` on a `Named` would panic.
#[test]
fn unresolved_named_is_an_error() {
    let reg = Registry::new();
    let t = parse_type("not_a_real_type").unwrap();
    assert_eq!(
        reg.resolve(&t),
        Err(ResolveError::Unknown("not_a_real_type".into()))
    );
}
