//! Task-5 requirement 1 & 2: decode of numpy/pint-tagged binary structured
//! arrays (`__structured_array__`) into the columnar `State`, and unique-store
//! `metadata` initialisation from the max existing `unique_index`.

use bigraph_schema_rs::codec::from_json;
use bigraph_schema_rs::state::{from_initial, StoreNode};

/// A real `.pbg` delivers the bulk store as a tagged structured array; it must
/// lift into a `Bulk` store with the decoded `count`/`id` columns.
#[test]
fn decodes_real_bulk_structured_array() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ecoli_partitioned_slice.pbg"
    ))
    .unwrap();
    let spec = from_json(&text).unwrap();
    let state = from_initial(&spec);

    let bulk = state.get(&["agents", "0", "bulk"]).expect("bulk store");
    let StoreNode::Bulk(b) = bulk else {
        panic!("bulk did not decode into a Bulk store: {bulk:?}");
    };
    // From the committed slice: 8 molecules, count[2] == 123.
    assert_eq!(b.count.len(), 8);
    assert_eq!(b.count[2], 123);
    assert_eq!(b.ids[0], "--TRANS-ACENAPHTHENE-12-DIOL[c]");
    // Submass columns decode as floats.
    assert!(b.floats.contains_key("protein_submass"));
}

/// A tagged unique structured array lifts into a `Unique` store, and its
/// `metadata` counter initialises to `max(unique_index) + 1` so the first `add`
/// cannot collide with a pre-existing molecule's index.
#[test]
fn decodes_unique_and_inits_metadata_from_max_index() {
    // `_entryState` int8, `unique_index` int64 (the real dtypes), two molecules.
    let json = r#"{
      "u": {
        "__structured_array__": true,
        "dtype": [["domain_index","int32"],["_entryState","int8"],["unique_index","int64"]],
        "columns": {
          "domain_index": [1, 2],
          "_entryState": [1, 1],
          "unique_index": [41, 99]
        },
        "length": 2
      }
    }"#;
    let spec = from_json(json).unwrap();
    let state = from_initial(&spec);

    let u = state.get(&["u"]).expect("unique store");
    let StoreNode::Unique(store) = u else {
        panic!("did not decode into a Unique store: {u:?}");
    };
    assert_eq!(store.len(), 2);
    // metadata = max(unique_index) + 1 = 99 + 1.
    assert_eq!(store.metadata, 100);
    assert_eq!(store.int_col_vec("unique_index"), vec![41, 99]);
    assert_eq!(store.int_col_vec("domain_index"), vec![1, 2]);
    // The int8/int64 field widths are preserved in the interleaved layout.
    assert_eq!(store.layout.itemsize, 4 + 1 + 8); // int32 + int8 + int64
}
