use bigraph_schema_rs::codec::from_json;
use bigraph_schema_rs::doc::{Node, Value};
use bigraph_schema_rs::grammar::{parse_type, render_type};
use bigraph_schema_rs::registry::{ApplyKey, ReconcileKey, Registry};
use bigraph_schema_rs::types::Type;

const FIXTURE: &str = include_str!("fixtures/ecoli_partitioned_slice.pbg");

/// Port/store type notation the E. coli model declares (v2ecoli
/// `library/schema_types.py` + process `inputs()`/`outputs()`); the fixture
/// carries only values, so these are the notations its stores correspond to.
const NOTATIONS: &[&str] = &[
    "float",
    "integer",
    "boolean",
    "string",
    "node",
    "integer{1}",
    "float[s]",
    "integer[nt]",
    "overwrite[float]",
    "overwrite[integer]",
    "overwrite[float[s]]",
    "list[string]",
    "map[float]",
    "map[map[list[integer]]]",
    "list[map[node]]",
    "array[float]",
    "array[integer]",
    "array[boolean]",
    "array[23,boolean]",
    "array[float[uM]]",
    "overwrite[array[integer]]",
    "overwrite[array[float[mM]]]",
    "quantity[float,fg]",
    "quantity[g/L]",
    "overwrite[quantity[float,fg]]",
    "quantity[array[float],1/s]",
    "map[quantity[float,nucleotide/s]]",
    "bulk_array",
    "bulk_array[id:string|count:integer|rRNA_submass:float|tRNA_submass:float|mRNA_submass:float|miscRNA_submass:float|nonspecific_RNA_submass:float|protein_submass:float|metabolite_submass:float|water_submass:float|DNA_submass:float]",
    "unique_array[domain_index:integer|coordinates:integer|is_forward:boolean|massDiff_rRNA:float|_entryState:integer|unique_index:integer]",
    "unique_array[TU_index:integer|coordinates:integer|domain_index:integer|bound_TF:array[23,boolean]|unique_index:integer]",
    "promoter",
    "rna",
    "active_RNAP",
    "DnaA_box",
];

#[test]
fn parse_and_resolve_fixture_types() {
    let reg = Registry::new();
    for n in NOTATIONS {
        let t = parse_type(n).unwrap_or_else(|e| panic!("parse {n}: {e}"));
        assert_eq!(&render_type(&t), n, "render round-trip");
        assert_eq!(parse_type(&render_type(&t)).unwrap(), t);
        let r = reg
            .resolve(&t)
            .unwrap_or_else(|e| panic!("resolve {n}: {e}"));
        assert!(!matches!(r, Type::Named(_)), "{n} unresolved");
    }

    // structure assertions
    assert_eq!(
        parse_type("overwrite[array[float[mM]]]").unwrap(),
        Type::Overwrite(Box::new(Type::Array {
            shape: vec![],
            elem: Box::new(Type::Float {
                units: Some("mM".into())
            })
        }))
    );
    assert_eq!(
        parse_type("array[23,boolean]").unwrap(),
        Type::Array {
            shape: vec![23],
            elem: Box::new(Type::Boolean)
        }
    );
    assert_eq!(
        parse_type("quantity[float,fg]").unwrap(),
        Type::Quantity {
            inner: Some(Box::new(Type::Float { units: None })),
            units: Some("fg".into())
        }
    );
}

fn find_bulk(m: &bigraph_schema_rs::doc::Map<Node>) -> Option<&bigraph_schema_rs::doc::Map<Value>> {
    for (k, n) in m {
        match n {
            Node::Leaf(Value::Tagged(_, t)) if k == "bulk" => return Some(t),
            Node::Store(s) => {
                if let Some(t) = find_bulk(s) {
                    return Some(t);
                }
            }
            _ => {}
        }
    }
    None
}

/// The fixture's real `bulk` structured array must match the registry's
/// `bulk_array` field names and dtype kinds, field for field.
#[test]
fn bulk_type_matches_fixture_structured_array() {
    let spec = from_json(FIXTURE).unwrap();
    let tagged = find_bulk(&spec.root).expect("bulk array in fixture");
    let Some(Value::Array(dtype)) = tagged.get("dtype") else {
        panic!("dtype")
    };
    let fixture: Vec<(String, String)> = dtype
        .iter()
        .map(|p| {
            let Value::Array(p) = p else { panic!() };
            let (Value::Str(n), Value::Str(d)) = (&p[0], &p[1]) else {
                panic!()
            };
            (n.clone(), d.clone())
        })
        .collect();
    assert_eq!(fixture.len(), 11);

    let reg = Registry::new();
    let t = reg.resolve(&parse_type("bulk_array").unwrap()).unwrap();
    let Type::BulkArray { fields } = t else {
        panic!("not bulk: {t:?}")
    };
    assert_eq!(fields.len(), fixture.len());
    for (f, (name, dt)) in fields.iter().zip(&fixture) {
        assert_eq!(&f.name, name);
        let kind = match (&f.ty, dt.as_str()) {
            (Type::String, d) if d.starts_with("<U") => true,
            (Type::Integer { .. }, "int64") => true,
            (Type::Float { .. }, "float64") => true,
            _ => false,
        };
        assert!(kind, "field {name}: {:?} vs {dt}", f.ty);
    }
}

#[test]
fn unique_record_structure() {
    let reg = Registry::new();
    let t = reg.resolve(&parse_type("active_RNAP").unwrap()).unwrap();
    let Type::UniqueArray { fields } = t else {
        panic!()
    };
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(&names[..3], ["domain_index", "coordinates", "is_forward"]);
    assert_eq!(names.last(), Some(&"unique_index"));
    assert_eq!(fields.len(), 3 + 9 + 2);
    assert_eq!(fields[2].ty, Type::Boolean);
    let p = reg.resolve(&parse_type("promoter").unwrap()).unwrap();
    let Type::UniqueArray { fields } = p else {
        panic!()
    };
    assert_eq!(
        fields[3].ty,
        Type::Array {
            shape: vec![23],
            elem: Box::new(Type::Boolean)
        }
    );
}

#[test]
fn dispatch_keys() {
    let reg = Registry::new();
    let d = |s: &str| reg.dispatch(&reg.resolve(&parse_type(s).unwrap()).unwrap());
    assert_eq!(
        (d("float").apply, d("float").reconcile),
        (ApplyKey::Additive, ReconcileKey::Sum)
    );
    assert_eq!(d("overwrite[array[integer]]").apply, ApplyKey::Overwrite);
    assert_eq!(
        d("overwrite[array[integer]]").reconcile,
        ReconcileKey::LastWins
    );
    assert_eq!(d("string").apply, ApplyKey::Replace);
    assert_eq!(d("bulk_array").apply, ApplyKey::BulkDelta);
    assert_eq!(d("bulk_array").reconcile, ReconcileKey::ConcatDeltas);
    assert_eq!(d("promoter").apply, ApplyKey::UniqueArray);
    assert_eq!(d("map[float]").apply, ApplyKey::Map);
    assert_eq!(d("array[float]").reconcile, ReconcileKey::ArrayMerge);
    assert_eq!(d("integer{1}").apply, ApplyKey::Additive);
}
