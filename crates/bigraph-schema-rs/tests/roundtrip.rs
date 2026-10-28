use bigraph_schema_rs::codec::{from_json, to_json};
use bigraph_schema_rs::doc::{Node, TagKind, Value};

const FIXTURE: &str = include_str!("fixtures/ecoli_partitioned_slice.pbg");

/// Replace bare non-finite tokens (outside strings) with sentinel strings so
/// strict serde_json can parse them. Independent of the codec under test.
fn sanitize(s: &str) -> String {
    let mut out = String::new();
    let b: Vec<char> = s.chars().collect();
    let (mut i, mut in_str) = (0, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c);
            if c == '\\' {
                out.push(b[i + 1]);
                i += 1;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        let rest: String = b[i..b.len().min(i + 9)].iter().collect();
        if c == '"' {
            in_str = true;
            out.push(c);
        } else if rest.starts_with("-Infinity") {
            out.push_str("\"@-Infinity\"");
            i += 8;
        } else if rest.starts_with("Infinity") {
            out.push_str("\"@Infinity\"");
            i += 7;
        } else if rest.starts_with("NaN") {
            out.push_str("\"@NaN\"");
            i += 2;
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

fn tree(s: &str) -> serde_json::Value {
    serde_json::from_str(&sanitize(s)).expect("sanitized json parses strictly")
}

#[test]
fn strict_serde_rejects_bare_infinity() {
    assert!(serde_json::from_str::<serde_json::Value>(FIXTURE).is_err());
}

#[test]
fn roundtrip_structural() {
    assert!(FIXTURE.contains("Infinity") && FIXTURE.contains("NaN"));
    let spec = from_json(FIXTURE).expect("lenient parse");
    let out = to_json(&spec);
    // Non-finite values survive as bare tokens.
    assert_eq!(
        out.matches("Infinity").count(),
        FIXTURE.matches("Infinity").count()
    );
    assert_eq!(out.matches("NaN").count(), FIXTURE.matches("NaN").count());
    // Same decoded tree (compared as parsed trees, not bytes).
    assert_eq!(tree(FIXTURE), tree(&out));
    // Second pass is a fixed point at the model level.
    assert_eq!(from_json(&out).unwrap(), spec);
}

#[test]
fn edges_and_tagged_dicts_are_modeled() {
    let spec = from_json(FIXTURE).unwrap();
    let edges = spec.edges();
    assert_eq!(edges.len(), 5);
    let (path, eq) = edges
        .iter()
        .find(|(_, e)| e.address.ends_with("equilibrium.Equilibrium"))
        .unwrap();
    assert_eq!(path.join("/"), "agents/0/ecoli-equilibrium");
    assert_eq!(eq.edge_type.as_deref(), Some("step"));
    assert!(eq.inputs.is_some() && eq.outputs.is_some());

    let Some(Node::Store(agents)) = spec.root.get("agents") else {
        panic!()
    };
    let Some(Node::Store(a0)) = agents.get("0") else {
        panic!()
    };
    let Some(Node::Leaf(Value::Tagged(kind, _))) = a0.get("bulk") else {
        panic!()
    };
    assert_eq!(*kind, TagKind::StructuredArray);
}

#[test]
fn edge_optional_fields_roundtrip() {
    let src = r#"{"p": {"_type": "process", "address": "local:X", "config": {"a": 1, "b": 2.0},
      "inputs": {"x": ["s"]}, "outputs": {}, "interval": 0.5, "_contract": {"k": [1, 2]},
      "extra_key": NaN},
      "q": {"__pint__": true, "magnitude": -Infinity, "units": "mol"},
      "r": {"__set__": true, "data": [1, 2]}, "n": {"__numpy__": true, "dtype": "int64", "shape": [2], "data": [1, 2]},
      "bb": {"__bytes__": true, "data": "ff"}, "big": 18446744073709551615, "u": "café\n"}"#;
    let spec = from_json(src).unwrap();
    let e = &spec.edges()[0].1;
    assert_eq!(e.interval, Some(Value::Float(0.5)));
    assert!(e.config.is_some() && e.contract.is_some());
    assert_eq!(tree(src), tree(&to_json(&spec)));
    assert_eq!(from_json(&to_json(&spec)).unwrap(), spec);
}

#[test]
fn rejects_malformed() {
    assert!(from_json("{\"a\": }").is_err());
    assert!(from_json("{\"a\": 1} x").is_err());
    assert!(from_json("[1]").is_err());
}
