use bigraph_schema_rs::apply::Op;
use bigraph_schema_rs::codec::from_json;
use bigraph_schema_rs::doc::Value;
use bigraph_schema_rs::state::StoreNode;
use process_bigraph_rs::engine::Engine;
use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};
use process_bigraph_rs::registry::ProcessRegistry;
use process_bigraph_rs::schedule::wire_step_layers;

fn layer_names(spec_json: &str) -> Vec<Vec<String>> {
    let spec = from_json(spec_json).unwrap();
    wire_step_layers(&spec)
        .iter()
        .map(|l| {
            let mut v: Vec<String> = l.steps.iter().map(|s| s.path.join("/")).collect();
            v.sort();
            v
        })
        .collect()
}

/// Expected order comes from pb 1.8.3's own scheduling functions:
/// `tests/oracle/gen_layers.py` -> `tests/fixtures/layers_expected.json`.
#[test]
fn layer_order_matches_pb() {
    let spec = include_str!("fixtures/layers_spec.json");
    let expected: Vec<Vec<String>> =
        serde_json::from_str(include_str!("fixtures/layers_expected.json")).unwrap();
    assert_eq!(layer_names(spec), expected);
}

#[test]
fn layer0_triggers_are_global_time() {
    let spec = from_json(include_str!("fixtures/layers_spec.json")).unwrap();
    let layers = wire_step_layers(&spec);
    for s in &layers[0].steps {
        assert_eq!(s.triggers, vec!["global_time".to_string()], "{:?}", s.path);
    }
    // explicit `_triggers` on a later-layer step is preserved
    let silent = layers[1]
        .steps
        .iter()
        .find(|s| s.path == ["s_silent"])
        .unwrap();
    assert_eq!(silent.triggers, vec!["x".to_string()]);
}

// ---- native test processes -------------------------------------------------

/// process: adds `rate * interval` to port `total`.
struct Accumulate {
    rate: f64,
}
impl Process for Accumulate {
    fn inputs(&self) -> Ports {
        vec![]
    }
    fn outputs(&self) -> Ports {
        vec![("total".into(), "float".into())]
    }
    fn update(&mut self, _v: &PortView, dt: f64) -> Result<Vec<PortUpdate>, ProcessError> {
        Ok(vec![PortUpdate::new(
            &["total"],
            Op::FloatDelta(self.rate * dt),
        )])
    }
}

/// step: mirrors `total` into `mirror` (overwrite).
struct Mirror;
impl Process for Mirror {
    fn inputs(&self) -> Ports {
        vec![("total".into(), "float".into())]
    }
    fn outputs(&self) -> Ports {
        vec![("mirror".into(), "overwrite[float]".into())]
    }
    fn update(&mut self, v: &PortView, _dt: f64) -> Result<Vec<PortUpdate>, ProcessError> {
        let Some(StoreNode::Leaf(t)) = v.get(&["total"]) else {
            return Err(ProcessError("no total".into()));
        };
        Ok(vec![PortUpdate::new(&["mirror"], Op::Overwrite(t.clone()))])
    }
}

const RUN_SPEC: &str = r#"{
  "global_time": 0.0,
  "total": 0.0,
  "mirror": -1.0,
  "acc": {"_type": "process", "address": "local:test.Accumulate", "config": {"rate": 2.0},
          "interval": 1.0, "inputs": {}, "outputs": {"total": ["total"]}},
  "mir": {"_type": "step", "address": "local:test.Mirror",
          "inputs": {"total": ["total"]}, "outputs": {"mirror": ["mirror"]}},
  "emitter": {"_type": "step", "address": "local:process_bigraph.emitter.RAMEmitter",
          "inputs": {"global_time": ["global_time"], "total": ["total"], "mirror": ["mirror"]},
          "outputs": {}}
}"#;

fn registry() -> ProcessRegistry {
    let mut r = ProcessRegistry::with_builtin_emitters();
    r.register("local:test.Accumulate", |cfg| {
        let rate = match cfg {
            Some(Value::Map(m)) => match m.get("rate") {
                Some(Value::Float(f)) => *f,
                _ => 1.0,
            },
            _ => 1.0,
        };
        Ok(Box::new(Accumulate { rate }))
    });
    r.register("local:test.Mirror", |_| Ok(Box::new(Mirror)));
    r
}

fn leaf_f64(e: &Engine, key: &str) -> f64 {
    match e.state.get(&[key]) {
        Some(StoreNode::Leaf(Value::Float(f))) => *f,
        other => panic!("{key}: {other:?}"),
    }
}

#[test]
fn run_advances_clock_and_applies() {
    let spec = from_json(RUN_SPEC).unwrap();
    let mut e = Engine::new(&spec, &registry()).unwrap();
    e.run(5).unwrap();
    assert_eq!(leaf_f64(&e, "global_time"), 5.0);
    assert_eq!(leaf_f64(&e, "total"), 10.0);
    // the step layer runs after each apply, so it sees the post-apply total
    assert_eq!(leaf_f64(&e, "mirror"), 10.0);
}

#[test]
fn run_until_stops_at_end_time() {
    let spec = from_json(RUN_SPEC).unwrap();
    let mut e = Engine::new(&spec, &registry()).unwrap();
    e.run_until(3.0).unwrap();
    assert_eq!(leaf_f64(&e, "global_time"), 3.0);
    assert_eq!(leaf_f64(&e, "total"), 6.0);
}

#[test]
fn emitter_captures_each_tick() {
    let spec = from_json(RUN_SPEC).unwrap();
    let mut e = Engine::new(&spec, &registry()).unwrap();
    e.run(3).unwrap();
    let h = e.emitter_history(&["emitter"]).expect("emitter history");
    // pb runs the step layers once before the first tick (t=0) and after each
    // applied tick: 1 + 3 rows, like RAMEmitter.history.
    assert_eq!(h.len(), 4);
    let get = |row: usize, k: &str| match h[row].get(k) {
        Some(StoreNode::Leaf(Value::Float(f))) => *f,
        o => panic!("{o:?}"),
    };
    let times: Vec<f64> = (0..4).map(|i| get(i, "global_time")).collect();
    assert_eq!(times, vec![0.0, 1.0, 2.0, 3.0]);
    let totals: Vec<f64> = (0..4).map(|i| get(i, "total")).collect();
    assert_eq!(totals, vec![0.0, 2.0, 4.0, 6.0]);
    // t=0 row: initial steps ran before any process applied (mirror was -1.0 ->
    // overwritten with total 0.0 by the Mirror step in the same layer pass;
    // the emitter is a later layer so it sees it).
    assert_eq!(get(0, "mirror"), 0.0);
    assert_eq!(get(3, "mirror"), 6.0);
}
