use bigraph_schema_rs::apply::Op;
use bigraph_schema_rs::codec::from_json;
use bigraph_schema_rs::doc::Value;
use process_bigraph_rs::engine::Engine;
use process_bigraph_rs::parallel::run_ensemble;
use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};
use process_bigraph_rs::registry::ProcessRegistry;

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

/// One ensemble member = one seed; the seed sets the process rate.
fn build(seed: &u64) -> Result<Engine, process_bigraph_rs::engine::EngineError> {
    let spec = from_json(&format!(
        r#"{{"global_time": 0.0, "total": 0.0,
  "acc": {{"_type": "process", "address": "local:test.Accumulate",
           "config": {{"rate": {}.37}}, "interval": 1.0,
           "inputs": {{}}, "outputs": {{"total": ["total"]}}}},
  "emitter": {{"_type": "step", "address": "local:ram-emitter",
           "inputs": {{"global_time": ["global_time"], "total": ["total"]}}, "outputs": {{}}}}}}"#,
        seed
    ))
    .unwrap();
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
    Engine::new(&spec, &r)
}

#[test]
fn ensemble_bit_identical_to_serial() {
    let seeds: Vec<u64> = (0..64).collect();
    let n_ticks = 50;
    let par = run_ensemble(&seeds, n_ticks, build).unwrap();
    assert_eq!(par.len(), seeds.len());
    for (seed, got) in seeds.iter().zip(&par) {
        let mut e = build(seed).unwrap();
        e.run(n_ticks).unwrap();
        assert_eq!(&e.state, got, "member {seed} differs from serial run");
    }
    // members are genuinely distinct (not all the same state)
    assert_ne!(par[1], par[2]);
}
