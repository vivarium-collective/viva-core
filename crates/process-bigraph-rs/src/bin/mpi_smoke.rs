//! MPI smoke: scatter seeds across ranks, run a real `Engine` per seed on
//! each rank, gather one scalar (final `total`) per seed on rank 0.
//! Run: `mpirun -n 2 target/debug/mpi_smoke` (see tests/mpi_smoke.sh).
use bigraph_schema_rs::apply::Op;
use bigraph_schema_rs::codec::from_json;
use bigraph_schema_rs::doc::Value;
use bigraph_schema_rs::state::StoreNode;
use mpi::traits::*;
use process_bigraph_rs::engine::Engine;
use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};
use process_bigraph_rs::registry::ProcessRegistry;

struct Accumulate(f64);
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
            Op::FloatDelta(self.0 * dt),
        )])
    }
}

fn run_seed(seed: i64) -> f64 {
    let spec = from_json(
        r#"{"global_time": 0.0, "total": 0.0,
  "acc": {"_type": "process", "address": "local:test.Accumulate", "interval": 1.0,
           "inputs": {}, "outputs": {"total": ["total"]}}}"#,
    )
    .unwrap();
    let mut r = ProcessRegistry::new();
    let rate = seed as f64;
    r.register("local:test.Accumulate", move |_| {
        Ok(Box::new(Accumulate(rate)))
    });
    let mut e = Engine::new(&spec, &r).unwrap();
    e.run(10).unwrap();
    match e.state.get(&["total"]) {
        Some(StoreNode::Leaf(Value::Float(f))) => *f,
        o => panic!("total: {o:?}"),
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let (size, rank) = (world.size() as usize, world.rank());
    let root = world.process_at_rank(0);
    let n = 8 * size; // divisible by rank count
    let per = n / size;
    let mut local = vec![0i64; per];
    if rank == 0 {
        let seeds: Vec<i64> = (0..n as i64).collect();
        root.scatter_into_root(&seeds[..], &mut local[..]);
    } else {
        root.scatter_into(&mut local[..]);
    }
    let results: Vec<f64> = local.iter().map(|&s| run_seed(s)).collect();
    if rank == 0 {
        let mut all = vec![0f64; n];
        root.gather_into_root(&results[..], &mut all[..]);
        let sum: f64 = all.iter().sum();
        let expect: f64 = (0..n as i64).map(|s| s as f64 * 10.0).sum();
        println!("RANK0: gathered {n} results from {size} ranks, sum = {sum}");
        assert_eq!(sum, expect);
        println!("MPI_SCATTER_GATHER_OK");
    } else {
        root.gather_into(&results[..]);
        println!("RANK{rank}: computed {} local results", results.len());
    }
}
