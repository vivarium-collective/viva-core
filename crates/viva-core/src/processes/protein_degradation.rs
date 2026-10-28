//! Native Rust port of v2ecoli `ProteinDegradation`, bit-identical to the Python
//! twin (`v2ecoli/processes/protein_degradation.py`).
//!
//! Per tick the Python process does:
//!
//! ```text
//! n_to_degrade = np.fmin(RandomState.poisson(rate * dt * protein_counts), protein_counts)
//! delta_metabolites = S @ n_to_degrade          # S: (n_aa + 1) x n_proteins, int64
//! return {"bulk": [(metabolite_idx, delta_metabolites), (protein_idx, -n_to_degrade)]}
//! ```
//!
//! The Rust twin reproduces each step exactly: the same `RandomState(seed)`
//! stream (via [`bigraph_schema_rs::rng`]), one vector-`lam` poisson draw per
//! protein in order (numpy consumes left to right; `lam == 0` draws nothing),
//! the same `fmin` cap, and the same integer S-matrix product (wrapping int64,
//! as numpy). Config is the serialized `get_protein_degradation_config` dict.

use std::collections::HashMap;

use bigraph_schema_rs::apply::Op;
use bigraph_schema_rs::doc::{Map, TagKind, Value};
use bigraph_schema_rs::rng::RandomState;
use bigraph_schema_rs::state::StoreNode;
use process_bigraph_rs::process::{PortUpdate, PortView, Ports, Process, ProcessError};

fn err(msg: impl Into<String>) -> ProcessError {
    ProcessError(msg.into())
}

pub struct ProteinDegradationCore {
    rate: Vec<f64>,
    protein_ids: Vec<String>,
    metabolite_ids: Vec<String>,
    n_proteins: usize,
    n_metabolites: usize,
    /// Degradation stoichiometry, row-major `(n_metabolites x n_proteins)` int64.
    s_matrix: Vec<i64>,
    rng: RandomState,
    /// Resolved on the first update, as the Python process does.
    protein_idx: Option<Vec<usize>>,
    metabolite_idx: Option<Vec<usize>>,
}

impl ProteinDegradationCore {
    /// Engine-registry constructor (the serialized `get_protein_degradation_config`).
    pub fn create(config: Option<&Value>) -> Result<Box<dyn Process>, ProcessError> {
        Ok(Box::new(Self::from_config(config)?))
    }

    pub fn from_config(config: Option<&Value>) -> Result<Self, ProcessError> {
        let empty = Map::new();
        let cfg = match config {
            Some(Value::Map(m)) => m,
            None | Some(Value::Null) => &empty,
            Some(_) => return Err(err("ProteinDegradation config must be a map")),
        };

        let rate = floats(cfg.get("raw_degradation_rate"), "raw_degradation_rate")?;
        let water_id = match cfg.get("water_id") {
            Some(Value::Str(s)) => s.clone(),
            None => "h2o".to_string(), // config_schema default
            Some(_) => return Err(err("water_id must be a string")),
        };
        let amino_acid_ids = strings(cfg.get("amino_acid_ids"), "amino_acid_ids")?;
        let protein_ids = strings(cfg.get("protein_ids"), "protein_ids")?;
        let aa_counts = int_matrix(cfg.get("amino_acid_counts"), "amino_acid_counts")?;
        let seed = match cfg.get("seed") {
            Some(Value::Int(i)) => u32::try_from(*i)
                .map_err(|_| err(format!("seed {i} out of numpy RandomState range")))?,
            None => 0, // config_schema default
            Some(_) => return Err(err("seed must be an integer")),
        };

        let n_aa = amino_acid_ids.len();
        let n_proteins = protein_ids.len();
        if rate.len() != n_proteins {
            return Err(err("raw_degradation_rate length != number of proteins"));
        }
        if aa_counts.len() != n_proteins || aa_counts.iter().any(|r| r.len() != n_aa) {
            return Err(err(
                "amino_acid_counts must be (n_proteins x n_amino_acids)",
            ));
        }

        let mut metabolite_ids = amino_acid_ids;
        metabolite_ids.push(water_id.clone());
        let n_metabolites = metabolite_ids.len();
        // `self.metabolite_ids.index(water_id)`: first match.
        let water_index = metabolite_ids
            .iter()
            .position(|m| *m == water_id)
            .expect("water id was just appended");

        // S[:n_aa, :] = aa_counts.T ; S[water, :] = -(colsum(S[:n_aa]) - 1).
        // (Same order as Python: the water row is set from the AA rows after
        // the transpose assignment; if water_index < n_aa it overwrites an AA
        // row exactly as numpy would.)
        let mut s = vec![0i64; n_metabolites * n_proteins];
        for (p, row) in aa_counts.iter().enumerate() {
            for (a, &c) in row.iter().enumerate() {
                s[a * n_proteins + p] = c;
            }
        }
        for p in 0..n_proteins {
            let mut sum = 0i64;
            for a in 0..n_aa {
                sum = sum.wrapping_add(s[a * n_proteins + p]);
            }
            s[water_index * n_proteins + p] = -(sum.wrapping_sub(1));
        }

        Ok(Self {
            rate,
            protein_ids,
            metabolite_ids,
            n_proteins,
            n_metabolites,
            s_matrix: s,
            rng: RandomState::new(seed),
            protein_idx: None,
            metabolite_idx: None,
        })
    }

    /// `bulk_name_to_idx(names, bulk_ids)` (strict).
    fn name_to_idx(
        names: &[String],
        index: &HashMap<&str, usize>,
    ) -> Result<Vec<usize>, ProcessError> {
        names
            .iter()
            .map(|n| {
                index
                    .get(n.as_str())
                    .copied()
                    .ok_or_else(|| err(format!("bulk id {n:?} not found")))
            })
            .collect()
    }
}

impl Process for ProteinDegradationCore {
    fn inputs(&self) -> Ports {
        vec![
            ("bulk".into(), "bulk_array".into()),
            ("timestep".into(), "integer".into()),
        ]
    }

    fn outputs(&self) -> Ports {
        vec![("bulk".into(), "bulk_array".into())]
    }

    fn update(&mut self, view: &PortView, _interval: f64) -> Result<Vec<PortUpdate>, ProcessError> {
        let bulk = match view.get(&["bulk"]) {
            Some(StoreNode::Bulk(b)) => b,
            _ => return Err(err("ProteinDegradation: `bulk` port is not a bulk store")),
        };
        let dt = match view.get(&["timestep"]) {
            Some(StoreNode::Leaf(Value::Int(i))) => *i as f64,
            Some(StoreNode::Leaf(Value::Float(f))) => *f,
            _ => return Err(err("ProteinDegradation: `timestep` port is not a number")),
        };

        if self.metabolite_idx.is_none() {
            let index: HashMap<&str, usize> = bulk
                .ids
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str(), i))
                .collect();
            self.protein_idx = Some(Self::name_to_idx(&self.protein_ids, &index)?);
            self.metabolite_idx = Some(Self::name_to_idx(&self.metabolite_ids, &index)?);
        }
        let protein_idx = self.protein_idx.as_ref().expect("resolved above");
        let metabolite_idx = self.metabolite_idx.as_ref().expect("resolved above");

        let counts: Vec<i64> = protein_idx.iter().map(|&i| bulk.count[i]).collect();
        // `raw_degradation_rate * dt * protein_counts`: (rate * dt) first.
        let lams: Vec<f64> = self
            .rate
            .iter()
            .zip(&counts)
            .map(|(&r, &c)| r * dt * c as f64)
            .collect();
        if lams.iter().any(|l| l.is_nan() || *l < 0.0) {
            return Err(err("ProteinDegradation: poisson lam must be >= 0"));
        }
        let n_to_degrade: Vec<i64> = self
            .rng
            .poisson_each(&lams)
            .into_iter()
            .zip(&counts)
            .map(|(d, &c)| d.min(c))
            .collect();

        let np = self.n_proteins;
        let mut deltas: Vec<(usize, i128)> = Vec::with_capacity(self.n_metabolites + np);
        for (m, &mi) in metabolite_idx.iter().enumerate() {
            let row = &self.s_matrix[m * np..(m + 1) * np];
            let mut acc = 0i64;
            for (&s, &n) in row.iter().zip(&n_to_degrade) {
                acc = acc.wrapping_add(s.wrapping_mul(n));
            }
            deltas.push((mi, acc as i128));
        }
        for (&pi, &n) in protein_idx.iter().zip(&n_to_degrade) {
            deltas.push((pi, -(n as i128)));
        }
        Ok(vec![PortUpdate::new(&["bulk"], Op::BulkDelta(deltas))])
    }
}

// ---- config deserialization (plain JSON lists or `__numpy__` tagged) --------

/// Unwrap a `__numpy__` tagged array to `(flat data, shape)`; plain values pass.
fn untag(v: &Value) -> (&Value, Option<Vec<usize>>) {
    if let Value::Tagged(TagKind::Numpy, m) = v {
        let shape = match m.get("shape") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| match x {
                    Value::Int(i) => usize::try_from(*i).ok(),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        if let Some(d) = m.get("data") {
            return (d, Some(shape));
        }
    }
    (v, None)
}

fn flatten<'a>(v: &'a Value, out: &mut Vec<&'a Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten(x, out)),
        other => out.push(other),
    }
}

fn num_f64(v: &Value, what: &str) -> Result<f64, ProcessError> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::UInt(u) => Ok(*u as f64),
        _ => Err(err(format!("{what}: expected a number"))),
    }
}

fn num_i64(v: &Value, what: &str) -> Result<i64, ProcessError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Float(f) if f.fract() == 0.0 => Ok(*f as i64),
        _ => Err(err(format!("{what}: expected an integer"))),
    }
}

fn floats(v: Option<&Value>, what: &str) -> Result<Vec<f64>, ProcessError> {
    let Some(v) = v else { return Ok(Vec::new()) };
    let (v, _) = untag(v);
    let mut flat = Vec::new();
    flatten(v, &mut flat);
    flat.into_iter().map(|x| num_f64(x, what)).collect()
}

fn strings(v: Option<&Value>, what: &str) -> Result<Vec<String>, ProcessError> {
    match v {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| match x {
                Value::Str(s) => Ok(s.clone()),
                _ => Err(err(format!("{what}: expected strings"))),
            })
            .collect(),
        Some(_) => Err(err(format!("{what}: expected a list"))),
    }
}

fn int_matrix(v: Option<&Value>, what: &str) -> Result<Vec<Vec<i64>>, ProcessError> {
    let Some(v) = v else { return Ok(Vec::new()) };
    let (v, shape) = untag(v);
    if let Some(shape) = shape.filter(|s| s.len() == 2) {
        let mut flat = Vec::new();
        flatten(v, &mut flat);
        let ints: Vec<i64> = flat
            .into_iter()
            .map(|x| num_i64(x, what))
            .collect::<Result<_, _>>()?;
        if ints.len() != shape[0] * shape[1] {
            return Err(err(format!("{what}: data length != shape")));
        }
        return Ok(ints.chunks(shape[1].max(1)).map(<[i64]>::to_vec).collect());
    }
    match v {
        Value::Array(rows) => rows
            .iter()
            .map(|r| match r {
                Value::Array(cols) => cols.iter().map(|c| num_i64(c, what)).collect(),
                _ => Err(err(format!("{what}: expected a 2-D matrix"))),
            })
            .collect(),
        _ => Err(err(format!("{what}: expected a 2-D matrix"))),
    }
}
