//! Address -> process factory registry. Task 5 registers a PyO3-backed
//! factory for `local:` Python addresses here.

use crate::emitter::RamEmitter;
use crate::process::{Process, ProcessError};
use bigraph_schema_rs::doc::Value;
use std::collections::HashMap;

pub type Factory = Box<dyn Fn(Option<&Value>) -> Result<Box<dyn Process>, ProcessError>>;

#[derive(Default)]
pub struct ProcessRegistry {
    factories: HashMap<String, Factory>,
}

impl ProcessRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry preloaded with the RAM/JSON emitter addresses.
    pub fn with_builtin_emitters() -> Self {
        let mut r = Self::new();
        for addr in [
            "local:process_bigraph.emitter.RAMEmitter",
            "local:process_bigraph.emitter.JSONEmitter",
            "local:ram-emitter",
            "local:json-emitter",
        ] {
            r.register(addr, |cfg| Ok(Box::new(RamEmitter::from_config(cfg))));
        }
        r
    }

    pub fn register<F>(&mut self, address: &str, f: F)
    where
        F: Fn(Option<&Value>) -> Result<Box<dyn Process>, ProcessError> + 'static,
    {
        self.factories.insert(address.to_string(), Box::new(f));
    }

    pub fn create(
        &self,
        address: &str,
        config: Option<&Value>,
    ) -> Result<Box<dyn Process>, ProcessError> {
        let f = self
            .factories
            .get(address)
            .ok_or_else(|| ProcessError(format!("no process registered for {address:?}")))?;
        f(config)
    }
}
