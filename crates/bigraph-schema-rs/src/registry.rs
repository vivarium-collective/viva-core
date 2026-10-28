//! Name -> [`Type`] registry plus the static apply/reconcile dispatch table.
//!
//! In Python (bigraph-schema 1.6.0) `apply`/`reconcile` are plum-dispatched on
//! the schema's class (`methods/apply.py`, `methods/reconcile.py`). Here the
//! dispatch is a static match from a resolved [`Type`] to an
//! ([`ApplyKey`], [`ReconcileKey`]) pair; Task 3 implements one function per key.

use crate::grammar::parse_type;
use crate::types::{Field, Type};
use std::collections::HashMap;

/// How an update is applied to a store of this type
/// (the Python `apply(schema, state, update, path)` overload chosen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApplyKey {
    /// `apply(Overwrite)`: replace; `None` update keeps state.
    Overwrite,
    /// `apply(Float/Integer/Atom)`: `state + update` (updates are deltas).
    Additive,
    /// `apply(String/Boolean)`: replace with update.
    Replace,
    /// `apply(Array)`: numpy array delta add / sparse / `{'set': ..}` forms.
    ArrayDelta,
    /// v2ecoli `apply(BulkNumpyUpdate)`: `state["count"][idx] += value`.
    BulkDelta,
    /// v2ecoli `apply(UniqueNumpyUpdate)`: add/set/delete unique molecules.
    UniqueArray,
    /// `apply(Map)`: per-key apply plus `_add`/`_remove` structural ops.
    Map,
    /// `apply(Tree)`.
    Tree,
    /// `apply(List)`: append/extend semantics.
    List,
    /// `apply(Node)` generic fallback (dict recursion by key).
    NodeDefault,
}

/// How concurrent updates in one layer are merged
/// (the Python `reconcile(schema, updates)` overload chosen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReconcileKey {
    /// `reconcile(Overwrite/Boolean/String)`: last non-None wins.
    LastWins,
    /// `reconcile(Float/Integer)`: sum of deltas (zero => None).
    Sum,
    /// `reconcile(Array)`: element-wise sum; `set` absorbing.
    ArrayMerge,
    /// v2ecoli `reconcile(BulkNumpyUpdate)`: concatenate `(idx, delta)` lists.
    ConcatDeltas,
    /// `reconcile(Map)`.
    MapMerge,
    /// `reconcile(Tree)`.
    TreeMerge,
    /// `reconcile(List)`.
    ListMerge,
    /// `reconcile(Node)` fallback: recursive dict merge, else last non-None.
    NodeDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dispatch {
    pub apply: ApplyKey,
    pub reconcile: ReconcileKey,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ResolveError {
    #[error("unknown type name {0:?}")]
    Unknown(String),
}

/// Canonical `bulk_array` fields (v2ecoli `library/schema_types.py::BULK_ARRAY`).
const BULK_FIELDS: &str = "id:string|count:integer|rRNA_submass:float|tRNA_submass:float|mRNA_submass:float|miscRNA_submass:float|nonspecific_RNA_submass:float|protein_submass:float|metabolite_submass:float|water_submass:float|DNA_submass:float";

const SUBMASS: &str = "massDiff_rRNA:float|massDiff_tRNA:float|massDiff_mRNA:float|massDiff_miscRNA:float|massDiff_nonspecific_RNA:float|massDiff_protein:float|massDiff_metabolite:float|massDiff_water:float|massDiff_DNA:float|_entryState:integer|unique_index:integer";

/// Biological unique-molecule type names -> their leading fields
/// (`BIOLOGICAL_UNIQUE_TYPES`; the shared submass/`_entryState`/`unique_index`
/// tail is appended).
const UNIQUE_TYPES: &[(&str, &str)] = &[
    ("promoter", "TU_index:integer|coordinates:integer|domain_index:integer|bound_TF:array[23,boolean]"),
    ("rna", "TU_index:integer|transcript_length:integer|is_mRNA:boolean|is_full_transcript:boolean|can_translate:boolean|RNAP_index:integer"),
    ("active_RNAP", "domain_index:integer|coordinates:integer|is_forward:boolean"),
    ("active_ribosome", "protein_index:integer|peptide_length:integer|mRNA_index:integer|pos_on_mRNA:integer"),
    ("active_replisome", "domain_index:integer|right_replichore:boolean|coordinates:integer"),
    ("full_chromosome", "division_time:float|has_triggered_division:boolean|domain_index:integer"),
    ("oriC", "domain_index:integer"),
    ("chromosome_domain", "domain_index:integer|child_domains:array[2,integer]"),
    ("chromosomal_segment", "boundary_molecule_indexes:array[2,integer]|boundary_coordinates:array[2,integer]|domain_index:integer|linking_number:float"),
    ("gene", "cistron_index:integer|coordinates:integer|domain_index:integer"),
    ("DnaA_box", "coordinates:integer|domain_index:integer|DnaA_bound:boolean|pool_label:integer|DnaA_bound_form:integer"),
];

fn fields_of(notation: &str) -> Vec<Field> {
    match parse_type(&format!("({notation})")).expect("builtin notation parses") {
        Type::Record(f) => f,
        _ => unreachable!(),
    }
}

pub struct Registry {
    named: HashMap<String, Type>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// A registry preloaded with the model's named types.
    pub fn new() -> Self {
        let mut named = HashMap::new();
        named.insert(
            "bulk_array".into(),
            Type::BulkArray {
                fields: fields_of(BULK_FIELDS),
            },
        );
        named.insert("unique_array".into(), Type::UniqueArray { fields: vec![] });
        for (name, head) in UNIQUE_TYPES {
            named.insert(
                (*name).into(),
                Type::UniqueArray {
                    fields: fields_of(&format!("{head}|{SUBMASS}")),
                },
            );
        }
        Self { named }
    }

    pub fn register(&mut self, name: &str, ty: Type) {
        self.named.insert(name.to_string(), ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&Type> {
        self.named.get(name)
    }

    /// Replace every [`Type::Named`] (recursively) with its definition.
    pub fn resolve(&self, t: &Type) -> Result<Type, ResolveError> {
        let b = |x: &Type| self.resolve(x).map(Box::new);
        Ok(match t {
            Type::Named(n) => self
                .named
                .get(n)
                .cloned()
                .ok_or_else(|| ResolveError::Unknown(n.clone()))?,
            Type::Quantity { inner, units } => Type::Quantity {
                inner: inner.as_deref().map(b).transpose()?,
                units: units.clone(),
            },
            Type::Array { shape, elem } => Type::Array {
                shape: shape.clone(),
                elem: b(elem)?,
            },
            Type::Map { value } => Type::Map { value: b(value)? },
            Type::Tree { leaf } => Type::Tree { leaf: b(leaf)? },
            Type::List { element } => Type::List {
                element: b(element)?,
            },
            Type::Overwrite(i) => Type::Overwrite(b(i)?),
            Type::WithDefault(i, d) => Type::WithDefault(b(i)?, d.clone()),
            Type::Record(f) => Type::Record(self.resolve_fields(f)?),
            Type::BulkArray { fields } if fields.is_empty() => self.named["bulk_array"].clone(),
            Type::BulkArray { fields } => Type::BulkArray {
                fields: self.resolve_fields(fields)?,
            },
            Type::UniqueArray { fields } => Type::UniqueArray {
                fields: self.resolve_fields(fields)?,
            },
            other => other.clone(),
        })
    }

    fn resolve_fields(&self, f: &[Field]) -> Result<Vec<Field>, ResolveError> {
        f.iter()
            .map(|x| {
                Ok(Field {
                    name: x.name.clone(),
                    ty: self.resolve(&x.ty)?,
                })
            })
            .collect()
    }

    /// The (apply, reconcile) dispatch keys for a resolved type. Wrappers
    /// delegate (`WithDefault`) or carry their own overload (`Overwrite`).
    pub fn dispatch(&self, t: &Type) -> Dispatch {
        use ApplyKey as A;
        use ReconcileKey as R;
        let (apply, reconcile) = match t {
            Type::WithDefault(i, _) => return self.dispatch(i),
            Type::Overwrite(_) => (A::Overwrite, R::LastWins),
            Type::Integer { .. } | Type::Float { .. } => (A::Additive, R::Sum),
            Type::String | Type::Boolean => (A::Replace, R::LastWins),
            Type::Array { .. } => (A::ArrayDelta, R::ArrayMerge),
            Type::BulkArray { .. } => (A::BulkDelta, R::ConcatDeltas),
            Type::UniqueArray { .. } => (A::UniqueArray, R::NodeDefault),
            Type::Map { .. } => (A::Map, R::MapMerge),
            Type::Tree { .. } => (A::Tree, R::TreeMerge),
            Type::List { .. } => (A::List, R::ListMerge),
            Type::Node | Type::Quantity { .. } | Type::Record(_) => {
                // Quantity has no custom apply in bigraph-schema 1.6.0: it
                // inherits `apply(Node)`, whose scalar branch is `result =
                // update` (replace). So a quantity leaf update replaces.
                (A::NodeDefault, R::NodeDefault)
            }
            // An unresolved name must never silently fall through to a default
            // overload — resolve it (`Registry::resolve`) first. Dispatching one
            // is a programming error, surfaced loudly rather than mis-applied.
            Type::Named(n) => {
                panic!("dispatch on unresolved type {n:?}; call Registry::resolve first")
            }
        };
        Dispatch { apply, reconcile }
    }
}
