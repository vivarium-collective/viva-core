//! The subset of bigraph-schema types the E. coli model's stores exercise.
//!
//! Deliberately NOT a port of all ~35 Python `Node` types (design §8):
//! only what `v2ecoli`'s port/store notation uses.

/// A named, typed field of a record / structured array.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// `node`: untyped / any.
    Node,
    Boolean,
    String,
    /// `integer` / `integer[units]`.
    Integer {
        units: Option<String>,
    },
    /// `float` / `float[units]`.
    Float {
        units: Option<String>,
    },
    /// `quantity`, `quantity[units]`, `quantity[float,units]`.
    Quantity {
        inner: Option<Box<Type>>,
        units: Option<String>,
    },
    /// `array[dtype]` / `array[d0,d1,dtype]`; `elem` is a scalar type, or a
    /// [`Type::Record`] for a structured (mixed-dtype) array.
    Array {
        shape: Vec<usize>,
        elem: Box<Type>,
    },
    /// `bulk_array[...]`: per-molecule counts (additive `(index, delta)` updates).
    BulkArray {
        fields: Vec<Field>,
    },
    /// `unique_array[...]` / named unique-molecule types (structured-record array).
    UniqueArray {
        fields: Vec<Field>,
    },
    /// `map[value]` (string keys).
    Map {
        value: Box<Type>,
    },
    /// `tree[leaf]`.
    Tree {
        leaf: Box<Type>,
    },
    /// `list[element]`.
    List {
        element: Box<Type>,
    },
    /// `overwrite[T]`: replace instead of merge.
    Overwrite(Box<Type>),
    /// `T{default}`: a default-value annotation (kept as raw text, like the
    /// Python grammar does).
    WithDefault(Box<Type>, String),
    /// `a:int|b:float`: a record / nested-dict schema.
    Record(Vec<Field>),
    /// A bare symbol not yet resolved by the registry (e.g. `promoter`).
    Named(String),
}
