//! Parser/renderer for bigraph-schema's compact type notation.
//!
//! Mirrors the PEG in `bigraph_schema/parse.py` (parsimonious):
//! ```text
//! expression = merge / union / tree / nothing
//! merge  = tree ('|' tree)+      union = tree ('~' tree)+
//! tree   = group / nest / type_name
//! group  = '(' expression ')'    nest  = symbol ':' tree
//! type_name = symbol ('[' expression (',' expression)* ']')? ('{' [^}]* '}')?
//! symbol = [\w\d-_/<>*&^%$#@!`+. ]+
//! ```
//! Implemented subset: merge-of-nests (records), groups, type parameters and
//! default blocks. Unions (`~`) and tuples (non-record merges) are unused by
//! the model and rejected with a clear error.

use crate::types::{Field, Type};

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("type notation error at {pos} in {input:?}: {msg}")]
pub struct ParseError {
    pub input: String,
    pub pos: usize,
    pub msg: String,
}

/// Parse a type-notation string.
pub fn parse_type(s: &str) -> Result<Type, ParseError> {
    let mut p = Parser {
        src: s.chars().collect(),
        pos: 0,
        input: s,
    };
    let e = p.expression()?;
    if p.pos != p.src.len() {
        return Err(p.err("unexpected trailing input"));
    }
    match e {
        Expr::Type(t) => Ok(t),
        Expr::Fields(f) => Ok(Type::Record(f)),
    }
}

enum Expr {
    Type(Type),
    Fields(Vec<Field>),
}

struct Parser<'a> {
    src: Vec<char>,
    pos: usize,
    input: &'a str,
}

fn is_symbol_char(c: char) -> bool {
    c.is_alphanumeric() || "_-/<>*&^%$#@!`+. ".contains(c)
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            input: self.input.to_string(),
            pos: self.pos,
            msg: msg.to_string(),
        }
    }
    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// expression: tree ('|' tree)*  (all-nests => record)
    fn expression(&mut self) -> Result<Expr, ParseError> {
        let first = self.tree()?;
        match self.peek() {
            Some('|') => {
                let mut fields = Vec::new();
                self.push_field(&mut fields, first)?;
                while self.eat('|') {
                    let t = self.tree()?;
                    self.push_field(&mut fields, t)?;
                }
                Ok(Expr::Fields(fields))
            }
            Some('~') => Err(self.err("union (~) is not supported (unused by the model)")),
            _ => Ok(match first {
                Tree::Type(t) => Expr::Type(t),
                Tree::Nest(n, t) => Expr::Fields(vec![Field { name: n, ty: t }]),
            }),
        }
    }

    fn push_field(&mut self, out: &mut Vec<Field>, t: Tree) -> Result<(), ParseError> {
        match t {
            Tree::Nest(name, ty) => {
                out.push(Field { name, ty });
                Ok(())
            }
            Tree::Type(_) => Err(self.err("tuple merge (non key:value) is not supported")),
        }
    }

    fn tree(&mut self) -> Result<Tree, ParseError> {
        if self.eat('(') {
            let e = self.expression()?;
            if !self.eat(')') {
                return Err(self.err("expected ')'"));
            }
            return Ok(match e {
                Expr::Type(t) => Tree::Type(t),
                Expr::Fields(f) => Tree::Type(Type::Record(f)),
            });
        }
        let sym = self.symbol()?;
        if self.peek() == Some(':') {
            self.pos += 1;
            return match self.tree()? {
                Tree::Type(t) => Ok(Tree::Nest(sym.trim().to_string(), t)),
                Tree::Nest(k, t) => Ok(Tree::Nest(
                    sym.trim().to_string(),
                    Type::Record(vec![Field { name: k, ty: t }]),
                )),
            };
        }
        let mut params = Vec::new();
        if self.eat('[') {
            loop {
                params.push(self.expression()?);
                if self.eat(',') {
                    continue;
                }
                if self.eat(']') {
                    break;
                }
                return Err(self.err("expected ',' or ']'"));
            }
        }
        let mut default = None;
        if self.eat('{') {
            let s = self.pos;
            while self.peek().is_some_and(|c| c != '}') {
                self.pos += 1;
            }
            let d: String = self.src[s..self.pos].iter().collect();
            if !self.eat('}') {
                return Err(self.err("expected '}'"));
            }
            default = Some(d);
        }
        let t = self.build(sym.trim(), params)?;
        Ok(Tree::Type(match default {
            Some(d) => Type::WithDefault(Box::new(t), d),
            None => t,
        }))
    }

    fn symbol(&mut self) -> Result<String, ParseError> {
        let s = self.pos;
        while self.peek().is_some_and(is_symbol_char) {
            self.pos += 1;
        }
        if s == self.pos {
            return Err(self.err("expected a type symbol"));
        }
        Ok(self.src[s..self.pos].iter().collect())
    }

    /// A parameter that is a bare unit/dim symbol (no nesting).
    fn as_symbol(e: &Expr) -> Option<String> {
        match e {
            Expr::Type(Type::Named(n)) => Some(n.clone()),
            _ => None,
        }
    }

    fn param_type(&self, e: Expr) -> Type {
        match e {
            Expr::Type(t) => t,
            Expr::Fields(f) => Type::Record(f),
        }
    }

    fn build(&self, name: &str, mut params: Vec<Expr>) -> Result<Type, ParseError> {
        let one = |s: &Self, mut p: Vec<Expr>, what: &str| -> Result<Type, ParseError> {
            if p.len() != 1 {
                return Err(s.err(&format!("{what} takes exactly one parameter")));
            }
            Ok(s.param_type(p.remove(0)))
        };
        Ok(match (name, params.len()) {
            ("node", 0) => Type::Node,
            ("boolean", 0) => Type::Boolean,
            ("string", 0) => Type::String,
            ("integer", 0) => Type::Integer { units: None },
            ("float", 0) => Type::Float { units: None },
            ("integer" | "float", 1) => {
                let units = Self::as_symbol(&params[0])
                    .ok_or_else(|| self.err("scalar unit parameter must be a bare symbol"))?;
                if name == "integer" {
                    Type::Integer { units: Some(units) }
                } else {
                    Type::Float { units: Some(units) }
                }
            }
            ("quantity", 0) => Type::Quantity {
                inner: None,
                units: None,
            },
            ("quantity", 1) => Type::Quantity {
                inner: None,
                units: Some(
                    Self::as_symbol(&params[0])
                        .ok_or_else(|| self.err("quantity units must be a bare symbol"))?,
                ),
            },
            ("quantity", 2) => {
                let units = Self::as_symbol(&params[1])
                    .ok_or_else(|| self.err("quantity units must be a bare symbol"))?;
                let inner = self.param_type(params.remove(0));
                Type::Quantity {
                    inner: Some(Box::new(inner)),
                    units: Some(units),
                }
            }
            ("array", n) if n >= 1 => {
                let elem = self.param_type(params.pop().unwrap());
                let mut shape = Vec::new();
                for p in &params {
                    let d = Self::as_symbol(p)
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .ok_or_else(|| self.err("array shape dims must be integers"))?;
                    shape.push(d);
                }
                Type::Array {
                    shape,
                    elem: Box::new(elem),
                }
            }
            // Bare `bulk_array` / `unique_array` (no field list) is the form a
            // v2ecoli process `outputs()` declares for a molecule port (the
            // concrete fields live on the store, not the port type). It must
            // still route updates to the additive bulk / path-keyed unique ops.
            ("bulk_array", 0) => Type::BulkArray { fields: Vec::new() },
            ("unique_array", 0) => Type::UniqueArray { fields: Vec::new() },
            ("bulk_array", 1) => match self.param_type(params.remove(0)) {
                Type::Record(fields) => Type::BulkArray { fields },
                _ => return Err(self.err("bulk_array parameter must be fields")),
            },
            ("unique_array", 1) => match self.param_type(params.remove(0)) {
                Type::Record(fields) => Type::UniqueArray { fields },
                _ => return Err(self.err("unique_array parameter must be fields")),
            },
            ("map", 0) => Type::Map {
                value: Box::new(Type::Node),
            },
            ("map", _) => Type::Map {
                value: Box::new(one(self, params, "map")?),
            },
            ("tree", 0) => Type::Tree {
                leaf: Box::new(Type::Node),
            },
            ("tree", _) => Type::Tree {
                leaf: Box::new(one(self, params, "tree")?),
            },
            ("list", 0) => Type::List {
                element: Box::new(Type::Node),
            },
            ("list", _) => Type::List {
                element: Box::new(one(self, params, "list")?),
            },
            ("overwrite", _) => Type::Overwrite(Box::new(one(self, params, "overwrite")?)),
            (n, 0) => Type::Named(n.to_string()),
            (n, _) => return Err(self.err(&format!("unsupported parameterized type {n:?}"))),
        })
    }
}

enum Tree {
    Type(Type),
    Nest(String, Type),
}

/// Render a type back to canonical bigraph notation.
pub fn render_type(t: &Type) -> String {
    fn fields(fs: &[Field]) -> String {
        fs.iter()
            .map(|f| format!("{}:{}", f.name, render_type(&f.ty)))
            .collect::<Vec<_>>()
            .join("|")
    }
    match t {
        Type::Node => "node".into(),
        Type::Boolean => "boolean".into(),
        Type::String => "string".into(),
        Type::Integer { units: None } => "integer".into(),
        Type::Integer { units: Some(u) } => format!("integer[{u}]"),
        Type::Float { units: None } => "float".into(),
        Type::Float { units: Some(u) } => format!("float[{u}]"),
        Type::Quantity {
            inner: None,
            units: None,
        } => "quantity".into(),
        Type::Quantity {
            inner: None,
            units: Some(u),
        } => format!("quantity[{u}]"),
        Type::Quantity {
            inner: Some(i),
            units: Some(u),
        } => {
            format!("quantity[{},{u}]", render_type(i))
        }
        Type::Quantity {
            inner: Some(i),
            units: None,
        } => format!("quantity[{}]", render_type(i)),
        Type::Array { shape, elem } => {
            let mut parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            parts.push(match &**elem {
                Type::Record(f) => fields(f),
                e => render_type(e),
            });
            format!("array[{}]", parts.join(","))
        }
        Type::BulkArray { fields: f } if f.is_empty() => "bulk_array".into(),
        Type::BulkArray { fields: f } => format!("bulk_array[{}]", fields(f)),
        Type::UniqueArray { fields: f } => format!("unique_array[{}]", fields(f)),
        Type::Map { value } => format!("map[{}]", render_type(value)),
        Type::Tree { leaf } => format!("tree[{}]", render_type(leaf)),
        Type::List { element } => format!("list[{}]", render_type(element)),
        Type::Overwrite(i) => format!("overwrite[{}]", render_type(i)),
        Type::WithDefault(i, d) => format!("{}{{{d}}}", render_type(i)),
        Type::Record(f) => format!("({})", fields(f)),
        Type::Named(n) => n.clone(),
    }
}
