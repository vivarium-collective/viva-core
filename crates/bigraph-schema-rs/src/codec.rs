//! Lenient JSON codec: accepts and emits bare `Infinity` / `-Infinity` / `NaN`
//! (Python `json` `allow_nan` default), which strict JSON parsers reject.
//! Hand-written so non-finite tokens are handled at the token level with no
//! sentinel-string collisions.

use crate::doc::{CompositeSpec, Map, Node, ProcessEdge, TagKind, Value};
use std::fmt::Write;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("JSON parse error at byte {pos}: {msg}")]
    Parse { pos: usize, msg: String },
    #[error("top-level JSON value must be an object")]
    NotAnObject,
}

/// Parse a `.pbg` composite spec (lenient about `Infinity`/`NaN`).
pub fn from_json(s: &str) -> Result<CompositeSpec, Error> {
    let mut p = Parser {
        s: s.as_bytes(),
        src: s,
        i: 0,
        depth: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing characters"));
    }
    match v {
        Value::Map(m) => match Node::from_map(m) {
            Node::Store(root) => Ok(CompositeSpec { root }),
            // The document root itself looks like an edge: keep it as a store of its fields.
            Node::Edge(e) => Ok(CompositeSpec {
                root: edge_fields(*e),
            }),
            Node::Leaf(_) => unreachable!(),
        },
        Value::Tagged(..) => Err(Error::NotAnObject),
        _ => Err(Error::NotAnObject),
    }
}

fn edge_fields(e: ProcessEdge) -> Map<Node> {
    let mut m = Map::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            m.insert(k.to_string(), Node::from_value(v));
        }
    };
    put("_type", e.edge_type.map(Value::Str));
    put("address", Some(Value::Str(e.address)));
    put("config", e.config);
    put("inputs", e.inputs);
    put("outputs", e.outputs);
    put("interval", e.interval);
    put("_contract", e.contract);
    for (k, v) in e.extra {
        m.insert(k, Node::from_value(v));
    }
    m
}

/// Serialize compactly, emitting bare `Infinity`/`-Infinity`/`NaN`.
pub fn to_json(spec: &CompositeSpec) -> String {
    let mut out = String::new();
    write_store(&mut out, &spec.root);
    out
}

// ---- writer ---------------------------------------------------------------

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
    } else if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        // `{:?}` is shortest round-trip and always keeps a `.0` / exponent,
        // so floats stay floats on re-read.
        let _ = write!(out, "{f:?}");
    }
}

fn write_obj<'a, T: 'a>(
    out: &mut String,
    it: impl Iterator<Item = (&'a String, &'a T)>,
    mut f: impl FnMut(&mut String, &'a T),
) {
    out.push('{');
    for (n, (k, v)) in it.enumerate() {
        if n > 0 {
            out.push(',');
        }
        write_str(out, k);
        out.push(':');
        f(out, v);
    }
    out.push('}');
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::UInt(u) => {
            let _ = write!(out, "{u}");
        }
        Value::Float(f) => write_float(out, *f),
        Value::Str(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (n, x) in a.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Map(m) | Value::Tagged(_, m) => write_obj(out, m.iter(), write_value),
    }
}

fn write_node(out: &mut String, n: &Node) {
    match n {
        Node::Store(s) => write_store(out, s),
        Node::Leaf(v) => write_value(out, v),
        Node::Edge(e) => {
            out.push('{');
            let mut first = true;
            let mut sep = |out: &mut String, k: &str| {
                if !first {
                    out.push(',');
                }
                first = false;
                write_str(out, k);
                out.push(':');
            };
            if let Some(t) = &e.edge_type {
                sep(out, "_type");
                write_str(out, t);
            }
            sep(out, "address");
            write_str(out, &e.address);
            for (k, v) in [
                ("config", &e.config),
                ("inputs", &e.inputs),
                ("outputs", &e.outputs),
                ("interval", &e.interval),
                ("_contract", &e.contract),
            ] {
                if let Some(v) = v {
                    sep(out, k);
                    write_value(out, v);
                }
            }
            for (k, v) in &e.extra {
                sep(out, k);
                write_value(out, v);
            }
            out.push('}');
        }
    }
}

fn write_store(out: &mut String, m: &Map<Node>) {
    write_obj(out, m.iter(), write_node);
}

// ---- parser ---------------------------------------------------------------

const MAX_DEPTH: usize = 512;

struct Parser<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> Error {
        Error::Parse {
            pos: self.i,
            msg: msg.to_string(),
        }
    }

    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, Error> {
        match self.s.get(self.i) {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(_) if self.eat("true") => Ok(Value::Bool(true)),
            Some(_) if self.eat("false") => Ok(Value::Bool(false)),
            Some(_) if self.eat("null") => Ok(Value::Null),
            Some(_) if self.eat("NaN") => Ok(Value::Float(f64::NAN)),
            Some(_) if self.eat("Infinity") => Ok(Value::Float(f64::INFINITY)),
            Some(_) if self.eat("-Infinity") => Ok(Value::Float(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn enter(&mut self) -> Result<(), Error> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        Ok(())
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut m = Map::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
        } else {
            loop {
                self.ws();
                if self.s.get(self.i) != Some(&b'"') {
                    return Err(self.err("expected object key"));
                }
                let k = self.string()?;
                self.ws();
                if !self.eat(":") {
                    return Err(self.err("expected ':'"));
                }
                self.ws();
                let v = self.value()?;
                m.insert(k, v);
                self.ws();
                if self.eat(",") {
                    continue;
                }
                if self.eat("}") {
                    break;
                }
                return Err(self.err("expected ',' or '}'"));
            }
        }
        self.depth -= 1;
        Ok(match TagKind::detect(&m) {
            Some(kind) => Value::Tagged(kind, m),
            None => Value::Map(m),
        })
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
        } else {
            loop {
                self.ws();
                a.push(self.value()?);
                self.ws();
                if self.eat(",") {
                    continue;
                }
                if self.eat("]") {
                    break;
                }
                return Err(self.err("expected ',' or ']'"));
            }
        }
        self.depth -= 1;
        Ok(Value::Array(a))
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let h = self
            .src
            .get(self.i..self.i + 4)
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            // Copy a run of plain bytes (valid UTF-8 since src is &str and we
            // only stop on ASCII delimiters).
            let start = self.i;
            while let Some(&c) = self.s.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(&self.src[start..self.i]);
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else {
                        return Err(self.err("unterminated escape"));
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.eat("\\u") {
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    return Err(self.err("bad surrogate pair"));
                                }
                            }
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("bad code point"))?);
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                Some(_) => return Err(self.err("control character in string")),
            }
        }
    }

    fn number(&mut self) -> Result<Value, Error> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        let mut is_float = false;
        while let Some(&c) = self.s.get(self.i) {
            match c {
                b'0'..=b'9' => {}
                b'.' | b'e' | b'E' | b'+' | b'-' => is_float = true,
                _ => break,
            }
            self.i += 1;
        }
        let t = &self.src[start..self.i];
        if !is_float {
            if let Ok(i) = t.parse::<i64>() {
                return Ok(Value::Int(i));
            }
            if let Ok(u) = t.parse::<u64>() {
                return Ok(Value::UInt(u));
            }
        }
        t.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| Error::Parse {
                pos: start,
                msg: format!("bad number {t:?}"),
            })
    }
}
