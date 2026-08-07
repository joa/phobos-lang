use std::collections::HashMap;

use anyhow::{Context, Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    // Note: The codes are part of the file format and MUST NOT be renumbered.
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_code(code: u32) -> Result<ValueType> {
        Ok(match code {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            other => bail!("unknown GGUF metadata value type {other}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

impl Value {
    pub fn as_int(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v.into(),
            Value::I8(v) => v.into(),
            Value::U16(v) => v.into(),
            Value::I16(v) => v.into(),
            Value::U32(v) => v.into(),
            Value::I32(v) => v.into(),
            Value::U64(v) => i64::try_from(v).ok()?, // we reject u64 that doesn't fit
            Value::I64(v) => v,
            Value::Bool(v) => v.into(),
            _ => return None,
        })
    }

    pub fn as_float(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_) => ValueType::Array,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
        }
    }

    pub fn preview(&self) -> String {
        const MAX_CHARS: usize = 96;
        const MAX_ITEMS: usize = 6;

        match self {
            Value::String(s) => {
                let mut out: String = s.chars().take(MAX_CHARS).collect();
                if out.chars().count() < s.chars().count() {
                    out.push_str("...");
                }
                format!("{out:?}")
            }
            Value::Array(a) => {
                let items: Vec<String> = a.previews().take(MAX_ITEMS).collect();
                let tail = if a.len() > items.len() { ", ..." } else { "" };
                format!("[{}{}] ({} items)", items.join(", "), tail, a.len())
            }
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Nested(Vec<Value>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::I8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::I16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::Bool(v) => v.len(),
            Array::String(v) => v.len(),
            Array::U64(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F64(v) => v.len(),
            Array::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_int_vec(&self) -> Option<Vec<i64>> {
        Some(match self {
            Array::U8(v) => v.iter().map(|&x| x.into()).collect(),
            Array::I8(v) => v.iter().map(|&x| x.into()).collect(),
            Array::U16(v) => v.iter().map(|&x| x.into()).collect(),
            Array::I16(v) => v.iter().map(|&x| x.into()).collect(),
            Array::U32(v) => v.iter().map(|&x| x.into()).collect(),
            Array::I32(v) => v.iter().map(|&x| x.into()).collect(),
            Array::U64(v) => v
                .iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<_>>()?,
            Array::I64(v) => v.clone(),
            Array::Bool(v) => v.iter().map(|&x| x.into()).collect(),
            _ => return None,
        })
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Array::String(v) => Some(v),
            _ => None,
        }
    }

    fn previews(&self) -> Box<dyn Iterator<Item = String> + '_> {
        match self {
            Array::String(v) => Box::new(v.iter().map(|s| format!("{s:?}"))),
            Array::F32(v) => Box::new(v.iter().map(f32::to_string)),
            Array::F64(v) => Box::new(v.iter().map(f64::to_string)),
            Array::Nested(v) => Box::new(v.iter().map(Value::preview)),
            _ => match self.to_int_vec() {
                Some(ints) => Box::new(ints.into_iter().map(|x| x.to_string())),
                None => Box::new(std::iter::empty()),
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Metadata {
    entries: Vec<(String, Value)>,
    index: HashMap<String, usize>,
}

impl Metadata {
    pub fn insert(&mut self, key: String, value: Value) {
        // A duplicate key replaces the earlier value in place.
        // Same as llama.cpp
        match self.index.get(&key) {
            Some(&at) => self.entries[at].1 = value,
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, value));
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&at| &self.entries[at].1)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Key/value pairs in the order they appear in the file.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn require(&self, key: &str) -> Result<&Value> {
        self.get(key)
            .with_context(|| format!("GGUF metadata has no key '{key}'"))
    }

    pub fn string(&self, key: &str) -> Result<&str> {
        self.require(key)?
            .as_str()
            .with_context(|| format!("metadata '{key}' is not a string"))
    }

    pub fn int(&self, key: &str) -> Result<i64> {
        self.require(key)?
            .as_int()
            .with_context(|| format!("metadata '{key}' is not an integer"))
    }

    /// A non-negative integer key as a `usize`.
    pub fn count(&self, key: &str) -> Result<usize> {
        let n = self.int(key)?;
        usize::try_from(n).with_context(|| format!("metadata '{key}' is negative ({n})"))
    }

    pub fn float(&self, key: &str) -> Result<f32> {
        let v = self.require(key)?;
        v.as_float()
            .map(|f| f as f32)
            .or_else(|| v.as_int().map(|i| i as f32))
            .with_context(|| format!("metadata '{key}' is not a number"))
    }

    pub fn boolean(&self, key: &str) -> Result<bool> {
        self.require(key)?
            .as_bool()
            .with_context(|| format!("metadata '{key}' is not a bool"))
    }

    pub fn strings(&self, key: &str) -> Result<&[String]> {
        self.require(key)?
            .as_array()
            .and_then(Array::as_strings)
            .with_context(|| format!("metadata '{key}' is not a string array"))
    }

    pub fn ints(&self, key: &str) -> Result<Vec<i64>> {
        self.require(key)?
            .as_array()
            .and_then(Array::to_int_vec)
            .with_context(|| format!("metadata '{key}' is not an integer array"))
    }

    pub fn architecture(&self) -> Result<&str> {
        self.string("general.architecture")
    }

    pub fn arch_key(&self, suffix: &str) -> Result<String> {
        // arch_key("block_count") yields "qwen35.block_count" for a Qwen3.5
        Ok(format!("{}.{suffix}", self.architecture()?))
    }

    pub fn arch_int(&self, suffix: &str) -> Result<i64> {
        self.int(&self.arch_key(suffix)?)
    }

    pub fn arch_count(&self, suffix: &str) -> Result<usize> {
        self.count(&self.arch_key(suffix)?)
    }

    pub fn arch_float(&self, suffix: &str) -> Result<f32> {
        self.float(&self.arch_key(suffix)?)
    }

    pub fn arch_get(&self, suffix: &str) -> Option<&Value> {
        self.get(&self.arch_key(suffix).ok()?)
    }
}
