use std::collections::HashMap;

use anyhow::{Result, bail, ensure};

use crate::meta::{Array, Metadata, Value, ValueType};
use crate::tensor::{GgmlType, TensorInfo};

const MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];
const DEFAULT_ALIGNMENT: u64 = 32;

pub struct Container {
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    pub index: HashMap<String, usize>,
    pub data_offset_bytes: usize,
    pub alignment_bytes: u64,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context_overflow()?;
        ensure!(
            end <= self.bytes.len(),
            "GGUF header truncated: wanted {n} bytes at offset {}, file has {}",
            self.pos,
            self.bytes.len()
        );
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?).context_overflow()?;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec())
            .map_err(|e| anyhow::anyhow!("GGUF string at offset {} is not UTF-8: {e}", self.pos))
    }

    fn checked_len(&mut self) -> Result<usize> {
        let len = usize::try_from(self.u64()?).context_overflow()?;
        ensure!(
            len <= self.bytes.len() - self.pos,
            "GGUF declares {len} elements but only {} bytes remain",
            self.bytes.len() - self.pos
        );
        Ok(len)
    }

    fn value(&mut self, value_type: ValueType) -> Result<Value> {
        Ok(match value_type {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.u8()? as i8),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.u16()? as i16),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.u32()? as i32),
            ValueType::F32 => Value::F32(f32::from_bits(self.u32()?)),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.u64()? as i64),
            ValueType::F64 => Value::F64(f64::from_bits(self.u64()?)),
            ValueType::Bool => Value::Bool(self.u8()? != 0),
            ValueType::String => Value::String(self.string()?),
            ValueType::Array => Value::Array(self.array()?),
        })
    }

    fn array(&mut self) -> Result<Array> {
        let elem_type = ValueType::from_code(self.u32()?)?;
        let len = self.checked_len()?;

        Ok(match elem_type {
            ValueType::U8 => Array::U8(self.collect(len, Reader::u8)?),
            ValueType::I8 => Array::I8(self.collect(len, |r| Ok(r.u8()? as i8))?),
            ValueType::U16 => Array::U16(self.collect(len, Reader::u16)?),
            ValueType::I16 => Array::I16(self.collect(len, |r| Ok(r.u16()? as i16))?),
            ValueType::U32 => Array::U32(self.collect(len, Reader::u32)?),
            ValueType::I32 => Array::I32(self.collect(len, |r| Ok(r.u32()? as i32))?),
            ValueType::F32 => Array::F32(self.collect(len, |r| Ok(f32::from_bits(r.u32()?)))?),
            ValueType::U64 => Array::U64(self.collect(len, Reader::u64)?),
            ValueType::I64 => Array::I64(self.collect(len, |r| Ok(r.u64()? as i64))?),
            ValueType::F64 => Array::F64(self.collect(len, |r| Ok(f64::from_bits(r.u64()?)))?),
            ValueType::Bool => Array::Bool(self.collect(len, |r| Ok(r.u8()? != 0))?),
            ValueType::String => Array::String(self.collect(len, Reader::string)?),
            ValueType::Array => Array::Nested(self.collect(len, |r| Ok(Value::Array(r.array()?)))?),
        })
    }

    fn collect<T>(
        &mut self,
        len: usize,
        mut read: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(read(self)?);
        }
        Ok(out)
    }
}

trait ContextOverflow<T> {
    fn context_overflow(self) -> Result<T>;
}

impl<T, E> ContextOverflow<T> for std::result::Result<T, E> {
    fn context_overflow(self) -> Result<T> {
        self.map_err(|_| anyhow::anyhow!("GGUF header declares a size too large for this platform"))
    }
}

impl<T> ContextOverflow<T> for Option<T> {
    fn context_overflow(self) -> Result<T> {
        self.ok_or_else(|| anyhow::anyhow!("GGUF header size arithmetic overflowed"))
    }
}

pub fn parse(bytes: &[u8]) -> Result<Container> {
    let mut reader = Reader::new(bytes);

    let magic = reader.take(4)?;
    ensure!(magic == MAGIC, "not a GGUF file (magic {magic:02x?})");

    let version = reader.u32()?;
    ensure!(
        SUPPORTED_VERSIONS.contains(&version),
        "unsupported GGUF version {version}; this reader handles {SUPPORTED_VERSIONS:?}"
    );

    let tensor_count = reader.checked_len()?;
    let kv_count = reader.checked_len()?;

    let mut metadata = Metadata::default();
    for _ in 0..kv_count {
        let key = reader.string()?;
        let value_type = ValueType::from_code(reader.u32()?)?;
        let value = reader.value(value_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut index = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let n_dims = reader.u32()? as usize;
        ensure!(
            n_dims <= 4,
            "tensor '{name}' declares {n_dims} dimensions; ggml allows at most 4"
        );
        let dims = reader.collect(n_dims, Reader::u64)?;
        let ggml_type = GgmlType::from_code(reader.u32()?)?;
        let offset_bytes = reader.u64()?;

        if let Some(prior) = index.insert(name.clone(), tensors.len()) {
            bail!(
                "tensor '{name}' appears twice in the directory (at {prior} and {})",
                tensors.len()
            );
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset_bytes,
        });
    }

    let alignment_bytes = match metadata.get("general.alignment") {
        Some(v) => {
            let n = v.as_int().context_overflow()?;
            let n = u64::try_from(n).context_overflow()?;
            ensure!(
                n.is_power_of_two(),
                "general.alignment {n} is not a power of two"
            );
            n
        }
        None => DEFAULT_ALIGNMENT,
    };

    let header_end = reader.pos as u64;
    let data_offset_bytes = header_end.next_multiple_of(alignment_bytes);
    let data_offset_bytes = usize::try_from(data_offset_bytes).context_overflow()?;

    Ok(Container {
        version,
        metadata,
        tensors,
        index,
        data_offset_bytes,
        alignment_bytes,
    })
}
