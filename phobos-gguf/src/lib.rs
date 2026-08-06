pub mod backend;
pub mod bpe;
mod layers;
pub mod llama;
pub mod meta;
pub mod model;
pub mod qwen35;
pub mod read;
pub mod runtime;
pub mod tensor;
pub mod vocab;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;

pub use backend::{Backend, Buf, HostBackend};
pub use bpe::Bpe;
pub use meta::{Array, Metadata, Value, ValueType};
pub use model::Decoder;
pub use runtime::GgufModel;
pub use tensor::{GgmlType, TensorInfo, dequantize_into, f16_to_f32};
pub use vocab::{TokenType, Vocab};

/// Where a [`Gguf`]'s bytes live.
enum Backing {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

pub struct Gguf {
    backing: Backing,
    version: u32,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    data_offset_bytes: usize,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("version", &self.version)
            .field("metadata_keys", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("data_offset_bytes", &self.data_offset_bytes)
            .finish()
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // safe as long as the file is not modified while mapped
        let map =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        Gguf::new(Backing::Mapped(map)).with_context(|| format!("parse {}", path.display()))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Gguf> {
        Gguf::new(Backing::Owned(bytes))
    }

    fn new(backing: Backing) -> Result<Gguf> {
        let container = read::parse(&backing)?;
        ensure!(
            container.data_offset_bytes <= backing.len(),
            "GGUF header ends at {} but the file is only {} bytes",
            container.data_offset_bytes,
            backing.len()
        );
        Ok(Gguf {
            backing,
            version: container.version,
            metadata: container.metadata,
            tensors: container.tensors,
            index: container.index,
            data_offset_bytes: container.data_offset_bytes,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The model architecture, which selects the forward pass.
    pub fn architecture(&self) -> Result<&str> {
        self.metadata.architecture()
    }

    /// Tensor descriptors in directory order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&at| &self.tensors[at])
    }

    fn require_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name)
            .with_context(|| format!("GGUF file has no tensor '{name}'"))
    }

    /// The raw, still-quantized bytes backing a tensor.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let len = info.storage_bytes()?;
        let start = self
            .data_offset_bytes
            .checked_add(usize::try_from(info.offset_bytes).unwrap_or(usize::MAX))
            .context("tensor offset overflows the address space")?;
        let end = start
            .checked_add(len)
            .context("tensor extent overflows the address space")?;
        ensure!(
            end <= self.backing.len(),
            "tensor '{}' spans bytes {start}..{end} but the file is {} bytes",
            info.name,
            self.backing.len()
        );
        Ok(&self.backing[start..end])
    }

    /// Dequantize a tensor into a fresh f32 buffer, in ggml element order (the
    /// fastest-varying axis of [`TensorInfo::dims`] first).
    pub fn dequantize(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.require_tensor(name)?;
        let numel =
            usize::try_from(info.numel()).context("tensor is too large for this platform")?;
        let mut out = vec![0.0; numel];
        self.dequantize_into(info, &mut out)?;
        Ok(out)
    }

    /// Dequantize into storage holding exactly the tensor's element count.
    pub fn dequantize_into(&self, info: &TensorInfo, out: &mut [f32]) -> Result<()> {
        let numel =
            usize::try_from(info.numel()).context("tensor is too large for this platform")?;
        ensure!(
            out.len() == numel,
            "tensor '{}' has {numel} elements, destination holds {}",
            info.name,
            out.len()
        );
        let bytes = self.tensor_bytes(info)?;
        dequantize_into(info.ggml_type, bytes, out)
            .with_context(|| format!("dequantize tensor '{}'", info.name))
    }

    pub fn vocab(&self) -> Result<Vocab> {
        Vocab::from_metadata(&self.metadata)
    }

    pub fn parameter_count(&self) -> u64 {
        self.tensors.iter().map(TensorInfo::numel).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF writer, enough to round-trip the reader.q
    #[derive(Default)]
    struct Builder {
        kv: Vec<u8>,
        kv_count: u64,
        tensors: Vec<u8>,
        tensor_count: u64,
        data: Vec<u8>,
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend((s.len() as u64).to_le_bytes());
        buf.extend(s.as_bytes());
    }

    impl Builder {
        fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
            push_str(&mut self.kv, key);
            self.kv.extend(8u32.to_le_bytes());
            push_str(&mut self.kv, value);
            self.kv_count += 1;
            self
        }

        fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
            push_str(&mut self.kv, key);
            self.kv.extend(4u32.to_le_bytes());
            self.kv.extend(value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
            push_str(&mut self.kv, key);
            self.kv.extend(9u32.to_le_bytes());
            self.kv.extend(8u32.to_le_bytes());
            self.kv.extend((values.len() as u64).to_le_bytes());
            for v in values {
                push_str(&mut self.kv, v);
            }
            self.kv_count += 1;
            self
        }

        fn tensor_f32(&mut self, name: &str, dims: &[u64], values: &[f32]) -> &mut Self {
            push_str(&mut self.tensors, name);
            self.tensors.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                self.tensors.extend(d.to_le_bytes());
            }
            self.tensors.extend(0u32.to_le_bytes()); // F32
            self.tensors.extend((self.data.len() as u64).to_le_bytes());
            self.tensor_count += 1;
            for v in values {
                self.data.extend(v.to_le_bytes());
            }
            self
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(b"GGUF");
            out.extend(3u32.to_le_bytes());
            out.extend(self.tensor_count.to_le_bytes());
            out.extend(self.kv_count.to_le_bytes());
            out.extend(&self.kv);
            out.extend(&self.tensors);
            out.resize(out.len().next_multiple_of(32), 0);
            out.extend(&self.data);
            out
        }
    }

    #[test]
    fn round_trips_metadata_and_tensors() {
        let bytes = Builder::default()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.block_count", 12)
            .kv_string_array("tokenizer.ggml.tokens", &["a", "b"])
            .kv_string("tokenizer.ggml.model", "gpt2")
            .tensor_f32(
                "token_embd.weight",
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .tensor_f32("output_norm.weight", &[2], &[0.25, 0.5])
            .build();

        let gguf = Gguf::from_bytes(bytes).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.architecture().unwrap(), "llama");
        assert_eq!(gguf.metadata().arch_count("block_count").unwrap(), 12);
        assert_eq!(gguf.tensors().len(), 2);

        let embd = gguf.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.dims, vec![2, 3]);
        // ggml stores the fastest axis first, so this is 3 rows of 2.
        assert_eq!(embd.row_major_dims(), vec![3, 2]);
        assert_eq!(embd.numel(), 6);
        assert_eq!(
            gguf.dequantize("token_embd.weight").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            gguf.dequantize("output_norm.weight").unwrap(),
            vec![0.25, 0.5]
        );
        assert_eq!(gguf.parameter_count(), 8);

        let vocab = gguf.vocab().unwrap();
        assert_eq!(vocab.tokens, vec!["a", "b"]);
    }

    #[test]
    fn rejects_a_non_gguf_file() {
        let err = Gguf::from_bytes(b"ONNX and some padding bytes".to_vec()).unwrap_err();
        assert!(err.to_string().contains("not a GGUF file"), "{err}");
    }

    #[test]
    fn rejects_a_truncated_header() {
        let mut bytes = Builder::default()
            .kv_string("general.architecture", "llama")
            .build();
        bytes.truncate(20);
        assert!(Gguf::from_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_a_tensor_running_past_the_file() {
        let mut bytes = Builder::default()
            .kv_string("general.architecture", "llama")
            .tensor_f32("w", &[4], &[1.0, 2.0, 3.0, 4.0])
            .build();
        // Drop half the payload; the directory still claims 16 bytes.
        bytes.truncate(bytes.len() - 8);

        let gguf = Gguf::from_bytes(bytes).unwrap();
        let err = gguf.dequantize("w").unwrap_err();
        assert!(format!("{err:#}").contains("but the file is"), "{err:#}");
    }

    #[test]
    fn reports_unknown_tensors() {
        let bytes = Builder::default()
            .kv_string("general.architecture", "llama")
            .build();
        let gguf = Gguf::from_bytes(bytes).unwrap();
        assert!(gguf.tensor("missing").is_none());
        assert!(
            gguf.dequantize("missing")
                .unwrap_err()
                .to_string()
                .contains("no tensor")
        );
    }
}
