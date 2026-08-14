pub mod backend;
pub mod bpe;
mod layers;
pub mod llama;
pub mod meta;
pub mod model;
pub mod quant;
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
pub use quant::{Packed, Planes, Quant, Spec};
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
mod tests;
