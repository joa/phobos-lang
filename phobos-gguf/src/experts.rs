//! A mixture-of-experts block's expert weights: `[count, n, k]` tensors read
//! out of the file's mapping rather than copied out of it, since a model's
//! experts are most of its bytes and do not all fit on a device at once.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::quant::Quant;
use crate::quant::grouped::{group_rows_into, grouped_len};
use crate::{Gguf, Window};

/// One `[count, n, k]` quantized tensor: `count` experts' `[n, k]` matrices
/// back to back in storage order, each `n` rows of `k / block` blocks, so
/// expert `e` is the byte range `e * expert_bytes .. (e + 1) * expert_bytes`.
pub struct ExpertStack {
    bytes: Window,
    quant: Quant,
    count: usize,
    n: usize,
    k: usize,
}

impl ExpertStack {
    pub fn load(gguf: &Gguf, name: &str, count: usize, n: usize, k: usize) -> Result<ExpertStack> {
        let info = gguf
            .tensor(name)
            .with_context(|| format!("missing tensor '{name}'"))?;
        ensure!(
            info.dims == [k as u64, n as u64, count as u64],
            "tensor '{name}' has ggml extents {:?}, expected [{k}, {n}, {count}]",
            info.dims
        );
        let quant = info.ggml_type.quant().with_context(|| {
            format!(
                "'{name}' is {}, which is not a quantized format experts are held in",
                info.ggml_type.name()
            )
        })?;
        ensure!(
            k.is_multiple_of(quant.spec().block),
            "'{name}': a {} block of {} does not divide k = {k}",
            quant.name(),
            quant.spec().block
        );
        let bytes = gguf.tensor_window(info)?;
        Ok(ExpertStack { bytes, quant, count, n, k })
    }

    pub fn quant(&self) -> Quant {
        self.quant
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Blocks along one row.
    pub fn blocks_per_row(&self) -> usize {
        self.k / self.quant.spec().block
    }

    /// Bytes one expert's matrix occupies, which is also the stride between
    /// experts.
    pub fn expert_bytes(&self) -> usize {
        self.n * self.blocks_per_row() * self.quant.spec().block_bytes
    }

    /// Bytes of the whole stack.
    pub fn byte_len(&self) -> usize {
        self.count * self.expert_bytes()
    }

    /// Expert `e`'s blocks, `[n, k]` in storage order.
    pub fn expert(&self, e: usize) -> &[u8] {
        assert!(e < self.count, "expert {e} of {}", self.count);
        let stride = self.expert_bytes();
        &self.bytes[e * stride..(e + 1) * stride]
    }

    /// Expert `e` decoded into `out`, `[n, k]` row-major: `out[j * k + i]` is
    /// output `j`'s weight on input `i`.
    pub fn dequantize(&self, e: usize, out: &mut [f32]) -> Result<()> {
        ensure!(
            out.len() == self.n * self.k,
            "a [{}, {}] expert does not fit a {}-element slice",
            self.n,
            self.k,
            out.len()
        );
        (self.quant.spec().dequantize)(self.expert(e), out);
        Ok(())
    }

    /// Bytes expert `e` takes in the grouped layout the device kernels read:
    /// [`grouped_len`] of its rows at the format's device block.
    pub fn grouped_bytes(&self) -> usize {
        grouped_len(self.n, self.blocks_per_row(), self.quant.device_block().1)
    }

    /// Expert `e` in the grouped layout, into a buffer of
    /// [`ExpertStack::grouped_bytes`]: each block trimmed to its device
    /// window, then rows regrouped by eights. Padding rows past `n` are left
    /// as they are.
    pub fn grouped_into(&self, e: usize, out: &mut [u8]) {
        let (skip, dev) = self.quant.device_block();
        let block_bytes = self.quant.spec().block_bytes;
        let nb = self.blocks_per_row();
        let bytes = self.expert(e);
        if skip == 0 && dev == block_bytes {
            return group_rows_into(bytes, self.n, nb, dev, out);
        }
        let trimmed: Vec<u8> = bytes
            .chunks_exact(block_bytes)
            .flat_map(|block| &block[skip..skip + dev])
            .copied()
            .collect();
        group_rows_into(&trimmed, self.n, nb, dev, out);
    }
}

/// The three stacks of one block's routed feed-forward: `gate` and `up` map
/// `d_model` to `d_ff`, `down` maps back.
pub struct ExpertSet {
    pub gate: ExpertStack,
    pub up: ExpertStack,
    pub down: ExpertStack,
}

impl ExpertSet {
    pub fn load(gguf: &Gguf, prefix: &str, count: usize, d_model: usize, d_ff: usize) -> Result<Arc<ExpertSet>> {
        let stack = |name: &str, n, k| ExpertStack::load(gguf, &format!("{prefix}.ffn_{name}_exps.weight"), count, n, k);
        Ok(Arc::new(ExpertSet {
            gate: stack("gate", d_ff, d_model)?,
            up: stack("up", d_ff, d_model)?,
            down: stack("down", d_model, d_ff)?,
        }))
    }

    pub fn count(&self) -> usize {
        self.gate.count()
    }

    /// Bytes of all three stacks, as the file holds them.
    pub fn byte_len(&self) -> usize {
        self.gate.byte_len() + self.up.byte_len() + self.down.byte_len()
    }

    /// Bytes one expert's three matrices take, as the file holds them.
    pub fn expert_bytes(&self) -> usize {
        self.gate.expert_bytes() + self.up.expert_bytes() + self.down.expert_bytes()
    }
}

#[cfg(test)]
pub(crate) mod tests;
