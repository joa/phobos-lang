//! A mixture-of-experts block's expert weights: `[count, n, k]` tensors read
//! in place from the file mapping, not copied. A model's experts are most of
//! its bytes and do not all fit on a device at once.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::quant::grouped::{group_rows_into, grouped_len};
use crate::quant::{Q8_0_UNIT, Q8_0_UNIT_BYTES, Quant, q8_0_units};
use crate::{Gguf, Window};

/// One `[count, n, k]` quantized tensor: `count` experts' `[n, k]` matrices
/// back to back, each `n` rows of `k / block` blocks. Expert `e` is the byte
/// range `e * expert_bytes .. (e + 1) * expert_bytes`.
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
        let stack = ExpertStack { bytes, quant, count, n, k };
        ensure!(
            k.is_multiple_of(stack.unit().0),
            "'{name}': the kernels read {} experts {} elements at a time, which does not divide k = {k}",
            quant.name(),
            stack.unit().0
        );
        Ok(stack)
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

    /// The unit the device slabs and the host kernels read an expert in,
    /// as elements and bytes. A K-quant's unit is one block at its device
    /// stride. Q8_0's is eight file blocks re-laid ([`q8_0_units`]), since
    /// the kernels take 256 elements at a time with sixteen-aligned loads.
    pub fn unit(&self) -> (usize, usize) {
        match self.quant {
            Quant::Q8_0 => (Q8_0_UNIT, Q8_0_UNIT_BYTES),
            quant => (quant.spec().block, quant.device_block().1),
        }
    }

    /// Bytes of one unit, the stride of the device's grouped layout.
    pub fn unit_bytes(&self) -> usize {
        self.unit().1
    }

    /// Bytes one unit's elements occupy in the file: the blocks it is made
    /// of, as stored.
    pub fn file_unit_bytes(&self) -> usize {
        self.unit().0 / self.quant.spec().block * self.quant.spec().block_bytes
    }

    /// Units along one row.
    pub fn blocks_per_row(&self) -> usize {
        self.k / self.unit().0
    }

    /// The file's blocks along one row.
    fn file_blocks_per_row(&self) -> usize {
        self.k / self.quant.spec().block
    }

    /// Bytes one expert's matrix occupies, which is also the stride between
    /// experts.
    pub fn expert_bytes(&self) -> usize {
        self.n * self.file_blocks_per_row() * self.quant.spec().block_bytes
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

    /// Bytes one expert takes in the grouped layout the device kernels
    /// read: [`grouped_len`] of its rows at the format's unit.
    pub fn grouped_bytes(&self) -> usize {
        grouped_len(self.n, self.blocks_per_row(), self.unit_bytes())
    }

    /// Elements of one expert's grouped scale plane: one `f16` per unit,
    /// rows regrouped by eights like the blocks. Kernels read it only for a
    /// format whose device block lacks the scale (Q6_K). The others take it
    /// as an operand and ignore it; Q8_0 keeps its scales inside the unit.
    pub fn grouped_scales(&self) -> usize {
        grouped_len(self.n, self.blocks_per_row(), 1)
    }

    /// Expert `e`'s scale plane in the grouped layout, into a buffer of
    /// [`ExpertStack::grouped_scales`]. A format with no header scale gets
    /// zeros.
    pub fn grouped_scales_into(&self, e: usize, out: &mut [u16]) {
        let Some(split) = self.quant.spec().raw_scales else {
            out.fill(0);
            return;
        };
        let scales = split(self.expert(e), self.k, self.n);
        group_rows_into(&scales.d, self.n, self.blocks_per_row(), 1, out);
    }

    /// Expert `e` in the grouped layout, into a buffer of
    /// [`ExpertStack::grouped_bytes`]: each block trimmed to its device
    /// window, or Q8_0's re-laid into units, then rows regrouped by eights.
    /// Padding rows past `n` are not written.
    pub fn grouped_into(&self, e: usize, out: &mut [u8]) {
        let nb = self.blocks_per_row();
        let bytes = self.expert(e);
        if self.quant == Quant::Q8_0 {
            return group_rows_into(&q8_0_units(bytes), self.n, nb, Q8_0_UNIT_BYTES, out);
        }
        let (skip, dev) = self.quant.device_block();
        let block_bytes = self.quant.spec().block_bytes;
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

/// One of the three, in the order the set holds them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stack {
    Gate,
    Up,
    Down,
}

impl Stack {
    pub const ALL: [Stack; 3] = [Stack::Gate, Stack::Up, Stack::Down];
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

    pub fn stack(&self, stack: Stack) -> &ExpertStack {
        match stack {
            Stack::Gate => &self.gate,
            Stack::Up => &self.up,
            Stack::Down => &self.down,
        }
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
