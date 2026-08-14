use anyhow::{Context, Result, ensure};

use crate::GgmlType;

mod q4_0;
mod q4_1;
mod q4_k;
mod q5_0;
mod q5_1;
mod q6_k;
mod q8_0;
mod q8_1;

pub use q8_0::{BLOCK as Q8_0_BLOCK, pack as pack_q8_0, quantize_row};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q4_K,
    Q6_K,
}

impl Quant {
    pub const ALL: [Quant; 8] = [
        Quant::Q4_0,
        Quant::Q4_1,
        Quant::Q5_0,
        Quant::Q5_1,
        Quant::Q8_0,
        Quant::Q8_1,
        Quant::Q4_K,
        Quant::Q6_K,
    ];

    pub fn spec(self) -> &'static Spec {
        match self {
            Quant::Q4_0 => &q4_0::SPEC,
            Quant::Q4_1 => &q4_1::SPEC,
            Quant::Q5_0 => &q5_0::SPEC,
            Quant::Q5_1 => &q5_1::SPEC,
            Quant::Q8_0 => &q8_0::SPEC,
            Quant::Q8_1 => &q8_1::SPEC,
            Quant::Q4_K => &q4_k::SPEC,
            Quant::Q6_K => &q6_k::SPEC,
        }
    }

    pub fn name(self) -> &'static str {
        self.spec().name
    }
}

/// What one block of a format holds, and the two things a caller can do with
/// it. The registry is [`Quant::spec`].
pub struct Spec {
    pub name: &'static str,
    /// Elements one stored block covers.
    pub block: usize,
    /// Bytes that block occupies.
    pub block_bytes: usize,
    /// Elements sharing one scale. Equal to `block` for the legacy formats; a
    /// K-quant block is a super-block carrying several runs.
    pub scale_run: usize,
    /// Quants are unsigned with a per-block minimum rather than symmetric
    /// around zero, so decoding them needs the minimum as well as the scale.
    pub has_min: bool,
    /// Decodes whole blocks in storage order. `out` is `bytes.len() /
    /// block_bytes * block` elements; both are checked before this runs.
    pub dequantize: fn(bytes: &[u8], out: &mut [f32]),
    /// Splits a `[n, k]` weight into the planes the quantized kernels index,
    /// or `None` for a format no kernel unpacks yet: those dequantize at
    /// upload and take the dense contraction.
    ///
    /// The two planes are transposed relative to each other, each for its own
    /// access pattern, and a kernel indexes against exactly this:
    ///
    /// - `qs` is `[n, k]`, so `qs[j * k + p]` is the quant for output `j` and
    ///   input `p`, putting the contraction axis contiguous.
    /// - `scales` is `[k / scale_run, n]`, so `scales[(p / scale_run) * n + j]`
    ///   scales that quant, putting one run's scales for a stretch of outputs
    ///   contiguous.
    pub planes: Option<Split>,
}

/// What [`Spec::planes`] holds: a `[n, k]` weight's blocks into its planes.
pub type Split = fn(bytes: &[u8], k: usize, n: usize) -> Planes;

/// A quantized weight split into the planes [`Spec::planes`] describes.
#[derive(Debug)]
pub struct Planes {
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Planes {
    /// One quant per element, one scale per run of them. The runs go down `k`,
    /// so `k` has to tile evenly.
    pub fn check(&self, quant: Quant, k: usize, n: usize) -> Result<()> {
        let run = quant.spec().scale_run;
        ensure!(
            k.is_multiple_of(run),
            "a {} weight needs k ({k}) to be a multiple of {run}",
            quant.name()
        );
        ensure!(
            self.qs.len() == k * n,
            "a [{k}, {n}] {} weight needs {} quants, got {}",
            quant.name(),
            k * n,
            self.qs.len()
        );
        ensure!(
            self.scales.len() == (k / run) * n,
            "a [{k}, {n}] {} weight needs {} scales, got {}",
            quant.name(),
            (k / run) * n,
            self.scales.len()
        );
        Ok(())
    }
}

/// A `[n, k]` weight held in its file format between load and upload.
///
/// The blocks are kept exactly as GGUF stores them, `[n, k / block]`, so one
/// output's inputs are contiguous. Nothing here requantizes: stacking,
/// permuting and reading a row all move whole blocks, which is what makes them
/// format-independent.
pub struct Packed {
    quant: Quant,
    blocks: Vec<u8>,
    k: usize,
    n: usize,
}

impl Packed {
    pub fn new(quant: Quant, blocks: Vec<u8>, k: usize, n: usize) -> Result<Packed> {
        let want = byte_len(quant, k, n)?;
        ensure!(
            blocks.len() == want,
            "a [{k}, {n}] {} weight needs {want} bytes, got {}",
            quant.name(),
            blocks.len()
        );
        Ok(Packed {
            quant,
            blocks,
            k,
            n,
        })
    }

    /// [`Packed::new`] taking what it needs off the front of a tensor's bytes,
    /// which a GGUF file hands over as a window of the whole mapping.
    pub fn from_bytes(quant: Quant, bytes: &[u8], k: usize, n: usize) -> Result<Packed> {
        let want = byte_len(quant, k, n)?;
        ensure!(
            bytes.len() >= want,
            "a [{k}, {n}] {} weight needs {want} bytes, got {}",
            quant.name(),
            bytes.len()
        );
        Packed::new(quant, bytes[..want].to_vec(), k, n)
    }

    pub fn quant(&self) -> Quant {
        self.quant
    }

    pub fn spec(&self) -> &'static Spec {
        self.quant.spec()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn byte_len(&self) -> usize {
        self.blocks.len()
    }

    /// Bytes one output row occupies.
    fn row_bytes(&self) -> usize {
        let spec = self.spec();
        self.k / spec.block * spec.block_bytes
    }

    fn row(&self, j: usize) -> &[u8] {
        let stride = self.row_bytes();
        &self.blocks[j * stride..][..stride]
    }

    /// Whether a quantized kernel can read this format, which decides between
    /// the quantized contraction and dequantizing at upload.
    pub fn has_planes(&self) -> bool {
        self.spec().planes.is_some()
    }

    /// The planes [`Spec::planes`] describes, for a backend uploading this.
    pub fn planes(&self) -> Result<Planes> {
        let split = self
            .spec()
            .planes
            .with_context(|| format!("no kernel unpacks {}", self.quant.name()))?;
        let planes = split(&self.blocks, self.k, self.n);
        planes.check(self.quant, self.k, self.n)?;
        Ok(planes)
    }

    /// Output row `j` decoded into `out`, which is `k` elements.
    pub fn row_into(&self, j: usize, out: &mut [f32]) -> Result<()> {
        ensure!(
            j < self.n && out.len() == self.k,
            "row {j} of a [{}, {}] weight does not fit a {}-element slice",
            self.n,
            self.k,
            out.len()
        );
        (self.spec().dequantize)(self.row(j), out);
        Ok(())
    }

    /// The whole weight decoded into the `[k, n]` layout [`crate::backend::Backend::matmul`]
    /// reads, which is the transpose of how it is stored.
    pub fn dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.k * self.n];
        let mut row = vec![0.0f32; self.k];
        for j in 0..self.n {
            (self.spec().dequantize)(self.row(j), &mut row);
            for (i, &v) in row.iter().enumerate() {
                out[i * self.n + j] = v;
            }
        }
        out
    }

    /// The same weight with its outputs permuted: output `j` of the result is
    /// output `order[j]` of this one.
    pub fn select_outputs(&self, order: &[usize]) -> Result<Packed> {
        ensure!(
            order.iter().all(|&j| j < self.n),
            "a reordering of a [{}, {}] weight names an output it does not have",
            self.k,
            self.n
        );
        let mut blocks = Vec::with_capacity(order.len() * self.row_bytes());
        for &from in order {
            blocks.extend_from_slice(self.row(from));
        }
        Packed::new(self.quant, blocks, self.k, order.len())
    }

    /// Weights sharing an input stacked along the output axis, padded out to
    /// `n` outputs.
    ///
    /// The padding is zero blocks, which decode to zero in every format here
    /// because a zero scale zeroes the whole block. It exists so a fused width
    /// tiles evenly; nothing reads it.
    pub fn stack(parts: &[&Packed], n: usize) -> Result<Packed> {
        let (first, rest) = parts.split_first().context("stacking needs a weight")?;
        let (quant, k) = (first.quant, first.k);
        ensure!(
            rest.iter().all(|p| p.quant == quant && p.k == k),
            "stacked weights must share their format and input width"
        );
        let stacked: usize = parts.iter().map(|p| p.n).sum();
        ensure!(
            stacked <= n,
            "{stacked} outputs do not fit a {n}-output weight"
        );
        let mut blocks = Vec::with_capacity(n * first.row_bytes());
        for part in parts {
            blocks.extend_from_slice(&part.blocks);
        }
        blocks.resize(n * first.row_bytes(), 0);
        Packed::new(quant, blocks, k, n)
    }
}

/// Bytes a `[n, k]` weight occupies in its format, which needs `k` to be a
/// whole number of blocks: a block never straddles two outputs.
fn byte_len(quant: Quant, k: usize, n: usize) -> Result<usize> {
    let spec = quant.spec();
    ensure!(
        k.is_multiple_of(spec.block),
        "a {} weight needs k ({k}) to be a multiple of {}",
        quant.name(),
        spec.block
    );
    Ok(k / spec.block * spec.block_bytes * n)
}

impl GgmlType {
    /// The compute format for this type code, for the types this crate
    /// contracts against. Everything else, the scalar types included, is
    /// `None` and goes down a dense path.
    pub fn quant(self) -> Option<Quant> {
        Some(match self {
            GgmlType::Q4_0 => Quant::Q4_0,
            GgmlType::Q4_1 => Quant::Q4_1,
            GgmlType::Q5_0 => Quant::Q5_0,
            GgmlType::Q5_1 => Quant::Q5_1,
            GgmlType::Q8_0 => Quant::Q8_0,
            GgmlType::Q8_1 => Quant::Q8_1,
            GgmlType::Q4_K => Quant::Q4_K,
            GgmlType::Q6_K => Quant::Q6_K,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests;
