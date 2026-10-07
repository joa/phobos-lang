use anyhow::{Context, Result, ensure};

use crate::GgmlType;

mod iq1_m;
mod iq1_s;
mod iq2_s;
mod iq2_xs;
mod iq2_xxs;
mod iq3_s;
mod iq3_xxs;
mod iq4_xs;
pub(crate) mod ptq1_0;
mod q2_k;
mod q3_k;
mod q4_0;
mod q4_1;
pub(crate) mod q4_k;
mod q5_k;
mod q5_0;
mod q5_1;
mod q6_k;
mod q8_0;
mod q8_1;
mod tables;

// Used only by the cuda backend; `iq1s_signed_grid` also has a host test.
#[cfg(any(test, feature = "cuda"))]
pub(crate) use iq1_s::signed_grid as iq1s_signed_grid;
#[cfg(feature = "cuda")]
pub(crate) use iq1_s::{
    flat_grid as iq1s_flat_grid, grid2 as iq1s_grid2, grid4 as iq1s_grid4,
    packed_grid as iq1s_packed_grid,
};
#[cfg(feature = "cuda")]
pub(crate) use iq2_s::{
    flat_grid as iq2s_flat_grid, flat_signs as iq2s_flat_signs, packed_grid as iq2s_packed_grid,
    packed_sign_masks as iq2s_sign_masks, packed_signs as iq2s_packed_signs,
};
#[cfg(feature = "cuda")]
pub(crate) use iq2_xs::{flat_grid as iq2xs_flat_grid, packed_grid as iq2xs_packed_grid};
#[cfg(feature = "cuda")]
pub(crate) use iq2_xxs::{
    flat_grid as iq2xxs_flat_grid, flat_signs as iq2xxs_flat_signs,
    packed_grid as iq2xxs_packed_grid, packed_sign_masks as iq2xxs_sign_masks,
    packed_signs as iq2xxs_packed_signs,
};
#[cfg(feature = "cuda")]
pub(crate) use iq3_s::{flat_grid as iq3s_flat_grid, packed_grid as iq3s_packed_grid};
#[cfg(feature = "cuda")]
pub(crate) use iq3_xxs::{flat_grid as iq3xxs_flat_grid, packed_grid as iq3xxs_packed_grid};
#[cfg(feature = "cuda")]
pub(crate) use iq4_xs::flat_codebook as iq4xs_flat_codebook;
pub use q8_0::{BLOCK as Q8_0_BLOCK, pack as pack_q8_0, quantize_row};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quant {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q4_K,
    Q5_K,
    Q6_K,
    Q2_K,
    Q3_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_XS,
    PTQ1_0,
}

impl Quant {
    pub const ALL: [Quant; 20] = [
        Quant::Q4_0,
        Quant::Q4_1,
        Quant::Q5_0,
        Quant::Q5_1,
        Quant::Q8_0,
        Quant::Q8_1,
        Quant::Q4_K,
        Quant::Q5_K,
        Quant::Q6_K,
        Quant::Q2_K,
        Quant::Q3_K,
        Quant::IQ1_S,
        Quant::IQ1_M,
        Quant::IQ2_XXS,
        Quant::IQ2_XS,
        Quant::IQ2_S,
        Quant::IQ3_XXS,
        Quant::IQ3_S,
        Quant::IQ4_XS,
        Quant::PTQ1_0,
    ];

    /// Bytes a block occupies once uploaded. Differs from
    /// `spec().block_bytes` where [`Quant::device_block`] pads or trims.
    ///
    /// Q3_K pads 110 to 112 because `q3k_qdot_t` reads its `qs` and `hmask`
    /// planes eight bytes at a time. Disk and host keep 110; only
    /// `constant_raw` and the device kernels see the padding.
    pub fn device_block_bytes(self) -> usize {
        self.device_block().1
    }

    /// How a block reaches the device: the leading bytes dropped, and the
    /// stride it occupies.
    ///
    /// Q3_K pads to 112 so its planes are eight-byte aligned. The listed IQ
    /// formats drop their leading f16 `d`, because kernels read the scale
    /// from the separate plane `constant_raw` uploads. Q6_K drops its
    /// trailing `d` the same way, so 210 bytes become 208, sixteen-aligned.
    ///
    /// Q5_0 drops its `d` as well, leaving the `qh` word and the nibbles,
    /// 20 four-aligned bytes, which `q50_qdot_t` and `q50_qmma_t` read beside
    /// the scale planes of [`Spec::planes`].
    ///
    /// Q4_K and Q5_K keep their headers. 144 and 176 are already
    /// sixteen-aligned, and the kernels read `d` and `dmin` in the same load
    /// as the scales. PTQ1_0 keeps its size, but [`Packed::device_blocks`]
    /// reorders each block (see `ptq1_0.rs`).
    pub fn device_block(self) -> (usize, usize) {
        let bytes = self.spec().block_bytes;
        match self {
            Quant::Q3_K => (0, 112),
            Quant::Q6_K => (0, 208),
            Quant::Q5_0 => (2, 20),
            Quant::IQ1_S | Quant::IQ2_XXS | Quant::IQ2_S => (2, bytes - 2),
            Quant::IQ2_XS | Quant::IQ3_XXS | Quant::IQ3_S => (2, bytes - 2),
            _ => (0, bytes),
        }
    }

    /// Whether the device holds this format's payload and scale planes
    /// grouped by eight columns, as `[n / 8][nb][8][block]` and
    /// `[n / 8][nb][8]`, zero-padded. The compiler's `RAW_GROUP` readers
    /// expect this layout.
    pub fn grouped_rows(self) -> bool {
        matches!(
            self,
            Quant::IQ1_S
                | Quant::IQ1_M
                | Quant::IQ2_XXS
                | Quant::IQ2_XS
                | Quant::IQ2_S
                | Quant::IQ3_XXS
                | Quant::IQ3_S
                | Quant::Q4_K
                | Quant::Q5_K
                | Quant::Q6_K
                | Quant::PTQ1_0
        )
    }

    pub fn spec(self) -> &'static Spec {
        match self {
            Quant::Q4_0 => &q4_0::SPEC,
            Quant::Q4_1 => &q4_1::SPEC,
            Quant::Q5_0 => &q5_0::SPEC,
            Quant::Q5_1 => &q5_1::SPEC,
            Quant::Q8_0 => &q8_0::SPEC,
            Quant::Q8_1 => &q8_1::SPEC,
            Quant::Q4_K => &q4_k::SPEC,
            Quant::Q5_K => &q5_k::SPEC,
            Quant::Q6_K => &q6_k::SPEC,
            Quant::Q2_K => &q2_k::SPEC,
            Quant::Q3_K => &q3_k::SPEC,
            Quant::IQ1_S => &iq1_s::SPEC,
            Quant::IQ1_M => &iq1_m::SPEC,
            Quant::IQ2_XXS => &iq2_xxs::SPEC,
            Quant::IQ2_XS => &iq2_xs::SPEC,
            Quant::IQ2_S => &iq2_s::SPEC,
            Quant::IQ3_XXS => &iq3_xxs::SPEC,
            Quant::IQ3_S => &iq3_s::SPEC,
            Quant::IQ4_XS => &iq4_xs::SPEC,
            Quant::PTQ1_0 => &ptq1_0::SPEC,
        }
    }

    pub fn name(self) -> &'static str {
        self.spec().name
    }
}

/// Describes one block format and how to decode it. [`Quant::spec`] is the
/// registry.
pub struct Spec {
    pub name: &'static str,
    /// Elements one stored block covers.
    pub block: usize,
    /// Bytes that block occupies.
    pub block_bytes: usize,
    /// Elements sharing one scale. Equals `block` for the legacy formats; a
    /// K-quant super-block carries several runs.
    pub scale_run: usize,
    /// Quants are unsigned with a per-block minimum instead of symmetric
    /// around zero, so decoding needs the minimum as well as the scale.
    pub has_min: bool,
    /// Decodes whole blocks in storage order. `out` is `bytes.len() /
    /// block_bytes * block` elements; both are checked before this runs.
    pub dequantize: fn(bytes: &[u8], out: &mut [f32]),
    /// Splits a `[n, k]` weight into the planes the quantized kernels index.
    /// `None` for a format without such a kernel, which dequantizes at upload
    /// instead.
    ///
    /// The two planes are transposed relative to each other. `qs` is
    /// `[n, k]`: `qs[j * k + p]` is the quant for output `j`, input `p`.
    /// `scales` is `[k / scale_run, n]`: `scales[(p / scale_run) * n + j]`
    /// scales that quant.
    pub planes: Option<Split>,
    /// Pulls the per-block `d` and `dmin` header fields out of a `[n, k]`
    /// weight, as raw f16 bits ([`RawScales`]). Used by kernels that decode
    /// everything else straight from the uploaded block bytes. `None` for a
    /// format with no such kernel.
    ///
    /// Unlike `planes`, the blocks stay as the file orders them,
    /// `[n, k / block * block_bytes]`. Only `d` and `dmin` are copied out,
    /// into their own `[n, k / block]` planes.
    pub raw_scales: Option<RawSplit>,
}

/// What [`Spec::planes`] holds: a `[n, k]` weight's blocks into its planes.
pub type Split = fn(bytes: &[u8], k: usize, n: usize) -> Planes;

/// What [`Spec::raw_scales`] holds: a `[n, k]` weight's blocks into its
/// header-field planes.
pub type RawSplit = fn(bytes: &[u8], k: usize, n: usize) -> RawScales;

/// A quantized weight's per-super-block scale and minimum, pulled out of its
/// blocks as raw `f16` bit patterns: [`Spec::raw_scales`].
#[derive(Debug)]
pub struct RawScales {
    pub d: Vec<u16>,
    /// Empty for a format with no minimum ([`Spec::has_min`] false).
    pub dmin: Vec<u16>,
}

/// A quantized weight split into the planes [`Spec::planes`] describes.
#[derive(Debug)]
pub struct Planes {
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Planes {
    /// Checks the plane sizes: one quant per element and one scale per run.
    /// Runs go along `k`, so `k` must be a multiple of the run.
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

/// A `[n, k]` weight kept in its file format between load and upload.
///
/// Blocks stay exactly as GGUF stores them, `[n, k / block]`, so one output's
/// inputs are contiguous. Nothing here requantizes. Stacking, permuting and
/// reading a row all move whole blocks, so they work for any format.
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

    /// [`Packed::new`] from the front of a longer slice, such as a tensor's
    /// window into the file mapping.
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

    /// Whether a raw kernel decodes this format from its own bytes.
    pub fn has_raw_scales(&self) -> bool {
        self.spec().raw_scales.is_some()
    }

    /// The blocks as the device wants them. Signed because the kernel language
    /// has no unsigned byte type; kernels map each byte back to 0..255.
    ///
    /// Each block is trimmed or zero-padded to its device stride, see
    /// [`Quant::device_block`]. PTQ1_0 blocks are also reordered.
    pub fn device_blocks(&self) -> Vec<i8> {
        let packed = self.spec().block_bytes;
        if self.quant == Quant::PTQ1_0 {
            let mut out = vec![0u8; self.blocks.len()];
            for (src, dst) in self.blocks.chunks_exact(packed).zip(out.chunks_exact_mut(packed)) {
                ptq1_0::device_block(src, dst);
            }
            return out.into_iter().map(|b| b as i8).collect();
        }
        let (skip, dev) = self.quant().device_block();
        if skip == 0 && dev == packed {
            return self.blocks().iter().map(|&b| b as i8).collect();
        }
        let mut out = vec![0i8; self.blocks().len() / packed * dev];
        for (i, blk) in self.blocks().chunks_exact(packed).enumerate() {
            let src = &blk[skip..packed.min(skip + dev)];
            for (o, &b) in src.iter().enumerate() {
                out[i * dev + o] = b as i8;
            }
        }
        out
    }

    /// The block bytes exactly as the file holds them, `[n, k / block *
    /// block_bytes]`, for a raw kernel to upload verbatim.
    pub fn blocks(&self) -> &[u8] {
        &self.blocks
    }

    /// The header-field planes [`Spec::raw_scales`] describes, for a backend
    /// uploading this alongside [`Packed::blocks`].
    pub fn raw_scales(&self) -> Result<RawScales> {
        let split = self
            .spec()
            .raw_scales
            .with_context(|| format!("no raw kernel decodes {}", self.quant.name()))?;
        Ok(split(&self.blocks, self.k, self.n))
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
    /// Padding is zero blocks, which decode to zero in every format because
    /// their scale is zero. It only makes a fused width tile evenly and is
    /// never read.
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
            GgmlType::Q5_K => Quant::Q5_K,
            GgmlType::Q6_K => Quant::Q6_K,
            GgmlType::Q2_K => Quant::Q2_K,
            GgmlType::Q3_K => Quant::Q3_K,
            GgmlType::IQ1_S => Quant::IQ1_S,
            GgmlType::IQ1_M => Quant::IQ1_M,
            GgmlType::IQ2_XXS => Quant::IQ2_XXS,
            GgmlType::IQ2_XS => Quant::IQ2_XS,
            GgmlType::IQ2_S => Quant::IQ2_S,
            GgmlType::IQ3_XXS => Quant::IQ3_XXS,
            GgmlType::IQ3_S => Quant::IQ3_S,
            GgmlType::IQ4_XS => Quant::IQ4_XS,
            GgmlType::PTQ1_0 => Quant::PTQ1_0,
            _ => return None,
        })
    }
}

pub mod grouped;

#[cfg(test)]
mod tests;
