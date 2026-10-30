// Q8_0: `{ f16 d; int8 qs[32]; }`, every element `d * qs[i]`.
//
// Activations are always quantized to this format, whatever format the
// weight is in; see [`quantize_row`].

use anyhow::Result;

use phobos_base::half::{f16_to_f32, f32_to_f16};

use super::{Packed, Planes, Quant, Spec};

pub const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 34;

/// Elements of the unit the expert kernels read Q8_0 in: eight blocks, the
/// K-quants' super-block, so one lane mapping serves every format.
pub const UNIT: usize = 8 * BLOCK;
/// Bytes of that unit: the eight `f16` scales, then the 256 quants.
pub const UNIT_BYTES: usize = 8 * BLOCK_BYTES;

pub static SPEC: Spec = Spec {
    name: "Q8_0",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: BLOCK,
    has_min: false,
    dequantize,
    planes: Some(planes),
    raw_scales: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (y, &q) in dst.iter_mut().zip(&block[2..]) {
            *y = d * (q as i8) as f32;
        }
    }
}

/// Splits the blocks into quants and scales without decoding, so a device
/// uploads a quarter of the bytes and multiplies in the kernel.
fn planes(bytes: &[u8], k: usize, n: usize) -> Planes {
    super::scaled_planes::<BLOCK>(bytes, k, n, BLOCK_BYTES, |block| std::array::from_fn(|i| block[2 + i] as i8))
}

/// The inverse of [`planes`]: builds blocks from given quants and scales.
/// Checks and benchmarks use it to make a weight without a file.
///
/// Scales are stored as halves, so a scale a half cannot hold comes back
/// rounded. Weights read from a file always round-trip exactly.
pub fn pack(qs: &[i8], scales: &[f32], k: usize, n: usize) -> Result<Packed> {
    let planes = Planes {
        qs: qs.to_vec(),
        scales: scales.to_vec(),
    };
    planes.check(Quant::Q8_0, k, n)?;

    let blocks = k / BLOCK;
    let mut bytes = Vec::with_capacity(n * blocks * BLOCK_BYTES);
    for j in 0..n {
        for b in 0..blocks {
            bytes.extend_from_slice(&f32_to_f16(scales[b * n + j]).to_le_bytes());
            bytes.extend(qs[j * k + b * BLOCK..][..BLOCK].iter().map(|&q| q as u8));
        }
    }
    Packed::new(Quant::Q8_0, bytes, k, n)
}

/// Re-lays rows of blocks into units of eight ([`UNIT_BYTES`] each): the
/// eight scales first, then the 256 quants, so every block's quants sit
/// sixteen-aligned for the device's vector loads. The rows must be whole
/// units.
pub fn units(bytes: &[u8]) -> Vec<u8> {
    assert!(bytes.len().is_multiple_of(UNIT_BYTES), "{} bytes of Q8_0 are not whole units", bytes.len());
    let mut out = vec![0; bytes.len()];
    for (unit, into) in bytes.chunks_exact(UNIT_BYTES).zip(out.chunks_exact_mut(UNIT_BYTES)) {
        unit_into(unit, into);
    }
    out
}

/// One unit of [`units`]: eight file blocks into `out`, both
/// [`UNIT_BYTES`] long.
pub fn unit_into(unit: &[u8], out: &mut [u8]) {
    let (scales, quants) = out.split_at_mut(2 * UNIT / BLOCK);
    for ((block, scale), qs) in unit.chunks_exact(BLOCK_BYTES).zip(scales.chunks_exact_mut(2)).zip(quants.chunks_exact_mut(BLOCK)) {
        scale.copy_from_slice(&block[..2]);
        qs.copy_from_slice(&block[2..]);
    }
}

/// Quantizes one block of [`BLOCK`] activations to int8 and returns the
/// shared scale. Symmetric and round to nearest; the largest magnitude maps
/// to 127.
///
/// The device kernel must round the same way: ties go to even, as the
/// hardware's rounding instruction does.
pub fn quantize_row(x: &[f32], qs: &mut [i8]) -> f32 {
    let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let inv = 127.0 / (absmax + 1e-8);
    for (q, &v) in qs.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even() as i8;
    }
    absmax / 127.0
}
