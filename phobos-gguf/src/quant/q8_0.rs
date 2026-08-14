// Q8_0: `{ f16 d; int8 qs[32]; }`, every element `d * qs[i]`.
//
// The only format a kernel unpacks today, and the one activations are always
// quantized to whatever a weight is held in; see [`quantize_row`].

use anyhow::Result;

use phobos_base::half::{f16_to_f32, f32_to_f16};

use super::{Packed, Planes, Quant, Spec};

pub const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 34;

pub static SPEC: Spec = Spec {
    name: "Q8_0",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: BLOCK,
    has_min: false,
    dequantize,
    planes: Some(planes),
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (y, &q) in dst.iter_mut().zip(&block[2..]) {
            *y = d * (q as i8) as f32;
        }
    }
}

/// Pulls the blocks apart without decoding, which is what lets a device upload
/// a quarter of the bytes and do the multiply in the kernel.
fn planes(bytes: &[u8], k: usize, n: usize) -> Planes {
    let blocks = k / BLOCK;
    let mut qs = vec![0i8; k * n];
    let mut scales = vec![0.0f32; blocks * n];
    for (index, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let (j, b) = (index / blocks, index % blocks);
        scales[b * n + j] = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (dst, &q) in qs[j * k + b * BLOCK..][..BLOCK].iter_mut().zip(&block[2..]) {
            *dst = q as i8;
        }
    }
    Planes { qs, scales }
}

/// The inverse of [`planes`]: blocks assembled from chosen quants and scales,
/// which is how the checks and the benchmarks make a weight rather than reading
/// one from a file.
///
/// A scale is stored as a half, so one that is not representable comes back
/// rounded. Weights out of a file always round-trip exactly, having been halves
/// to begin with.
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

/// Quantize one block of [`BLOCK`] activations to int8 with a shared scale:
/// symmetric, round to nearest, the extreme element landing on 127.
///
/// The device kernel reproduces this and has to round the same way. Ties go to
/// even because that is what the hardware's rounding instruction does.
pub fn quantize_row(x: &[f32], qs: &mut [i8]) -> f32 {
    let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let inv = 127.0 / (absmax + 1e-8);
    for (q, &v) in qs.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even() as i8;
    }
    absmax / 127.0
}
