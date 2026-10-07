// Q5_0: `{ f16 d; uint8 qh[4]; uint8 qs[16]; }`.
//
// Q4_0 with a fifth bit per quant in `qh`: bit `j` for the low nibble and bit
// `j + 16` for the high one.

use phobos_base::half::f16_to_f32;

use super::{Planes, Spec};

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 22;

pub static SPEC: Spec = Spec {
    name: "Q5_0",
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
        for (y, q) in dst.iter_mut().zip(quants(block)) {
            *y = d * f32::from(q);
        }
    }
}

/// A block's 32 quants, centred: each is in -16..=15.
fn quants(block: &[u8]) -> [i8; BLOCK] {
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let mut out = [0i8; BLOCK];
    for (j, &q) in block[6..].iter().enumerate() {
        let hi_lo = ((qh >> j) << 4) & 0x10;
        let hi_hi = (qh >> (j + 12)) & 0x10;
        out[j] = (i32::from(q & 0x0f) | hi_lo as i32) as i8 - 16;
        out[j + 16] = (i32::from(q >> 4) | hi_hi as i32) as i8 - 16;
    }
    out
}

/// The quants and scales without decoding, as [`super::Spec::planes`]
/// describes: one centred quant per element, one scale per block.
fn planes(bytes: &[u8], k: usize, n: usize) -> Planes {
    let blocks = k / BLOCK;
    let mut qs = vec![0i8; k * n];
    let mut scales = vec![0.0f32; blocks * n];
    for (index, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let (j, b) = (index / blocks, index % blocks);
        scales[b * n + j] = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        qs[j * k + b * BLOCK..][..BLOCK].copy_from_slice(&quants(block));
    }
    Planes { qs, scales }
}
