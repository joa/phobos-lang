// IQ4_XS: `{ f16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }`,
// 4.25 bits a weight.
//
// Eight 32-element runs, each with a 6-bit signed scale split like Q4_K's
// (4 bits from `scales_l`, 2 from `scales_h`), offset by 32. Nibbles index
// the non-linear codebook [`super::tables::iq3::KVALUES_IQ4NL`] rather than
// scaling linearly.

use phobos_base::half::f16_to_f32;

use super::tables::iq3::KVALUES_IQ4NL;
use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 136;
const SCALES_H: usize = 2;
const SCALES_L: usize = SCALES_H + 2;
const QS: usize = SCALES_L + 4;

pub static SPEC: Spec = Spec {
    name: "IQ4_XS",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: 32,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        d.push(u16::from_le_bytes([block[0], block[1]]));
    }
    RawScales { d, dmin: Vec::new() }
}

/// [`KVALUES_IQ4NL`] widened to `i32`, one entry a nibble value.
pub(crate) fn flat_codebook() -> Vec<i32> {
    KVALUES_IQ4NL.iter().map(|&v| i32::from(v)).collect()
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales_h = u16::from_le_bytes([block[SCALES_H], block[SCALES_H + 1]]);
        let scales_l = &block[SCALES_L..QS];
        let qs = &block[QS..];

        for (ib, group) in dst.chunks_mut(32).enumerate() {
            let low = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f);
            let high = i32::from((scales_h >> (2 * ib)) & 3);
            let ls = low | (high << 4);
            let dl = d * (ls - 32) as f32;
            let plane = &qs[16 * ib..][..16];
            for (j, &b) in plane.iter().enumerate() {
                group[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(b & 0x0f)]);
                group[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(b >> 4)]);
            }
        }
    }
}
