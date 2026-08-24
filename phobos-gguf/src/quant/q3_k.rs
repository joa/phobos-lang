// Q3_K: `{ uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; f16 d; }`.
//
// Two bits per element from `qs` (same four-pass layout as Q2_K), the third
// from `hmask` (one bit per element, set for the low half of the range).
// `scales` packs sixteen signed 6-bit values into 12 bytes like Q6_K's (see
// [`unpack_scales`]), offset by 32.

use phobos_base::half::f16_to_f32;

use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 110;
/// Elements sharing one scale.
const RUN: usize = 16;

const HMASK: usize = 0;
const QS: usize = HMASK + BLOCK / 8;
const SCALES: usize = QS + BLOCK / 4;
const D: usize = SCALES + 12;

pub static SPEC: Spec = Spec {
    name: "Q3_K",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: RUN,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

/// `dmin` stays empty: Q3_K has no minimum term.
fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        d.push(u16::from_le_bytes([block[D], block[D + 1]]));
    }
    RawScales { d, dmin: Vec::new() }
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d_all = f16_to_f32(u16::from_le_bytes([block[D], block[D + 1]]));
        let hm = &block[HMASK..QS];
        let q = &block[QS..SCALES];
        let scales = unpack_scales(&block[SCALES..D]);

        let mut is = 0;
        let mut m = 1u8;
        for (n, half) in dst.chunks_mut(128).enumerate() {
            let qplane = &q[n * 32..][..32];
            for j in 0..4 {
                let shift = 2 * j as u32;
                for half2 in 0..2 {
                    let dl = d_all * f32::from(scales[is]);
                    is += 1;
                    let src = &qplane[half2 * 16..][..16];
                    let hsrc = &hm[half2 * 16..][..16];
                    let dst16 = &mut half[(j * 32 + half2 * 16)..][..16];
                    for ((y, &b), &hb) in dst16.iter_mut().zip(src).zip(hsrc) {
                        let low = i32::from((b >> shift) & 3);
                        let bit = i32::from((hb & m) != 0);
                        *y = dl * (low - 4 + bit * 4) as f32;
                    }
                }
                m = m.wrapping_shl(1);
            }
        }
    }
}

/// Sixteen signed 6-bit scales packed into 12 bytes: the first two groups of
/// four take their low nibble from the first eight bytes, the last two the
/// high nibble; all sixteen take their top two bits from the last four bytes.
fn unpack_scales(b: &[u8]) -> [i8; 16] {
    let mut scales = [0i8; 16];
    for c in 0..4 {
        let hi = b[8 + c];
        scales[c] = (((b[c] & 0x0f) | ((hi & 3) << 4)) as i32 - 32) as i8;
        scales[4 + c] = (((b[4 + c] & 0x0f) | (((hi >> 2) & 3) << 4)) as i32 - 32) as i8;
        scales[8 + c] = ((((b[c] >> 4) & 0x0f) | (((hi >> 4) & 3) << 4)) as i32 - 32) as i8;
        scales[12 + c] = ((((b[4 + c] >> 4) & 0x0f) | (((hi >> 6) & 3) << 4)) as i32 - 32) as i8;
    }
    scales
}
