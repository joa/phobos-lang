// Q2_K: `{ uint8 scales[16]; uint8 qs[64]; f16 d; f16 dmin; }`.
//
// 16 runs of 16 elements, each with a scale and min packed into one byte
// (`scale = byte & 0xf`, `min = byte >> 4`). `qs` packs four 2-bit values a
// byte; each 32-byte half serves 128 elements, read four times with the
// shift advancing by two each pass.

use phobos_base::half::f16_to_f32;

use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 84;
/// Elements sharing one scale and minimum.
const RUN: usize = 16;

const SCALES: usize = 0;
const QS: usize = SCALES + BLOCK / 16;
const D: usize = QS + BLOCK / 4;
const DMIN: usize = D + 2;

pub static SPEC: Spec = Spec {
    name: "Q2_K",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: RUN,
    has_min: true,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    let mut dmin = Vec::with_capacity(d.capacity());
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        d.push(u16::from_le_bytes([block[D], block[D + 1]]));
        dmin.push(u16::from_le_bytes([block[DMIN], block[DMIN + 1]]));
    }
    RawScales { d, dmin }
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[D], block[D + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[DMIN], block[DMIN + 1]]));
        let scales = &block[SCALES..QS];
        let qs = &block[QS..D];

        let mut is = 0;
        for (h, half) in dst.chunks_mut(128).enumerate() {
            let q = &qs[h * 32..][..32];
            for j in 0..4 {
                let shift = 2 * j as u32;
                for half2 in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let (dl, ml) = (d * f32::from(sc & 0x0f), dmin * f32::from(sc >> 4));
                    let src = &q[half2 * 16..][..16];
                    let dst16 = &mut half[(j * 32 + half2 * 16)..][..16];
                    for (y, &b) in dst16.iter_mut().zip(src) {
                        *y = dl * f32::from((b >> shift) & 3) - ml;
                    }
                }
            }
        }
    }
}
