// IQ1_M: `{ uint8 qs[32]; uint8 qh[16]; uint8 scales[8]; }`, 1.75 bits a weight.
//
// Same grid lookup as IQ1_S (see [`super::iq1_s`]), but with no `f16` scale
// of its own: `d` is packed across the top nibble of each `scales` halfword,
// and each group of 32 gets its own 3-bit scale instead of sharing one.

use phobos_base::half::f16_to_f32;

use super::tables::iq1::IQ1S_GRID;
use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 56;
const DELTA: f32 = 0.125;

const QS: usize = 0;
const QH: usize = QS + 32;
const SCALES: usize = QH + 16;

pub static SPEC: Spec = Spec {
    name: "IQ1_M",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: 32,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

/// Reassembles the per-block `d` the kernel needs from the `scales`
/// halfwords, since IQ1_M has no stored `f16` scale field.
fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let sc = &block[SCALES..];
        let sc16 = |i: usize| u16::from_le_bytes([sc[2 * i], sc[2 * i + 1]]);
        let scale_bits = (sc16(0) >> 12)
            | ((sc16(1) >> 8) & 0x00f0)
            | ((sc16(2) >> 4) & 0x0f00)
            | (sc16(3) & 0xf000);
        d.push(scale_bits);
    }
    RawScales { d, dmin: Vec::new() }
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let qs = &block[QS..QH];
        let qh = &block[QH..SCALES];
        let sc = &block[SCALES..];
        let sc16 = |i: usize| u16::from_le_bytes([sc[2 * i], sc[2 * i + 1]]);
        let scale_bits =
            (sc16(0) >> 12) | ((sc16(1) >> 8) & 0x00f0) | ((sc16(2) >> 4) & 0x0f00) | (sc16(3) & 0xf000);
        let d = f16_to_f32(scale_bits);

        for (ib, group) in dst.chunks_mut(32).enumerate() {
            let word = sc16(ib / 2);
            let shift = 6 * (ib % 2) as u32;
            let dl1 = d * f32::from(2 * ((word >> shift) & 7) + 1);
            let dl2 = d * f32::from(2 * ((word >> (shift + 3)) & 7) + 1);

            let qs4 = &qs[4 * ib..][..4];
            let (qh0, qh1) = (qh[2 * ib], qh[2 * ib + 1]);
            let idx = [
                usize::from(qs4[0]) | ((usize::from(qh0) << 8) & 0x700),
                usize::from(qs4[1]) | ((usize::from(qh0) << 4) & 0x700),
                usize::from(qs4[2]) | ((usize::from(qh1) << 8) & 0x700),
                usize::from(qs4[3]) | ((usize::from(qh1) << 4) & 0x700),
            ];
            let delta = [
                if qh0 & 0x08 != 0 { -DELTA } else { DELTA },
                if qh0 & 0x80 != 0 { -DELTA } else { DELTA },
                if qh1 & 0x08 != 0 { -DELTA } else { DELTA },
                if qh1 & 0x80 != 0 { -DELTA } else { DELTA },
            ];

            for (l, lane) in group.chunks_mut(8).enumerate() {
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for (y, &g) in lane.iter_mut().zip(&grid) {
                    *y = dl * (g as i8 as f32 + delta[l]);
                }
            }
        }
    }
}
