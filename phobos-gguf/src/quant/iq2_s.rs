// IQ2_S: `{ f16 d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]; }`, 2.5625
// bits a weight.
//
// The first half of `qs` holds an 8-bit grid index per lane, extended to 10
// bits by 2 bits of a `qh` byte shared by four lanes. The second half holds
// a sign byte per lane, tested directly against
// [`super::tables::iq2::KMASK_IQ2XS`] instead of going through the parity
// table IQ2_XXS and IQ2_XS use.

use phobos_base::half::f16_to_f32;

use super::tables::iq2::{IQ2S_GRID, KMASK_IQ2XS};
use super::{RawHeader, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 82;
const QS: usize = 2;
const QH: usize = QS + 64;
const SCALES: usize = QH + 8;

pub static SPEC: Spec = Spec {
    name: "IQ2_S",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: 16,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(RawHeader::At { d: 0, dmin: None }),
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let grid_lo = &block[QS..QS + 32];
        let signs = &block[QS + 32..QH];
        let qh = &block[QH..SCALES];
        let scales = &block[SCALES..];

        for (ib32, group) in dst.chunks_mut(32).enumerate() {
            let sc = scales[ib32];
            let db = [
                d * (0.5 + f32::from(sc & 0x0f)) * 0.25,
                d * (0.5 + f32::from(sc >> 4)) * 0.25,
            ];
            let qh_byte = usize::from(qh[ib32]);
            for (l, lane) in group.chunks_mut(8).enumerate() {
                let idx = usize::from(grid_lo[4 * ib32 + l]) | ((qh_byte << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let sb = signs[4 * ib32 + l];
                let dl = db[l / 2];
                for (j, (y, &g)) in lane.iter_mut().zip(&grid).enumerate() {
                    let sign = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    *y = dl * f32::from(g) * sign;
                }
            }
        }
    }
}
