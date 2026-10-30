// IQ2_XS: `{ f16 d; uint16 qs[32]; uint8 scales[8]; }`, 2.3125 bits a weight.
//
// Each `qs` halfword holds a 9-bit index into
// [`super::tables::iq2::IQ2XS_GRID`] and a 7-bit sign parity in its top
// bits. `scales` gives each half of a 32-element group its own 4-bit scale.

use phobos_base::half::f16_to_f32;

use super::tables::iq2::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use super::{RawHeader, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 74;
const QS: usize = 2;
const SCALES: usize = QS + 64;

pub static SPEC: Spec = Spec {
    name: "IQ2_XS",
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
        let qs = &block[QS..SCALES];
        let scales = &block[SCALES..];

        for (ib32, group) in dst.chunks_mut(32).enumerate() {
            let sc = scales[ib32];
            let db = [
                d * (0.5 + f32::from(sc & 0x0f)) * 0.25,
                d * (0.5 + f32::from(sc >> 4)) * 0.25,
            ];
            for (l, lane) in group.chunks_mut(8).enumerate() {
                let at = 2 * (4 * ib32 + l);
                let q16 = u16::from_le_bytes([qs[at], qs[at + 1]]);
                let grid = IQ2XS_GRID[usize::from(q16 & 511)].to_le_bytes();
                let signs = KSIGNS_IQ2XS[usize::from(q16 >> 9)];
                let dl = db[l / 2];
                for (j, (y, &g)) in lane.iter_mut().zip(&grid).enumerate() {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    *y = dl * f32::from(g) * sign;
                }
            }
        }
    }
}
