// IQ2_XXS: `{ f16 d; uint16 qs[32]; }`, 2.0625 bits a weight.
//
// Each 32-element group owns 8 bytes of `qs`: four grid index bytes, then a
// 32-bit aux word. The aux word's top 4 bits are the group's scale. Below
// them sit four 7-bit sign parities, one per lane, indexing
// [`super::tables::iq2::KSIGNS_IQ2XS`].

use phobos_base::half::f16_to_f32;

use super::tables::iq2::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use super::{RawHeader, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 66;
const QS: usize = 2;

pub static SPEC: Spec = Spec {
    name: "IQ2_XXS",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: 32,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(RawHeader::At { d: 0, dmin: None }),
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[QS..];

        for (ib32, group) in dst.chunks_mut(32).enumerate() {
            let chunk = &qs[8 * ib32..][..8];
            let aux_hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let db = d * (0.5 + (aux_hi >> 28) as f32) * 0.25;
            for (l, lane) in group.chunks_mut(8).enumerate() {
                let grid = IQ2XXS_GRID[usize::from(chunk[l])].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux_hi >> (7 * l)) & 127) as usize];
                for (j, (y, &g)) in lane.iter_mut().zip(&grid).enumerate() {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    *y = db * f32::from(g) * sign;
                }
            }
        }
    }
}
