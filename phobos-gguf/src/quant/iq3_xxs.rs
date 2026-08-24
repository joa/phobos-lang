// IQ3_XXS: `{ f16 d; uint8 qs[96]; }`, 3.0625 bits a weight.
//
// Eight 32-element groups. `qs`'s first 64 bytes hold two grid-index bytes
// per lane into [`super::tables::iq3::IQ3XXS_GRID`] (four elements per
// entry); its last 32 bytes hold the same scale-and-parity word IQ2_XXS
// reads (see [`super::iq2_xxs`]).

use phobos_base::half::f16_to_f32;

use super::tables::iq2::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use super::tables::iq3::IQ3XXS_GRID;
use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 98;
const QS: usize = 2;
const SS: usize = QS + 64;

pub static SPEC: Spec = Spec {
    name: "IQ3_XXS",
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

/// [`IQ3XXS_GRID`] flattened to one magnitude byte a slot, four bytes an
/// entry (a lane's eight elements come from two grid entries):
/// `flat_grid()[i * 4 + j]` is byte `j` of `IQ3XXS_GRID[i]`.
pub(crate) fn flat_grid() -> Vec<i32> {
    IQ3XXS_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(i32::from))
        .collect()
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[QS..SS];
        let ss = &block[SS..];

        for (ib32, group) in dst.chunks_mut(32).enumerate() {
            let s = &ss[4 * ib32..][..4];
            let aux32 = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let base = &qs[8 * ib32..][..8];
            for (l, lane) in group.chunks_mut(8).enumerate() {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[usize::from(base[2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(base[2 * l + 1])].to_le_bytes();
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    lane[j] = db * f32::from(g1[j]) * s0;
                    lane[j + 4] = db * f32::from(g2[j]) * s1;
                }
            }
        }
    }
}
