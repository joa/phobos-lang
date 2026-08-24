// IQ3_S: `{ f16 d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8
// scales[4]; }`, 3.3125 bits a weight.
//
// Four 64-element groups, each split into two 32-element halves with their
// own scale nibble. A half's eight lanes read two grid-index bytes from
// `qs`, extended to 9 bits by a bit of `qh`, into
// [`super::tables::iq3::IQ3S_GRID`]; signs come from `signs`, tested against
// [`super::tables::iq2::KMASK_IQ2XS`] like IQ2_S (see [`super::iq2_s`]).

use phobos_base::half::f16_to_f32;

use super::tables::iq2::KMASK_IQ2XS;
use super::tables::iq3::IQ3S_GRID;
use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 110;
const QS: usize = 2;
const QH: usize = QS + 64;
const SIGNS: usize = QH + 8;
const SCALES: usize = SIGNS + 32;

pub static SPEC: Spec = Spec {
    name: "IQ3_S",
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

/// [`IQ3S_GRID`] flattened to one magnitude byte a slot, four bytes an
/// entry (a lane's eight elements come from two grid entries). Signs reuse
/// [`super::iq2_s::flat_signs`].
pub(crate) fn flat_grid() -> Vec<i32> {
    IQ3S_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(i32::from))
        .collect()
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[QS..QH];
        let qh = &block[QH..SIGNS];
        let signs = &block[SIGNS..SCALES];
        let scales = &block[SCALES..];

        for (o, group) in dst.chunks_mut(64).enumerate() {
            let sc = scales[o];
            let db = [d * f32::from(1 + 2 * (sc & 0x0f)), d * f32::from(1 + 2 * (sc >> 4))];
            let qs16 = &qs[16 * o..][..16];
            let qh2 = [usize::from(qh[2 * o]), usize::from(qh[2 * o + 1])];
            let sg = &signs[8 * o..][..8];

            for half in 0..2 {
                let qh_byte = qh2[half];
                let qs8 = &qs16[8 * half..][..8];
                let sg4 = &sg[4 * half..][..4];
                let dl = db[half];
                let out_half = &mut group[32 * half..][..32];
                for l in 0..4 {
                    let idx1 = usize::from(qs8[2 * l]) | ((qh_byte << (8 - 2 * l)) & 256);
                    let idx2 = usize::from(qs8[2 * l + 1]) | ((qh_byte << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[idx1].to_le_bytes();
                    let g2 = IQ3S_GRID[idx2].to_le_bytes();
                    let signs_byte = sg4[l];
                    let lane = &mut out_half[8 * l..][..8];
                    for j in 0..4 {
                        let s0 = if signs_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s1 = if signs_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        lane[j] = dl * f32::from(g1[j]) * s0;
                        lane[j + 4] = dl * f32::from(g2[j]) * s1;
                    }
                }
            }
        }
    }
}
