// IQ1_S: `{ f16 d; uint8 qs[32]; uint16 qh[16]; }`, 1.5625 bits a weight.
//
// Eight groups of 32 elements, each with a 3-bit scale (`qh[ib] >> 12`) and
// sign bit (`qh[ib] & 0x8000`), and four 8-element lanes looked up in
// [`super::tables::iq1::IQ1S_GRID`] by a 9-bit index from a `qs` byte and
// three bits of `qh`. `DELTA` shifts the grid's signed -1/0/1 lanes off zero.

use phobos_base::half::f16_to_f32;

use super::tables::iq1::IQ1S_GRID;
use super::{RawScales, Spec};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 50;
const DELTA: f32 = 0.125;

const QS: usize = 2;
const QH: usize = QS + 32;

pub static SPEC: Spec = Spec {
    name: "IQ1_S",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: 32,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

/// `d` leads this block rather than trailing it, unlike Q2_K and Q3_K.
fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        d.push(u16::from_le_bytes([block[0], block[1]]));
    }
    RawScales { d, dmin: Vec::new() }
}

/// [`IQ1S_GRID`] flattened to sign-extended lanes: `flat_grid()[i * 8 + j]`
/// is byte `j` of `IQ1S_GRID[i]` read as `i8`.
pub(crate) fn flat_grid() -> Vec<i32> {
    IQ1S_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(|b| i32::from(b as i8)))
        .collect()
}

/// [`IQ1S_GRID`] as raw signed bytes, the layout `iq1s_qdot_t` reads:
/// `packed_grid()[i * 8 + j]` is byte `j` of `IQ1S_GRID[i]`, the same value
/// [`flat_grid`] widens to `i32`. A lane's eight entries are eight contiguous
/// bytes here, so it takes one 64-bit load instead of eight 32-bit ones, and
/// the whole table is 16 KB rather than 64.
pub(crate) fn packed_grid() -> Vec<i8> {
    IQ1S_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(|b| b as i8))
        .collect()
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[QS..QH];

        for (ib, group) in dst.chunks_mut(32).enumerate() {
            let qh = u16::from_le_bytes([block[QH + 2 * ib], block[QH + 2 * ib + 1]]);
            let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
            let delta = if qh & 0x8000 != 0 { -DELTA } else { DELTA };
            for (l, lane) in group.chunks_mut(8).enumerate() {
                let idx = usize::from(qs[4 * ib + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
                let grid = IQ1S_GRID[idx].to_le_bytes();
                for (y, &g) in lane.iter_mut().zip(&grid) {
                    *y = dl * (g as i8 as f32 + delta);
                }
            }
        }
    }
}
