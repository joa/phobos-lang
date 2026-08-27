// IQ2_XXS: `{ f16 d; uint16 qs[32]; }`, 2.0625 bits a weight.
//
// Each 32-element group's 8-byte span of `qs` packs 4 grid indices plus a
// 32-bit aux word: its top 4 bits are the group's scale, the rest packs four
// 7-bit sign parities (one per lane) indexing
// [`super::tables::iq2::KSIGNS_IQ2XS`].

use phobos_base::half::f16_to_f32;

use super::tables::iq2::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use super::{RawScales, Spec};

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
    raw_scales: Some(raw_scales),
};

fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let mut d = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        d.push(u16::from_le_bytes([block[0], block[1]]));
    }
    RawScales { d, dmin: Vec::new() }
}

/// [`IQ2XXS_GRID`] flattened to one magnitude byte a slot: `flat_grid()[i *
/// 8 + j]` is byte `j` of `IQ2XXS_GRID[i]`.
pub(crate) fn flat_grid() -> Vec<i32> {
    IQ2XXS_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(i32::from))
        .collect()
}

/// [`KSIGNS_IQ2XS`] expanded to per-element +-1 multipliers via
/// [`KMASK_IQ2XS`]: `flat_signs()[i * 8 + j]` is element `j`'s sign under
/// sign index `i`.
pub(crate) fn flat_signs() -> Vec<i32> {
    KSIGNS_IQ2XS
        .iter()
        .flat_map(|&byte| KMASK_IQ2XS.map(move |bit| if byte & bit != 0 { -1 } else { 1 }))
        .collect()
}

/// [`flat_grid`]'s values as raw bytes, the layout `iq2xxs_qdot_t` reads: a
/// lane's eight magnitudes are eight contiguous bytes, so it takes one 64-bit
/// load instead of eight 32-bit ones. Magnitudes top out at 43, so `i8` holds
/// them exactly.
pub(crate) fn packed_grid() -> Vec<i8> {
    IQ2XXS_GRID
        .iter()
        .flat_map(|entry| entry.to_le_bytes().map(|b| b as i8))
        .collect()
}

/// [`flat_signs`] packed the same way; the multipliers are +-1.
/// [`packed_signs`] as a bitwise mask: `0` where the sign is positive, `-1`
/// where it is negative. The dp4a decode applies signs with `and`, since an
/// elementwise i8 multiply has no hardware form and scalarizes.
pub(crate) fn packed_sign_masks() -> Vec<i8> {
    KSIGNS_IQ2XS
        .iter()
        .flat_map(|&byte| KMASK_IQ2XS.map(move |bit| if byte & bit != 0 { -1i8 } else { 0 }))
        .collect()
}

pub(crate) fn packed_signs() -> Vec<i8> {
    KSIGNS_IQ2XS
        .iter()
        .flat_map(|&byte| KMASK_IQ2XS.map(move |bit| if byte & bit != 0 { -1i8 } else { 1 }))
        .collect()
}

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
