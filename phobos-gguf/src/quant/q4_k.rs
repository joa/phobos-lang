// Q4_K: `{ f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128]; }`.
//
// A 256-element super-block holding eight runs of 32. Each run has its own
// scale and minimum, but stored as six-bit indices into the super-block's `d`
// and `dmin` rather than as halves of their own, which is where the format
// spends less than Q4_1 for the same nibble quants. The twelve scale bytes
// pack sixteen six-bit values; see [`scale_min`].
//
// Nibble order follows Q4_0: a byte plane of 32 supplies one run from its low
// nibbles and the next run from its high ones.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 144;
/// Elements sharing one scale and minimum.
const RUN: usize = 32;
const RUNS: usize = BLOCK / RUN;
const SCALE_BYTES: usize = 12;

pub static SPEC: Spec = Spec {
    name: "Q4_K",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: RUN,
    has_min: true,
    dequantize,
    planes: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..4 + SCALE_BYTES];
        let qs = &block[4 + SCALE_BYTES..];

        for run in 0..RUNS {
            let (s, m) = scale_min(run, packed);
            let (scale, min) = (d * f32::from(s), dmin * f32::from(m));
            let plane = &qs[(run / 2) * RUN..][..RUN];
            let shift = 4 * (run as u32 % 2);
            for (y, &q) in dst[run * RUN..][..RUN].iter_mut().zip(plane) {
                *y = scale * f32::from((q >> shift) & 0x0f) - min;
            }
        }
    }
}

/// The six-bit scale and minimum indices of one run, unpacked from the twelve
/// scale bytes.
///
/// The first four runs get a byte each for the scale and a byte each for the
/// minimum, both with two bits spare. The last four are split: their low four
/// bits come from `packed[8..12]` and their top two are the spare bits of the
/// first four runs' bytes.
pub(super) fn scale_min(run: usize, packed: &[u8]) -> (u8, u8) {
    if run < 4 {
        (packed[run] & 63, packed[run + 4] & 63)
    } else {
        (
            (packed[run + 4] & 0x0f) | ((packed[run - 4] >> 6) << 4),
            (packed[run + 4] >> 4) | ((packed[run] >> 6) << 4),
        )
    }
}

/// The inverse of [`scale_min`] over a whole super-block, for a caller building
/// blocks. Every index has to fit six bits.
#[cfg(test)]
pub(super) fn pack_scales(scales: &[u8; RUNS], mins: &[u8; RUNS]) -> [u8; SCALE_BYTES] {
    let mut packed = [0u8; SCALE_BYTES];
    for run in 0..4 {
        packed[run] = scales[run] & 63;
        packed[run + 4] = mins[run] & 63;
    }
    for run in 4..RUNS {
        packed[run + 4] = (scales[run] & 0x0f) | ((mins[run] & 0x0f) << 4);
        packed[run - 4] |= (scales[run] >> 4) << 6;
        packed[run] |= (mins[run] >> 4) << 6;
    }
    packed
}
