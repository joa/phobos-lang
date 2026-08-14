// Q5_1: `{ f16 d; f16 m; uint8 qh[4]; uint8 qs[16]; }`.
//
// Q5_0's fifth bit with Q4_1's offset.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 24;

pub static SPEC: Spec = Spec {
    name: "Q5_1",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: BLOCK,
    has_min: true,
    dequantize,
    planes: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for (j, &q) in block[8..].iter().enumerate() {
            let hi_lo = ((qh >> j) << 4) & 0x10;
            let hi_hi = (qh >> (j + 12)) & 0x10;
            dst[j] = d * (u32::from(q & 0x0f) | hi_lo) as f32 + m;
            dst[j + 16] = d * (u32::from(q >> 4) | hi_hi) as f32 + m;
        }
    }
}
