// Q5_0: `{ f16 d; uint8 qh[4]; uint8 qs[16]; }`.
//
// Q4_0 with a fifth bit per quant in `qh`: bit `j` for the low nibble and bit
// `j + 16` for the high one.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 22;

pub static SPEC: Spec = Spec {
    name: "Q5_0",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: BLOCK,
    has_min: false,
    dequantize,
    planes: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for (j, &q) in block[6..].iter().enumerate() {
            let hi_lo = ((qh >> j) << 4) & 0x10;
            let hi_hi = (qh >> (j + 12)) & 0x10;
            dst[j] = d * ((i32::from(q & 0x0f) | hi_lo as i32) - 16) as f32;
            dst[j + 16] = d * ((i32::from(q >> 4) | hi_hi as i32) - 16) as f32;
        }
    }
}
