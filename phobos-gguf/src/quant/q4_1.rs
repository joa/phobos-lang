// Q4_1: `{ f16 d; f16 m; uint8 qs[16]; }`.
//
// Q4_0's nibble halves, but the quants are unsigned with a per-block offset
// `m` rather than symmetric around eight.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 20;

pub static SPEC: Spec = Spec {
    name: "Q4_1",
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
        for (j, &q) in block[4..].iter().enumerate() {
            dst[j] = d * f32::from(q & 0x0f) + m;
            dst[j + 16] = d * f32::from(q >> 4) + m;
        }
    }
}
