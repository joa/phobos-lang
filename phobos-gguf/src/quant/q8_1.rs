// Q8_1: `{ f16 d; f16 s; int8 qs[32]; }`.
//
// Q8_0 with the block sum alongside the scale, for a contraction that wants
// it precomputed. Decoding ignores `s`.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 36;

pub static SPEC: Spec = Spec {
    name: "Q8_1",
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
        for (y, &q) in dst.iter_mut().zip(&block[4..]) {
            *y = d * (q as i8) as f32;
        }
    }
}
