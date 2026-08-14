// Q4_0: `{ f16 d; uint8 qs[16]; }`, symmetric around eight.
//
// The two nibbles of `qs[j]` are elements `j` and `j + 16`, not `2j` and
// `2j + 1`.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 18;

pub static SPEC: Spec = Spec {
    name: "Q4_0",
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
        for (j, &q) in block[2..].iter().enumerate() {
            dst[j] = d * (i32::from(q & 0x0f) - 8) as f32;
            dst[j + 16] = d * (i32::from(q >> 4) - 8) as f32;
        }
    }
}
