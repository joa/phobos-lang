// Q5_K: `{ f16 d; f16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }`.
//
// Q4_K with a fifth bit. The same eight runs of 32, the same 6-bit scale and
// minimum indices packed the same way (see [`super::q4_k::scale_min`]), and
// the same nibble planes in `qs`. Run `r`'s element `l` takes its top bit
// from bit `r` of `qh[l]`, so the one 32-byte `qh` plane serves every run.

use phobos_base::half::f16_to_f32;

use super::Spec;
use super::q4_k::scale_min;

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 176;
/// Elements sharing one scale and minimum.
const RUN: usize = 32;
const RUNS: usize = BLOCK / RUN;

const SCALES: usize = 4;
const QH: usize = SCALES + 12;
const QS: usize = QH + BLOCK / 8;

pub static SPEC: Spec = Spec {
    name: "Q5_K",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: RUN,
    has_min: true,
    dequantize,
    planes: None,
    raw_scales: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[SCALES..QH];
        let qh = &block[QH..QS];
        let qs = &block[QS..];

        for run in 0..RUNS {
            let (s, m) = scale_min(run, packed);
            let (scale, min) = (d * f32::from(s), dmin * f32::from(m));
            let plane = &qs[(run / 2) * RUN..][..RUN];
            let shift = 4 * (run as u32 % 2);
            let high = 1u8 << run;
            for ((y, &q), &h) in dst[run * RUN..][..RUN].iter_mut().zip(plane).zip(qh) {
                let q = ((q >> shift) & 0x0f) | if h & high != 0 { 16 } else { 0 };
                *y = scale * f32::from(q) - min;
            }
        }
    }
}
