// Q6_K: `{ uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d; }`.
//
// A 256-element super-block of six-bit quants, four low bits in `ql` and two
// high ones in `qh`, symmetric around 32. The sixteen runs of 16 scale by a
// signed eight-bit index into the super-block's `d`, so unlike Q4_K there is
// no minimum to subtract.
//
// A Q4_K_M file is mostly Q4_K with its more sensitive tensors left here, so
// the two formats arrive together.

use phobos_base::half::f16_to_f32;

use super::Spec;

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 210;
/// Elements sharing one scale.
const RUN: usize = 16;
/// Elements whose quants are interleaved across one stretch of `ql` and `qh`.
const GROUP: usize = 128;
/// One byte of `qh` covers this many elements' worth of stride: the group is
/// decoded a quarter at a time.
const QUARTER: usize = GROUP / 4;

const QL: usize = 0;
const QH: usize = QL + BLOCK / 2;
const SCALES: usize = QH + BLOCK / 4;
const D: usize = SCALES + BLOCK / RUN;

pub static SPEC: Spec = Spec {
    name: "Q6_K",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: RUN,
    has_min: false,
    dequantize,
    planes: None,
};

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[D], block[D + 1]]));
        let scales = &block[SCALES..D];

        for (g, group) in dst.chunks_mut(GROUP).enumerate() {
            let ql = &block[QL + g * GROUP / 2..];
            let qh = &block[QH + g * GROUP / 4..];
            // One byte of `qh` carries the top two bits of four quants, one
            // from each quarter of the group; the low nibbles of those four sit
            // in two bytes of `ql`, one nibble each.
            for (l, &high) in qh[..QUARTER].iter().enumerate() {
                let sources = [
                    (l, 0, false),
                    (l + QUARTER, 2, false),
                    (l, 4, true),
                    (l + QUARTER, 6, true),
                ];
                for (quarter, (byte, shift, top)) in sources.into_iter().enumerate() {
                    let nibble = if top { ql[byte] >> 4 } else { ql[byte] & 0x0f };
                    let q = i32::from(nibble | (((high >> shift) & 3) << 4)) - 32;
                    let at = quarter * QUARTER + l;
                    let scale = scales[(g * GROUP + at) / RUN] as i8;
                    group[at] = d * f32::from(scale) * q as f32;
                }
            }
        }
    }
}
