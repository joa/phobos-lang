// PTQ1_0, PrismML's ternary format: `{ uint8 qs[24]; uint8 qh[2]; f16 d; }`
// over 128 weights, each `d * (t - 1)` for a trit `t` in 0..2, 1.75 bits a
// weight. A byte holds five trits (`qh` four) in the TQ1_0 encoding: trit
// `n` of byte `b` is the top trit of `b * 3^n mod 256`, read as
// `(x * 3) >> 8`.
//
// The file's element order is a staging artifact: qs[0..16] carry elements
// `16 n + m`, qs[16..24] `80 + 8 n + m`, qh `120 + 2 n + h`. The registry
// block is a pair of them, 256 weights in 56 bytes, so every width here
// divides; the device holds the pair re-laid ([`device_block`]) so that each
// 64-weight quarter decodes on its own:
//
// - bytes `12 q .. 12 q + 12`: three words of quarter `q`; trit `n` of byte
//   `b` of word `w` is weight `4 (5 w + n) + b` of the quarter, so one trit
//   of a word is four consecutive weights, a `dp4a` operand;
// - byte `48 + q`: the quarter's last four weights, trit `n` weight `60 + n`;
// - bytes 52 and 54: the two file blocks' `d`, the first scaling quarters 0
//   and 1.

use phobos_base::half::f16_to_f32;

use super::{RawScales, Spec};

pub(crate) const FILE_BLOCK: usize = 128;
pub(crate) const FILE_BLOCK_BYTES: usize = 28;
const BLOCK: usize = 2 * FILE_BLOCK;
const BLOCK_BYTES: usize = 2 * FILE_BLOCK_BYTES;
const QH: usize = 24;
const D: usize = 26;

/// Where the device block keeps the quarters' tails and the two scales.
const DEV_TAILS: usize = 48;
const DEV_D: usize = 52;

pub static SPEC: Spec = Spec {
    name: "PTQ1_0",
    block: BLOCK,
    block_bytes: BLOCK_BYTES,
    scale_run: FILE_BLOCK,
    has_min: false,
    dequantize,
    planes: None,
    raw_scales: Some(raw_scales),
};

/// Trit `n` (0 the most significant) of a TQ1_0-encoded byte.
pub(crate) fn trit(byte: u8, n: usize) -> u8 {
    let q = byte.wrapping_mul(3u8.pow(n as u32));
    ((u16::from(q) * 3) >> 8) as u8
}

/// Five trits, the first most significant, as a byte [`trit`] reads back.
pub(crate) fn encode(trits: [u8; 5]) -> u8 {
    let v = trits.iter().fold(0u16, |acc, &t| acc * 3 + u16::from(t));
    (v * 256).div_ceil(243) as u8
}

/// A file block's 128 trits in element order.
fn file_trits(block: &[u8]) -> [u8; FILE_BLOCK] {
    let mut t = [0u8; FILE_BLOCK];
    for n in 0..5 {
        for m in 0..16 {
            t[16 * n + m] = trit(block[m], n);
        }
        for m in 0..8 {
            t[80 + 8 * n + m] = trit(block[16 + m], n);
        }
    }
    for n in 0..4 {
        for h in 0..2 {
            t[120 + 2 * n + h] = trit(block[QH + h], n);
        }
    }
    t
}

fn file_d(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[D], block[D + 1]]))
}

fn dequantize(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes
        .chunks_exact(FILE_BLOCK_BYTES)
        .zip(out.chunks_mut(FILE_BLOCK))
    {
        let d = file_d(block);
        for (y, &t) in dst.iter_mut().zip(&file_trits(block)) {
            *y = d * (f32::from(t) - 1.0);
        }
    }
}

/// The first file block's `d` of each pair: the plane every raw format
/// uploads. The kernels read both from the block itself.
fn raw_scales(bytes: &[u8], _k: usize, _n: usize) -> RawScales {
    let d = bytes
        .chunks_exact(BLOCK_BYTES)
        .map(|b| u16::from_le_bytes([b[D], b[D + 1]]))
        .collect();
    RawScales { d, dmin: Vec::new() }
}

/// A pair of file blocks re-laid for the device; see the module comment.
pub(crate) fn device_block(pair: &[u8], out: &mut [u8]) {
    let mut t = [0u8; BLOCK];
    t[..FILE_BLOCK].copy_from_slice(&file_trits(&pair[..FILE_BLOCK_BYTES]));
    t[FILE_BLOCK..].copy_from_slice(&file_trits(&pair[FILE_BLOCK_BYTES..]));
    for q in 0..4 {
        let quarter = &t[64 * q..][..64];
        for w in 0..3 {
            for b in 0..4 {
                let trits = std::array::from_fn(|n| quarter[4 * (5 * w + n) + b]);
                out[12 * q + 4 * w + b] = encode(trits);
            }
        }
        out[DEV_TAILS + q] = encode([quarter[60], quarter[61], quarter[62], quarter[63], 0]);
    }
    for h in 0..2 {
        let at = h * FILE_BLOCK_BYTES + D;
        out[DEV_D + 2 * h..][..2].copy_from_slice(&pair[at..at + 2]);
    }
}

/// A device block decoded, the host's reading of what the kernels see.
#[cfg(test)]
pub(crate) fn dequantize_device(block: &[u8], out: &mut [f32]) {
    for q in 0..4 {
        let at = DEV_D + 2 * (q / 2);
        let d = f16_to_f32(u16::from_le_bytes([block[at], block[at + 1]]));
        let dst = &mut out[64 * q..][..64];
        for w in 0..3 {
            for n in 0..5 {
                for b in 0..4 {
                    let t = trit(block[12 * q + 4 * w + b], n);
                    dst[4 * (5 * w + n) + b] = d * (f32::from(t) - 1.0);
                }
            }
        }
        for n in 0..4 {
            dst[60 + n] = d * (f32::from(trit(block[DEV_TAILS + q], n)) - 1.0);
        }
    }
}
