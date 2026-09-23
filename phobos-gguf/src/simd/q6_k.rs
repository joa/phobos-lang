// Q6_K against Q8: sixteen runs of 16 with a signed 8-bit scale apiece and
// no minimum, the quants six bits around 32. The offset comes off through
// the Q8 sums, so the quants are dotted as they are stored, unsigned.

use super::{Acc, Block, Q8Block, Quants, SUMS, SUM_RUN};

/// Elements whose quants interleave across one stretch of `ql` and `qh`.
const GROUP: usize = 128;
const QUARTER: usize = GROUP / 4;
const OFFSET: f32 = 32.0;

const QH: usize = 128;
const SCALES: usize = QH + 64;
const D: usize = SCALES + SUMS;

#[derive(Default)]
pub(super) struct Unpacked {
    d: f32,
    scales: [i16; SUMS],
    q: Quants,
}

impl Unpacked {
    /// The header: the scale from `d` where the layout keeps it apart,
    /// else the block's trailing half.
    fn read_header(&mut self, bytes: &[u8], d: Option<u16>, half: impl Fn(u16) -> f32) {
        self.d = half(d.unwrap_or_else(|| u16::from_le_bytes([bytes[D], bytes[D + 1]])));
        for (scale, &b) in self.scales.iter_mut().zip(&bytes[SCALES..D]) {
            *scale = i16::from(b as i8);
        }
    }
}

pub(super) struct Scalar;

impl Block for Scalar {
    const AVX2: bool = false;
    type Unpacked = Unpacked;

    unsafe fn unpack(bytes: &[u8], d: Option<u16>, u: &mut Unpacked) {
        u.read_header(bytes, d, phobos_base::half::f16_to_f32);
        for g in 0..2 {
            let ql = &bytes[g * GROUP / 2..];
            let qh = &bytes[QH + g * GROUP / 4..];
            let group = &mut u.q.0[g * GROUP..][..GROUP];
            for (l, &high) in qh[..QUARTER].iter().enumerate() {
                let sources = [(l, 0, false), (l + QUARTER, 2, false), (l, 4, true), (l + QUARTER, 6, true)];
                for (quarter, (byte, shift, top)) in sources.into_iter().enumerate() {
                    let nibble = if top { ql[byte] >> 4 } else { ql[byte] & 0x0f };
                    group[quarter * QUARTER + l] = nibble | (((high >> shift) & 3) << 4);
                }
            }
        }
    }

    unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        // A run of 16 at a time, its scaled dot less the offset times its
        // Q8 sum, scaled by its activation run's scale.
        for (i, (&scale, (w, a))) in u.scales.iter().zip(u.q.0.chunks_exact(SUM_RUN).zip(block.qs.chunks_exact(SUM_RUN))).enumerate() {
            let dot = super::dot_i32(w, a);
            acc.0[0] += block.d[i / 2] * u.d * f32::from(scale) * (dot as f32 - OFFSET * f32::from(block.sums[i]));
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::{dot_runs16, half, min_term};
    use super::super::{Acc, Q8Block, avx2_block};
    use super::{GROUP, OFFSET, QH, Unpacked};

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn unpack(bytes: &[u8], d: Option<u16>, u: &mut Unpacked) {
        // A closure, since a `#[target_feature]` function is not `Fn`.
        u.read_header(bytes, d, |bits| half(bits));
        // SAFETY: the caller has AVX2; a block holds both groups.
        unsafe {
            let m4 = _mm256_set1_epi8(0x0f);
            let m3 = _mm256_set1_epi8(3);
            for g in 0..2 {
                let ql = bytes.as_ptr().add(g * GROUP / 2);
                let low = [_mm256_loadu_si256(ql.cast()), _mm256_loadu_si256(ql.add(32).cast())];
                let high = _mm256_loadu_si256(bytes.as_ptr().add(QH + g * GROUP / 4).cast());
                // Quarter `i` takes its nibble from plane `i % 2`, the top
                // nibble for the last two, and its high bits from bit pair
                // `i` of the shared high plane.
                let nibbles = [low[0], low[1], _mm256_srli_epi16(low[0], 4), _mm256_srli_epi16(low[1], 4)];
                let highs = [high, _mm256_srli_epi16(high, 2), _mm256_srli_epi16(high, 4), _mm256_srli_epi16(high, 6)];
                for (i, (nibble, high)) in nibbles.into_iter().zip(highs).enumerate() {
                    let q = _mm256_or_si256(_mm256_and_si256(nibble, m4), _mm256_slli_epi16(_mm256_and_si256(high, m3), 4));
                    _mm256_store_si256(u.q.0.as_mut_ptr().add(g * GROUP + i * 32).cast(), q);
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        // SAFETY: the caller has AVX2 and FMA.
        unsafe {
            dot_runs16(&u.q, &u.scales, block.qs, u.d, block.d, acc);
            min_term(&u.scales, block.sums, u.d * OFFSET, block.d, acc);
        }
    }

    avx2_block!(Avx2, Unpacked, unpack, dot);
}
