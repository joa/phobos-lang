// Q6_K against Q8: sixteen runs of 16 with a signed 8-bit scale apiece and
// no minimum, the quants six bits around 32. The offset comes off through
// the Q8 sums, so the quants are dotted as they are stored, unsigned.

use phobos_base::half::f16_to_f32;

use super::{Block, Quants};

const RUNS: usize = 16;
/// Elements whose quants interleave across one stretch of `ql` and `qh`.
const GROUP: usize = 128;
const QUARTER: usize = GROUP / 4;
const OFFSET: i32 = 32;

const QL: usize = 0;
const QH: usize = QL + 128;
const SCALES: usize = QH + 64;
const D: usize = SCALES + RUNS;

#[derive(Default)]
pub(crate) struct Unpacked {
    d: f32,
    scales: [i16; RUNS],
    q: Quants,
}

impl Unpacked {
    fn read_header(&mut self, bytes: &[u8]) {
        self.d = f16_to_f32(u16::from_le_bytes([bytes[D], bytes[D + 1]]));
        for (scale, &b) in self.scales.iter_mut().zip(&bytes[SCALES..D]) {
            *scale = i16::from(b as i8);
        }
    }

    /// The offset's term: each run's scale times its Q8 sum, times 32.
    fn offset_term(&self, sums: &[i16]) -> i32 {
        OFFSET * self.scales.iter().zip(sums).map(|(&s, &sum)| i32::from(s) * i32::from(sum)).sum::<i32>()
    }
}

pub(super) struct Scalar;

impl Block for Scalar {
    const AVX2: bool = false;
    type Unpacked = Unpacked;

    unsafe fn unpack(bytes: &[u8], u: &mut Unpacked) {
        u.read_header(bytes);
        for g in 0..2 {
            let ql = &bytes[QL + g * GROUP / 2..];
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

    fn scales(u: &Unpacked) -> (f32, f32) {
        (u.d, u.d)
    }

    unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16]) -> (i32, i32) {
        let sumi = u
            .scales
            .iter()
            .zip(u.q.0.chunks_exact(16).zip(qs.chunks_exact(16)))
            .map(|(&scale, (w, a))| i32::from(scale) * w.iter().zip(a).map(|(&w, &a)| i32::from(w) * i32::from(a)).sum::<i32>())
            .sum();
        (sumi, u.offset_term(sums))
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::dot_runs16;
    use super::{Block, GROUP, QH, QL, Unpacked};

    pub(crate) struct Avx2;

    #[target_feature(enable = "avx2")]
    unsafe fn unpack(bytes: &[u8], u: &mut Unpacked) {
        u.read_header(bytes);
        // SAFETY: the caller has AVX2; a block holds both groups.
        unsafe {
            let m4 = _mm256_set1_epi8(0x0f);
            let m3 = _mm256_set1_epi8(3);
            for g in 0..2 {
                let ql = bytes.as_ptr().add(QL + g * GROUP / 2);
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

    impl Block for Avx2 {
        const AVX2: bool = true;
        type Unpacked = Unpacked;

        #[inline(always)]
        unsafe fn unpack(bytes: &[u8], u: &mut Unpacked) {
            // SAFETY: the caller has AVX2.
            unsafe { unpack(bytes, u) }
        }

        fn scales(u: &Unpacked) -> (f32, f32) {
            (u.d, u.d)
        }

        #[inline(always)]
        unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16]) -> (i32, i32) {
            // SAFETY: the caller has AVX2.
            (unsafe { dot_runs16(&u.q, &u.scales, qs) }, u.offset_term(sums))
        }
    }
}
