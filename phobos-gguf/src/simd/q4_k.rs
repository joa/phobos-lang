// Q4_K against Q8: eight runs of 32, a 6-bit scale and minimum apiece; the
// low nibbles of a 32-byte plane are one run and the high nibbles the next.

use phobos_base::half::f16_to_f32;

use super::{Block, Quants};
use crate::quant::q4_k::scale_min;

const RUNS: usize = 8;
const RUN: usize = 32;
const SCALES: usize = 4;
pub(super) const QS: usize = SCALES + 12;

/// A block's header: `d`, `dmin` and the runs' scale and minimum indices.
#[derive(Default)]
pub(super) struct Header {
    pub(super) d: f32,
    pub(super) dmin: f32,
    pub(super) scales: [i16; RUNS],
    pub(super) mins: [i16; RUNS],
}

impl Header {
    pub(super) fn read(&mut self, bytes: &[u8]) {
        self.d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        self.dmin = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
        for run in 0..RUNS {
            let (s, m) = scale_min(run, &bytes[SCALES..QS]);
            self.scales[run] = i16::from(s);
            self.mins[run] = i16::from(m);
        }
    }

    /// The minimums' term: each run's minimum times its Q8 sums.
    pub(super) fn min_term(&self, sums: &[i16]) -> i32 {
        self.mins
            .iter()
            .zip(sums.chunks_exact(2))
            .map(|(&m, s)| i32::from(m) * (i32::from(s[0]) + i32::from(s[1])))
            .sum()
    }
}

/// The run-scaled dot of unpacked quants with a Q8 block, without vectors.
pub(super) fn dot_runs32(q: &Quants, scales: &[i16; RUNS], qs: &[i8]) -> i32 {
    scales
        .iter()
        .zip(q.0.chunks_exact(RUN).zip(qs.chunks_exact(RUN)))
        .map(|(&scale, (w, a))| i32::from(scale) * w.iter().zip(a).map(|(&w, &a)| i32::from(w) * i32::from(a)).sum::<i32>())
        .sum()
}

#[derive(Default)]
pub(crate) struct Unpacked {
    pub(super) header: Header,
    pub(super) q: Quants,
}

pub(super) struct Scalar;

impl Block for Scalar {
    const AVX2: bool = false;
    type Unpacked = Unpacked;

    unsafe fn unpack(bytes: &[u8], u: &mut Unpacked) {
        u.header.read(bytes);
        for run in 0..RUNS {
            let plane = &bytes[QS + (run / 2) * RUN..][..RUN];
            let shift = 4 * (run as u32 % 2);
            for (q, &b) in u.q.0[run * RUN..][..RUN].iter_mut().zip(plane) {
                *q = (b >> shift) & 0x0f;
            }
        }
    }

    fn scales(u: &Unpacked) -> (f32, f32) {
        (u.header.d, u.header.dmin)
    }

    unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16]) -> (i32, i32) {
        (dot_runs32(&u.q, &u.header.scales, qs), u.header.min_term(sums))
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::dot_runs32;
    use super::{Block, QS, Unpacked};

    pub(crate) struct Avx2;

    #[target_feature(enable = "avx2")]
    unsafe fn unpack(bytes: &[u8], u: &mut Unpacked) {
        u.header.read(bytes);
        // SAFETY: the caller has AVX2; a block holds the four planes.
        unsafe {
            let m4 = _mm256_set1_epi8(0x0f);
            let q = u.q.0.as_mut_ptr();
            for j in 0..4 {
                let v = _mm256_loadu_si256(bytes.as_ptr().add(QS + 32 * j).cast());
                _mm256_store_si256(q.add(64 * j).cast(), _mm256_and_si256(v, m4));
                _mm256_store_si256(q.add(64 * j + 32).cast(), _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
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
            (u.header.d, u.header.dmin)
        }

        #[inline(always)]
        unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16]) -> (i32, i32) {
            // SAFETY: the caller has AVX2.
            (unsafe { dot_runs32(&u.q, &u.header.scales, qs) }, u.header.min_term(sums))
        }
    }
}
