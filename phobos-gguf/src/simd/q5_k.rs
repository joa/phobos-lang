// Q5_K against Q8: Q4_K's runs, scales and nibble planes with a fifth bit,
// run `r`'s element `l` taking it from bit `r` of `qh[l]`.

use super::q4_k::{Unpacked, dot_plain};
use super::{Block, Lanes};

const RUNS: usize = 8;
const RUN: usize = 32;
const QH: usize = 16;
pub(super) const QS: usize = QH + 32;

pub(super) struct Scalar;

impl Block for Scalar {
    const AVX2: bool = false;
    type Unpacked = Unpacked;

    unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
        u.header.read(bytes, phobos_base::half::f16_to_f32);
        let qh = &bytes[QH..QS];
        for run in 0..RUNS {
            let plane = &bytes[QS + (run / 2) * RUN..][..RUN];
            let shift = 4 * (run as u32 % 2);
            let high = 1u8 << run;
            for ((q, &b), &h) in u.q.0[run * RUN..][..RUN].iter_mut().zip(plane).zip(qh) {
                *q = ((b >> shift) & 0x0f) | if h & high != 0 { 16 } else { 0 };
            }
        }
    }

    unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16], da: &[f32], acc: &mut Lanes) {
        dot_plain(&u.header, &u.q, qs, sums, da, acc);
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::{dot_runs32, half, min_term};
    use super::{Block, Lanes, QH, QS, RUNS, Unpacked};

    pub(crate) struct Avx2;

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
        u.header.read(bytes, |bits| half(bits));
        // SAFETY: the caller has AVX2; a block holds the planes.
        unsafe {
            let m4 = _mm256_set1_epi8(0x0f);
            let m1 = _mm256_set1_epi8(1);
            let q = u.q.0.as_mut_ptr();
            // Each run takes bit zero of the high plane and shifts it down
            // for the next; a byte's neighbour only ever reaches its top
            // bits, which the mask drops.
            let mut high = _mm256_loadu_si256(bytes.as_ptr().add(QH).cast());
            for run in 0..RUNS {
                let v = _mm256_loadu_si256(bytes.as_ptr().add(QS + (run / 2) * 32).cast());
                let low = if run % 2 == 0 { v } else { _mm256_srli_epi16(v, 4) };
                let fifth = _mm256_slli_epi16(_mm256_and_si256(high, m1), 4);
                _mm256_store_si256(q.add(run * 32).cast(), _mm256_or_si256(_mm256_and_si256(low, m4), fifth));
                high = _mm256_srli_epi16(high, 1);
            }
        }
    }

    impl Block for Avx2 {
        const AVX2: bool = true;
        type Unpacked = Unpacked;

        #[inline(always)]
        unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
            // SAFETY: the caller has AVX2.
            unsafe { unpack(bytes, _d, u) }
        }

        #[inline(always)]
        unsafe fn dot(u: &Unpacked, qs: &[i8], sums: &[i16], da: &[f32], acc: &mut Lanes) {
            // SAFETY: the caller has AVX2 and FMA.
            unsafe {
                dot_runs32(&u.q, &u.header.scales, qs, u.header.d, da, acc);
                min_term(&u.header.mins, sums, u.header.dmin, da, acc);
            }
        }
    }
}
