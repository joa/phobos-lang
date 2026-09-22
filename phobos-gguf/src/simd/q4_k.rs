// Q4_K against Q8: eight runs of 32, a 6-bit scale and minimum apiece; the
// low nibbles of a 32-byte plane are one run and the high nibbles the next.

use super::{Block, Lanes, Quants};

const RUNS: usize = 8;
const RUN: usize = 32;
const SCALES: usize = 4;
pub(super) const QS: usize = SCALES + 12;

/// A block's header: `d`, `dmin`, the runs' scale indices, and the minimum
/// indices doubled up to one a Q8 sum of sixteen, for the multiply-add
/// that takes the minimums off.
#[derive(Default)]
pub(super) struct Header {
    pub(super) d: f32,
    pub(super) dmin: f32,
    pub(super) scales: [i16; RUNS],
    pub(super) mins: [i16; 2 * RUNS],
}

impl Header {
    /// The header of `bytes`, `d` and `dmin` through `half`.
    pub(super) fn read(&mut self, bytes: &[u8], half: impl Fn(u16) -> f32) {
        self.d = half(u16::from_le_bytes([bytes[0], bytes[1]]));
        self.dmin = half(u16::from_le_bytes([bytes[2], bytes[3]]));
        // The first four runs get a byte each for the scale and the
        // minimum, two bits spare; the last four take their low nibbles
        // from the last four bytes and their top bits from those spares.
        let p = &bytes[SCALES..QS];
        let mut set = |run: usize, s: u8, m: u8| {
            self.scales[run] = i16::from(s);
            self.mins[2 * run] = i16::from(m);
            self.mins[2 * run + 1] = i16::from(m);
        };
        for run in 0..4 {
            set(run, p[run] & 63, p[run + 4] & 63);
        }
        for run in 4..RUNS {
            set(run, (p[run + 4] & 0x0f) | ((p[run - 4] >> 6) << 4), (p[run + 4] >> 4) | ((p[run] >> 6) << 4));
        }
    }

}

/// The dot without vectors: each run's scaled dot less its minimum times
/// its Q8 sums, times the run's activation scale, into the first lane.
pub(super) fn dot_plain(h: &Header, q: &Quants, qs: &[i8], sums: &[i16], da: &[f32], acc: &mut Lanes) {
    for run in 0..RUNS {
        let (w, a) = (&q.0[run * RUN..][..RUN], &qs[run * RUN..][..RUN]);
        let dot = w.iter().zip(a).map(|(&w, &a)| i32::from(w) * i32::from(a)).sum::<i32>();
        let min = i32::from(sums[2 * run]) + i32::from(sums[2 * run + 1]);
        acc.0[0] += da[run] * (h.d * f32::from(h.scales[run]) * dot as f32 - h.dmin * f32::from(h.mins[2 * run]) * min as f32);
    }
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

    unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
        u.header.read(bytes, phobos_base::half::f16_to_f32);
        for run in 0..RUNS {
            let plane = &bytes[QS + (run / 2) * RUN..][..RUN];
            let shift = 4 * (run as u32 % 2);
            for (q, &b) in u.q.0[run * RUN..][..RUN].iter_mut().zip(plane) {
                *q = (b >> shift) & 0x0f;
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
    use super::{Block, Lanes, QS, Unpacked};

    pub(crate) struct Avx2;

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
        u.header.read(bytes, |bits| half(bits));
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
