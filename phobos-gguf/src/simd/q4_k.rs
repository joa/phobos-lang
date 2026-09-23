// Q4_K against Q8: eight runs of 32, a 6-bit scale and minimum apiece; the
// low nibbles of a 32-byte plane are one run and the high nibbles the next.

use super::{Acc, Block, Q8Block, Quants, RUN, RUNS, SUMS};

const SCALES: usize = 4;
const QS: usize = SCALES + 12;

/// A block's header: `d`, `dmin`, the runs' scale indices, and the minimum
/// indices doubled up to one a Q8 sum of sixteen, for the multiply-add
/// that takes the minimums off.
#[derive(Default)]
pub(super) struct Header {
    pub(super) d: f32,
    pub(super) dmin: f32,
    pub(super) scales: [i16; RUNS],
    pub(super) mins: [i16; SUMS],
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

#[derive(Default)]
pub(super) struct Unpacked {
    pub(super) header: Header,
    pub(super) q: Quants,
}

/// The dot without vectors: each run's scaled dot less its minimum times
/// its Q8 sums, times the run's activation scale, into the first lane.
pub(super) fn dot_plain(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
    let h = &u.header;
    for run in 0..RUNS {
        let (w, a) = (&u.q.0[run * RUN..][..RUN], &block.qs[run * RUN..][..RUN]);
        let dot = super::dot_i32(w, a);
        let min = i32::from(block.sums[2 * run]) + i32::from(block.sums[2 * run + 1]);
        acc.0[0] += block.d[run] * (h.d * f32::from(h.scales[run]) * dot as f32 - h.dmin * f32::from(h.mins[2 * run]) * min as f32);
    }
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

    unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        dot_plain(u, block, acc);
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::{dot_runs32, half, min_term};
    use super::super::{Acc, Q8Block, avx2_block};
    use super::{QS, Unpacked};

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn unpack(bytes: &[u8], _d: Option<u16>, u: &mut Unpacked) {
        // A closure, since a `#[target_feature]` function is not `Fn`.
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

    /// The dot for Q4_K's and Q5_K's unpacked quants alike.
    #[target_feature(enable = "avx2,fma")]
    pub(in crate::simd) unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        let h = &u.header;
        // SAFETY: the caller has AVX2 and FMA.
        unsafe {
            dot_runs32(&u.q, &h.scales, block.qs, h.d, block.d, acc);
            min_term(&h.mins, block.sums, h.dmin, block.d, acc);
        }
    }

    avx2_block!(Avx2, Unpacked, unpack, dot);
}
