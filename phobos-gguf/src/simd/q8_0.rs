// Q8_0 against Q8: eight runs of 32, each a signed byte per element under
// one `f16` scale, with no minimum. A unit is eight file blocks. The file
// leads each block with its scale; the device layout leads the unit with all
// eight (`quant::q8_0_units`), which `d` being present tells apart.

use super::{Acc, BLOCK, Block, Q8Block, RUN, RUNS};

/// Bytes of one file block, and of a unit in either layout.
const FILE_BLOCK: usize = 2 + RUN;
const SCALES: usize = 2 * RUNS;

/// A unit's quants as stored, signed, aligned for vector loads.
#[repr(C, align(32))]
pub(super) struct Signed([i8; BLOCK]);

impl Default for Signed {
    fn default() -> Signed {
        Signed([0; BLOCK])
    }
}

#[derive(Default)]
pub(super) struct Unpacked {
    d: [f32; RUNS],
    q: Signed,
}

impl Unpacked {
    /// Reads a unit in the device layout when `grouped`, else in the
    /// file's, converting each scale with `half`.
    fn read(&mut self, bytes: &[u8], grouped: bool, half: impl Fn(u16) -> f32) {
        for run in 0..RUNS {
            let (scale, quants) = if grouped { (2 * run, SCALES + run * RUN) } else { (run * FILE_BLOCK, run * FILE_BLOCK + 2) };
            self.d[run] = half(u16::from_le_bytes([bytes[scale], bytes[scale + 1]]));
            for (q, &b) in self.q.0[run * RUN..][..RUN].iter_mut().zip(&bytes[quants..quants + RUN]) {
                *q = b as i8;
            }
        }
    }
}

pub(super) struct Scalar;

impl Block for Scalar {
    const AVX2: bool = false;
    type Unpacked = Unpacked;

    unsafe fn unpack(bytes: &[u8], d: Option<u16>, u: &mut Unpacked) {
        u.read(bytes, d.is_some(), phobos_base::half::f16_to_f32);
    }

    unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        for (run, (w, a)) in u.q.0.chunks_exact(RUN).zip(block.qs.chunks_exact(RUN)).enumerate() {
            let dot: i32 = w.iter().zip(a).map(|(&w, &a)| i32::from(w) * i32::from(a)).sum();
            acc.0[0] += block.d[run] * u.d[run] * dot as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::super::x86::half;
    use super::super::{Acc, Q8Block, RUNS, avx2_block};
    use super::Unpacked;

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn unpack(bytes: &[u8], d: Option<u16>, u: &mut Unpacked) {
        // A closure, since a `#[target_feature]` function is not `Fn`.
        u.read(bytes, d.is_some(), |bits| half(bits));
    }

    /// Per run: the signed dot as `maddubs` of the weight's magnitudes with
    /// the activation carrying the weight's signs, which cannot saturate at
    /// two products of at most 127 by 128, summed to eight lanes and scaled.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot(u: &Unpacked, block: &Q8Block, acc: &mut Acc) {
        // SAFETY: the caller has AVX2 and FMA; `q` and `acc` are aligned.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut sum = _mm256_load_ps(acc.0.as_ptr());
            for run in 0..RUNS {
                let w = _mm256_load_si256(u.q.0.as_ptr().add(run * 32).cast());
                let a = _mm256_loadu_si256(block.qs.as_ptr().add(run * 32).cast());
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w)), ones);
                sum = _mm256_fmadd_ps(_mm256_set1_ps(u.d[run] * block.d[run]), _mm256_cvtepi32_ps(p), sum);
            }
            _mm256_store_ps(acc.0.as_mut_ptr(), sum);
        }
    }

    avx2_block!(Avx2, Unpacked, unpack, dot);
}
