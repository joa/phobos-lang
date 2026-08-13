/// Widen an IEEE binary16 bit pattern, subnormals and non-finites included.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);

    let widened = match exp {
        0 if mant == 0 => sign,
        // Subnormal: renormalize so the leading one moves into the implicit bit.
        0 => {
            let lz = mant.leading_zeros();
            sign | ((134 - lz) << 23) | ((mant << (lz - 8)) & 0x007f_ffff)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(widened)
}

/// Narrow to an IEEE binary16 bit pattern, rounding to nearest with ties to
/// even, which is what the hardware's `cvt.rn.f16.f32` does. A magnitude past
/// the format's range becomes an infinity and one below its subnormals becomes
/// a zero of the same sign.
///
/// The key and value caches are held this way, so the host reference rounds
/// through the same conversion the device kernel does and the two stay
/// comparable.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        let payload = if mant == 0 {
            0
        } else {
            0x0200 | (mant >> 13) as u16
        };
        return sign | 0x7c00 | payload;
    }

    let shifted = exp - 127 + 15;
    if shifted >= 0x1f {
        return sign | 0x7c00;
    }

    if shifted < -10 {
        return sign;
    }

    let (significand, dropped) = if shifted > 0 {
        (mant | (shifted as u32) << 23, 13)
    } else {
        (mant | 0x0080_0000, (14 - shifted) as u32)
    };

    let kept = (significand >> dropped) as u16;
    let round = significand >> (dropped - 1) & 1;
    let sticky = significand & ((1 << (dropped - 1)) - 1);
    sign | (kept + u16::from(round == 1 && (sticky != 0 || kept & 1 == 1)))
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    #[test]
    fn widens_half_precision() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        // The largest finite half, and the subnormal run below the smallest.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn narrows_to_half_precision() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
        // Past the range at either end: an infinity, and a signed zero.
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(1.0e20), 0x7c00);
        assert_eq!(f32_to_f16(-1.0e-20), 0x8000);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());

        // Halves step by 2^-10 either side of 1.0, so both midpoints below and
        // above 0x3c01 land on the neighbour with an even mantissa.
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
        // A tie that carries all the way out of the mantissa.
        assert_eq!(f32_to_f16(2047.5), 0x6800);
    }

    #[test]
    fn round_trips_every_half() {
        for bits in 0..=u16::MAX {
            let widened = f16_to_f32(bits);
            if widened.is_nan() {
                continue;
            }
            assert_eq!(
                f32_to_f16(widened),
                bits,
                "half {bits:#06x} did not survive"
            );
        }
    }
}
