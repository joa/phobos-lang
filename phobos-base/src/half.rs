/// Widen an IEEE binary16 bit pattern, subnormals and non-finites included.
///
/// Both front ends need this: GGUF stores block scales as halves and an ONNX
/// export can carry f16 initializers.
///
/// Inline because GGUF dequantization calls it once per block, so a call across
/// the crate boundary would show up in model load time.
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

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

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
}
