use anyhow::{Result, bail, ensure};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
}

impl GgmlType {
    pub fn from_code(code: u32) -> Result<GgmlType> {
        Ok(match code {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::IQ2_XXS,
            17 => GgmlType::IQ2_XS,
            18 => GgmlType::IQ3_XXS,
            19 => GgmlType::IQ1_S,
            20 => GgmlType::IQ4_NL,
            21 => GgmlType::IQ3_S,
            22 => GgmlType::IQ2_S,
            23 => GgmlType::IQ4_XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1_M,
            30 => GgmlType::BF16,
            other => bail!("unknown or retired ggml tensor type {other}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q8_1 => "Q8_1",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q3_K => "Q3_K",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::Q8_K => "Q8_K",
            GgmlType::IQ2_XXS => "IQ2_XXS",
            GgmlType::IQ2_XS => "IQ2_XS",
            GgmlType::IQ3_XXS => "IQ3_XXS",
            GgmlType::IQ1_S => "IQ1_S",
            GgmlType::IQ4_NL => "IQ4_NL",
            GgmlType::IQ3_S => "IQ3_S",
            GgmlType::IQ2_S => "IQ2_S",
            GgmlType::IQ4_XS => "IQ4_XS",
            GgmlType::I8 => "I8",
            GgmlType::I16 => "I16",
            GgmlType::I32 => "I32",
            GgmlType::I64 => "I64",
            GgmlType::F64 => "F64",
            GgmlType::IQ1_M => "IQ1_M",
            GgmlType::BF16 => "BF16",
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::F64
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::IQ4_NL => 32,
            _ => 256,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 => 2,
            GgmlType::I8 => 1,
            GgmlType::F64 | GgmlType::I64 => 8,
            GgmlType::Q4_0 | GgmlType::IQ4_NL => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2_K => 84,
            GgmlType::Q3_K => 110,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            GgmlType::Q8_K => 292,
            GgmlType::IQ2_XXS => 66,
            GgmlType::IQ2_XS => 74,
            GgmlType::IQ3_XXS => 98,
            GgmlType::IQ1_S => 50,
            GgmlType::IQ3_S => 110,
            GgmlType::IQ2_S => 82,
            GgmlType::IQ4_XS => 136,
            GgmlType::IQ1_M => 56,
        }
    }

    pub fn is_dequantizable(self) -> bool {
        matches!(
            self,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::F64
                | GgmlType::I8
                | GgmlType::I16
                | GgmlType::I32
                | GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
        )
    }

    pub fn storage_bytes(self, numel: usize) -> Result<usize> {
        let block = self.block_size();
        ensure!(
            numel.is_multiple_of(block),
            "{} tensor has {numel} elements, not a multiple of the {block}-element block",
            self.name()
        );
        Ok(numel / block * self.type_size())
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Extents in ggml order, fastest-varying axis first. A `[1024, 248320]`
    /// embedding is 248320 rows of 1024 elements, the reverse of the row-major
    /// shape (see [`TensorInfo::row_major_dims`]).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the tensor data section, not the file.
    pub offset_bytes: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Extents outermost-first, the order row-major consumers expect.
    pub fn row_major_dims(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        let numel = usize::try_from(self.numel())
            .map_err(|_| anyhow::anyhow!("tensor '{}' element count exceeds usize", self.name))?;
        self.ggml_type.storage_bytes(numel)
    }
}

pub fn dequantize_into(ggml_type: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = ggml_type.storage_bytes(out.len())?;
    ensure!(
        bytes.len() == expected,
        "{} needs {expected} bytes for {} elements, got {}",
        ggml_type.name(),
        out.len(),
        bytes.len()
    );

    match ggml_type {
        GgmlType::F32 => copy_scalars(bytes, out, 4, |b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }),
        GgmlType::F64 => copy_scalars(bytes, out, 8, |b| {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }),
        GgmlType::F16 => copy_scalars(bytes, out, 2, |b| {
            f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
        }),
        GgmlType::BF16 => copy_scalars(bytes, out, 2, |b| {
            f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
        }),
        GgmlType::I8 => copy_scalars(bytes, out, 1, |b| b[0] as i8 as f32),
        GgmlType::I16 => copy_scalars(bytes, out, 2, |b| i16::from_le_bytes([b[0], b[1]]) as f32),
        GgmlType::I32 => copy_scalars(bytes, out, 4, |b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32
        }),
        GgmlType::Q4_0 => dequantize_q4_0(bytes, out),
        GgmlType::Q4_1 => dequantize_q4_1(bytes, out),
        GgmlType::Q5_0 => dequantize_q5_0(bytes, out),
        GgmlType::Q5_1 => dequantize_q5_1(bytes, out),
        GgmlType::Q8_0 => dequantize_q8_0(bytes, out),
        GgmlType::Q8_1 => dequantize_q8_1(bytes, out),
        other => bail!(
            "dequantizing {} is not implemented; re-quantize the model to Q8_0 or a legacy Q4/Q5 type",
            other.name()
        ),
    }

    Ok(())
}

fn copy_scalars(bytes: &[u8], out: &mut [f32], width: usize, decode: fn(&[u8]) -> f32) {
    for (dst, src) in out.iter_mut().zip(bytes.chunks_exact(width)) {
        *dst = decode(src);
    }
}

fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) {
    // `{ f16 d; int8 qs[32]; }`: every element is `d * qs[i]`.
    for (block, dst) in bytes.chunks_exact(34).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (y, &q) in dst.iter_mut().zip(&block[2..]) {
            *y = d * (q as i8) as f32;
        }
    }
}

pub const Q8_0_BLOCK: usize = 32;

/// The signed bytes and per-block scales of a Q8_0 tensor, without decoding.
///
/// Returns `(qs, scales)` in storage order, so element `i` dequantizes to
/// `qs[i] as f32 * scales[i / Q8_0_BLOCK]`. Keeping the two apart lets a backend
/// upload a quarter of the bytes and do the multiply in the kernel.
pub fn q8_0_blocks(bytes: &[u8], numel: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    ensure!(
        numel.is_multiple_of(Q8_0_BLOCK),
        "a Q8_0 tensor has {numel} elements, which is not a multiple of {Q8_0_BLOCK}"
    );
    let blocks = numel / Q8_0_BLOCK;
    ensure!(
        bytes.len() >= blocks * 34,
        "a Q8_0 tensor of {numel} elements needs {} bytes, got {}",
        blocks * 34,
        bytes.len()
    );
    let mut qs = vec![0i8; numel];
    let mut scales = vec![0.0f32; blocks];
    for (b, block) in bytes.chunks_exact(34).take(blocks).enumerate() {
        scales[b] = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (dst, &q) in qs[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK]
            .iter_mut()
            .zip(&block[2..])
        {
            *dst = q as i8;
        }
    }
    Ok((qs, scales))
}

/// `{ f16 d; f16 s; int8 qs[32]; }`: `s` is the block sum, unused when decoding.
fn dequantize_q8_1(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(36).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (y, &q) in dst.iter_mut().zip(&block[4..]) {
            *y = d * (q as i8) as f32;
        }
    }
}

/// `{ f16 d; uint8 qs[16]; }`. The two nibbles of `qs[j]` are elements `j` and
/// `j + 16`, not `2j` and `2j + 1`.
fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(18).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (j, &q) in block[2..].iter().enumerate() {
            dst[j] = d * (i32::from(q & 0x0f) - 8) as f32;
            dst[j + 16] = d * (i32::from(q >> 4) - 8) as f32;
        }
    }
}

/// `{ f16 d; f16 m; uint8 qs[16]; }`: quants are unsigned with an offset `m`.
fn dequantize_q4_1(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(20).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        for (j, &q) in block[4..].iter().enumerate() {
            dst[j] = d * f32::from(q & 0x0f) + m;
            dst[j + 16] = d * f32::from(q >> 4) + m;
        }
    }
}

/// `{ f16 d; uint8 qh[4]; uint8 qs[16]; }`: `qh` supplies a fifth bit per quant,
/// bit `j` for the low nibble and bit `j + 16` for the high one.
fn dequantize_q5_0(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(22).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for (j, &q) in block[6..].iter().enumerate() {
            let hi_lo = ((qh >> j) << 4) & 0x10;
            let hi_hi = (qh >> (j + 12)) & 0x10;
            dst[j] = d * ((i32::from(q & 0x0f) | hi_lo as i32) - 16) as f32;
            dst[j + 16] = d * ((i32::from(q >> 4) | hi_hi as i32) - 16) as f32;
        }
    }
}

/// `{ f16 d; f16 m; uint8 qh[4]; uint8 qs[16]; }`: Q5_0's fifth bit with Q4_1's
/// offset.
fn dequantize_q5_1(bytes: &[u8], out: &mut [f32]) {
    for (block, dst) in bytes.chunks_exact(24).zip(out.chunks_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for (j, &q) in block[8..].iter().enumerate() {
            let hi_lo = ((qh >> j) << 4) & 0x10;
            let hi_hi = (qh >> (j + 12)) & 0x10;
            dst[j] = d * (u32::from(q & 0x0f) | hi_lo) as f32 + m;
            dst[j + 16] = d * (u32::from(q >> 4) | hi_hi) as f32 + m;
        }
    }
}

pub fn f16_to_f32(bits: u16) -> f32 {
    // Widen an IEEE binary16 bit pattern, subnormals and non-finites included.
    
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
    use super::*;

    #[test]
    fn widens_half_precision() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        // Largest normal and smallest positive subnormal.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn decodes_q8_0_blocks() {
        // One block: scale 0.5, quants 0, 1, -1, 2, then zeros.
        let mut block = vec![0x00, 0x38]; // f16 0.5
        block.extend([0i8, 1, -1, 2].iter().map(|&q| q as u8));
        block.extend(std::iter::repeat_n(0u8, 28));

        let mut out = vec![0.0; 32];
        dequantize_into(GgmlType::Q8_0, &block, &mut out).unwrap();
        assert_eq!(&out[..4], &[0.0, 0.5, -0.5, 1.0]);
        assert!(out[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn decodes_q4_0_nibble_halves() {
        // Scale 1.0; qs[0] = 0x9A puts quant 10 at element 0 and 9 at element 16.
        let mut block = vec![0x00, 0x3c];
        block.push(0x9a);
        block.extend(std::iter::repeat_n(0x88u8, 15));

        let mut out = vec![0.0; 32];
        dequantize_into(GgmlType::Q4_0, &block, &mut out).unwrap();
        assert_eq!(out[0], 2.0); // 0xA - 8
        assert_eq!(out[16], 1.0); // 0x9 - 8
        // 0x88 decodes to zero in both halves.
        assert!(out[1..16].iter().all(|&v| v == 0.0));
        assert!(out[17..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rejects_short_buffers_and_unsupported_types() {
        let mut out = vec![0.0; 32];
        assert!(dequantize_into(GgmlType::Q8_0, &[0; 33], &mut out).is_err());
        assert!(dequantize_into(GgmlType::Q4_K, &[0; 144], &mut vec![0.0; 256]).is_err());
    }

    #[test]
    fn reports_block_alignment() {
        // 20 elements cannot fill whole 32-element Q8_0 blocks.
        assert!(GgmlType::Q8_0.storage_bytes(20).is_err());
        assert_eq!(GgmlType::Q8_0.storage_bytes(64).unwrap(), 68);
        assert_eq!(GgmlType::F32.storage_bytes(20).unwrap(), 80);
    }
}
