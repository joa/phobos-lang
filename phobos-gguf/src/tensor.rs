use anyhow::{Context, Result, bail, ensure};

/// Block scales are stored as halves throughout the GGUF quantizations.
pub use phobos_base::half::f16_to_f32;

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
        self.scalar_decode().is_some() || self.quant().is_some()
    }

    /// How to widen one element of a type that is not blocked at all, which
    /// are the types [`dequantize_into`] handles without the quant registry.
    fn scalar_decode(self) -> Option<ScalarDecode> {
        Some(match self {
            GgmlType::F32 => (4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            GgmlType::F64 => (8, |b| {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            }),
            GgmlType::F16 => (2, |b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            GgmlType::BF16 => (2, |b| {
                f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
            }),
            GgmlType::I8 => (1, |b| b[0] as i8 as f32),
            GgmlType::I16 => (2, |b| i16::from_le_bytes([b[0], b[1]]) as f32),
            GgmlType::I32 => (4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
            _ => return None,
        })
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

/// Bytes one element of an unblocked type occupies, and how to widen it.
type ScalarDecode = (usize, fn(&[u8]) -> f32);

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

    if let Some(quant) = ggml_type.quant() {
        (quant.spec().dequantize)(bytes, out);
        return Ok(());
    }

    let (width, decode) = ggml_type.scalar_decode().with_context(|| {
        format!(
            "dequantizing {} is not implemented; re-quantize the model to Q8_0, Q4_K or a legacy Q4/Q5 type",
            ggml_type.name()
        )
    })?;
    for (dst, src) in out.iter_mut().zip(bytes.chunks_exact(width)) {
        *dst = decode(src);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // The K-quants a Q4_K_M file holds are in the registry; Q5_K is not.
        assert!(dequantize_into(GgmlType::Q4_K, &[0; 144], &mut vec![0.0; 256]).is_ok());
        assert!(dequantize_into(GgmlType::Q6_K, &[0; 210], &mut vec![0.0; 256]).is_ok());
        assert!(dequantize_into(GgmlType::Q5_K, &[0; 176], &mut vec![0.0; 256]).is_err());
    }

    #[test]
    fn reports_block_alignment() {
        // 20 elements cannot fill whole 32-element Q8_0 blocks.
        assert!(GgmlType::Q8_0.storage_bytes(20).is_err());
        assert_eq!(GgmlType::Q8_0.storage_bytes(64).unwrap(), 68);
        assert_eq!(GgmlType::F32.storage_bytes(20).unwrap(), 80);
    }
}
