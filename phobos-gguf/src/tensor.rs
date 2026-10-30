use anyhow::{Context, Result, bail, ensure};
use phobos_base::half::f16_to_f32;

use crate::quant::Quant;
use crate::quant::ptq1_0::{FILE_BLOCK, FILE_BLOCK_BYTES};

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
    /// PrismML's ternary format; no upstream ggml type carries this code.
    PTQ1_0,
}

/// One [`GgmlType`]: its code in the file, its name, the elements and bytes
/// of one block, and the format [`crate::quant`] contracts it as, `None` for
/// a type that goes down a dense path.
struct Row {
    ty: GgmlType,
    code: u32,
    name: &'static str,
    block: usize,
    bytes: usize,
    quant: Option<Quant>,
}

const fn row(ty: GgmlType, code: u32, name: &'static str, block: usize, bytes: usize, quant: Option<Quant>) -> Row {
    Row { ty, code, name, block, bytes, quant }
}

/// Every type the reader knows, in the enum's order.
const TYPES: [Row; 30] = [
    row(GgmlType::F32, 0, "F32", 1, 4, None),
    row(GgmlType::F16, 1, "F16", 1, 2, None),
    row(GgmlType::Q4_0, 2, "Q4_0", 32, 18, Some(Quant::Q4_0)),
    row(GgmlType::Q4_1, 3, "Q4_1", 32, 20, Some(Quant::Q4_1)),
    row(GgmlType::Q5_0, 6, "Q5_0", 32, 22, Some(Quant::Q5_0)),
    row(GgmlType::Q5_1, 7, "Q5_1", 32, 24, Some(Quant::Q5_1)),
    row(GgmlType::Q8_0, 8, "Q8_0", 32, 34, Some(Quant::Q8_0)),
    row(GgmlType::Q8_1, 9, "Q8_1", 32, 36, Some(Quant::Q8_1)),
    row(GgmlType::Q2_K, 10, "Q2_K", 256, 84, Some(Quant::Q2_K)),
    row(GgmlType::Q3_K, 11, "Q3_K", 256, 110, Some(Quant::Q3_K)),
    row(GgmlType::Q4_K, 12, "Q4_K", 256, 144, Some(Quant::Q4_K)),
    row(GgmlType::Q5_K, 13, "Q5_K", 256, 176, Some(Quant::Q5_K)),
    row(GgmlType::Q6_K, 14, "Q6_K", 256, 210, Some(Quant::Q6_K)),
    row(GgmlType::Q8_K, 15, "Q8_K", 256, 292, None),
    row(GgmlType::IQ2_XXS, 16, "IQ2_XXS", 256, 66, Some(Quant::IQ2_XXS)),
    row(GgmlType::IQ2_XS, 17, "IQ2_XS", 256, 74, Some(Quant::IQ2_XS)),
    row(GgmlType::IQ3_XXS, 18, "IQ3_XXS", 256, 98, Some(Quant::IQ3_XXS)),
    row(GgmlType::IQ1_S, 19, "IQ1_S", 256, 50, Some(Quant::IQ1_S)),
    row(GgmlType::IQ4_NL, 20, "IQ4_NL", 32, 18, None),
    row(GgmlType::IQ3_S, 21, "IQ3_S", 256, 110, Some(Quant::IQ3_S)),
    row(GgmlType::IQ2_S, 22, "IQ2_S", 256, 82, Some(Quant::IQ2_S)),
    row(GgmlType::IQ4_XS, 23, "IQ4_XS", 256, 136, Some(Quant::IQ4_XS)),
    row(GgmlType::I8, 24, "I8", 1, 1, None),
    row(GgmlType::I16, 25, "I16", 1, 2, None),
    row(GgmlType::I32, 26, "I32", 1, 4, None),
    row(GgmlType::I64, 27, "I64", 1, 8, None),
    row(GgmlType::F64, 28, "F64", 1, 8, None),
    row(GgmlType::IQ1_M, 29, "IQ1_M", 256, 56, Some(Quant::IQ1_M)),
    row(GgmlType::BF16, 30, "BF16", 1, 2, None),
    row(GgmlType::PTQ1_0, 143, "PTQ1_0", FILE_BLOCK, FILE_BLOCK_BYTES, Some(Quant::PTQ1_0)),
];

// `GgmlType::row` indexes the table by discriminant.
const _: () = {
    let mut i = 0;
    while i < TYPES.len() {
        assert!(TYPES[i].ty as usize == i, "TYPES lists the types in the enum's order");
        i += 1;
    }
};

impl GgmlType {
    fn row(self) -> &'static Row {
        &TYPES[self as usize]
    }

    pub fn from_code(code: u32) -> Result<GgmlType> {
        match TYPES.iter().find(|row| row.code == code) {
            Some(row) => Ok(row.ty),
            None => bail!("unknown or retired ggml tensor type {code}"),
        }
    }

    /// The type's number in the file format.
    pub fn code(self) -> u32 {
        self.row().code
    }

    pub fn name(self) -> &'static str {
        self.row().name
    }

    pub fn block_size(self) -> usize {
        self.row().block
    }

    pub fn type_size(self) -> usize {
        self.row().bytes
    }

    /// The compute format for this type, for the types this crate contracts
    /// against. Everything else, the scalar types included, is `None` and
    /// goes down a dense path.
    pub fn quant(self) -> Option<Quant> {
        self.row().quant
    }

    /// Every type, for the tests that walk them all.
    #[cfg(test)]
    pub(crate) fn all() -> impl Iterator<Item = GgmlType> {
        TYPES.iter().map(|row| row.ty)
    }

    pub fn is_dequantizable(self) -> bool {
        self.scalar_decode().is_some() || self.quant().is_some()
    }

    /// How to widen one element of an unblocked type. These are the types
    /// [`dequantize_into`] handles without the quant registry.
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
        (numel / block)
            .checked_mul(self.type_size())
            .with_context(|| format!("{} tensor of {numel} elements overflows the address space", self.name()))
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
    /// The element count. The reader refuses dimensions whose product
    /// overflows, so this saturates only for one built by hand, and the
    /// saturated count then fails any bounds check rather than passing it.
    pub fn numel(&self) -> u64 {
        self.dims.iter().fold(1u64, |n, &d| n.saturating_mul(d))
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
        // The three K-quants a Q4_K_M file holds are all in the registry.
        assert!(dequantize_into(GgmlType::Q4_K, &[0; 144], &mut vec![0.0; 256]).is_ok());
        assert!(dequantize_into(GgmlType::Q5_K, &[0; 176], &mut vec![0.0; 256]).is_ok());
        assert!(dequantize_into(GgmlType::Q6_K, &[0; 210], &mut vec![0.0; 256]).is_ok());
    }

    #[test]
    fn reports_block_alignment() {
        // 20 elements cannot fill whole 32-element Q8_0 blocks.
        assert!(GgmlType::Q8_0.storage_bytes(20).is_err());
        assert_eq!(GgmlType::Q8_0.storage_bytes(64).unwrap(), 68);
        assert_eq!(GgmlType::F32.storage_bytes(20).unwrap(), 80);
    }
}
