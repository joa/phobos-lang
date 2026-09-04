// The staged prompt projections, one kernel a raw format; see
// `phobos-lang`'s `codegen/tile/qgemm.rs`.

use crate::quant::Quant;

/// Rows, columns and threads of a `<fmt>_qgemm` kernel, fixed by the
/// intrinsic: the whole batch down, 64 columns across, eight warps.
pub(crate) const QGEMM_TM: usize = 128;
pub(crate) const QGEMM_TN: usize = 64;
pub(crate) const QGEMM_CTA: usize = 256;

/// The ternary grid at two bits a lane, shared by IQ1_S and IQ1_M.
pub(crate) const IQ1_GRID2_LEN: usize = 2048 * 2;

/// The same grid at a nibble a lane, which the decode matvecs read.
pub(crate) const IQ1_GRID4_LEN: usize = 2048 * 4;

/// The table operands a format's staged kernel reads after `(qb, d)`, by
/// length: the magnitude grid, then the 0/-1 sign masks where the format
/// carries signs apart. Mirrors `QgFormat::tables` in the compiler.
pub(crate) fn qgemm_tables(quant: Quant) -> Option<&'static [usize]> {
    Some(match quant {
        Quant::IQ1_S | Quant::IQ1_M => &[IQ1_GRID2_LEN],
        Quant::IQ2_XXS => &[256 * 8, 128 * 8],
        Quant::IQ2_XS => &[512 * 8, 128 * 8],
        Quant::IQ2_S => &[1024 * 8, 256 * 8],
        Quant::IQ3_XXS => &[256 * 4, 128 * 8],
        Quant::IQ3_S => &[512 * 4, 256 * 8],
        // The K-quants decode from the block bytes alone.
        Quant::Q4_K | Quant::Q5_K | Quant::Q6_K => &[],
        _ => return None,
    })
}

/// The kernel and intrinsic a format's staged projection is named by.
pub(crate) fn qgemm_name(quant: Quant) -> Option<&'static str> {
    Some(match quant {
        Quant::IQ1_S => "iq1s",
        Quant::IQ1_M => "iq1m",
        Quant::IQ2_XXS => "iq2xxs",
        Quant::IQ2_XS => "iq2xs",
        Quant::IQ2_S => "iq2s",
        Quant::IQ3_XXS => "iq3xxs",
        Quant::IQ3_S => "iq3s",
        Quant::Q4_K => "q4k",
        Quant::Q5_K => "q5k",
        Quant::Q6_K => "q6k",
        _ => return None,
    })
}

/// The kernel a format's staged source declares.
pub(crate) fn qgemm_kernel(quant: Quant) -> Option<&'static str> {
    Some(match quant {
        Quant::IQ1_S => "iq1s_qgemm",
        Quant::IQ1_M => "iq1m_qgemm",
        Quant::IQ2_XXS => "iq2xxs_qgemm",
        Quant::IQ2_XS => "iq2xs_qgemm",
        Quant::IQ2_S => "iq2s_qgemm",
        Quant::IQ3_XXS => "iq3xxs_qgemm",
        Quant::IQ3_S => "iq3s_qgemm",
        Quant::Q4_K => "q4k_qgemm",
        Quant::Q5_K => "q5k_qgemm",
        Quant::Q6_K => "q6k_qgemm",
        _ => return None,
    })
}

/// The source of a format's staged projection, at two CTAs a
/// multiprocessor (128 registers).
pub(crate) fn qgemm_src(quant: Quant) -> Option<String> {
    let name = qgemm_name(quant)?;
    let tables = qgemm_tables(quant)?;
    // A format with no tables declares none and passes none: the joins
    // below leave no stray comma behind.
    let params: String = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}], "))
        .collect();
    let args: String = (0..tables.len())
        .map(|i| format!(",\n                                                T{i}[0 :+ 1, :]"))
        .collect();
    Some(format!(
        "@launch({QGEMM_CTA}, 2)
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}])
@aligned(M = TM, N = TN, K = 256)
kernel {name}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  {params}C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = {name}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :]{args})
}}
"
    ))
}
