// Writes the generated raw-format kernel sources to a directory as `.ph`
// files. `phobos-lang`'s `emit` example can then diff a codegen change over
// them, which it cannot do for sources built in `format!` strings.
//
//   PHOBOS_DUMP_DIR=/some/dir cargo test -p phobos-gguf --features cuda \
//       dump_raw_kernel_sources -- --ignored

use super::*;
use crate::quant::Quant;

/// One dp4a decode matvec: the source builder, its wide and narrow output
/// tiles, and the kernel's name.
type I8Row = (fn(usize) -> String, usize, usize, &'static str);

#[test]
#[ignore = "writes files; run by hand with PHOBOS_DUMP_DIR set"]
fn dump_raw_kernel_sources() {
    let Ok(dir) = std::env::var("PHOBOS_DUMP_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut sources: Vec<(String, String)> = Vec::new();
    for quant in Quant::ALL {
        if let (Some(name), Some(src)) = (qgemm_kernel(quant), qgemm_src(quant)) {
            sources.push((name.to_string(), src));
        }
        // The 4B's deep projection, split three ways.
        let block = quant.spec().block;
        if let (Some(name), Some(src)) = (qgemm_split_kernel(quant), qgemm_split_src(quant, 2560, 9216, block, 3)) {
            sources.push((name.to_string(), src));
        }
    }
    sources.push(("qgemm_reduce".to_string(), qgemm_reduce_src(2560, 3, false)));
    sources.push(("qgemm_reduce_add".to_string(), qgemm_reduce_src(2560, 3, true)));
    sources.push(("attn_tc".to_string(), attn_tc_src(256, 4)));
    sources.push(("delta_scan_reg".to_string(), delta_scan_reg_src(128)));
    // The 4B's attention heads: 256 wide, 64 of them rotated.
    for cache in [false, true] {
        let name = if cache { "attn_prep_kv" } else { "attn_prep_q" };
        sources.push((name.to_string(), attn_prep_src(256, 32, 1e-6, cache)));
    }
    sources.push(("gate_q".to_string(), swiglu_q_src(ELEM_TILE / RMS_LANE, true, true)));
    // The 4B's fused MLP planned for a 170-SM card, its down projection
    // split along k.
    let chain = crate::backend::fuse::mlp_chain_raw(
        crate::backend::Buf(0),
        crate::backend::Buf(1),
        (crate::backend::RawBuf(0), Quant::Q4_K),
        (crate::backend::RawBuf(1), Quant::Q4_K),
        (crate::backend::RawBuf(2), Quant::Q6_K),
        2560,
        9216,
        1e-6,
    )
    .expect("the 4B's formats fuse");
    let plan = chain.key(340).plan().expect("well-formed").expect("fused");
    sources.push(("fused_mlp_340".to_string(), plan.source));
    // The 4B's down projections split six ways.
    for quant in [Quant::Q4_K, Quant::Q6_K] {
        let src = kquant_qdot_i8_split_src(quant, 2560, 9216, 6).expect("a K-quant splits");
        sources.push((kquant_qdot_i8_split_name(quant).to_string(), src));
    }
    // The 4B's convolution: 32 value heads over 16 key heads, and ungrouped.
    let batch = delta_conv_batch(512);
    let scale = 1.0 / 128f32.sqrt();
    sources.push(("delta_conv".to_string(), delta_conv_src(32, 16, 128, 4, 128, 2048, batch, true, scale)));
    sources.push(("delta_conv_mha".to_string(), delta_conv_src(16, 16, 128, 4, 128, 2048, batch, true, scale)));
    sources.push(("rms_norm_gated".to_string(), rms_norm_src(128, 1e-6, NormForm::GatedQuantized)));
    let i8: [I8Row; 11] = [
        (
            iq1s_qdot_i8_matvec_src,
            IQ1S_I8_TN,
            IQ1S_I8_NARROW_TN,
            "iq1s",
        ),
        (
            iq1m_qdot_i8_matvec_src,
            IQ1M_I8_TN,
            IQ1M_I8_NARROW_TN,
            "iq1m",
        ),
        (
            iq2xxs_qdot_i8_matvec_src,
            IQ2XXS_I8_TN,
            IQ2XXS_I8_NARROW_TN,
            "iq2xxs",
        ),
        (
            iq2xs_qdot_i8_matvec_src,
            IQ2XS_I8_TN,
            IQ2XS_I8_NARROW_TN,
            "iq2xs",
        ),
        (
            iq2s_qdot_i8_matvec_src,
            IQ2S_I8_TN,
            IQ2S_I8_NARROW_TN,
            "iq2s",
        ),
        (
            iq3xxs_qdot_i8_matvec_src,
            IQ3XXS_I8_TN,
            IQ3XXS_I8_NARROW_TN,
            "iq3xxs",
        ),
        (
            iq3s_qdot_i8_matvec_src,
            IQ3S_I8_TN,
            IQ3S_I8_NARROW_TN,
            "iq3s",
        ),
        (q4k_qdot_i8_matvec_src, Q4K_I8_TN, Q4K_I8_NARROW_TN, "q4k"),
        (ptq1_qdot_i8_matvec_src, PTQ1_I8_TN, PTQ1_I8_NARROW_TN, "ptq1"),
        (q5k_qdot_i8_matvec_src, Q5K_I8_TN, Q5K_I8_NARROW_TN, "q5k"),
        (q6k_qdot_i8_matvec_src, Q6K_I8_TN, Q6K_I8_NARROW_TN, "q6k"),
    ];
    for (src, wide, narrow, name) in i8 {
        sources.push((format!("{name}_qdot_i8_matvec_{wide}"), src(wide)));
        sources.push((format!("{name}_qdot_i8_matvec_{narrow}"), src(narrow)));
    }
    for accumulate in [false, true] {
        sources.push((format!("q50_qdot_{accumulate}"), q50_qdot_src(accumulate)));
    }
    for tn in Q8_QMMA_WIDTHS {
        sources.push((format!("q50_qmma_{tn}"), q50_qmma_src(Q8_QMMA_CTA, Q8_QMMA_TM, tn)));
    }
    sources.push(("q50_qmma_shallow".into(), q50_qmma_src(Q8_QMMA_CTA, Q8_QMMA_SHALLOW, Q8_QMMA_TN)));
    for (name, src) in sources {
        std::fs::write(dir.join(format!("{name}.ph")), src).unwrap();
    }
}
