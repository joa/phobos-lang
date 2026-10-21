// Writes the generated raw-format kernel sources to a directory as `.ph`
// files. `phobos-lang`'s `emit` example can then diff a codegen change over
// them, which it cannot do for sources built in `format!` strings.
// The test beside it checks that decode attention compiles the same with
// its written grid barrier left to the compiler.
//
//   PHOBOS_DUMP_DIR=/some/dir cargo test -p phobos-gguf --features cuda \
//       dump_raw_kernel_sources -- --ignored

use super::*;
use crate::quant::Quant;

/// One dp4a decode matvec: the source builder, its wide and narrow output
/// tiles, and the kernel's name.
/// A source builder taking its tile width.
type Src = fn(usize) -> String;


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
    sources.push(("attn_merge".to_string(), attn_merge_src(256, 4)));
    for (name, form) in [("attn_flash", FlashForm::Streamed), ("attn_flash_resident", FlashForm::Resident)] {
        let shape = FlashShape { form, keys: FLASH_KEYS, pipeline: false };
        sources.push((name.to_string(), attn_flash_src(256, 4, shape, 1)));
        sources.push((format!("{name}_split"), attn_flash_src(256, 4, shape, 4)));
    }
    sources.push(("attn_gemm".to_string(), attn_gemm_src(256, ATTN_GEMM_TILE)));
    sources.push(("q8_split".to_string(), Q8_SPLIT_SRC.replace("{ALIGNED}", "@aligned(N = TN, K = 32)")));
    sources.push(("moe_topk".to_string(), moe_topk_src()));
    sources.push(("q8_qdot_lanes".to_string(), q8_qdot_lanes_src(false)));
    sources.push(("q4k_qdot_lanes".to_string(), q4k_qdot_lanes_src()));
    sources.push(("q5k_qdot_lanes".to_string(), q5k_qdot_lanes_src()));
    sources.push(("q6k_qdot_lanes".to_string(), q6k_qdot_lanes_src()));
    sources.push(("iq2xxs_qdot_lanes".to_string(), iq2xxs_qdot_lanes_src()));
    sources.push(("iq2xs_qdot_lanes".to_string(), iq2xs_qdot_lanes_src()));
    sources.push(("iq2s_qdot_lanes".to_string(), iq2s_qdot_lanes_src()));
    sources.push(("iq3xxs_qdot_lanes".to_string(), iq3xxs_qdot_lanes_src()));
    sources.push(("iq3s_qdot_lanes".to_string(), iq3s_qdot_lanes_src()));
    sources.push(("iq1s_qdot_lanes".to_string(), iq1s_qdot_lanes_src()));
    sources.push(("iq1m_qdot_lanes".to_string(), iq1m_qdot_lanes_src()));
    sources.push(("ptq1_qdot_lanes".to_string(), ptq1_qdot_lanes_src()));
    sources.push(("q2k_qdot_lanes".to_string(), q2k_qdot_lanes_src()));
    sources.push(("q3k_qdot_lanes".to_string(), q3k_qdot_lanes_src()));
    sources.push(("iq4xs_qdot_lanes".to_string(), iq4xs_qdot_lanes_src()));
    sources.push(("moe_combine".to_string(), moe_combine_src()));
    sources.push(("delta_scan_reg".to_string(), delta_scan_reg_src(128)));
    // The IQ formats' f32 decode matvecs and weight expansions.
    let iq: [(&str, Src, Src, usize); 7] = [
        ("iq1s", iq1s_qdot_matvec_src, iq1s_qdecode_src, IQ1S_TN),
        ("iq1m", iq1m_qdot_matvec_src, iq1m_qdecode_src, IQ1M_TN),
        ("iq2xxs", iq2xxs_qdot_matvec_src, iq2xxs_qdecode_src, IQ2XXS_TN),
        ("iq2xs", iq2xs_qdot_matvec_src, iq2xs_qdecode_src, IQ2XS_TN),
        ("iq2s", iq2s_qdot_matvec_src, iq2s_qdecode_src, IQ2S_TN),
        ("iq3xxs", iq3xxs_qdot_matvec_src, iq3xxs_qdecode_src, IQ3XXS_TN),
        ("iq3s", iq3s_qdot_matvec_src, iq3s_qdecode_src, IQ3S_TN),
    ];
    for (name, qdot, qdecode, tn) in iq {
        sources.push((format!("{name}_qdot_matvec"), qdot(tn)));
        sources.push((format!("{name}_qdecode"), qdecode(tn)));
    }
    // Decode attention at the Qwen3.5 head layouts, a 128-wide one and a
    // 64-wide one: (heads, group, head dim).
    for (heads, group, head_dim) in [(8, 4, 256), (16, 4, 256), (32, 4, 128), (16, 1, 64)] {
        let qgroup = attention_qgroup(group);
        let name = format!("{heads}x{group}x{head_dim}");
        let split = attention_split_src(heads, group, head_dim, qgroup, ATTN_SPLITS, ATTN_WARP_SPLITS);
        sources.push((format!("attention_split_{name}"), split));
        let persist = attention_persist_src(heads, group, head_dim, qgroup, ATTN_SPLITS, ATTN_WARP_SPLITS, 46);
        sources.push((format!("attention_persist_{name}"), persist));
    }
    sources.push(("delta_wy".to_string(), delta_wy_src(32, 128, DELTA_CHUNK)));
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
        let src = kquant_lanes_split_src(quant, 2560, 9216, 6).expect("a K-quant splits");
        sources.push((kquant_lanes_split_name(quant).to_string(), src));
    }
    // The 4B's convolution: 32 value heads over 16 key heads, and ungrouped.
    let batch = delta_conv_batch(512);
    let scale = 1.0 / 128f32.sqrt();
    sources.push(("delta_conv".to_string(), delta_conv_src(32, 16, 128, 4, 128, 2048, batch, true, scale)));
    sources.push(("delta_conv_mha".to_string(), delta_conv_src(16, 16, 128, 4, 128, 2048, batch, true, scale)));
    sources.push(("rms_norm_gated".to_string(), rms_norm_src(128, 1e-6, NormForm::GatedQuantized)));
    // The kernels that take an absmax or a SiLU in tile form: the 4B's
    // widths, and 160, which no intrinsic norm divides.
    sources.push(("quantize".to_string(), QUANTIZE_SRC.to_string()));
    sources.push(("pointwise".to_string(), POINTWISE_SRC.to_string()));
    let delta = DELTA_SRC.replace("{H}", "32").replace("{D}", "128");
    sources.push(("delta".to_string(), delta.replace("{TN}", &DELTA_TN.to_string())));
    for width in [2560, 160] {
        for (form, name) in [(NormForm::Plain, "plain"), (NormForm::Quantized, "q"), (NormForm::GatedQuantized, "gated")] {
            sources.push((format!("rms_norm_{name}_{width}"), rms_norm_src(width, 1e-6, form)));
        }
    }
    sources.push(("delta_gates".to_string(), delta_gates_src(32, (ELEM_TILE / 4 / 32).max(1))));
    sources.push(("hadamard_q".to_string(), hadamard_src(2560, None, HadamardForm::Quantized)));
    for accumulate in [false, true] {
        sources.push((format!("q50_qdot_{accumulate}"), q50_qdot_src(accumulate)));
    }
    for tn in Q8_QMMA_WIDTHS {
        sources.push((format!("q50_qmma_{tn}"), q50_qmma_src(Q8_QMMA_CTA, Q8_QMMA_TM, tn)));
    }
    sources.push(("q50_qmma_shallow".into(), q50_qmma_src(Q8_QMMA_CTA, Q8_QMMA_SHALLOW, Q8_QMMA_TN)));
    // The Q8_0 prompt projections, at the tiles init.rs and matmul.rs set.
    let tile = |src: String, tm: usize, tn: usize| {
        src.replace(
            "@autotune(TM in [64], TN in [64])",
            &format!("@autotune(TM in [{tm}], TN in [{tn}])"),
        )
    };
    for tn in Q8_QMMA_WIDTHS {
        sources.push((format!("q8_qmma_{tn}"), tile(q8_qmma_src(Q8_QMMA_CTA), Q8_QMMA_TM, tn)));
    }
    sources.push(("q8_qmma_shallow".into(), q8_qmma_src(Q8_QMMA_CTA)));
    let narrow = tile(q8_qmma_src(Q8_QMMA_NARROW_CTA), Q8_QMMA_TM, Q8_QMMA_NARROW_TN);
    sources.push(("q8_qmma_narrow".into(), narrow));
    let split = tile(q8_qmma_split_src(Q8_QMMA_CTA, 2560, 4), Q8_QMMA_TM, Q8_QMMA_WIDTHS[0]);
    sources.push(("q8_qmma_split".into(), split));
    let mma = Q8_MMA_SRC.replace("{ALIGNED}", "@aligned(M = TM, N = TN, K = 32)");
    sources.push(("q8_mma".into(), mma));
    sources.extend(crate::backend::device::formats::all_sources());
    for (name, src) in sources {
        std::fs::write(dir.join(format!("{name}.ph")), src).unwrap();
    }
}

/// Decode attention's split and merge order through device memory, so the
/// compiler has to place the barrier the source writes between them.
#[test]
fn the_compiler_places_the_attention_barrier() {
    use phobos_kernels::compile::ptx_for;
    for (heads, group, head_dim) in [(8, 4, 256), (16, 4, 256), (32, 4, 128), (16, 1, 64)] {
        let qgroup = attention_qgroup(group);
        let src = attention_persist_src(heads, group, head_dim, qgroup, ATTN_SPLITS, ATTN_WARP_SPLITS, 46);
        let stripped: String = src.lines().filter(|l| l.trim() != "grid_barrier()").map(|l| format!("{l}\n")).collect();
        assert_ne!(src.len(), stripped.len(), "the source writes a barrier");
        let compile = |s: &str| ptx_for("sm_75", s, "attention_persist").unwrap_or_else(|e| panic!("{e:#}"));
        assert!(compile(&src) == compile(&stripped), "{heads}x{group}x{head_dim}: the placed barrier differs");
    }
}
