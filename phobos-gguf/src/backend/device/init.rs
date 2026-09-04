// Bringing a `DeviceBackend` up: the CUDA context, every kernel source
// compiled (the independent ones in parallel), the constant tables the
// grid-coded formats gather from, and the knobs read from the environment.
// Nothing here runs after construction; it lives apart from `mod.rs` for
// length alone.

use super::*;

impl DeviceBackend {
    pub fn new() -> Result<DeviceBackend> {
        let _ctx = cust::quick_init().context("initializing CUDA")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        residency::vram_mark("cuda context up");

        let matmul = Variants::compile(
            MATMUL_SRC,
            &[("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)],
            "matmul",
            ("@aligned(M = TILE_M, N = TILE_N)", ""),
        )?;
        let matmul_tc = compile(
            MATMUL_TC_SRC,
            &[
                ("TILE_M", TC_TILE_M),
                ("TILE_N", TC_TILE_N),
                ("TILE_K", TC_TILE_K),
            ],
            "matmul_tc",
        )?;
        let matmul_f16w_body = matmul_f16w_src();
        let matmul_f16w = Variants::compile(
            &matmul_f16w_body,
            &[("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)],
            "matmul",
            ("@aligned(M = TILE_M, N = TILE_N)", ""),
        )?;
        let matmul_tc_f16w = compile(
            &matmul_tc_f16w_src(),
            &[
                ("TILE_M", TC_TILE_M),
                ("TILE_N", TC_TILE_N),
                ("TILE_K", TC_TILE_K),
            ],
            "matmul_tc",
        )?;
        let matvec = Variants::compile(
            MATVEC_SRC,
            &[("TILE_N", MV_TN), ("TILE_K", TILE_K)],
            "matvec",
            ("@aligned(N = TILE_N)", ""),
        )?;
        // matmul_quant requires k to be a whole number of Q8_0 blocks, so the k
        // loop never has a remainder to split off.
        let q8_dp4a = Variants::compile(
            Q8_DP4A_SRC,
            &[("TN", Q8_TN)],
            "q8_dp4a",
            ("@aligned(N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let q8_mma = Variants::compile(
            Q8_MMA_SRC,
            &[("TM", Q8_MMA_TM), ("TN", Q8_MMA_TN)],
            "q8_mma",
            ("@aligned(M = TM, N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let qmma_src = q8_qmma_src(Q8_QMMA_CTA);
        let q8_qmma = compile(
            &qmma_src,
            &[("TM", Q8_QMMA_SHALLOW), ("TN", Q8_QMMA_TN)],
            "q8_qmma",
        )?;
        let mut q8_qmma_deep = HashMap::new();
        for tn in Q8_QMMA_WIDTHS {
            let module = compile(&qmma_src, &[("TM", Q8_QMMA_TM), ("TN", tn)], "q8_qmma")?;
            q8_qmma_deep.insert(tn, module);
        }
        let q8_split = Variants::compile(
            Q8_SPLIT_SRC,
            &[("TN", Q8_TN), ("RT", Q8_REDUCE_TN)],
            "q8_split",
            ("@aligned(N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let q8_qdot = compile(Q8_QDOT_SRC, &[("TN", Q8_QDOT_TN)], "q8_qdot")?;
        let q8_qdot_add = compile(Q8_QDOT_ADD_SRC, &[("TN", Q8_QDOT_TN)], "q8_qdot_add")?;
        let quantize = compile(QUANTIZE_SRC, &[("TB", QUANT_TB)], "quantize")?;
        let quantize_wide = compile(QUANTIZE_SRC, &[("TB", QUANT_TB_WIDE)], "quantize")?;
        let pointwise = compile(POINTWISE_SRC, &[("TILE", ELEM_TILE)], "pointwise")?;
        let pointwise_wide = compile(POINTWISE_SRC, &[("TILE", ELEM_TILE_WIDE)], "pointwise")?;
        let argmax_finish = compile(ARGMAX_FINISH_SRC, &[], "argmax_finish")?;
        // Independent kernels compiled in parallel (`compile_parallel`), so
        // MLIR-to-PTX lowering runs concurrently. `Vec::remove(0)` keeps the
        // destructure in the jobs' own order without cloning `Module`s.
        let q2k_src = q2k_matvec_src(Q2K_TN);
        let q3k_src = q3k_matvec_src(Q3K_TN);
        let iq1s_src = iq1s_matvec_src(IQ1S_TN);
        let iq2xxs_src = iq2xxs_matvec_src(IQ2XXS_TN);
        let iq1m_src = iq1m_matvec_src(IQ1M_TN);
        let iq2s_src = iq2s_matvec_src(IQ2S_TN);
        let iq2xs_src = iq2xs_matvec_src(IQ2XS_TN);
        let iq3xxs_src = iq3xxs_matvec_src(IQ3XXS_TN);
        let iq3s_src = iq3s_matvec_src(IQ3S_TN);
        let iq4xs_src = iq4xs_matvec_src(IQ4XS_TN);
        let iq1s_dequant_body = iq1s_dequant_src(IQ1S_TN);
        let iq2xxs_dequant_body = iq2xxs_dequant_src(IQ2XXS_TN);
        let iq1m_dequant_body = iq1m_dequant_src(IQ1M_TN);
        let iq2s_dequant_body = iq2s_dequant_src(IQ2S_TN);
        let iq2xs_dequant_body = iq2xs_dequant_src(IQ2XS_TN);
        let iq3xxs_dequant_body = iq3xxs_dequant_src(IQ3XXS_TN);
        let iq3s_dequant_body = iq3s_dequant_src(IQ3S_TN);
        let iq4xs_dequant_body = iq4xs_dequant_src(IQ4XS_TN);
        let q2k_dequant_body = q2k_dequant_src(Q2K_TN);
        let (qtm, qtn, qcta) = qmma_tile();
        let qsubs = [("TM", qtm), ("TN", qtn)];
        let iq1s_qmma_body = iq1s_qmma_src(qcta, qtm, qtn);
        let iq2xxs_qmma_body = iq2xxs_qmma_src(qcta, qtm, qtn);
        let iq2s_qmma_body = iq2s_qmma_src(qcta, qtm, qtn);
        let iq2xs_qmma_body = iq2xs_qmma_src(qcta, qtm, qtn);
        let iq3xxs_qmma_body = iq3xxs_qmma_src(qcta, qtm, qtn);
        let iq3s_qmma_body = iq3s_qmma_src(qcta, qtm, qtn);
        let iq1s_qdecode_body = iq1s_qdecode_src(IQ1S_TN);
        let iq2xxs_qdecode_body = iq2xxs_qdecode_src(IQ2XXS_TN);
        let iq1m_qdecode_body = iq1m_qdecode_src(IQ1M_TN);
        let iq2s_qdecode_body = iq2s_qdecode_src(IQ2S_TN);
        let iq2xs_qdecode_body = iq2xs_qdecode_src(IQ2XS_TN);
        let iq3xxs_qdecode_body = iq3xxs_qdecode_src(IQ3XXS_TN);
        let iq3s_qdecode_body = iq3s_qdecode_src(IQ3S_TN);
        // The f16 strip is the same kernel with a narrower destination.
        let f16_scratch =
            |src: &str| src.replace("SCRATCH: tensor<f32>[K, N]", "SCRATCH: tensor<f16>[K, N]");
        let iq1s_qdecode_f16_body = f16_scratch(&iq1s_qdecode_body);
        let iq2xxs_qdecode_f16_body = f16_scratch(&iq2xxs_qdecode_body);
        let iq1m_qdecode_f16_body = f16_scratch(&iq1m_qdecode_body);
        let iq2s_qdecode_f16_body = f16_scratch(&iq2s_qdecode_body);
        let iq2xs_qdecode_f16_body = f16_scratch(&iq2xs_qdecode_body);
        let iq3xxs_qdecode_f16_body = f16_scratch(&iq3xxs_qdecode_body);
        let iq3s_qdecode_f16_body = f16_scratch(&iq3s_qdecode_body);
        let iq1s_qdot_body = iq1s_qdot_matvec_src(IQ1S_TN);
        let iq2xxs_qdot_body = iq2xxs_qdot_matvec_src(IQ2XXS_TN);
        let iq1m_qdot_body = iq1m_qdot_matvec_src(IQ1M_TN);
        let iq2s_qdot_body = iq2s_qdot_matvec_src(IQ2S_TN);
        let iq2xs_qdot_body = iq2xs_qdot_matvec_src(IQ2XS_TN);
        let iq3xxs_qdot_body = iq3xxs_qdot_matvec_src(IQ3XXS_TN);
        let iq3s_qdot_body = iq3s_qdot_matvec_src(IQ3S_TN);
        let iq4xs_qdot_body = iq4xs_qdot_matvec_src(IQ4XS_TN);
        let q2k_qdot_body = q2k_qdot_matvec_src(Q2K_TN);
        let q3k_qdot_body = q3k_qdot_matvec_src(Q3K_TN);
        // The dp4a decode matvecs, wide tile then narrow. One table drives the
        // sources, the compile entries and the `remove`s below, so their order
        // cannot drift apart.
        let i8: [I8Row; 10] = [
            (
                iq1s_qdot_i8_matvec_src,
                IQ1S_I8_TN,
                IQ1S_I8_NARROW_TN,
                "iq1s_qdot_i8_matvec",
            ),
            (
                iq3s_qdot_i8_matvec_src,
                IQ3S_I8_TN,
                IQ3S_I8_NARROW_TN,
                "iq3s_qdot_i8_matvec",
            ),
            (
                iq3xxs_qdot_i8_matvec_src,
                IQ3XXS_I8_TN,
                IQ3XXS_I8_NARROW_TN,
                "iq3xxs_qdot_i8_matvec",
            ),
            (
                iq2xxs_qdot_i8_matvec_src,
                IQ2XXS_I8_TN,
                IQ2XXS_I8_NARROW_TN,
                "iq2xxs_qdot_i8_matvec",
            ),
            (
                iq1m_qdot_i8_matvec_src,
                IQ1M_I8_TN,
                IQ1M_I8_NARROW_TN,
                "iq1m_qdot_i8_matvec",
            ),
            (
                iq2xs_qdot_i8_matvec_src,
                IQ2XS_I8_TN,
                IQ2XS_I8_NARROW_TN,
                "iq2xs_qdot_i8_matvec",
            ),
            (
                iq2s_qdot_i8_matvec_src,
                IQ2S_I8_TN,
                IQ2S_I8_NARROW_TN,
                "iq2s_qdot_i8_matvec",
            ),
            (
                q4k_qdot_i8_matvec_src,
                Q4K_I8_TN,
                Q4K_I8_NARROW_TN,
                "q4k_qdot_i8_matvec",
            ),
            (
                q5k_qdot_i8_matvec_src,
                Q5K_I8_TN,
                Q5K_I8_NARROW_TN,
                "q5k_qdot_i8_matvec",
            ),
            (
                q6k_qdot_i8_matvec_src,
                Q6K_I8_TN,
                Q6K_I8_NARROW_TN,
                "q6k_qdot_i8_matvec",
            ),
        ];
        let i8_srcs: Vec<OwnedEntry> = i8
            .iter()
            .flat_map(|&(src, w, n, name)| {
                let w = qdot_i8_tn(w);
                [(src(w), [("TN", w)], name), (src(n), [("TN", n)], name)]
            })
            .collect();

        let mut raw_entries: Vec<Entry> = vec![
            (q2k_src.as_str(), &[("TN", Q2K_TN)], "q2k_matvec"),
            (q3k_src.as_str(), &[("TN", Q3K_TN)], "q3k_matvec"),
            (iq1s_src.as_str(), &[("TN", IQ1S_TN)], "iq1s_matvec"),
            (iq2xxs_src.as_str(), &[("TN", IQ2XXS_TN)], "iq2xxs_matvec"),
            (iq1m_src.as_str(), &[("TN", IQ1M_TN)], "iq1m_matvec"),
            (iq2s_src.as_str(), &[("TN", IQ2S_TN)], "iq2s_matvec"),
            (iq2xs_src.as_str(), &[("TN", IQ2XS_TN)], "iq2xs_matvec"),
            (iq3xxs_src.as_str(), &[("TN", IQ3XXS_TN)], "iq3xxs_matvec"),
            (iq3s_src.as_str(), &[("TN", IQ3S_TN)], "iq3s_matvec"),
            (iq4xs_src.as_str(), &[("TN", IQ4XS_TN)], "iq4xs_matvec"),
            (
                iq1s_dequant_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_dequant",
            ),
            (
                iq2xxs_dequant_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_dequant",
            ),
            (
                iq1m_dequant_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_dequant",
            ),
            (
                iq2s_dequant_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_dequant",
            ),
            (
                iq2xs_dequant_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_dequant",
            ),
            (
                iq3xxs_dequant_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_dequant",
            ),
            (
                iq3s_dequant_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_dequant",
            ),
            (
                iq4xs_dequant_body.as_str(),
                &[("TN", IQ4XS_TN)],
                "iq4xs_dequant",
            ),
            (q2k_dequant_body.as_str(), &[("TN", Q2K_TN)], "q2k_dequant"),
            (
                iq1s_qdecode_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdecode",
            ),
            (
                iq2xxs_qdecode_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdecode",
            ),
            (
                iq1m_qdecode_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdecode",
            ),
            (
                iq2s_qdecode_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdecode",
            ),
            (
                iq2xs_qdecode_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdecode",
            ),
            (
                iq3xxs_qdecode_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdecode",
            ),
            (
                iq3s_qdecode_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdecode",
            ),
            (
                iq1s_qdecode_f16_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdecode",
            ),
            (
                iq2xxs_qdecode_f16_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdecode",
            ),
            (
                iq1m_qdecode_f16_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdecode",
            ),
            (
                iq2s_qdecode_f16_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdecode",
            ),
            (
                iq2xs_qdecode_f16_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdecode",
            ),
            (
                iq3xxs_qdecode_f16_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdecode",
            ),
            (
                iq3s_qdecode_f16_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdecode",
            ),
            (
                iq1s_qdot_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdot_matvec",
            ),
            (
                iq2xxs_qdot_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdot_matvec",
            ),
            (
                iq1m_qdot_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdot_matvec",
            ),
            (
                iq2s_qdot_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdot_matvec",
            ),
            (
                iq2xs_qdot_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdot_matvec",
            ),
            (
                iq3xxs_qdot_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdot_matvec",
            ),
            (
                iq3s_qdot_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdot_matvec",
            ),
            (
                iq4xs_qdot_body.as_str(),
                &[("TN", IQ4XS_TN)],
                "iq4xs_qdot_matvec",
            ),
            (q2k_qdot_body.as_str(), &[("TN", Q2K_TN)], "q2k_qdot_matvec"),
            (q3k_qdot_body.as_str(), &[("TN", Q3K_TN)], "q3k_qdot_matvec"),
        ];
        raw_entries.extend(
            i8_srcs
                .iter()
                .map(|(b, d, n)| (b.as_str(), d.as_slice(), *n)),
        );
        raw_entries.push((
            iq1s_qmma_body.as_str(),
            &qsubs,
            "iq1s_qmma",
        ));
        raw_entries.push((
            iq2xxs_qmma_body.as_str(),
            &qsubs,
            "iq2xxs_qmma",
        ));
        raw_entries.push((
            iq2s_qmma_body.as_str(),
            &qsubs,
            "iq2s_qmma",
        ));
        raw_entries.push((
            iq2xs_qmma_body.as_str(),
            &qsubs,
            "iq2xs_qmma",
        ));
        raw_entries.push((iq3xxs_qmma_body.as_str(), &qsubs, "iq3xxs_qmma"));
        raw_entries.push((iq3s_qmma_body.as_str(), &qsubs, "iq3s_qmma"));
        let mut raw_matvecs = compile_parallel(&raw_entries)?;
        let q2k_matvec = raw_matvecs.remove(0);
        let q3k_matvec = raw_matvecs.remove(0);
        let iq1s_matvec = raw_matvecs.remove(0);
        let iq2xxs_matvec = raw_matvecs.remove(0);
        let iq1m_matvec = raw_matvecs.remove(0);
        let iq2s_matvec = raw_matvecs.remove(0);
        let iq2xs_matvec = raw_matvecs.remove(0);
        let iq3xxs_matvec = raw_matvecs.remove(0);
        let iq3s_matvec = raw_matvecs.remove(0);
        let iq4xs_matvec = raw_matvecs.remove(0);
        let iq1s_dequant = raw_matvecs.remove(0);
        let iq2xxs_dequant = raw_matvecs.remove(0);
        let iq1m_dequant = raw_matvecs.remove(0);
        let iq2s_dequant = raw_matvecs.remove(0);
        let iq2xs_dequant = raw_matvecs.remove(0);
        let iq3xxs_dequant = raw_matvecs.remove(0);
        let iq3s_dequant = raw_matvecs.remove(0);
        let iq4xs_dequant = raw_matvecs.remove(0);
        let q2k_dequant = raw_matvecs.remove(0);
        let iq1s_qdecode = raw_matvecs.remove(0);
        let iq2xxs_qdecode = raw_matvecs.remove(0);
        let iq1m_qdecode = raw_matvecs.remove(0);
        let iq2s_qdecode = raw_matvecs.remove(0);
        let iq2xs_qdecode = raw_matvecs.remove(0);
        let iq3xxs_qdecode = raw_matvecs.remove(0);
        let iq3s_qdecode = raw_matvecs.remove(0);
        let iq1s_qdecode_f16 = raw_matvecs.remove(0);
        let iq2xxs_qdecode_f16 = raw_matvecs.remove(0);
        let iq1m_qdecode_f16 = raw_matvecs.remove(0);
        let iq2s_qdecode_f16 = raw_matvecs.remove(0);
        let iq2xs_qdecode_f16 = raw_matvecs.remove(0);
        let iq3xxs_qdecode_f16 = raw_matvecs.remove(0);
        let iq3s_qdecode_f16 = raw_matvecs.remove(0);
        let iq1s_qdot_matvec = raw_matvecs.remove(0);
        let iq2xxs_qdot_matvec = raw_matvecs.remove(0);
        let iq1m_qdot_matvec = raw_matvecs.remove(0);
        let iq2s_qdot_matvec = raw_matvecs.remove(0);
        let iq2xs_qdot_matvec = raw_matvecs.remove(0);
        let iq3xxs_qdot_matvec = raw_matvecs.remove(0);
        let iq3s_qdot_matvec = raw_matvecs.remove(0);
        let iq4xs_qdot_matvec = raw_matvecs.remove(0);
        let q2k_qdot_matvec = raw_matvecs.remove(0);
        let q3k_qdot_matvec = raw_matvecs.remove(0);
        let iq1s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq3s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq3xxs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2xxs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq1m_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2xs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let q4k_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let q5k_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let q6k_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq1s_qmma = raw_matvecs.remove(0);
        let iq2xxs_qmma = raw_matvecs.remove(0);
        let iq2s_qmma = raw_matvecs.remove(0);
        let iq2xs_qmma = raw_matvecs.remove(0);
        let iq3xxs_qmma = raw_matvecs.remove(0);
        let iq3s_qmma = raw_matvecs.remove(0);
        let iq1s_grid = DeviceBuffer::from_slice(&crate::quant::iq1s_flat_grid())?;
        let iq2xxs_grid = DeviceBuffer::from_slice(&crate::quant::iq2xxs_flat_grid())?;
        let iq2xxs_signs = DeviceBuffer::from_slice(&crate::quant::iq2xxs_flat_signs())?;
        let iq2s_grid = DeviceBuffer::from_slice(&crate::quant::iq2s_flat_grid())?;
        let iq2s_signs = DeviceBuffer::from_slice(&crate::quant::iq2s_flat_signs())?;
        let iq2xs_grid = DeviceBuffer::from_slice(&crate::quant::iq2xs_flat_grid())?;
        let iq3xxs_grid = DeviceBuffer::from_slice(&crate::quant::iq3xxs_flat_grid())?;
        let iq3s_grid = DeviceBuffer::from_slice(&crate::quant::iq3s_flat_grid())?;
        let iq4xs_codebook = DeviceBuffer::from_slice(&crate::quant::iq4xs_flat_codebook())?;
        let iq1s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq1s_packed_grid())?;
        let iq1s_signed_grid = DeviceBuffer::from_slice(&crate::quant::iq1s_signed_grid())?;
        let iq2xxs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2xxs_packed_grid())?;
        let iq2xxs_signs_packed = DeviceBuffer::from_slice(
            &[
                crate::quant::iq2xxs_packed_signs(),
                crate::quant::iq2xxs_sign_masks(),
            ]
            .concat(),
        )?;
        let iq2s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2s_packed_grid())?;
        let iq2s_signs_packed = DeviceBuffer::from_slice(
            &[
                crate::quant::iq2s_packed_signs(),
                crate::quant::iq2s_sign_masks(),
            ]
            .concat(),
        )?;
        let iq2xs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2xs_packed_grid())?;
        let iq3xxs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq3xxs_packed_grid())?;
        let iq3s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq3s_packed_grid())?;
        let iota: Vec<i32> = (0..8).collect();
        let iota8 = DeviceBuffer::from_slice(&iota)?;

        residency::vram_mark("kernels compiled and loaded");
        Ok(DeviceBackend {
            stream,
            matmul,
            matmul_tc,
            matmul_f16w,
            matmul_tc_f16w,
            matvec,
            q8_dp4a,
            q8_mma,
            q8_qmma,
            q8_qmma_deep,
            q8_qmma_split: RefCell::new(HashMap::new()),
            q8_qmma_narrow: RefCell::new(None),
            q8_split,
            q8_qdot,
            q8_qdot_add,
            q8_qdot_persist: RefCell::new(HashMap::new()),
            persist_blocks: Cell::new(0),
            persist_qdot: std::env::var_os("PHOBOS_PERSIST_QDOT").is_some(),
            iq1s_dp4a: Cell::new(env_flag_on("PHOBOS_IQ1S_DP4A")),
            qmma_split: env_flag("PHOBOS_QMMA_SPLIT"),
            qmma_narrow: env_flag("PHOBOS_QMMA_NARROW"),
            fused_plans: RefCell::new(HashMap::new()),
            fused_mlp: fused_stage("PHOBOS_FUSED_MLP"),
            fused_project: fused_stage("PHOBOS_FUSED_PROJ"),
            fused_mix: fused_stage("PHOBOS_FUSED_MIX"),
            fused_attn_out: fused_stage("PHOBOS_FUSED_ATTN_OUT"),
            fused_store2d: fused_stage("PHOBOS_FUSED_STORE2D"),
            fused_blocks: Cell::new(0),
            fused_scratch: RefCell::new(Vec::new()),
            fused_barrier: RefCell::new(None),
            fused_operands: RefCell::new(Vec::new()),
            functions: RefCell::new(HashMap::new()),
            func_shared: RefCell::new(HashMap::new()),
            func_threads: RefCell::new(HashMap::new()),
            eager: RefCell::new(Recorded::default()),
            recording: Cell::new(false),
            flushed: Cell::new(false),
            pending: RefCell::new(Vec::new()),
            recorded_len: Cell::new(0),
            pass: RefCell::new(None),
            // The fourth replay by default: past the prefill and past the
            // warmup passes whose scratch arenas are still growing.
            report_pass: Cell::new(match std::env::var("PHOBOS_PASS_REPORT") {
                Ok(v) if v.is_empty() => 4,
                Ok(v) => v.parse().unwrap_or(4),
                Err(_) => 0,
            }),
            report: RefCell::new(Vec::new()),
            reported: Cell::new(0),
            quantize,
            quantize_wide,
            pointwise,
            pointwise_wide,
            norms: RefCell::new(HashMap::new()),
            quant_norms: RefCell::new(HashMap::new()),
            gated_norms: RefCell::new(HashMap::new()),
            gated_swiglu: RefCell::new(HashMap::new()),
            swiglu_planes: RefCell::new(HashMap::new()),
            deltas: RefCell::new(HashMap::new()),
            chunks: RefCell::new(HashMap::new()),
            identities: RefCell::new(HashMap::new()),
            convs: RefCell::new(HashMap::new()),
            gates: RefCell::new(HashMap::new()),
            splits: RefCell::new(HashMap::new()),
            store_pairs: RefCell::new(HashMap::new()),
            ropes: RefCell::new(HashMap::new()),
            rope_gathers: RefCell::new(HashMap::new()),
            attentions: RefCell::new(HashMap::new()),
            split_attn: RefCell::new(HashMap::new()),
            attn_persist: !matches!(
                std::env::var("PHOBOS_ATTN_PERSIST").as_deref(),
                Ok("0" | "off" | "no" | "false")
            ),
            attn_persist_modules: RefCell::new(HashMap::new()),
            attn_partials: RefCell::new(None),
            readback: RefCell::new(None),
            argmax_reduce: RefCell::new(HashMap::new()),
            argmax_iota: RefCell::new(HashMap::new()),
            argmax_finish,
            argmax_scratch: RefCell::new(None),
            blocked: RefCell::new(HashMap::new()),
            attn_gemm: RefCell::new(HashMap::new()),
            slots: RefCell::new(Vec::new()),
            free_slots: RefCell::new(Vec::new()),
            pool: Pool::new(),
            dense_scratch: [const { Cell::new(None) }; 2],
            drop_scratch: Cell::new(false),
            pass_marks: Cell::new(0),
            alloc_hist: RefCell::new(HashMap::new()),
            constants: RefCell::new(HashMap::new()),
            quants: RefCell::new(Vec::new()),
            q_constants: RefCell::new(HashMap::new()),
            raw_quants: RefCell::new(Vec::new()),
            arena: arena::Arena::default(),
            hot: arena::Arena::with_slab(arena::HOT_SLAB_BYTES),
            state_arena: arena::Arena::default(),
            state_live: Cell::new(0),
            state_arena_on: env_flag("PHOBOS_STATE_ARENA"),
            arena_const: env_flag("PHOBOS_ARENA_CONST"),
            arena_weights: env_flag_on("PHOBOS_ARENA"),
            owned_quants: RefCell::new(Vec::new()),
            owned_raw: RefCell::new(Vec::new()),
            raw_constants: RefCell::new(HashMap::new()),
            q2k_matvec,
            q3k_matvec,
            iq1s_matvec,
            iq2xxs_matvec,
            iq1m_matvec,
            iq2s_matvec,
            iq2xs_matvec,
            iq3xxs_matvec,
            iq3s_matvec,
            iq4xs_matvec,
            iq1s_dequant,
            iq1s_qdecode,
            iq2xxs_qdecode,
            iq1m_qdecode,
            iq2s_qdecode,
            iq2xs_qdecode,
            iq3xxs_qdecode,
            iq3s_qdecode,
            iq1s_qdecode_f16,
            iq2xxs_qdecode_f16,
            iq1m_qdecode_f16,
            iq2s_qdecode_f16,
            iq2xs_qdecode_f16,
            iq3xxs_qdecode_f16,
            iq3s_qdecode_f16,
            iq2xxs_dequant,
            iq1m_dequant,
            iq2s_dequant,
            iq2xs_dequant,
            iq3xxs_dequant,
            iq3s_dequant,
            iq4xs_dequant,
            q2k_dequant,
            iq1s_qdot_matvec,
            iq1s_qdot_i8,
            iq3s_qdot_i8,
            iq3xxs_qdot_i8,
            iq2xxs_qdot_i8,
            iq1m_qdot_i8,
            iq2xs_qdot_i8,
            iq2s_qdot_i8,
            q4k_qdot_i8,
            q5k_qdot_i8,
            q6k_qdot_i8,
            iq2xxs_qdot_matvec,
            iq1m_qdot_matvec,
            iq2s_qdot_matvec,
            iq2xs_qdot_matvec,
            iq3xxs_qdot_matvec,
            iq3s_qdot_matvec,
            iq4xs_qdot_matvec,
            q2k_qdot_matvec,
            q3k_qdot_matvec,
            iq1s_grid,
            iq2xxs_grid,
            iq2xxs_signs_packed,
            iq2s_grid,
            iq2s_signs_packed,
            iq2xs_grid,
            iq3xxs_grid,
            iq3s_grid,
            iq4xs_codebook,
            iq1s_grid_packed,
            iq1s_signed_grid,
            iq1s_qmma,
            iq2xxs_qmma,
            iq2s_qmma,
            iq2xs_qmma,
            iq3xxs_qmma,
            iq3s_qmma,
            raw_qmma: Cell::new(!matches!(
                std::env::var("PHOBOS_RAW_QMMA").as_deref(),
                Ok("0" | "off" | "no" | "false")
            )),
            raw_qmma_formats: qmma_formats(),
            qgemm: qmma_raw::Qgemm::from_env(),
            dense_scratch_shared: env_flag_on("PHOBOS_DENSE_SCRATCH"),
            trim_after_dense: env_flag("PHOBOS_TRIM"),
            iq2xxs_grid_packed,
            iq2xxs_signs,
            iq2s_grid_packed,
            iq2s_signs,
            iq2xs_grid_packed,
            iq3xxs_grid_packed,
            iq3s_grid_packed,
            iota8,
            act_scratch: RefCell::new(Vec::new()),
            act_ring: Cell::new(0),
            act_next: Cell::new(0),
            split_scratch: RefCell::new(None),
            _ctx,
        })
    }
}
