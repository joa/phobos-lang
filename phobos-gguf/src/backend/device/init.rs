// Construction of a `DeviceBackend`: the CUDA context, the compiled kernels,
// the grid tables the IQ formats gather from, and the environment knobs.

use super::*;

impl DeviceBackend {
    /// A backend whose raw-format kernels cover `quants`, the formats of the
    /// model it will run. [`Quant::ALL`] builds every one.
    pub fn new(quants: &[Quant]) -> Result<DeviceBackend> {
        let _ctx = cust::quick_init().context("initializing CUDA")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let copy_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let sms = cust::device::Device::get_device(0)?
            .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
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
        // k is always a whole number of Q8_0 blocks, so the k loop has no
        // remainder.
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
        let argmax_finish = compile(&argmax_finish_src(), &[], "argmax_finish")?;
        let formats = formats::build(&formats::to_build(quants))?;
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
            sms,
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
            q50_kernels: RefCell::new(HashMap::new()),
            persist_blocks: Cell::new(0),
            persist_qdot: env_flag("PHOBOS_PERSIST_QDOT"),
            iq1s_dp4a: Cell::new(env_flag_on("PHOBOS_IQ1S_DP4A")),
            qmma_split: env_flag_on("PHOBOS_QMMA_SPLIT"),
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
            streamed: Cell::new(false),
            pending: RefCell::new(Vec::new()),
            recorded_len: Cell::new(0),
            copy_stream,
            moe_lookahead: env_flag("PHOBOS_MOE_LOOKAHEAD"),
            moe_host: env_flag_on("PHOBOS_MOE_HOST"),
            moe_host_decode: env_flag_on("PHOBOS_MOE_HOST_DECODE"),
            moe_grouped: env_flag_on("PHOBOS_MOE_GROUPED"),
            pass: RefCell::new(Vec::new()),
            segment: Cell::new(0),
            // Defaults to the fourth replay, after prefill and warmup.
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
            delta_reg: RefCell::new(HashMap::new()),
            delta_reg_on: env_flag_on("PHOBOS_DELTA_SCAN"),
            identities: RefCell::new(HashMap::new()),
            convs: RefCell::new(HashMap::new()),
            gates: RefCell::new(HashMap::new()),
            hadamards: RefCell::new(HashMap::new()),
            rows_matmuls: RefCell::new(HashMap::new()),
            splits: RefCell::new(HashMap::new()),
            store_pairs: RefCell::new(HashMap::new()),
            ropes: RefCell::new(HashMap::new()),
            rope_gathers: RefCell::new(HashMap::new()),
            attentions: RefCell::new(HashMap::new()),
            split_attn: RefCell::new(HashMap::new()),
            attn_persist: env_flag_on("PHOBOS_ATTN_PERSIST"),
            attn_persist_modules: RefCell::new(HashMap::new()),
            attn_partials: RefCell::new(None),
            readback: RefCell::new(None),
            argmax_reduce: RefCell::new(HashMap::new()),
            argmax_iota: RefCell::new(HashMap::new()),
            argmax_finish,
            argmax_scratch: RefCell::new(None),
            blocked: RefCell::new(HashMap::new()),
            attn_gemm: RefCell::new(HashMap::new()),
            attn_tc: RefCell::new(HashMap::new()),
            attn_tc_on: env_flag_on("PHOBOS_ATTN_TC"),
            slots: RefCell::new(Vec::new()),
            free_slots: RefCell::new(Vec::new()),
            pool: Pool::new(),
            kernels_reused: Cell::new(0),
            kernels_compiled: Cell::new(0),
            dense_scratch: [const { Cell::new(None) }; 2],
            drop_scratch: Cell::new(false),
            last_rows: Cell::new(0),
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
            experts: RefCell::new(experts::Experts::new()),
            expert_keys: RefCell::new(HashMap::new()),
            moe_topk: RefCell::new(HashMap::new()),
            moe_qdot: RefCell::new(HashMap::new()),
            moe_gateup: RefCell::new(HashMap::new()),
            moe_combine: RefCell::new(HashMap::new()),
            host_add: RefCell::new(None),
            moe_permute: RefCell::new(HashMap::new()),
            moe_qgemm: RefCell::new(HashMap::new()),
            moe_gather_add: RefCell::new(HashMap::new()),
            moe_shared: RefCell::new(HashMap::new()),
            formats,
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
            raw_qmma: Cell::new(env_flag_on("PHOBOS_RAW_QMMA")),
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
            act_arena: arena::Arena::with_slab(arena::ACT_SLAB_BYTES),
            act_ring: Cell::new(0),
            act_shared: Cell::new(mem::ACT_RING),
            act_next: Cell::new(0),
            split_scratch: RefCell::new(None),
            _ctx,
        })
    }
}

/// The card, as the driver describes it.
///
/// Call it once: reading the display driver version spawns a subprocess.
pub(super) fn device_info() -> Option<phobos_inference::DeviceInfo> {
    use cust::device::DeviceAttribute;

    let device = cust::device::Device::get_device(0).ok()?;
    let attribute = |which| device.get_attribute(which).ok().unwrap_or(0).max(0) as u32;
    let api = cust::CudaApiVersion::get().ok();
    Some(phobos_inference::DeviceInfo {
        name: device.name().ok()?,
        capability: (
            attribute(DeviceAttribute::ComputeCapabilityMajor),
            attribute(DeviceAttribute::ComputeCapabilityMinor),
        ),
        multiprocessors: attribute(DeviceAttribute::MultiprocessorCount),
        core_clock_khz: attribute(DeviceAttribute::ClockRate),
        memory_clock_khz: attribute(DeviceAttribute::MemoryClockRate),
        memory_bus_bits: attribute(DeviceAttribute::GlobalMemoryBusWidth),
        cuda: api
            .map(|v| (v.major().max(0) as u32, v.minor().max(0) as u32))
            .unwrap_or((0, 0)),
        driver: display_driver(),
    })
}

/// The display driver's release version, which the CUDA API cannot report.
///
/// Read from `nvidia-smi`. `None` if it is not on the path or fails.
fn display_driver() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| line.to_string())
}
