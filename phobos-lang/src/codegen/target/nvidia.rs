// NVIDIA, the one implementation of [`Isa`].
//
// Everything here is an nvgpu, nvvm or gpu-dialect op, a PTX string, or a
// number off the chip's data sheet. Nothing here decides which of them a
// contraction wants: that is the emitter's, and it asks the capability
// predicates first.

use super::*;

/// Element type and operand role of the wmma fragments, fixed at 16x16 on
/// every chip that has them.
const WMMA_A: &str = "!gpu.mma_matrix<16x16xf16, \"AOp\">";
const WMMA_B: &str = "!gpu.mma_matrix<16x16xf16, \"BOp\">";
const WMMA_C_F16: &str = "!gpu.mma_matrix<16x16xf16, \"COp\">";
const WMMA_C_F32: &str = "!gpu.mma_matrix<16x16xf32, \"COp\">";

/// Address space of tensor parameters -> GPU global memory.
const MEM_GLOBAL: i64 = 1;

/// Address space of tile buffers -> GPU shared memory (one per CTA).
const MEM_SHARED: i64 = 3;

/// Address space of dynamic tile buffers.
const MEM_SHARED_SYM: &str = "#gpu.address_space<workgroup>";

pub(super) struct Nvidia {
    /// Compute capability as a number: sm_75 is 75, sm_90a is 90.
    cc: u32,
    /// Width index values lower to. The nvgpu ops that form their own
    /// addresses, cp.async and mma.sync both, need the 64-bit one.
    index_bits: u32,
}

impl Nvidia {
    pub(super) fn new(cc: u32, index_bits: u32) -> Nvidia {
        Nvidia { cc, index_bits }
    }

    /// One PTX instruction over an f32, taken through llvm.inline_asm.
    ///
    /// The math dialect does not reach these: convert-math-to-llvm runs before
    /// the gpu-to-nvvm libdevice patterns and rewrites math.exp and friends to
    /// llvm.intr.*, which the NVPTX backend cannot select.
    fn ptx_f32<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        asm: &str,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("llvm.inline_asm", cg.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        cg.id("asm_string"),
                        StringAttribute::new(cg.ctx, asm).into(),
                    ),
                    (
                        cg.id("constraints"),
                        StringAttribute::new(cg.ctx, "=f,f").into(),
                    ),
                    (cg.id("has_side_effects"), Attribute::unit(cg.ctx)),
                ])
                .add_results(&[cg.f32_t])
                .build()?,
        )
    }

    /// A gpu index op (gpu.thread_id / gpu.block_id / gpu.block_dim /
    /// gpu.grid_dim) along one dimension.
    fn gpu_index<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        op: &str,
        dim: &str,
    ) -> Result<Value<'c, 'c>> {
        // Bracket form required for gpu::DimensionAttr.
        let attr = cg.parse_attr(&format!("#gpu<dim {dim}>"))?;
        cg.push(
            block,
            OperationBuilder::new(op, cg.loc)
                .add_attributes(&[(cg.id("dimension"), attr)])
                .add_results(&[cg.index_t])
                .build()?,
        )
    }

    fn async_token_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>> {
        cg.parse_type("!nvgpu.device.async.token")
    }
}

impl Isa for Nvidia {
    // ---- what the chip can do ----

    fn has_cp_async(&self) -> bool {
        self.cc >= 80 && self.index_bits == 64
    }

    fn has_wmma(&self) -> bool {
        self.cc >= 70
    }

    fn has_mma_sync(&self) -> bool {
        self.mma_sync_k().is_some() && self.index_bits == 64
    }

    fn mma_sync_k(&self) -> Option<i64> {
        match self.cc {
            cc if cc >= 80 => Some(16), // Ampere and later: m16n8k16
            cc if cc >= 75 => Some(8),  // Turing's op is m16n8k8
            _ => None,
        }
    }

    fn has_dp4a(&self) -> bool {
        self.cc >= 61
    }

    fn has_int8_mma(&self) -> bool {
        self.cc >= 75
    }

    fn smem_per_sm(&self) -> i64 {
        (match self.cc {
            cc if cc >= 90 => 228, // Hopper
            87 => 164,             // Orin
            cc if cc >= 86 => 100, // Ada / GA10x
            cc if cc >= 80 => 164, // A100
            cc if cc >= 75 => 64,  // Turing
            cc if cc >= 70 => 96,  // Volta
            _ => 48,
        }) * 1024
    }

    fn regs_per_sm(&self) -> i64 {
        64 * 1024 // 64K on every CUDA arch since Kepler
    }

    fn max_warps_per_sm(&self) -> i64 {
        match self.cc {
            75 => 32,           // Turing
            86 | 87 | 89 => 48, // Ada / GA10x / Orin
            _ => 64,            // Volta, A100, Hopper
        }
    }

    // ---- where memory lives ----

    fn global_space<'c>(&self, cg: &Codegen<'c>) -> Attribute<'c> {
        IntegerAttribute::new(cg.i64_t, MEM_GLOBAL).into()
    }

    fn shared_space<'c>(&self, cg: &Codegen<'c>, dynamic: bool) -> Result<Attribute<'c>> {
        if dynamic {
            Attribute::parse(cg.ctx, MEM_SHARED_SYM).context("workgroup address space")
        } else {
            Ok(IntegerAttribute::new(cg.i64_t, MEM_SHARED).into())
        }
    }

    fn mem_space_text(&self, shared: bool, dynamic: bool) -> String {
        match shared {
            true if dynamic => MEM_SHARED_SYM.to_string(),
            true => MEM_SHARED.to_string(),
            false => MEM_GLOBAL.to_string(),
        }
    }

    fn dynamic_shared_base<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        byte_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("gpu.dynamic_shared_memory", cg.loc)
                .add_results(&[byte_t])
                .build()?,
        )
    }

    // ---- how a thread finds itself ----

    fn thread_id<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.gpu_index(cg, block, "gpu.thread_id", "x")
    }

    fn block_dim<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.gpu_index(cg, block, "gpu.block_dim", "x")
    }

    fn grid_dim<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.gpu_index(cg, block, "gpu.grid_dim", "x")
    }

    fn block_id<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        dim: &str,
    ) -> Result<Value<'c, 'c>> {
        self.gpu_index(cg, block, "gpu.block_id", dim)
    }

    fn barrier<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()> {
        block.append_operation(OperationBuilder::new("gpu.barrier", cg.loc).build()?);
        Ok(())
    }

    fn kernel_return<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()> {
        block.append_operation(OperationBuilder::new("gpu.return", cg.loc).build()?);
        Ok(())
    }

    fn launch_attrs<'c>(
        &self,
        cg: &Codegen<'c>,
        launch: Launch,
    ) -> Vec<(Identifier<'c>, Attribute<'c>)> {
        let mut attrs = vec![(
            cg.id("nvvm.maxntid"),
            DenseI32ArrayAttribute::new(cg.ctx, &[launch.max_threads as i32]).into(),
        )];
        if let Some(min_blocks) = launch.min_blocks {
            attrs.push((
                cg.id("nvvm.minctasm"),
                IntegerAttribute::new(cg.i32_t, min_blocks).into(),
            ));
        }
        if let Some(max_nreg) = launch.max_nreg {
            attrs.push((
                cg.id("nvvm.maxnreg"),
                IntegerAttribute::new(cg.i32_t, max_nreg).into(),
            ));
        }
        attrs
    }

    // ---- what a warp can share ----

    /// gpu.shuffle xor, which is shfl.sync.bfly after convert-gpu-to-nvvm.
    fn shfl_xor_f32<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        mask: i64,
    ) -> Result<Value<'c, 'c>> {
        let c_i32 = |val: i64| {
            arith::constant(
                cg.ctx,
                IntegerAttribute::new(cg.i32_t, val).into(),
                cg.loc,
            )
        };
        let offset = cg.push(block, c_i32(mask))?;
        let width = cg.push(block, c_i32(WARP))?;
        cg.push(
            block,
            OperationBuilder::new("gpu.shuffle", cg.loc)
                .add_operands(&[v, offset, width])
                .add_attributes(&[(cg.id("mode"), cg.parse_attr("#gpu<shuffle_mode xor>")?)])
                .add_results(&[cg.f32_t, cg.bool_t])
                .build()?,
        )
    }

    // ---- which math is approximate ----

    /// ex2(x * log2e): the hardware primitive is base two, so the change of
    /// base rides on the outside.
    fn approx_exp<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let log2e = cg.push(
            block,
            arith::constant(
                cg.ctx,
                FloatAttribute::new(cg.ctx, cg.f32_t, std::f64::consts::LOG2_E).into(),
                cg.loc,
            ),
        )?;
        let t = cg.push(block, arith::mulf(x, log2e, cg.loc))?;
        self.ptx_f32(cg, block, "ex2.approx.ftz.f32 $0, $1;", t)
    }

    /// lg2(x) * ln2, the mirror of [`Isa::approx_exp`].
    fn approx_log<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let lg2 = self.ptx_f32(cg, block, "lg2.approx.ftz.f32 $0, $1;", x)?;
        let ln2 = cg.push(
            block,
            arith::constant(
                cg.ctx,
                FloatAttribute::new(cg.ctx, cg.f32_t, std::f64::consts::LN_2).into(),
                cg.loc,
            ),
        )?;
        cg.push(block, arith::mulf(lg2, ln2, cg.loc))
    }

    fn approx_sqrt<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.ptx_f32(cg, block, "sqrt.approx.f32 $0, $1;", x)
    }

    /// tanh.approx.f32, which is sm_75 and up.
    fn approx_tanh<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.ptx_f32(cg, block, "tanh.approx.f32 $0, $1;", x)
    }

    /// The hardware's own rounding, so unlike biasing into a positive range and
    /// truncating it loses nothing: a bias large enough to cover the range
    /// costs the low mantissa bits, enough at the top of an int8 range to cross
    /// a boundary.
    fn round_even<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.ptx_f32(cg, block, "cvt.rni.f32.f32 $0, $1;", x)
    }

    // ---- how bytes arrive ----

    fn async_copy<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        src: Value<'c, 'c>,
        src_idx: &[Value<'c, 'c>],
        dst: Value<'c, 'c>,
        dst_idx: &[Value<'c, 'c>],
        width: i64,
        dst_elem_bytes: i64,
    ) -> Result<()> {
        let token_t = self.async_token_type(cg)?;
        let mut operands = vec![dst];

        operands.extend_from_slice(dst_idx);
        operands.push(src);
        operands.extend_from_slice(src_idx);

        let mut attributes = vec![
            (
                cg.id("dstElements"),
                IntegerAttribute::new(cg.index_t, width).into(),
            ),
            (
                cg.id("operandSegmentSizes"),
                cg.i32_array(&[1, dst_idx.len() as i32, 1, src_idx.len() as i32, 0])?,
            ),
        ];

        // Only 16-byte copies can skip L1 (cp.async.cg), and staged tiles are
        // consumed from shared memory rather than re-read through it. A
        // vectorized f16 copy is only 8B, so gate on bytes, not elements.
        if width * dst_elem_bytes == 16 {
            attributes.push((cg.id("bypassL1"), Attribute::unit(cg.ctx)));
        }
        block.append_operation(
            OperationBuilder::new("nvgpu.device_async_copy", cg.loc)
                .add_operands(&operands)
                .add_attributes(&attributes)
                .add_results(&[token_t])
                .build()?,
        );
        Ok(())
    }

    fn async_create_group<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
    ) -> Result<Value<'c, 'c>> {
        let token_t = self.async_token_type(cg)?;
        cg.push(
            block,
            OperationBuilder::new("nvgpu.device_async_create_group", cg.loc)
                .add_results(&[token_t])
                .build()?,
        )
    }

    fn async_wait<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        group: Value<'c, 'c>,
    ) -> Result<()> {
        block.append_operation(
            OperationBuilder::new("nvgpu.device_async_wait", cg.loc)
                .add_operands(&[group])
                .build()?,
        );
        Ok(())
    }

    // ---- wmma ----

    fn wmma_a_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>> {
        cg.parse_type(WMMA_A)
    }

    fn wmma_b_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>> {
        cg.parse_type(WMMA_B)
    }

    fn wmma_c_type<'c>(&self, cg: &Codegen<'c>, elem: Type<'c>) -> Result<Type<'c>> {
        if elem == cg.f16_t {
            cg.parse_type(WMMA_C_F16)
        } else if elem == cg.f32_t {
            cg.parse_type(WMMA_C_F32)
        } else {
            bail!("WMMA accumulation needs an f16 or f32 type, got {elem}")
        }
    }

    fn wmma_const_frag<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        init: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("gpu.subgroup_mma_constant_matrix", cg.loc)
                .add_operands(&[init])
                .add_results(&[c_frag_t])
                .build()?,
        )
    }

    fn wmma_load<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        lead: i64,
        frag_t: Type<'c>,
        transpose: bool,
    ) -> Result<Value<'c, 'c>> {
        let mut operands = vec![mem];

        operands.extend_from_slice(indices);

        let mut attrs = vec![(
            cg.id("leadDimension"),
            IntegerAttribute::new(cg.index_t, lead).into(),
        )];

        if transpose {
            attrs.push((cg.id("transpose"), Attribute::unit(cg.ctx)));
        }

        cg.push(
            block,
            OperationBuilder::new("gpu.subgroup_mma_load_matrix", cg.loc)
                .add_operands(&operands)
                .add_attributes(&attrs)
                .add_results(&[frag_t])
                .build()?,
        )
    }

    fn wmma_store<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        frag: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        lead: i64,
    ) -> Result<()> {
        let mut operands = vec![frag, mem];
        operands.extend_from_slice(indices);
        block.append_operation(
            OperationBuilder::new("gpu.subgroup_mma_store_matrix", cg.loc)
                .add_operands(&operands)
                .add_attributes(&[(
                    cg.id("leadDimension"),
                    IntegerAttribute::new(cg.index_t, lead).into(),
                )])
                .build()?,
        );
        Ok(())
    }

    fn wmma_compute<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("gpu.subgroup_mma_compute", cg.loc)
                .add_operands(&[a, b, acc])
                .add_results(&[c_frag_t])
                .build()?,
        )
    }

    // ---- mma.sync ----

    fn mma_sync<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        shape: Attribute<'c>,
        acc_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("nvgpu.mma.sync", cg.loc)
                .add_operands(&[a, b, acc])
                .add_attributes(&[(cg.id("mmaShape"), shape)])
                .add_results(&[acc_t])
                .build()?,
        )
    }

    fn mma_shape<'c>(
        &self,
        cg: &Codegen<'c>,
        m: i64,
        n: i64,
        k: i64,
    ) -> Result<Attribute<'c>> {
        cg.parse_attr(&format!("[{m}, {n}, {k}]"))
    }

    fn ldmatrix<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: [Value<'c, 'c>; 2],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let trans = cg.parse_attr(if transpose { "true" } else { "false" })?;
        cg.push(
            block,
            OperationBuilder::new("nvgpu.ldmatrix", cg.loc)
                .add_operands(&[mem, indices[0], indices[1]])
                .add_attributes(&[
                    (cg.id("transpose"), trans),
                    (
                        cg.id("numTiles"),
                        IntegerAttribute::new(cg.i32_t, num_tiles).into(),
                    ),
                ])
                .add_results(&[frag_t])
                .build()?,
        )
    }

    // ---- dp4a ----

    fn dot4_accumulate<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let signed = cg.parse_attr("#nvvm.dot_accumulate_type<signed>")?;
        cg.push(
            block,
            OperationBuilder::new("nvvm.dot.accumulate.4way", cg.loc)
                .add_operands(&[a, b, acc])
                .add_attributes(&[(cg.id("a_type"), signed), (cg.id("b_type"), signed)])
                .add_results(&[cg.i32_t])
                .build()?,
        )
    }
}
