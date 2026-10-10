// The target seam: everything the emitter cannot say in a portable dialect.
//
// `Isa` is the vocabulary and `nvidia.rs` its one implementation. The
// forwarding methods below are what the rest of codegen/ calls. A second
// target is a second file here.
//
// Nothing above this module names an instruction, and nothing below it names
// a codegen type. `Isa` speaks only in MLIR values and plain integers.

use phobos_base::context::GpuConfig;

use super::*;

mod nvidia;

/// One GPU target's instruction vocabulary.
///
/// The capability methods say what may be emitted, and the rest emit it. A
/// capability that is absent is not an error; the emitter picks another path.
pub(super) trait Isa {
    // ---- what the chip can do ----

    /// Whether the chip has an asynchronous global-to-shared copy.
    fn has_cp_async(&self) -> bool;

    /// Whether the chip has tensor cores, ignoring what the kernel asked for.
    /// [`Codegen::has_wmma`] also checks `@tensorcore`.
    fn has_wmma(&self) -> bool;

    /// Whether the per-lane tensor path is usable. This can be false even when
    /// [`Isa::mma_sync_k`] is Some, for example under 32-bit indices.
    fn has_mma_sync(&self) -> bool;

    /// The k of the native f16 mma.sync, or None on a chip without one.
    fn mma_sync_k(&self) -> Option<i64>;

    /// Whether the chip has a four-way integer dot product.
    fn has_dp4a(&self) -> bool;

    /// Whether the chip has integer tensor cores.
    fn has_int8_mma(&self) -> bool;

    /// Whether to convert an i32 to f32 with `cvt` rather than an integer
    /// add and a float add. The pair keeps the conversion off its own
    /// narrow pipe; from sm_80 on, where an epilogue already fills the float
    /// and integer pipes, the one instruction measured faster. Untested
    /// below sm_80.
    fn cheap_int_to_float(&self) -> bool;

    /// The k of the int8 `mma.sync` the integer intrinsics issue: 16
    /// (`m8n8k16`) through Hopper, 32 (`m16n8k32`) from Blackwell on, which
    /// has no native `m8n8k16` and runs it as a half-used `m16n8k16`.
    fn int8_mma_k(&self) -> i64;

    /// Shared memory bytes one SM splits across its resident CTAs, plus the
    /// two other occupancy limits below. Only [`Codegen::wmma_should_pad`]
    /// reads them.
    fn smem_per_sm(&self) -> i64;
    fn regs_per_sm(&self) -> i64;
    fn max_warps_per_sm(&self) -> i64;

    // ---- where memory lives ----

    fn global_space<'c>(&self, cg: &Codegen<'c>) -> Attribute<'c>;

    /// The shared address space. A dynamic allocation and its views use the
    /// symbolic form, and a static global uses the integer form.
    fn shared_space<'c>(&self, cg: &Codegen<'c>, dynamic: bool) -> Result<Attribute<'c>>;

    /// The same spaces as spelled inside a memref type's text.
    fn mem_space_text(&self, shared: bool, dynamic: bool) -> String;

    /// The CTA's one dynamic shared allocation, as a byte buffer the tile views
    /// carve up.
    fn dynamic_shared_base<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        byte_t: Type<'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- how a thread finds itself ----

    fn thread_id<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>>;
    fn block_dim<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>>;
    fn grid_dim<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<Value<'c, 'c>>;

    /// The CTA's index in the grid along `dim`, which is what `program_id`
    /// resolves to.
    fn block_id<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        dim: &str,
    ) -> Result<Value<'c, 'c>>;

    fn barrier<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()>;

    /// Closes a kernel body.
    fn kernel_return<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()>;

    /// The function attributes `@launch` becomes, the occupancy bounds the
    /// assembler reads.
    fn launch_attrs<'c>(
        &self,
        cg: &Codegen<'c>,
        launch: Launch,
    ) -> Vec<(Identifier<'c>, Attribute<'c>)>;

    // ---- what a warp can share ----

    /// Prefetches the line holding `mem[indices]` into L2.
    fn prefetch_read<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<()>;

    /// The value of v on the lane whose id differs in the given xor mask bits.
    /// Every lane of the executing warp must reach this.
    fn shfl_xor_f32<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        mask: i64,
    ) -> Result<Value<'c, 'c>>;

    // ---- which math is approximate ----
    //
    // All five take and return f32. The approx ones may be less accurate than
    // the IEEE operation. The emitter widens f16 to f32 around them.

    fn approx_exp<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    fn approx_log<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    fn approx_sqrt<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    fn approx_tanh<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    /// The nearest integer to an f32, ties to even, as an f32. This must be
    /// exact, because quantization rounds with it.
    fn round_even<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- how bytes arrive ----

    /// One asynchronous copy of `width` elements from src[src_idx] into
    /// dst[dst_idx]. The target drops any token it produces. The caller
    /// commits with [`Isa::async_create_group`] and waits on that.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<()>;

    fn async_create_group<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
    ) -> Result<Value<'c, 'c>>;

    fn async_wait<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        group: Value<'c, 'c>,
    ) -> Result<()>;

    // ---- the opaque-fragment tensor path ----
    //
    // Fragments with a per-lane layout only the target knows. The emitter
    // loads, computes on and stores them, but never indexes inside one.

    fn wmma_a_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>>;
    fn wmma_b_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>>;
    fn wmma_c_type<'c>(&self, cg: &Codegen<'c>, elem: Type<'c>) -> Result<Type<'c>>;

    fn wmma_const_frag<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        init: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>>;

    /// Loads mem[indices] as a fragment.
    ///
    /// `lead` is the physical row stride, larger than the logical width when
    /// the tile is padded. `transpose` reads a transposed tile straight from a
    /// row-major buffer.
    #[allow(clippy::too_many_arguments)]
    fn wmma_load<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        lead: i64,
        frag_t: Type<'c>,
        transpose: bool,
    ) -> Result<Value<'c, 'c>>;

    /// mem[indices] = frag, the store side of [`Isa::wmma_load`].
    fn wmma_store<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        frag: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        lead: i64,
    ) -> Result<()>;

    /// acc + a * b over one fragment.
    fn wmma_compute<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- the per-lane tensor path ----
    //
    // The same arithmetic with the fragment layout exposed. Operands are
    // ordinary per-lane vectors, so the emitter picks the shape and places
    // the lanes itself.

    /// acc + a * b over one [`Isa::mma_shape`].
    #[allow(clippy::too_many_arguments)]
    fn mma_sync<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        shape: Attribute<'c>,
        acc_t: Type<'c>,
    ) -> Result<Value<'c, 'c>>;

    fn mma_shape<'c>(
        &self,
        cg: &Codegen<'c>,
        m: i64,
        n: i64,
        k: i64,
    ) -> Result<Attribute<'c>>;

    /// A warp-collective load of `num_tiles` 8x8 f16 fragments into per-lane
    /// registers, in the [`Isa::mma_sync`] operand layout. `indices` already
    /// includes each lane's address offset.
    #[allow(clippy::too_many_arguments)]
    fn ldmatrix<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: [Value<'c, 'c>; 2],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- the four-way integer dot ----

    /// acc + dot(a, b) over four signed bytes apiece.
    fn dot4_accumulate<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- byte permutation ----

    /// Picks four bytes out of the pair `lo`, `hi` using the four low nibbles
    /// of `sel`. Byte `n` of the result is byte `sel[4n..4n+3]` of the pair,
    /// where `lo` holds bytes 0 to 3 and `hi` bytes 4 to 7.
    fn byte_permute<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        sel: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;
}

/// The target a config selects. A second vendor adds an arm here and a file
/// beside `nvidia.rs`.
pub(super) fn isa_for(base: &phobos_base::context::Context) -> Box<dyn Isa> {
    match &base.gpu_config {
        GpuConfig::Nvidia(_) => Box::new(nvidia::Nvidia::new(
            base.gpu_config.compute_capability(),
            base.index_bitwidth,
        )),
    }
}

/// What the rest of codegen/ calls, one method per piece of the vocabulary.
/// Most just forward. The rest read codegen facts, such as a tile's stride or
/// element width, and pass them down as plain numbers.
impl<'c> Codegen<'c> {
    // ---- what the chip can do, with what the kernel asked for folded in ----

    pub(super) fn has_cp_async(&self) -> bool {
        self.isa.has_cp_async()
    }

    pub(super) fn has_wmma(&self) -> bool {
        self.tensorcore && self.isa.has_wmma()
    }

    pub(super) fn has_mma_sync(&self) -> bool {
        self.mma_sync && self.isa.has_mma_sync()
    }

    pub(super) fn has_dp4a(&self) -> bool {
        self.isa.has_dp4a()
    }

    pub(super) fn has_int8_mma(&self) -> bool {
        self.isa.has_int8_mma()
    }

    pub(super) fn cheap_int_to_float(&self) -> bool {
        self.isa.cheap_int_to_float()
    }

    pub(super) fn int8_mma_k(&self) -> i64 {
        self.isa.int8_mma_k()
    }

    /// The k of the native f16 mma.sync. Call only after
    /// [`Self::has_mma_sync`]; a chip without one is an error here.
    pub(super) fn mma_sync_k(&self) -> Result<i64> {
        self.isa
            .mma_sync_k()
            .ok_or_else(|| anyhow!("mma.sync needs a chip with a native shape (sm_75+)"))
    }

    // ---- where memory lives ----

    pub(super) fn global_space(&self) -> Attribute<'c> {
        self.isa.global_space(self)
    }

    pub(super) fn shared_space(&self) -> Result<Attribute<'c>> {
        self.isa.shared_space(self, self.dynamic_shared)
    }

    pub(in crate::codegen) fn mem_space(&self, shared: bool) -> String {
        self.isa.mem_space_text(shared, self.dynamic_shared)
    }

    pub(in crate::codegen) fn dynamic_shared_base(
        &self,
        block: &Block<'c>,
        byte_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.dynamic_shared_base(self, block, byte_t)
    }

    // ---- how a thread finds itself ----

    pub(in crate::codegen) fn thread_id(&self, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.isa.thread_id(self, block)
    }

    pub(in crate::codegen) fn block_dim(&self, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.isa.block_dim(self, block)
    }

    pub(in crate::codegen) fn grid_dim(&self, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        self.isa.grid_dim(self, block)
    }

    pub(in crate::codegen) fn block_id(
        &self,
        block: &Block<'c>,
        dim: &str,
    ) -> Result<Value<'c, 'c>> {
        self.isa.block_id(self, block, dim)
    }

    /// A CTA barrier. A recording emission logs it. A replaying emission
    /// skips the one barrier call the membar pass elided for this op.
    pub(in crate::codegen) fn barrier(&mut self, block: &Block<'c>) -> Result<()> {
        if matches!(self.policy, SharedPolicy::Record) {
            self.trace.barrier();
        }
        if let Some(k) = self.skip_barrier {
            self.barrier_calls += 1;
            if self.barrier_calls == k {
                return Ok(());
            }
        }
        self.isa.barrier(self, block)
    }

    pub(in crate::codegen) fn kernel_return(&self, block: &Block<'c>) -> Result<()> {
        self.isa.kernel_return(self, block)
    }

    pub(in crate::codegen) fn launch_attrs(
        &self,
        launch: Launch,
    ) -> Vec<(Identifier<'c>, Attribute<'c>)> {
        self.isa.launch_attrs(self, launch)
    }

    // ---- what a warp can share ----

    pub(in crate::codegen) fn shfl_xor_f32(
        &self,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        mask: i64,
    ) -> Result<Value<'c, 'c>> {
        self.isa.shfl_xor_f32(self, block, v, mask)
    }


    // ---- which math is approximate ----

    pub(in crate::codegen) fn approx_exp(
        &self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.approx_exp(self, block, x)
    }

    pub(in crate::codegen) fn approx_log(
        &self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.approx_log(self, block, x)
    }

    pub(in crate::codegen) fn approx_sqrt(
        &self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.approx_sqrt(self, block, x)
    }

    pub(in crate::codegen) fn approx_tanh(
        &self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.approx_tanh(self, block, x)
    }

    pub(in crate::codegen) fn round_even(
        &self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.round_even(self, block, x)
    }

    // ---- how bytes arrive ----

    pub(in crate::codegen) fn async_copy(
        &self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        src_idx: &[Value<'c, 'c>],
        dst: &MemVal<'c>,
        dst_idx: &[Value<'c, 'c>],
        width: i64,
    ) -> Result<()> {
        // The target decides whether the copy skips L1 by its size in bytes.
        let elem_bytes = self
            .elem_bytes(dst.elem)
            .map(i64::from)
            .ok_or_else(|| anyhow!("async copy of an element of unknown size"))?;
        self.isa.async_copy(
            self, block, src.mem, src_idx, dst.mem, dst_idx, width, elem_bytes,
        )
    }

    pub(in crate::codegen) fn async_create_group(
        &self,
        block: &Block<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.async_create_group(self, block)
    }

    pub(in crate::codegen) fn async_wait(
        &self,
        block: &Block<'c>,
        group: Value<'c, 'c>,
    ) -> Result<()> {
        self.isa.async_wait(self, block, group)
    }

    // ---- the opaque-fragment tensor path ----

    pub(in crate::codegen) fn wmma_a_type(&self) -> Result<Type<'c>> {
        self.isa.wmma_a_type(self)
    }

    pub(in crate::codegen) fn wmma_b_type(&self) -> Result<Type<'c>> {
        self.isa.wmma_b_type(self)
    }

    pub(in crate::codegen) fn wmma_c_type(&self, elem: Type<'c>) -> Result<Type<'c>> {
        self.isa.wmma_c_type(self, elem)
    }

    pub(in crate::codegen) fn wmma_const_frag(
        &self,
        block: &Block<'c>,
        init: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.wmma_const_frag(self, block, init, c_frag_t)
    }

    pub(in crate::codegen) fn wmma_load(
        &self,
        block: &Block<'c>,
        buf: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
        frag_t: Type<'c>,
        transpose: bool,
    ) -> Result<Value<'c, 'c>> {
        let lead = buf.row_stride.unwrap_or(buf.shape[1]);
        self.isa
            .wmma_load(self, block, buf.mem, indices, lead, frag_t, transpose)
    }

    pub(in crate::codegen) fn wmma_store(
        &self,
        block: &Block<'c>,
        frag: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        lead: i64,
    ) -> Result<()> {
        self.isa.wmma_store(self, block, frag, mem, indices, lead)
    }

    pub(in crate::codegen) fn wmma_compute(
        &self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.wmma_compute(self, block, a, b, acc, c_frag_t)
    }

    // ---- the per-lane tensor path ----

    pub(in crate::codegen) fn mma_sync(
        &self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        shape: Attribute<'c>,
        acc_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.mma_sync(self, block, a, b, acc, shape, acc_t)
    }

    pub(in crate::codegen) fn mma_shape(&self, m: i64, n: i64, k: i64) -> Result<Attribute<'c>> {
        self.isa.mma_shape(self, m, n, k)
    }

    pub(in crate::codegen) fn ldmatrix(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: [Value<'c, 'c>; 2],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa
            .ldmatrix(self, block, mem, indices, num_tiles, transpose, frag_t)
    }

    // ---- the four-way integer dot ----

    pub(in crate::codegen) fn prefetch_read(
        &self,
        block: &Block<'c>,
        mem: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<()> {
        self.isa.prefetch_read(self, block, mem, indices)
    }

    pub(in crate::codegen) fn dot4_accumulate(
        &self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.dot4_accumulate(self, block, a, b, acc)
    }

    pub(in crate::codegen) fn byte_permute(
        &self,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        sel: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.isa.byte_permute(self, block, lo, hi, sel)
    }
}

/// The chip facts the AST-to-IR build needs, read from the same [`Isa`] the
/// emitter uses so the two always agree.
pub fn build_target(base: &phobos_base::context::Context) -> crate::ir::build::Target {
    let isa = isa_for(base);
    crate::ir::build::Target {
        has_cp_async: isa.has_cp_async(),
        has_wmma: isa.has_wmma(),
        has_mma_sync: isa.has_mma_sync(),
        mma_sync_k: isa.mma_sync_k(),
        has_dp4a: isa.has_dp4a(),
        has_int8_mma: isa.has_int8_mma(),
    }
}
