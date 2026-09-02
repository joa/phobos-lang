// The target seam: everything the emitter cannot say in a portable dialect.
//
// `Isa` is the vocabulary, `nvidia.rs` is the one implementation of it, and the
// forwarding methods at the bottom are what the rest of codegen/ calls. The
// indirection buys one thing: a second target is a second file here rather than
// a match arm inside thirty emission sites.
//
// Two rules keep the seam honest. Nothing above this module names an
// instruction, and nothing below it names a codegen type: `Isa` speaks in MLIR
// values and plain integers, never in [`MemVal`] or [`Binding`], so the layout
// decisions that read a tile's stride stay on the emitter's side where a second
// target inherits them for free.

use phobos_base::context::GpuConfig;

use super::*;

mod nvidia;

/// One GPU target's instruction vocabulary.
///
/// The capability half is chip data and answers what may be emitted; the rest
/// emits it. A caller that asks the first half a question it does not need is
/// how the two stay separable: `mma_sync_k` returning None is a chip without
/// the instruction, not an error, and the emitter picks another path.
pub(super) trait Isa {
    // ---- what the chip can do ----

    /// Whether the chip has an asynchronous global-to-shared copy.
    fn has_cp_async(&self) -> bool;

    /// Whether the chip has tensor cores at all, ignoring what the kernel asked
    /// for; [`Codegen::has_wmma`] is the question with `@tensorcore` folded in.
    fn has_wmma(&self) -> bool;

    /// Whether the per-lane tensor path is usable. Not the same question as
    /// [`Isa::mma_sync_k`]: the chip can have the instruction and the target
    /// still decline it, which is what the 32-bit index width does here.
    fn has_mma_sync(&self) -> bool;

    /// The k of the native f16 mma.sync, or None on a chip without one.
    fn mma_sync_k(&self) -> Option<i64>;

    /// Whether the chip has a four-way integer dot product.
    fn has_dp4a(&self) -> bool;

    /// Whether the chip has integer tensor cores.
    fn has_int8_mma(&self) -> bool;

    /// Bytes of shared memory one SM hands out across its resident CTAs, and
    /// the two other occupancy limits beside it. The staging-pad decision in
    /// [`Codegen::wmma_should_pad`] is the only caller.
    fn smem_per_sm(&self) -> i64;
    fn regs_per_sm(&self) -> i64;
    fn max_warps_per_sm(&self) -> i64;

    // ---- where memory lives ----

    fn global_space<'c>(&self, cg: &Codegen<'c>) -> Attribute<'c>;

    /// The shared address space. A dynamic allocation names it symbolically and
    /// a view of one has to agree; a static global uses the integer form.
    fn shared_space<'c>(&self, cg: &Codegen<'c>, dynamic: bool) -> Result<Attribute<'c>>;

    /// The same spaces as they are spelled inside a memref type's text.
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

    /// The CTA's index in the grid, the one place a dimension other than x
    /// comes up: it is what `program_id` resolves to.
    fn block_id<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        dim: &str,
    ) -> Result<Value<'c, 'c>>;

    fn barrier<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()>;

    /// Closes a kernel body.
    fn kernel_return<'c>(&self, cg: &Codegen<'c>, block: &Block<'c>) -> Result<()>;

    /// What `@launch` becomes on the emitted function: the occupancy bounds the
    /// assembler reads out of the module.
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
    // All five are f32 in and f32 out, and all five are allowed to be less
    // accurate than the IEEE operation of the same name: the emitter widens f16
    // through them and narrows on the way out.

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

    /// The nearest integer to an f32, ties to even, as an f32. This one is not
    /// approximate and must not be: rounding a quantized value is where the
    /// last mantissa bit decides a byte.
    fn round_even<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;

    // ---- how bytes arrive ----

    /// One asynchronous transfer of `width` elements, src[src_idx] into
    /// dst[dst_idx]. Any token it produces is the target's to drop: the
    /// enclosing stage commits with [`Isa::async_create_group`] and waits on
    /// that.
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
    // Fragments whose per-lane layout the target keeps to itself: the emitter
    // loads one, folds it and stores it without ever indexing inside it.

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

    /// mem[indices] as a fragment. `lead` is the buffer's physical row stride,
    /// which exceeds the logical inner extent when the tile is bank-conflict
    /// padded. `transpose` reads a logically transposed tile out of a row-major
    /// buffer without a separate staging pass.
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
    // The same arithmetic with the fragment layout exposed: operands are
    // ordinary vectors a lane owns, so the emitter picks the shape and has to
    // place the lanes itself.

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
    /// registers, in the [`Isa::mma_sync`] operand layout. The caller has
    /// already folded each lane's address offset into `indices`.
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

    /// The four bytes of `lo` and `hi` picked by the four low nibbles of
    /// `sel`, byte `n` of the result being byte `sel[4n..4n+3]` of the pair
    /// (`lo` is bytes 0 to 3, `hi` 4 to 7). One instruction wherever a
    /// four-entry byte table has to be applied to four selectors at once.
    fn byte_permute<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        sel: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>>;
}

/// The target a config selects. The one arm is not an oversight: a second
/// vendor is a second arm here and a second file beside `nvidia.rs`, which is
/// the whole point of the trait above.
pub(super) fn isa_for(base: &phobos_base::context::Context) -> Box<dyn Isa> {
    match &base.gpu_config {
        GpuConfig::Nvidia(_) => Box::new(nvidia::Nvidia::new(
            base.gpu_config.compute_capability(),
            base.index_bitwidth,
        )),
    }
}

/// What the rest of codegen/ calls. Each of these is the emitter's name for one
/// piece of the vocabulary; the target decides what it becomes.
///
/// The few that are not a bare forward are the ones where a codegen type meets
/// the seam: a tile's stride and element width are the emitter's to know, so
/// they are read here and passed down as plain numbers.
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

    /// The k of the native f16 mma.sync. The caller has already asked
    /// [`Self::has_mma_sync`], so a chip without one is a bug rather than a
    /// fallback.
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

    pub(in crate::codegen) fn barrier(&self, block: &Block<'c>) -> Result<()> {
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
        // Whether the copy can skip L1 is the target's rule but it is keyed on
        // bytes, and only the emitter knows what a tile's element weighs.
        let elem_bytes = if dst.elem == self.f16_t { 2 } else { 4 };
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
