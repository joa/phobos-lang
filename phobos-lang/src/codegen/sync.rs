use super::*;

/// Grid-wide synchronization, for a kernel that runs several stages of a
/// pass with a barrier between them instead of one launch per stage.
///
/// `grid_barrier(bar)` takes an `i32` tensor of at least two elements. Slot 0
/// is the arrival counter and slot 1 the release generation. It may be
/// `tensor<i32>[2]` or the `tensor<i32>[2, 1]` column, for a host whose launch
/// ABI passes rank-2 descriptors. The caller zeroes it once before the launch
/// and leaves it alone while the kernel runs. The caller must also guarantee,
/// unchecked:
/// - every block of the grid is resident at once, or the barrier deadlocks.
///   That is what `@persistent` is for.
/// - no two concurrent kernels share the same barrier tensor.
///
/// The expansion is the standard two-phase arrive-and-wait. One thread per
/// block does the atomics, and CTA barriers carry the result to the rest:
///
///   gpu.barrier                       every thread in the CTA is done
///   if tid == 0 {
///     gen = atomic_add(bar, 1, 0)     read the generation we are leaving
///     if atomic_add(bar, 0, 1) == blocks - 1 {
///       atomic_add(bar, 0, -blocks)   last in: reset the counter
///       atomic_add(bar, 1, 1)         and release everyone
///     } else {
///       while atomic_add(bar, 1, 0) == gen {}
///     }
///   }
///   gpu.barrier                       the release reaches the whole CTA
///
/// The counter is reset by subtracting the block count, not by storing zero.
/// That keeps the reset atomic, so it cannot lose an arrival from a block
/// already racing into the next barrier.
///
/// The spin and the generation read use `atomic_add(_, 0)`, not a plain load.
/// A plain load could be hoisted out of the loop or read a stale cache line.
impl<'c> Codegen<'c> {

    /// A `memref.atomic_rmw addi` on slot `idx` of an `i32` tensor of any
    /// rank. The slot indexes the leading dimension and the other indices are
    /// zero, so both the rank-1 pair and the `[2, 1]` column work.
    pub(super) fn atomic_add_raw(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        rank: usize,
        idx: Value<'c, 'c>,
        val: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let elem = val.r#type();
        let mut operands = vec![val, mem, idx];
        for _ in 1..rank {
            operands.push(self.const_index(block, 0)?);
        }
        self.push(
            block,
            OperationBuilder::new("memref.atomic_rmw", self.loc)
                // AtomicRMWKindAttr is an I64EnumAttr, so the kind is a plain
                // i64. addf is case 0 and addi case 1 (mlir Arith/IR/ArithBase.td).
                .add_attributes(&[(
                    self.id("kind"),
                    IntegerAttribute::new(
                        IntegerType::new(self.ctx, 64).into(),
                        if self.is_float(elem) { 0 } else { 1 },
                    )
                    .into(),
                )])
                .add_operands(&operands)
                .add_results(&[elem])
                .build()?,
        )
    }

    /// A compare-and-swap on slot `idx`: `val` is stored when the slot holds
    /// `expected`, and the old value comes back either way. A
    /// `memref.generic_atomic_rmw`, which lowers to a `cmpxchg` loop.
    pub(super) fn atomic_cas_raw(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        rank: usize,
        idx: Value<'c, 'c>,
        expected: Value<'c, 'c>,
        val: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let elem = val.r#type();
        let mut operands = vec![mem, idx];
        for _ in 1..rank {
            operands.push(self.const_index(block, 0)?);
        }
        let body = Block::new(&[(elem, self.loc)]);
        let cur = detach(body.argument(0)?.into());
        let hit = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, cur, expected, self.loc),
        )?;
        let new = self.push(&body, arith::select(hit, val, cur, self.loc))?;
        body.append_operation(
            OperationBuilder::new("memref.atomic_yield", self.loc)
                .add_operands(&[new])
                .build()?,
        );
        let region = Region::new();
        region.append_block(body);
        self.push(
            block,
            OperationBuilder::new("memref.generic_atomic_rmw", self.loc)
                .add_operands(&operands)
                .add_results(&[elem])
                .add_regions([region])
                .build()?,
        )
    }

    /// One atomic for the whole CTA, which `rmw` emits into thread 0's
    /// branch. Every thread gets its result through shared memory, so what
    /// follows stays CTA-uniform. The leading barrier orders the CTA's
    /// earlier writes before the atomic.
    pub(super) fn cta_atomic(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        rmw: impl FnOnce(&mut Self, &Block<'c>) -> Result<Value<'c, 'c>>,
    ) -> Result<Value<'c, 'c>> {
        let slot = self.alloc_tile_shaped(block, elem, &[1])?;
        self.barrier(block)?;
        let zero = self.const_index(block, 0)?;
        self.leader(block, |cg, then| {
            let old = rmw(cg, then)?;
            then.append_operation(memref::store(old, slot.mem, &[zero], cg.loc));
            Ok(())
        })?;
        self.barrier(block)?;
        let old = self.push(block, memref::load(slot.mem, &[zero], self.loc))?;
        self.release(&slot);
        Ok(old)
    }

    /// Emits `body` into a branch only thread 0 takes.
    pub(super) fn leader(
        &mut self,
        block: &Block<'c>,
        body: impl FnOnce(&mut Self, &Block<'c>) -> Result<()>,
    ) -> Result<()> {
        let tid = self.thread_id(block)?;
        let zero = self.const_index(block, 0)?;
        let is_leader = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, tid, zero, self.loc),
        )?;
        let then = Block::new(&[]);
        body(self, &then)?;
        then.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then);
        block.append_operation(scf::r#if(is_leader, &[], then_region, Region::new(), self.loc));
        Ok(())
    }

    /// The barrier itself, over a resolved state tensor of `rank`.
    pub(super) fn grid_barrier_raw(
        &mut self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        rank: usize,
    ) -> Result<()> {
        // Every thread of the CTA must be done before thread 0 arrives.
        self.barrier(block)?;

        let tid = self.thread_id(block)?;
        let zero_i = self.const_index(block, 0)?;
        let is_leader = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, tid, zero_i, self.loc),
        )?;

        // The fences make the barrier order memory as well as time: every
        // block's global writes before it are visible to every block after.
        // Without them a block may read another's results from a stale cache.
        let then = Block::new(&[]);
        self.fence_gpu(&then)?;
        self.emit_arrive_and_wait(&then, mem, rank)?;
        self.fence_gpu(&then)?;
        then.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then);
        block.append_operation(scf::r#if(
            is_leader,
            &[],
            then_region,
            Region::new(),
            self.loc,
        ));

        // Only thread 0 saw the release. This barrier passes it to the CTA,
        // ordering the next stage's reads after every block's writes.
        self.barrier(block)
    }

    /// Thread 0's half of the barrier: arrive, then either release or spin.
    fn emit_arrive_and_wait(
        &mut self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        rank: usize,
    ) -> Result<()> {
        let count_at = self.const_index(block, 0)?;
        let gen_at = self.const_index(block, 1)?;
        let zero = self.const_i32(block, 0)?;
        let one = self.const_i32(block, 1)?;

        // A full barrier is gridDim.x arrivals, so the grid must be
        // one-dimensional.
        let blocks = self.grid_dim(block, "x")?;
        let blocks = self.push(
            block,
            arith::index_cast(blocks, IntegerType::new(self.ctx, 32).into(), self.loc),
        )?;
        let last = self.push(block, arith::subi(blocks, one, self.loc))?;

        let seen = self.atomic_add_raw(block, mem, rank, gen_at, zero)?;
        let arrived = self.atomic_add_raw(block, mem, rank, count_at, one)?;
        let is_last = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, arrived, last, self.loc),
        )?;

        // Last in: reset the counter, then bump the generation. This order
        // keeps a released block from arriving at a counter not yet reset.
        let release = Block::new(&[]);
        let neg = self.push(&release, arith::subi(zero, blocks, self.loc))?;
        self.atomic_add_raw(&release, mem, rank, count_at, neg)?;
        self.atomic_add_raw(&release, mem, rank, gen_at, one)?;
        release.append_operation(scf::r#yield(&[], self.loc));
        let release_region = Region::new();
        release_region.append_block(release);

        // Everyone else spins until the generation changes.
        let spin = Block::new(&[]);
        let before = Block::new(&[]);
        let now = self.atomic_add_raw(&before, mem, rank, gen_at, zero)?;
        let unchanged = self.push(
            &before,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, now, seen, self.loc),
        )?;
        before.append_operation(scf::condition(unchanged, &[], self.loc));
        let after = Block::new(&[]);
        after.append_operation(scf::r#yield(&[], self.loc));
        let before_region = Region::new();
        before_region.append_block(before);
        let after_region = Region::new();
        after_region.append_block(after);
        spin.append_operation(scf::r#while(
            &[],
            &[],
            before_region,
            after_region,
            self.loc,
        ));
        spin.append_operation(scf::r#yield(&[], self.loc));
        let spin_region = Region::new();
        spin_region.append_block(spin);

        block.append_operation(scf::r#if(
            is_last,
            &[],
            release_region,
            spin_region,
            self.loc,
        ));
        Ok(())
    }
}
