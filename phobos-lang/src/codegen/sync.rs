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
        let i32_t: Type<'c> = IntegerType::new(self.ctx, 32).into();
        let mut operands = vec![val, mem, idx];
        for _ in 1..rank {
            operands.push(self.const_index(block, 0)?);
        }
        self.push(
            block,
            OperationBuilder::new("memref.atomic_rmw", self.loc)
                // AtomicRMWKindAttr is an I64EnumAttr, so the kind is a plain
                // i64. addi is case 1 (mlir Arith/IR/ArithBase.td).
                .add_attributes(&[(
                    self.id("kind"),
                    IntegerAttribute::new(IntegerType::new(self.ctx, 64).into(), 1).into(),
                )])
                .add_operands(&operands)
                .add_results(&[i32_t])
                .build()?,
        )
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

        let then = Block::new(&[]);
        self.emit_arrive_and_wait(&then, mem, rank)?;
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
        let blocks = self.grid_dim(block)?;
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
