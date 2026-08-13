use super::*;

/// Grid-wide synchronization, for kernels that outlive one stage of a pass.
///
/// A decode step is a deep, narrow chain of tiny kernels, and a launch boundary
/// costs about twice what an in-kernel grid barrier does, so a kernel that spans
/// several stages and separates them with a barrier pays less than the same
/// stages launched one at a time.
///
/// `grid_barrier(bar)` takes an `i32` tensor of at least two elements: slot 0 is
/// the arrival counter and slot 1 the release generation. `tensor<i32>[2]` and
/// the `tensor<i32>[2, 1]` column both spell it, the latter for a host whose
/// launch ABI passes rank-2 descriptors. The caller owns it,
/// zeroed once before the launch, and must not touch it again while the kernel
/// runs. Two properties are the caller's to guarantee, and neither is checked:
/// - every block of the grid is resident at once, since a block still waiting to
///   be scheduled never arrives and the barrier deadlocks. That is what
///   `@persistent` is for.
/// - the same barrier tensor is not shared by two concurrent kernels.
///
/// The expansion is the standard two-phase arrive-and-wait, one thread per
/// block doing the atomics and the CTA barriers carrying the result to the rest:
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
/// The counter is reset by subtracting the block count rather than storing zero,
/// so the reset is itself atomic and cannot lose an arrival from a block that
/// has already raced ahead into the next barrier.
///
/// Both the spin and the generation read go through `atomic_add(_, 0)` rather
/// than a plain load: an ordinary load is free to be hoisted out of the spin
/// loop or served from a stale cache line, and the atomic is ordered against
/// the releasing block's writes.
impl<'p, 'c> Codegen<'p, 'c> {
    /// `atomic_add(t, i, v) -> old`: adds `v` to `t[i]` and returns the previous
    /// value, atomically across the whole device. `t` must be an `i32` tensor.
    pub(super) fn emit_atomic_add(&mut self, block: &Block<'c>, args: &[Expr]) -> Result<Rv<'c>> {
        let [t, i, v] = args else {
            bail!("atomic_add expects (tensor, index, value)");
        };
        let (mem, rank) = self.barrier_tensor(t, "atomic_add")?;
        let idx = self.emit_index(block, i, "atomic_add index")?;
        let val = self.emit_scalar(block, v)?;
        // Integer literals lower to index, so the common `atomic_add(B, 0, 1)`
        // arrives here needing a cast rather than a diagnostic.
        let i32_t: Type<'c> = IntegerType::new(self.ctx, 32).into();
        let val = match val.r#type() {
            t if t == i32_t => val,
            t if t == self.index_t || self.is_int(t) => {
                self.push(block, arith::index_cast(val, i32_t, self.loc))?
            }
            t => bail!("atomic_add value must be an integer, got {t}"),
        };
        let old = self.atomic_add_raw(block, mem, rank, idx, val)?;
        Ok(Rv::Scalar(old))
    }

    /// Resolves the atomic-state operand: a named `i32` tensor parameter, and
    /// nothing else. A tile lives in shared memory and a slice has an offset
    /// the atomic would have to fold in, so neither is accepted. Returns the
    /// memref and its rank, which decides how the slot index is subscripted.
    fn barrier_tensor(&self, e: &Expr, what: &str) -> Result<(Value<'c, 'c>, usize)> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named i32 tensor parameter");
        };
        let Some(Binding::Tensor(mem)) = self.lookup(name) else {
            bail!("{what} expects a tensor parameter, but '{name}' is not one");
        };
        if mem.elem != IntegerType::new(self.ctx, 32).into() {
            bail!(
                "{what} expects an i32 tensor, but '{name}' holds {}",
                mem.elem
            );
        }
        Ok((mem.mem, mem.shape.len()))
    }

    /// The `memref.atomic_rmw addi` itself, addressing slot `idx` of an `i32`
    /// tensor of any rank.
    ///
    /// The slot indexes the leading dimension and the rest are zero, so the
    /// state is a rank-1 pair or the `[2, 1]` column a rank-2 launch ABI can
    /// pass without a descriptor of its own.
    fn atomic_add_raw(
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
                // AtomicRMWKindAttr is an I64EnumAttr, so the kind travels as a
                // plain i64: addi is case 1 (mlir Arith/IR/ArithBase.td).
                .add_attributes(&[(
                    self.id("kind"),
                    IntegerAttribute::new(IntegerType::new(self.ctx, 64).into(), 1).into(),
                )])
                .add_operands(&operands)
                .add_results(&[i32_t])
                .build()?,
        )
    }

    /// `grid_barrier(bar)`: every block of the grid waits for every other.
    pub(super) fn emit_grid_barrier(&mut self, block: &Block<'c>, args: &[Expr]) -> Result<Rv<'c>> {
        let [bar] = args else {
            bail!("grid_barrier expects one argument, the barrier tensor");
        };
        let (mem, rank) = self.barrier_tensor(bar, "grid_barrier")?;

        // Publish this stage's writes to the rest of the CTA, and make sure no
        // thread of it is still working when thread 0 arrives.
        self.barrier(block)?;

        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
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

        // The release only reached thread 0; this is what hands it to the CTA,
        // and it orders the next stage's reads after every block's writes.
        self.barrier(block)?;
        Ok(Rv::Scalar(self.const_index(block, 0)?))
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

        // gridDim.x, as an i32, is how many arrivals make a full barrier. A
        // kernel whose grid is not one-dimensional cannot use this.
        let blocks = self.gpu_index(block, "gpu.grid_dim", "x")?;
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

        // Last in: undo every arrival and bump the generation, in that order, so
        // no released block can arrive at the next barrier against a counter
        // that has not been reset yet.
        let release = Block::new(&[]);
        let neg = self.push(&release, arith::subi(zero, blocks, self.loc))?;
        self.atomic_add_raw(&release, mem, rank, count_at, neg)?;
        self.atomic_add_raw(&release, mem, rank, gen_at, one)?;
        release.append_operation(scf::r#yield(&[], self.loc));
        let release_region = Region::new();
        release_region.append_block(release);

        // Everyone else spins until the generation moves off the one they saw.
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

    fn const_i32(&self, block: &Block<'c>, value: i64) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            arith::constant(
                self.ctx,
                IntegerAttribute::new(IntegerType::new(self.ctx, 32).into(), value).into(),
                self.loc,
            ),
        )
    }
}
