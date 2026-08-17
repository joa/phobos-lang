use super::*;

/// Conservative ceiling used only to decide whether doubling a staged tile's
/// buffers would still fit, before committing a bare/auto-attempted loop to
/// pipelining. Mirrors `phobos_kernels::launch::STATIC_SHARED_LIMIT` (48
/// KiB): phobos-lang cannot depend on phobos-kernels (phobos-kernels depends
/// on phobos-lang), so this is a plain literal kept in sync by hand rather
/// than a shared constant. It is the CUDA *static* (non-dynamic-shared)
/// per-block ceiling every target this codebase compiles for shares; a
/// kernel that opts into `@dynshared` can be raised past this at launch (see
/// `phobos-gguf`'s `compile_dynamic`), but pipelining eligibility is decided
/// here, at compile time, before that opt-in is visible, so staying with the
/// conservative static bound is the safe default even for a dynamic-shared
/// kernel.
const PIPELINE_SHARED_LIMIT_BYTES: i64 = 48 * 1024;

/// Why an eligible-looking loop body was not turned into a pipelined
/// (double-buffered) loop. Every variant is a decline, not an error: a bare,
/// auto-attempted loop that declines just falls through to the ordinary
/// codegen path (see `emit_for_inner`). The only place one of these turns
/// into a hard failure is an explicit `@pipeline` on the kernel finding zero
/// eligible loops anywhere in its body -- see `Codegen::pipelined_any` and
/// `emit`'s check of it.
pub(super) enum PipelineDecline {
    /// The body has no leading run of `var t = <static tensor slice>; ...`
    /// statements at all: a loop that opens with ordinary compute, or a
    /// kernel with no for loop shaped like this in the first place.
    NoStagedPrefix,
    /// A staged slice's shape cannot be proven to tile evenly (see
    /// `slice_is_partial`); double-buffering it would prefetch past the
    /// source on the last tile the same way the plain unmasked path would.
    PartialSlice,
    /// A statement after the staged prefix writes one of the staged names,
    /// so the value read at prefetch time would not be the one the compute
    /// half expects.
    StagedNameWritten,
    /// A staged slice's element type has no known byte width, so its
    /// footprint cannot be proven to fit even conservatively. Declining here
    /// (rather than assuming a width) keeps this check fail-safe.
    UnknownElementWidth,
    /// Doubling every staged slice's shared buffer would not fit the
    /// target's shared-memory ceiling.
    SharedMemoryBudget { needed_bytes: i64, limit_bytes: i64 },
}

impl std::fmt::Display for PipelineDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineDecline::NoStagedPrefix => {
                write!(f, "no leading run of staged tensor-slice `var` statements")
            }
            PipelineDecline::PartialSlice => write!(
                f,
                "a staged slice's shape cannot be proven to tile evenly (partial slice)"
            ),
            PipelineDecline::StagedNameWritten => write!(
                f,
                "a staged name is written again later in the loop body"
            ),
            PipelineDecline::UnknownElementWidth => write!(
                f,
                "a staged slice's element type has no known byte width"
            ),
            PipelineDecline::SharedMemoryBudget {
                needed_bytes,
                limit_bytes,
            } => write!(
                f,
                "doubling the staged buffers would need {needed_bytes} bytes of shared \
                 memory, over the {limit_bytes}-byte ceiling"
            ),
        }
    }
}

impl<'c> Codegen<'c> {
    /// Matches a loop body of the form "var t = <static tensor slice>; ..."
    /// whose remaining statements never write the staged names, and whose
    /// staged buffers would still fit shared memory once doubled. Returns the
    /// staged (name, slice expr) pairs and the compute statements, or the
    /// reason the loop is not (yet) pipelined.
    ///
    /// Deliberately does not check whether the loop's own bounds could be
    /// lane-divergent (which would make `emit_pipelined_for`'s barrier-in-
    /// an-`scf.if` guard a hang instead of a miscompute): every `.ph`-level
    /// loop bound is CTA-uniform by construction in this language, a
    /// structural property with no runtime check needed. See the block
    /// comment at the end of this file for the proof and what would have to
    /// change in this codegen for that to stop being true.
    #[allow(clippy::type_complexity)]
    pub(super) fn pipeline_candidate<'a>(
        &self,
        body: &'a [Stmt],
    ) -> Result<(Vec<(&'a str, &'a Expr)>, &'a [Stmt]), PipelineDecline> {
        let mut staged = Vec::new();
        let mut rest = body;
        while let [
            Stmt::Var {
                name,
                ty: None,
                value: Some(value),
            },
            tail @ ..,
        ] = rest
        {
            if self.slice_static_shape(value).is_none() {
                break;
            }
            staged.push((name.as_str(), value));
            rest = tail;
        }

        if staged.is_empty() {
            return Err(PipelineDecline::NoStagedPrefix);
        }
        if staged.iter().any(|(_, v)| self.slice_is_partial(v)) {
            return Err(PipelineDecline::PartialSlice);
        }
        let names: Vec<&str> = staged.iter().map(|(n, _)| *n).collect();
        if rest.iter().any(|s| s.writes_any(&names)) {
            return Err(PipelineDecline::StagedNameWritten);
        }

        // Gap 2 (shared-memory budget): double-buffering doubles every
        // staged slice's footprint. An opt-in `@pipeline` author was
        // implicitly asserting they had checked this fits; an auto-attempt
        // cannot make that assumption, so it is checked here. Conservative
        // on two axes: it ignores that a released buffer can be reused from
        // the pool (so it can decline a loop that would in fact fit), and it
        // is blind to static (non-`@dynshared`) globals this codegen never
        // tracks in `shared_bytes` at all (so a kernel already heavy on
        // static shared can still be pipelined past the real ceiling here --
        // that degrades to ptxas's own "uses too much shared data" compile
        // error, not a hang, so the blind spot is safe, just not caught
        // early with a good message).
        let mut needed_bytes = 0i64;
        for (_, expr) in &staged {
            let shape = self
                .slice_static_shape(expr)
                .expect("checked above: staged slices have a static shape");
            let elem = self
                .slice_tensor_elem(expr)
                .expect("checked above: staged slices index a named tensor");
            let Some(width) = self.elem_bytes(elem) else {
                return Err(PipelineDecline::UnknownElementWidth);
            };
            let bytes = i64::from(width) * shape.iter().product::<i64>();
            needed_bytes += (bytes + 15) & !15; // 16-byte align, as alloc_tile_shaped does
        }
        let doubled = needed_bytes * 2;
        if self.shared_bytes + doubled > PIPELINE_SHARED_LIMIT_BYTES {
            return Err(PipelineDecline::SharedMemoryBudget {
                needed_bytes: self.shared_bytes + doubled,
                limit_bytes: PIPELINE_SHARED_LIMIT_BYTES,
            });
        }

        Ok((staged, rest))
    }

    /// Whether a tensor-slice expression can reach past its source on the last
    /// tile: a static extent an aligned tile cannot tile evenly, or a dynamic
    /// extent the offset cannot be accounted for against.
    ///
    /// The specialized matmul, fragment and pipeline paths bail on such a slice
    /// so the generic masked path handles it. This has to agree with the mask
    /// [`Codegen::emit_subview`] builds, or a drain with no per-element guard
    /// reaches a partial tile.
    pub(super) fn slice_is_partial(&self, expr: &Expr) -> bool {
        self.slice_is_partial_within(expr, &[])
    }

    /// [`Codegen::slice_is_partial`] for a slice that sits inside loops whose
    /// induction variables are `ivs` and whose bodies have not been emitted yet.
    /// A prescan runs at the enclosing scope, where those variables are not yet
    /// bound, so without naming them every loop-offset slice would look
    /// unprovable and the fast paths would never be taken.
    pub(super) fn slice_is_partial_within(&self, expr: &Expr, ivs: &[&str]) -> bool {
        let Expr::Index { base, subs } = expr else {
            return false;
        };

        let Expr::Var(name) = &**base else {
            return false;
        };

        let mv = match self.lookup(name) {
            Some(Binding::Tensor(mv) | Binding::View(mv) | Binding::Tile(mv)) => mv,
            _ => return false,
        };

        if subs.len() != mv.shape.len() {
            return false;
        }

        subs.iter().enumerate().any(|(d, s)| {
            let (start, size) = match s {
                Sub::Full | Sub::Point(_) => return false,
                Sub::Span { start, len } => (start, self.const_fold(len).unwrap_or(DYN)),
                Sub::Range { start, end } => {
                    let size = match (self.const_fold(start), self.const_fold(end)) {
                        (Some(a), Some(b)) => b - a,
                        _ => DYN,
                    };
                    (start, size)
                }
            };

            if mv.shape[d] == DYN {
                return size != DYN && !self.dyn_in_bounds(start, size, mv.div_of(d), ivs);
            }

            !dim_in_bounds(mv.shape[d], size, self.expr_div(start))
        })
    }

    /// The static shape of a tensor-slice expression, if it has one.
    pub(super) fn slice_static_shape(&self, expr: &Expr) -> Option<Vec<i64>> {
        let Expr::Index { base, subs } = expr else {
            return None;
        };
        let Expr::Var(name) = &**base else {
            return None;
        };
        let Some(Binding::Tensor(src)) = self.lookup(name) else {
            return None;
        };
        if subs.len() != src.shape.len() {
            return None;
        }
        subs.iter()
            .enumerate()
            .map(|(d, s)| match s {
                Sub::Span { len, .. } => self.const_fold(len),
                Sub::Range { start, end } => Some(self.const_fold(end)? - self.const_fold(start)?),
                Sub::Full => (src.shape[d] != DYN).then_some(src.shape[d]),
                Sub::Point(_) => None,
            })
            .collect()
    }

    /// The element type of the tensor an A[...] slice reads from.
    pub(super) fn slice_tensor_elem(&self, expr: &Expr) -> Option<Type<'c>> {
        let Expr::Index { base, .. } = expr else {
            return None;
        };
        let Expr::Var(name) = &**base else {
            return None;
        };
        match self.lookup(name)? {
            Binding::Tensor(src) => Some(src.elem),
            _ => None,
        }
    }

    /// Double-buffered loop: each staged slice gets two shared buffers, and the
    /// loop is unrolled by two so each half references its buffers statically, a
    /// runtime select forcing dynamic shared addressing through the hot loop.
    /// Per original iteration the next tiles are prefetched without a barrier
    /// into the inactive buffers before the compute reads the active ones, so
    /// global loads fly while the CTA does FMA work, and one closing barrier
    /// publishes the prefetch and retires reads of the buffer it overwrites.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_pipelined_for(
        &mut self,
        block: &Block<'c>,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        staged: &[(&str, &Expr)],
        rest: &[Stmt],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(block, start, end, step)?;

        // Prologue: stage iteration 0 into each pair's first buffer.
        let mut bufs0 = Vec::with_capacity(staged.len());
        let mut bufs1 = Vec::with_capacity(staged.len());
        self.scopes.push(HashMap::new());
        self.bind(
            var,
            Binding::Let {
                value: lo,
                div: iv_div,
            },
        );
        for (_, expr) in staged {
            let Rv::Tile(src) = self.emit_expr(block, expr)? else {
                bail!("staged value must be a tensor slice");
            };
            let b0 = self.alloc_tile_shaped(block, src.elem, &src.shape)?;
            bufs1.push(self.alloc_tile_shaped(block, src.elem, &src.shape)?);
            self.tile_copy(block, &src, &b0, true, false)?;
            bufs0.push(b0);
        }
        self.scopes.pop();

        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let iv = detach(body_block.argument(0)?.into());
        let next = self.addi(&body_block, iv, st)?;

        // Half A: compute iteration iv from bufs0, prefetch iv+st -> bufs1.
        self.emit_pipeline_stage(
            &body_block,
            var,
            iv_div,
            iv,
            next,
            hi,
            staged,
            &bufs0,
            &bufs1,
            rest,
        )?;

        // Half B (when iteration iv+st exists): compute it from bufs1,
        // prefetch iv+2*st -> bufs0. The guard is CTA-uniform, so the
        // barrier inside is safe.
        let have_b = self.push(
            &body_block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, next, hi, self.loc),
        )?;
        let half_b = Block::new(&[]);
        let next2 = self.addi(&half_b, next, st)?;
        self.emit_pipeline_stage(
            &half_b, var, iv_div, next, next2, hi, staged, &bufs1, &bufs0, rest,
        )?;
        half_b.append_operation(scf::r#yield(&[], self.loc));
        let half_b_region = Region::new();
        half_b_region.append_block(half_b);
        body_block.append_operation(scf::r#if(
            have_b,
            &[],
            half_b_region,
            Region::new(),
            self.loc,
        ));
        body_block.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let two = self.const_index(block, 2)?;
        let st2 = self.muli(block, st, two)?;
        block.append_operation(scf::r#for(lo, hi, st2, region, self.loc));
        Ok(())
    }

    /// if guard_iv < hi { prefetch(...) }: a guard around a barrier-free
    /// prefetch of one more iteration. No thread ids leak into it, so it stays
    /// CTA-uniform and a barrier inside would be safe.
    pub(super) fn guarded_prefetch(
        &mut self,
        block: &Block<'c>,
        guard_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        prefetch: impl FnOnce(&mut Self, &Block<'c>) -> Result<()>,
    ) -> Result<()> {
        let more = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, guard_iv, hi, self.loc),
        )?;
        let then_block = Block::new(&[]);
        prefetch(self, &then_block)?;
        then_block.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then_block);
        block.append_operation(scf::r#if(more, &[], then_region, Region::new(), self.loc));
        Ok(())
    }

    /// One unrolled half of a pipelined loop: a guarded prefetch (no barrier)
    /// of iteration prefetch_iv into dst, compute of compute_iv from cur, one
    /// closing barrier.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_pipeline_stage(
        &mut self,
        block: &Block<'c>,
        var: &str,
        iv_div: i64,
        compute_iv: Value<'c, 'c>,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        staged: &[(&str, &Expr)],
        cur: &[MemVal<'c>],
        dst: &[MemVal<'c>],
        rest: &[Stmt],
    ) -> Result<()> {
        let use_async = self.has_cp_async();
        self.guarded_prefetch(block, prefetch_iv, hi, |cg, then| {
            cg.scopes.push(HashMap::new());
            cg.bind(
                var,
                Binding::Let {
                    value: prefetch_iv,
                    div: iv_div,
                },
            );
            for ((_, expr), d) in staged.iter().zip(dst) {
                let Rv::Tile(src) = cg.emit_expr(then, expr)? else {
                    bail!("staged value must be a tensor slice");
                };
                cg.tile_copy(then, &src, d, false, use_async)?;
            }
            cg.scopes.pop();
            Ok(())
        })?;

        // cp.async: commit everything this thread issued in the prefetch into
        // one group. Legal outside the guard, since an empty group is a no-op wait.
        let group = if use_async {
            Some(self.async_create_group(block)?)
        } else {
            None
        };

        // Compute against cur (synced by the previous stage's closing
        // barrier, or the prologue's).
        let mut bindings: Vec<(&str, Binding<'c>)> = vec![(
            var,
            Binding::Let {
                value: compute_iv,
                div: iv_div,
            },
        )];
        for ((name, _), c) in staged.iter().zip(cur) {
            bindings.push((name, Binding::View(c.clone())));
        }
        self.emit_scope(block, &bindings, rest)?;

        // Closing sync. With cp.async the wait must precede a barrier that
        // runs after it (the compute's own barriers don't help), so the
        // elision below doesn't apply.
        if let Some(group) = group {
            self.async_wait(block, group)?;
            self.barrier(block)?;
            return Ok(());
        }
        // Closing barrier: publishes the prefetch and retires reads of the
        // buffer the next prefetch overwrites, unless the compute's last
        // statement was a tile op, whose own trailing barrier already did.
        if !self.ends_with_tile_op(rest) {
            self.barrier(block)?;
        }
        Ok(())
    }

    /// Whether the last statement lowers to a tile op (which always ends
    /// with its own gpu.barrier).
    pub(super) fn ends_with_tile_op(&self, stmts: &[Stmt]) -> bool {
        match stmts.last() {
            Some(Stmt::Assign { target, .. }) => match target {
                Expr::Var(n) => matches!(self.lookup(n), Some(Binding::Tile(_))),
                Expr::Index { subs, .. } => subs.iter().any(|s| !matches!(s, Sub::Point(_))),
                _ => false,
            },
            Some(Stmt::Var {
                ty: Some(AstType::Tile(..)),
                ..
            }) => true,
            _ => false,
        }
    }

    // CTA-uniformity of a loop's bounds (why `pipeline_candidate`'s caller
    // needs no runtime divergence check of its own):
    //
    // `emit_pipelined_for`'s guards wrap a `gpu.barrier` in an `scf.if`,
    // which hangs rather than miscomputes if a CTA's threads take different
    // branches -- so a loop bound has to be provably the same value on
    // every thread of the block before it is safe to pipeline at all.
    //
    // A `.ph`-level loop bound (`start`/`end`/`step` in `for x in
    // range(...)`) always lowers through `Codegen::emit_index`, which
    // requires the emitted value to already carry MLIR's `index` type
    // (`expect_index` rejects anything else outright). Tracing every way an
    // `index` value can be produced in this codegen accounts for all of
    // them: integer literals (lower straight to `index`), `program_id` /
    // `Codegen::block_id` (the grid's block id, the same for every thread
    // in it by definition), a tensor's shape via `memref.dim` (host-set
    // before the launch, not a per-thread quantity), `warp_partial` and
    // `grid_barrier`'s nominal return values (both literally
    // `Rv::Scalar(self.const_index(block, 0))`, a compile-time constant,
    // never data-dependent), and arithmetic over any of those. What is
    // conspicuously absent from that list is anything that reads program
    // DATA: a tensor element load, a reduction result, or an atomic's
    // return value (`atomic_add`'s `Rv::Scalar(old)` is exactly such a
    // value -- the whole point of an RMW is that distinct threads racing
    // the same slot see distinct "previous" values).
    //
    // That absence is not a gap the compiler happens not to hit today; it
    // is structurally enforced. `Codegen::coerce` converts `index` to an
    // integer type (`index_cast`) but has no arm the other way, and
    // `Codegen::unify` (the join two binary operands go through) only
    // widens between float types, `bail!`ing on any int/index mismatch --
    // so an i32 value coming out of `atomic_add`, or a tensor load, can
    // never become `index`-typed by any expression this language can
    // write, whether alone or arithmetic-combined with something that is.
    // Concretely: `let n = atomic_add(BAR, 0, 1); for i in range(0, n) {}`
    // fails to parse-then-typecheck with "loop end must be an integer, got
    // i32" before codegen ever reaches a loop; `range(0, n * 1, 1)` fails
    // the same way one step earlier, in `unify`, with "mismatched operand
    // types: i32 vs index". See
    // `codegen::tests::pipeline::atomic_add_cannot_reach_a_loop_bound` --
    // both failures are pinned there as a tripwire, not asserted only here:
    // if a future language feature adds an int-to-index conversion, or
    // changes `warp_partial`/`grid_barrier` to return something other than
    // a literal constant zero (a real risk, since those two live in
    // `tile/warp_attn.rs` and `sync.rs`, files this pipelining code does
    // not own), that test goes red and is what forces a second look at
    // this comment and at whether `pipeline_candidate`'s callers still need
    // no divergence check of their own.
}
