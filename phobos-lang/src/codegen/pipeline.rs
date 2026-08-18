use super::*;

/// Ceiling for the "would the doubled buffers still fit" check below. Mirrors
/// `phobos_kernels::launch::STATIC_SHARED_LIMIT`, duplicated by hand because
/// phobos-kernels depends on phobos-lang and not the other way round. A
/// `@dynshared` kernel can be raised past this at launch, but eligibility is
/// decided here, before that opt-in is visible, so the static bound is the
/// safe default for every kernel.
const PIPELINE_SHARED_LIMIT_BYTES: i64 = 48 * 1024;

/// Why a loop body was not turned into a pipelined (double-buffered) loop.
/// Every variant is a decline, not an error: the loop just falls through to
/// the ordinary codegen path. Only an explicit `@pipeline` finding zero
/// eligible loops in the whole kernel is a hard failure, checked in `emit`
/// via `Codegen::pipelined_any`.
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
    /// so prefetching would read a value the compute half does not expect.
    StagedNameWritten,
    /// A staged slice's element type has no known byte width, so the budget
    /// below cannot be computed; declining keeps that check fail-safe.
    UnknownElementWidth,
    /// Doubling every staged slice's shared buffer would not fit.
    SharedMemoryBudget { needed_bytes: i64, limit_bytes: i64 },
}

impl std::fmt::Display for PipelineDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStagedPrefix => {
                f.write_str("no leading run of staged tensor-slice `var` statements")
            }
            Self::PartialSlice => f.write_str(
                "a staged slice's shape cannot be proven to tile evenly (partial slice)",
            ),
            Self::StagedNameWritten => {
                f.write_str("a staged name is written again later in the loop body")
            }
            Self::UnknownElementWidth => {
                f.write_str("a staged slice's element type has no known byte width")
            }
            Self::SharedMemoryBudget {
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
    /// Deliberately does not check the loop bounds for lane divergence, which
    /// would turn `emit_pipelined_for`'s barrier-in-an-`scf.if` guard into a
    /// hang: every `.ph`-level bound is CTA-uniform by construction. See the
    /// block comment at the end of this file for why.
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

        // Double-buffering doubles every staged slice's footprint, which an
        // auto-attempted loop has nobody to vouch for. Conservative both
        // ways: it ignores pool reuse of released buffers, so it can decline
        // a loop that would fit, and it does not see static globals absent
        // from `shared_bytes`, so it can admit one that does not. The latter
        // degrades to ptxas's own "uses too much shared data" error, not a
        // hang.
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

    // Why a loop bound is CTA-uniform by construction, and so needs no
    // runtime divergence check before pipelining it.
    //
    // `emit_pipelined_for` wraps a `gpu.barrier` in an `scf.if`, which hangs
    // if a CTA's threads take different branches, so a bound has to hold the
    // same value on every thread of the block.
    //
    // Every `.ph`-level bound lowers through `emit_index`, which demands an
    // already-`index`-typed value. The only producers of one are integer
    // literals, `program_id`, a tensor's shape via `memref.dim`, the constant
    // zero `warp_partial`/`grid_barrier` nominally return, and arithmetic
    // over those -- all block-uniform. Nothing that reads program data is on
    // that list, and nothing can join it: `coerce` casts `index` to an
    // integer but has no arm the other way, and `unify` bails on an
    // int/index mismatch instead of promoting. So `atomic_add`'s i32, or a
    // tensor load, can never reach a bound.
    //
    // `codegen::tests::pipeline::atomic_add_cannot_reach_a_loop_bound` pins
    // both failure modes. It goes red if some future change adds an
    // int-to-index conversion, or makes `warp_partial`/`grid_barrier` return
    // something data-dependent, which is when this argument needs revisiting.
}
