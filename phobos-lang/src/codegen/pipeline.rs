use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    /// Matches a loop body of the form "var t = <static tensor slice>; ..."
    /// whose remaining statements never write the staged names. Returns the
    /// staged (name, slice expr) pairs and the compute statements.
    #[allow(clippy::type_complexity)]
    pub(super) fn pipeline_candidate<'a>(
        &self,
        body: &'a [Stmt],
    ) -> Option<(Vec<(&'a str, &'a Expr)>, &'a [Stmt])> {
        let mut staged = Vec::new();
        let mut rest = body;
        while let [
            Stmt::Var {
                name,
                ty: None,
                value,
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

        let names: Vec<&str> = staged.iter().map(|(n, _)| *n).collect();

        (!staged.is_empty()
            && !staged.iter().any(|(_, v)| self.slice_is_partial(v))
            && !rest.iter().any(|s| s.writes_any(&names)))
        .then_some((staged, rest))
    }

    /// Whether a tensor-slice expression can reach past its source on the last
    /// tile (a statically known extent an aligned tile cannot tile evenly).
    ///
    /// The specialized matmul/fragment/pipeline paths bail on such a slice so
    /// the generic masked load/store path handles it (see [`dim_in_bounds`]).
    pub(super) fn slice_is_partial(&self, expr: &Expr) -> bool {
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
            let (size, off_div) = match s {
                Sub::Full | Sub::Point(_) => return false,
                Sub::Span { start, len } => {
                    (self.const_fold(len).unwrap_or(DYN), self.expr_div(start))
                }
                Sub::Range { start, end } => {
                    let size = match (self.const_fold(start), self.const_fold(end)) {
                        (Some(a), Some(b)) => b - a,
                        _ => DYN,
                    };
                    (size, self.expr_div(start))
                }
            };
            !dim_in_bounds(mv.shape[d], size, off_div)
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

    /// Double-buffered loop: each staged slice gets two shared buffers, and
    /// the loop is unrolled by two so each half references its buffers
    /// statically (a runtime select between buffers would force dynamic
    /// shared addressing through the hot loop). Per original iteration: the
    /// next tiles are prefetched (without a barrier) into the inactive buffers
    /// before the compute reads the active ones (global loads fly while
    /// the CTA does FMA work), and one closing barrier publishes the
    /// prefetch and retires reads of the buffer the next prefetch overwrites.
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

    /// if guard_iv < hi { prefetch(...) }, a CTA-uniform guard around a
    /// prefetch of one more iteration that runs without a barrier. The guard
    /// is uniform (no thread ids leak into it), so a barrier inside would be safe.
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
        let use_async = self.cp_async();
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
}
