use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn emit_stmt(&mut self, block: &Block<'c>, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { name, ty, value } => {
                if let Some(AstType::Tile(scalar, dims)) = ty {
                    let tile = self.emit_tile_decl(block, *scalar, dims, value)?;
                    self.bind(name, Binding::View(tile));
                } else {
                    let div = self.expr_div(value);
                    match self.emit_expr(block, value)? {
                        Rv::Scalar(v) => self.bind(name, Binding::Let { value: v, div }),
                        Rv::Tile(t) => self.bind(name, Binding::View(t)),
                    }
                }
            }
            Stmt::Var { name, ty, value } => {
                if let Some(AstType::Tile(scalar, dims)) = ty {
                    let tile = self.emit_tile_decl(block, *scalar, dims, value)?;
                    self.bind(name, Binding::Tile(tile));
                } else {
                    match self.emit_expr(block, value)? {
                        Rv::Scalar(v) => {
                            let elem = v.r#type();
                            let slot_t = MemRefType::new(elem, &[], None, None);
                            let slot = self.push(
                                block,
                                memref::alloca(self.ctx, slot_t, &[], &[], None, self.loc),
                            )?;
                            block.append_operation(memref::store(v, slot, &[], self.loc));
                            self.bind(name, Binding::Var { slot, elem });
                        }
                        // var t = <tile expr> -> a writable copy of the value
                        // (a fresh temp is already private: adopt its buffer)
                        Rv::Tile(src) => {
                            if src.owned {
                                self.bind(name, Binding::Tile(src));
                            } else {
                                let tile = self.alloc_tile_shaped(block, src.elem, &src.shape)?;
                                self.tile_copy(block, &src, &tile, true, false)?;
                                self.bind(name, Binding::Tile(tile));
                            }
                        }
                    }
                }
            }
            Stmt::Assign { target, op, value } => self.emit_assign(block, target, *op, value)?,
            Stmt::For {
                var,
                start,
                end,
                step,
                body,
            } => self.emit_for(block, var, start, end, step.as_ref(), body)?,
            Stmt::While { cond, body } => self.emit_while(block, cond, body)?,
            Stmt::If { cond, then, r#else } => {
                self.emit_if(block, cond, then, r#else.as_deref())?
            }
            Stmt::Expr(e) => {
                self.emit_expr(block, e)?;
            }
        }
        Ok(())
    }

    /// Emits a tile-typed let/var initializer. Initializers store_tile would
    /// not fuse into the target (the elementwise calls exp/tmax/rowmax/rowsum)
    /// are evaluated first: when the result is an owned temp of exactly the
    /// declared type, its buffer is adopted outright, eliding the copy pass
    /// and its shared allocation. Everything else takes the regular
    /// alloc-then-store_tile path, which fuses dot/binary/scalar initializers
    /// straight into the target.
    fn emit_tile_decl(
        &mut self,
        block: &Block<'c>,
        scalar: Scalar,
        dims: &[Dim],
        value: &Expr,
    ) -> Result<MemVal<'c>> {
        let unfused = matches!(
            value,
            Expr::Call { callee, .. } if matches!(callee.as_str(), "exp" | "tmax" | "rowmax" | "rowsum")
        );
        if !unfused {
            let tile = self.alloc_tile(block, scalar, dims)?;
            self.store_tile(block, &tile, AssignOp::Set, value)?;
            return Ok(tile);
        }

        let shape = self.tile_shape(dims)?;
        let elem = self.scalar_type(scalar);
        let Rv::Tile(t) = self.emit_expr(block, value)? else {
            bail!("expected a tile initializer");
        };
        if t.owned && t.elem == elem && t.shape == shape {
            return Ok(t);
        }
        self.check_shapes(&t.shape, &shape, "tile init")?;
        let tile = self.alloc_tile_shaped(block, elem, &shape)?;
        if t.elem != elem {
            self.tile_convert(block, &t, &tile)?;
        } else {
            self.tile_copy(block, &t, &tile, true, false)?;
        }
        self.release(&t);
        Ok(tile)
    }

    pub(super) fn emit_assign(
        &mut self,
        block: &Block<'c>,
        target: &Expr,
        op: AssignOp,
        value: &Expr,
    ) -> Result<()> {
        match target {
            Expr::Var(name) => match self.lookup(name) {
                Some(Binding::Var { slot, elem }) => {
                    let rhs = self.emit_scalar(block, value)?;
                    let rhs = match op {
                        AssignOp::Set => rhs,
                        AssignOp::Add => {
                            let cur = self.push(block, memref::load(slot, &[], self.loc))?;
                            self.emit_binop(block, BinOp::Add, cur, rhs)?
                        }
                    };
                    let rhs = self.coerce(block, rhs, elem)?;
                    block.append_operation(memref::store(rhs, slot, &[], self.loc));
                }
                Some(Binding::Tile(tile)) => self.store_tile(block, &tile, op, value)?,
                Some(Binding::Frags(fa)) => self.emit_frag_assign(block, name, &fa, op, value)?,
                Some(Binding::View(_)) => {
                    bail!("'{name}' is a read-only view; declare it with `var` to assign")
                }
                Some(_) => bail!("'{name}' is not assignable (declare it with `var`)"),
                None => bail!("unknown identifier '{name}'"),
            },
            Expr::Index { base, subs } => {
                let (mv, binding) = self.mem_base(base)?;
                if subs.iter().all(|s| matches!(s, Sub::Point(_))) {
                    if matches!(binding, Binding::View(_)) {
                        bail!("cannot assign through a read-only view");
                    }
                    let indices = self.emit_indices(block, subs, mv.shape.len())?;
                    let rhs = self.emit_scalar(block, value)?;
                    let rhs = match op {
                        AssignOp::Set => rhs,
                        AssignOp::Add => {
                            let cur = self.load_scalar(block, &mv, &indices)?;
                            self.emit_binop(block, BinOp::Add, cur, rhs)?
                        }
                    };
                    let rhs = self.coerce(block, rhs, mv.elem)?;
                    block.append_operation(memref::store(rhs, mv.mem, &indices, self.loc));
                } else {
                    if !matches!(binding, Binding::Tensor(_)) {
                        bail!("only tensors can be sliced");
                    }
                    let view = self.emit_subview(block, &mv, subs)?;
                    self.store_tile(block, &view, op, value)?;
                }
            }
            _ => bail!("invalid assignment target"),
        }
        Ok(())
    }

    /// Stores value into the tile target (= or +=).
    /// 
    /// This is where tile-level patterns get matched:
    /// 
    /// - t += dot(a, b) -> accumulating matmul loop into target
    /// - t = dot(a, b)  -> zero-fill plus that accumulating loop
    /// - t = s1*t1 + s2*t2 -> GEMM epilogue (single loop, no temp tiles)
    /// - t = x * y      -> elementwise loop writing target (no temp buffer)
    /// - t = <scalar>   -> fill loop
    /// - t = <tile>     -> copy loop
    /// - t += <tile>    -> elementwise add into target
    pub(super) fn store_tile(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        op: AssignOp,
        value: &Expr,
    ) -> Result<()> {
        // <tensor slice> = acc for a fragment accumulator: scatter the
        // lane fragments straight to the target, no shared hop.
        if let Expr::Var(n) = value
            && let Some(Binding::Frags(fa)) = self.lookup(n)
        {
            if op != AssignOp::Set {
                bail!("fragment accumulator '{n}' cannot be accumulated into a tile");
            }
            return self.frag_store(block, target, &fa);
        }

        // t = exp(a op b) on float tile operands broadcasting to the target
        // fuses the binary and the exponential into one sweep and barrier.
        // t may itself be an operand (the softmax t = exp(t - mnew)): each
        // thread reads and writes the same element, so it is race-free.
        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && callee == "exp"
            && let [Expr::Binary { op: bop, lhs, rhs }] = &args[..]
            && let (Expr::Var(ln), Expr::Var(rn)) = (lhs.as_ref(), rhs.as_ref())
            && let (Some(Binding::Tile(a)), Some(Binding::Tile(b))) =
                (self.lookup(ln), self.lookup(rn))
            && self.is_float(target.elem)
            && a.elem == target.elem
            && b.elem == target.elem
            && broadcast_shape(&a.shape, &b.shape).as_deref() == Some(&target.shape[..])
        {
            return self.tile_exp_binary_bc(block, *bop, &a, &b, target);
        }

        // t = exp(t) rewrites the tile in place: each thread reads and
        // writes the same element, so no temp buffer or copy-back pass.
        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && callee == "exp"
            && let [Expr::Var(n)] = &args[..]
            && let Some(Binding::Tile(src)) = self.lookup(n)
            && src.global.is_some()
            && src.global == target.global
        {
            return self.tile_exp_into(block, &src, target);
        }

        if let Expr::Call { callee, args } = value
            && (callee == "dot" || callee == "dot_t")
        {
            // The matmul kernels write the target in place (register/fragment
            // blocking or a strided sub-tile sweep), none of which carry the
            // per-element store guard a partial tile needs. Route such stores
            // through an accumulator tile instead.
            if target.is_masked() {
                bail!(
                    "writing a dot result directly into a partially out-of-bounds \
                     tensor slice is unsupported; accumulate into a tile first"
                );
            }
            let transpose = callee == "dot_t";
            let (a, b) = self.dot_operands(block, args)?;
            if transpose {
                if a.shape.len() != 2 || b.shape.len() != 2 {
                    bail!("dot_t expects rank-2 tiles");
                }
                self.check_shapes(&[a.shape[1]], &[b.shape[1]], "dot_t contraction dim")?;
                self.check_shapes(&[a.shape[0], b.shape[0]], &target.shape, "dot_t result")?;
            } else {
                self.check_matmul_shapes(&a, &b, target)?;
            }
            // On the tensor cores = and += fold straight into the f32 fragment
            // accumulators (zero- or target-seeded), with f16 or f32 operands;
            // else the vector / mixed-precision fallback.
            if self.wmma_dot(block, &a, &b, target, transpose, op == AssignOp::Add)? {
                self.release(&a);
                self.release(&b);
                return Ok(());
            }
            if transpose {
                if op == AssignOp::Add {
                    bail!("`tile += dot_t(...)` is not supported; assign with `=`");
                }
                self.tile_matmul_t(block, &a, &b, target)?;
            } else {
                if op == AssignOp::Set {
                    let zero = self.zero_scalar(block, target.elem)?;
                    self.tile_fill(block, zero, target)?;
                }
                self.tile_matmul(block, &a, &b, target)?;
            }
            self.release(&a);
            self.release(&b);
            return Ok(());
        }

        // Fused GEMM epilogue: target = s1 * t1 + s2 * t2 in one loop.
        // Avoids allocating two intermediate shared-memory tiles.
        if op == AssignOp::Set
            && let Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            } = value
            && let Some((s1_expr, t1_expr)) = self.as_scale_mul(lhs)
            && let Some((s2_expr, t2_expr)) = self.as_scale_mul(rhs)
        {
            let s1 = self.emit_scalar(block, s1_expr)?;
            let Rv::Tile(t1) = self.emit_expr(block, t1_expr)? else {
                bail!("GEMM epilogue: expected tile in first operand");
            };
            let s2 = self.emit_scalar(block, s2_expr)?;
            let Rv::Tile(t2) = self.emit_expr(block, t2_expr)? else {
                bail!("GEMM epilogue: expected tile in second operand");
            };
            self.check_shapes(&t1.shape, &target.shape, "GEMM epilogue lhs")?;
            self.check_shapes(&t2.shape, &target.shape, "GEMM epilogue rhs")?;
            self.tile_scaled_add_into(block, s1, &t1, s2, &t2, target)?;
            self.release(&t1);
            self.release(&t2);
            return Ok(());
        }

        // Fuse t = x * y directly into the target without temp buffer.
        if op == AssignOp::Set
            && let Expr::Binary { op: bop, lhs, rhs } = value
        {
            let l = self.emit_expr(block, lhs)?;
            return match (l, self.emit_expr(block, rhs)?) {
                (Rv::Tile(a), Rv::Tile(b)) => {
                    // The operands broadcast to the target ([R,C] * [R,1]).
                    self.tile_binary_dispatch(block, *bop, &a, &b, target)?;
                    self.release(&a);
                    self.release(&b);
                    Ok(())
                }
                (Rv::Scalar(a), Rv::Scalar(b)) => {
                    let v = self.emit_binop(block, *bop, a, b)?;
                    let v = self.coerce(block, v, target.elem)?;
                    self.tile_fill(block, v, target)
                }
                // t = tile * scalar (or scalar * tile): broadcast the
                // scalar over the target.
                (Rv::Tile(a), Rv::Scalar(b)) => {
                    self.tile_scalar_into(block, *bop, &a, b, false, target)?;
                    self.release(&a);
                    Ok(())
                }
                (Rv::Scalar(a), Rv::Tile(b)) => {
                    self.tile_scalar_into(block, *bop, &b, a, true, target)?;
                    self.release(&b);
                    Ok(())
                }
            };
        }

        match (op, self.emit_expr(block, value)?) {
            (AssignOp::Set, Rv::Scalar(v)) => {
                let v = self.coerce(block, v, target.elem)?;
                self.tile_fill(block, v, target)
            }
            (AssignOp::Add, Rv::Scalar(_)) => {
                bail!("`tile += scalar` is not supported; use a tile-typed operand")
            }
            (AssignOp::Set, Rv::Tile(src)) => {
                self.check_shapes(&src.shape, &target.shape, "tile store")?;
                if src.elem != target.elem {
                    // e.g. an f32 accumulator written to an f16 output tensor.
                    self.tile_convert(block, &src, target)?;
                } else {
                    self.tile_copy(block, &src, target, true, false)?;
                }
                self.release(&src);
                Ok(())
            }
            (AssignOp::Add, Rv::Tile(src)) => {
                self.check_shapes(&src.shape, &target.shape, "tile accumulate")?;
                self.tile_binary(block, BinOp::Add, target, &src, target)?;
                self.release(&src);
                Ok(())
            }
        }
    }

    pub(super) fn emit_for(
        &mut self,
        block: &Block<'c>,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
    ) -> Result<()> {
        // Stage loop-invariant dot operands into the preheader once instead
        // of every iteration (see codegen/hoist.rs); the frame is popped and
        // its buffers pooled after the loop, whose in-body dot barriers
        // order the last reads before any reuse.
        let frame = self.hoist_dot_staging(block, body)?;
        self.hoisted_stages.push(frame);
        let result = self.emit_for_inner(block, var, start, end, step, body);
        for (_, buf) in self.hoisted_stages.pop().expect("hoist frame") {
            self.release(&buf);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_for_inner(
        &mut self,
        block: &Block<'c>,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
    ) -> Result<()> {
        // Fragment accumulators the body assigns must ride the loop as
        // iter_args (SSA values, unlike memref-backed tiles), which neither
        // the pipelined nor the plain paths can carry.
        let carried = self.frag_carried(body);
        if !carried.is_empty() {
            return self.emit_frag_for(block, var, start, end, step, body, &carried);
        }

        // @pipeline: double-buffer leading staged slices in this loop.
        if self.pipeline
            && let Some((staged, rest)) = self.pipeline_candidate(body)
        {
            return self.emit_pipelined_for(block, var, start, end, step, &staged, rest);
        }

        let const_step = match step {
            None => Some(1),
            Some(e) => self.const_fold(e),
        };
        if let (Some(lo), Some(hi), Some(st)) =
            (self.const_fold(start), self.const_fold(end), const_step)
            && st > 0
        {
            return self.emit_affine_for(block, var, lo, hi, st, body);
        }

        // Dynamic bounds or step: scf.for.
        let (lo, hi, st, iv_div) = self.loop_bounds(block, start, end, step)?;
        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let iv = detach(body_block.argument(0)?.into());
        let binding = Binding::Let {
            value: iv,
            div: iv_div,
        };
        self.emit_scope(&body_block, &[(var, binding)], body)?;
        body_block.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body_block);
        block.append_operation(scf::r#for(lo, hi, st, region, self.loc));
        Ok(())
    }

    /// affine.for %iv = lo to hi step st with constant bound maps.
    pub(super) fn emit_affine_for(
        &mut self,
        block: &Block<'c>,
        var: &str,
        lo: i64,
        hi: i64,
        step: i64,
        body: &[Stmt],
    ) -> Result<()> {
        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let iv = detach(body_block.argument(0)?.into());
        let binding = Binding::Let {
            value: iv,
            div: gcd(lo, step),
        };
        self.emit_scope(&body_block, &[(var, binding)], body)?;
        body_block.append_operation(OperationBuilder::new("affine.yield", self.loc).build()?);

        let region = Region::new();
        region.append_block(body_block);

        let bound_map = |v: i64| {
            Attribute::parse(self.ctx, &format!("affine_map<() -> ({v})>"))
                .ok_or_else(|| anyhow!("failed to parse affine bound map for {v}"))
        };
        let op = OperationBuilder::new("affine.for", self.loc)
            .add_attributes(&[
                (self.id("lowerBoundMap"), bound_map(lo)?),
                (self.id("upperBoundMap"), bound_map(hi)?),
                (
                    self.id("step"),
                    IntegerAttribute::new(self.index_t, step).into(),
                ),
                // No bound operands and no iter args (AttrSizedOperandSegments).
                (self.id("operandSegmentSizes"), self.i32_array(&[0, 0, 0])?),
            ])
            .add_regions([region])
            .build()?;
        block.append_operation(op);
        Ok(())
    }

    pub(super) fn emit_while(
        &mut self,
        block: &Block<'c>,
        cond: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        let before = Block::new(&[]);
        let c = self.emit_scalar(&before, cond)?;
        let c = self.expect_bool(c, "while condition")?;
        before.append_operation(scf::condition(c, &[], self.loc));

        let after = Block::new(&[]);
        self.emit_scope(&after, &[], body)?;
        after.append_operation(scf::r#yield(&[], self.loc));

        let before_region = Region::new();
        before_region.append_block(before);
        let after_region = Region::new();
        after_region.append_block(after);
        block.append_operation(scf::r#while(
            &[],
            &[],
            before_region,
            after_region,
            self.loc,
        ));
        Ok(())
    }

    pub(super) fn emit_if(
        &mut self,
        block: &Block<'c>,
        cond: &Expr,
        then: &[Stmt],
        els: Option<&[Stmt]>,
    ) -> Result<()> {
        let c = self.emit_scalar(block, cond)?;
        let c = self.expect_bool(c, "if condition")?;

        let then_block = Block::new(&[]);
        self.emit_scope(&then_block, &[], then)?;
        then_block.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then_block);

        let else_region = Region::new();
        if let Some(els) = els {
            let else_block = Block::new(&[]);
            self.emit_scope(&else_block, &[], els)?;
            else_block.append_operation(scf::r#yield(&[], self.loc));
            else_region.append_block(else_block);
        }

        block.append_operation(scf::r#if(c, &[], then_region, else_region, self.loc));
        Ok(())
    }

    /// Emits stmts in a fresh scope, pre-populated with given bindings
    pub(super) fn emit_scope(
        &mut self,
        block: &Block<'c>,
        bindings: &[(&str, Binding<'c>)],
        stmts: &[Stmt],
    ) -> Result<()> {
        self.scopes.push(HashMap::new());
        for (name, binding) in bindings {
            self.bind(name, binding.clone());
        }
        let result = self.emit_stmts(block, stmts);
        self.scopes.pop();
        result
    }

    pub(super) fn emit_stmts(&mut self, block: &Block<'c>, stmts: &[Stmt]) -> Result<()> {
        let mut i = 0;
        while i < stmts.len() {
            if let Some(p) = self.matmul_candidate(&stmts[i..]) {
                let consumed = p.consumed;
                self.emit_register_matmul(block, &p)?;
                i += consumed;
            } else if let Some(plan) = self.frag_acc_candidate(&stmts[i..]) {
                self.bind_frag_acc(block, &plan)?;
                i += 1;
            } else {
                self.emit_stmt(block, &stmts[i])?;
                i += 1;
            }
        }
        Ok(())
    }
}
