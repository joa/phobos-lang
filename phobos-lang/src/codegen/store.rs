use super::*;

impl<'c> Codegen<'c> {
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

        // t = qmma_t(..) writes the accumulators where they are wanted rather
        // than through a tile.
        //
        // The accumulators are registers already, and a `[128, 64]` f32 tile
        // on the way out is 32 KB, which holds the kernel to one CTA per SM.
        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && callee == "qmma_t"
            && !target.is_masked()
            && target.elem == self.f32_t
        {
            let [a, asc, w, wsc] = self.qmma_operands(block, args)?;

            self.qmma_t_into(block, &a, &asc, &w, &wsc, target)?;

            for t in [&a, &asc, &w, &wsc] {
                self.release(t);
            }

            return Ok(());
        }

        // t = iq1s_qmma_t(..) likewise. Without this the fused projection
        // allocates its own [128, 64] f32 tile, which ptxas reports as 32 KB of
        // shared memory and holds the kernel to two CTAs per multiprocessor for
        // no gain: the accumulators are already registers at the rows they
        // belong in.
        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && callee == "iq1s_qmma_t"
            && !target.is_masked()
            && target.elem == self.f32_t
        {
            let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(block, args)?;

            self.iq1s_qmma_t_into(block, &a, &asc, &qb, &d, &grid, target)?;

            for t in [&a, &asc, &qb, &d, &grid] {
                self.release(t);
            }

            return Ok(());
        }

        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && let Some(fmt) = QgFormat::from_intrinsic(callee)
            && !target.is_masked()
            && target.elem == self.f32_t
        {
            let tiles = self.qgemm_operands(block, fmt, args)?;
            self.qgemm_into(block, fmt, &tiles[0], &tiles[1], &tiles[2], &tiles[3], &tiles[4..], target)?;
            for t in &tiles {
                self.release(t);
            }
            return Ok(());
        }

        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && callee == "iq1s_qmma_staged_t"
            && !target.is_masked()
            && target.elem == self.f32_t
        {
            let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(block, args)?;

            self.iq1s_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, target)?;

            for t in [&a, &asc, &qb, &d, &grid] {
                self.release(t);
            }

            return Ok(());
        }

        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && matches!(
                callee.as_str(),
                "iq2xxs_qmma_staged_t"
                    | "iq2s_qmma_staged_t"
                    | "iq2xs_qmma_staged_t"
                    | "iq3xxs_qmma_staged_t"
                    | "iq3s_qmma_staged_t"
            )
            && !target.is_masked()
            && target.elem == self.f32_t
        {
            let [a, asc, qb, d, grid, signs] = self.iq2xxs_qmma_operands(block, args)?;

            match callee.as_str() {
                "iq2s_qmma_staged_t" => {
                    self.iq2s_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &signs, target)?
                }
                "iq2xs_qmma_staged_t" => {
                    self.iq2xs_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &signs, target)?
                }
                "iq3xxs_qmma_staged_t" => {
                    self.iq3xxs_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &signs, target)?
                }
                "iq3s_qmma_staged_t" => {
                    self.iq3s_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &signs, target)?
                }
                _ => {
                    self.iq2xxs_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &signs, target)?
                }
            }

            for t in [&a, &asc, &qb, &d, &grid, &signs] {
                self.release(t);
            }

            return Ok(());
        }

        // t = <fmt>_qdecode_t(..) writes the scratch directly, for the same
        // reason qmma_t above does: the values are already in registers at the
        // rows they belong in.
        if op == AssignOp::Set
            && let Expr::Call { callee, args } = value
            && let Some(fmt) = QFormat::from_intrinsic(callee)
        {
            let tiles = self.qdecode_operands(block, fmt, args)?;
            self.qdecode_t_into(block, fmt, &tiles[0], &tiles[1], &tiles[2..], target)?;

            for t in &tiles {
                self.release(t);
            }

            return Ok(());
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

            // `p = dot(p, p)` is routed through a tmp since we create garbage otherwise.
            if target.global.is_some() && (target.global == a.global || target.global == b.global) {
                let temp = self.alloc_tile_shaped(block, target.elem, &target.shape)?;

                if transpose {
                    self.tile_matmul_t(block, &a, &b, &temp)?;
                } else {
                    let zero = self.zero_scalar(block, temp.elem)?;
                    self.tile_fill(block, zero, &temp)?;
                    self.tile_matmul(block, &a, &b, &temp)?;
                }

                self.release(&a);
                self.release(&b);

                match op {
                    AssignOp::Set => self.tile_copy(block, &temp, target, true, false)?,
                    AssignOp::Add => self.tile_binary(block, BinOp::Add, target, &temp, target)?,
                }

                self.release(&temp);

                return Ok(());
            }

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

        // target = i8(i32(round(t))) and its like: one sweep for the whole chain
        // rather than a shared tile per conversion. See codegen/elemwise.rs.
        if op == AssignOp::Set && self.store_elem_chain(block, target, value)? {
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

        // Anything else built out of arithmetic, tmax and the per-element math
        // calls: the whole tree in one sweep, whatever its depth. See
        // codegen/elemwise.rs.
        if self.store_fused(block, target, op, value)? {
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
}
