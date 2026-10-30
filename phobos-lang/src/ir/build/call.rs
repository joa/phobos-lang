use anyhow::{Result, anyhow, bail, ensure};

use super::{Binding, Build, DYN, Rv, broadcast_shape};
use crate::ast::{Expr, ExprKind, Scalar as AstScalar, ctx};
use crate::ir::{
    ElemStep, Intrinsic, Map, MathFn, OpKind, RawFmt, Reduce, Scalar, Space, Stage, ThreadIndex, Type,
    ValueId,
};

impl Build {
    /// Each arm evaluates its operands in source order and produces one op
    /// whose result type is the buffer the emitter allocates.
    pub(crate) fn emit_call(&mut self, callee: &str, args: &[Expr]) -> Result<Rv> {
        if let Some(rv) = self.emit_lane_builtin(callee, args)? {
            return Ok(rv);
        }
        match callee {
            ctx::BLOCK | ctx::BLOCKS => {
                let [Expr { kind: ExprKind::Int(d @ 0..=2), .. }] = args else {
                    unreachable!("the parser builds ctx fields with an axis 0..=2");
                };
                // A persistent grid's block count is compiled in.
                if callee == ctx::BLOCKS
                    && *d == 0
                    && let Some(n) = self.ir.kernel.persistent_blocks
                {
                    return Ok(Rv::Scalar(self.const_index(n)));
                }
                let kind = if callee == ctx::BLOCK { OpKind::ProgramId(*d as u8) } else { OpKind::GridDim(*d as u8) };
                Ok(Rv::Scalar(self.value(kind, &[], Type::INDEX)))
            }
            ctx::THREAD | ctx::WARP | ctx::LANE => {
                let which = match callee {
                    ctx::THREAD => ThreadIndex::Thread,
                    ctx::WARP => ThreadIndex::Warp,
                    _ => ThreadIndex::Lane,
                };
                Ok(Rv::Scalar(self.value(OpKind::Thread(which), &[], Type::INDEX)))
            }
            ctx::THREADS | ctx::LANES | ctx::WARPS => {
                let n = self.ctx_count(callee).ok_or_else(|| {
                    anyhow!(
                        "ctx.warps: {} threads are not a whole number of {}-lane warps",
                        self.cta_threads,
                        self.target.lanes
                    )
                })?;
                Ok(Rv::Scalar(self.const_index(n)))
            }
            super::extent::EXTENT => Ok(Rv::Scalar(self.emit_extent(args)?)),
            "atomic_add" => self.emit_atomic_add(args),
            "atomic_cas" => self.emit_atomic_cas(args),
            "load_acquire" | "store_release" => self.emit_system_scope(callee, args),
            "grid_barrier" => self.emit_grid_barrier(args),
            "dot" => {
                let (a, b) = self.dot_operands(args)?;
                let (ash, bsh) = (self.shape(a), self.shape(b));
                let shape = [ash[0], bsh[1]];
                if shape.contains(&DYN) {
                    bail!("`dot` result shape must be static; assign to a tile-typed var instead");
                }
                let acc_elem = super::expr::accumulator_elem(self.elem(a), self.elem(b))?;
                self.check_matmul_shapes(&ash, &bsh, &shape)?;
                let ty = self.shared_ty(acc_elem, &shape);
                Ok(Rv::Tile(self.value(OpKind::Dot { transpose: false }, &[a, b], ty)))
            }
            "dot_t" => {
                let (a, b) = self.dot_operands(args)?;
                let (ash, bsh) = (self.shape(a), self.shape(b));
                if ash.len() != 2 || bsh.len() != 2 {
                    bail!("`dot_t` expects rank-2 tiles");
                }
                self.check_shapes(&[ash[1]], &[bsh[1]], "dot_t contraction dim")?;
                let shape = [ash[0], bsh[0]];
                if shape.contains(&DYN) {
                    bail!("`dot_t` result shape must be static");
                }
                let acc_elem = super::expr::accumulator_elem(self.elem(a), self.elem(b))?;
                let ty = self.shared_ty(acc_elem, &shape);
                Ok(Rv::Tile(self.value(OpKind::Dot { transpose: true }, &[a, b], ty)))
            }
            "qdot_t" | "q50_qdot_t" => {
                let [a, asc, w, wsc] = args else {
                    bail!("`{callee}` expects (a, a_scales, w, w_scales)");
                };
                let a = self.tile_operand(a, callee)?;
                let asc = self.tile_operand(asc, callee)?;
                let w = self.tile_operand(w, callee)?;
                let wsc = self.tile_operand(wsc, callee)?;
                let shape = [self.shape(a)[0], self.shape(w)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                let op = Intrinsic::from_name(callee).expect("matched above");
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(op), &[a, asc, w, wsc], ty)))
            }
            "qmma_t" | "q50_qmma_t" | "q8_qgemm_t" => {
                let [a, asc, w, wsc] = self.qmma_operands(callee, args)?;
                let shape = [self.shape(a)[0], self.shape(w)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                let op = Intrinsic::from_name(callee).expect("matched above");
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(op), &[a, asc, w, wsc], ty)))
            }
            "iq1s_qmma_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(args)?;
                let shape = [self.shape(a)[0], self.shape(qb)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(
                    OpKind::Intrinsic(Intrinsic::RawQmma(RawFmt::Iq1s)),
                    &[a, asc, qb, d, grid],
                    ty,
                )))
            }
            "iq1s_qmma_staged_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(args)?;
                let shape = [self.shape(a)[0], self.shape(qb)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(
                    OpKind::Intrinsic(Intrinsic::RawQmmaStaged(RawFmt::Iq1s)),
                    &[a, asc, qb, d, grid],
                    ty,
                )))
            }
            _ if super::store::qgemm_format(callee).is_some() => {
                let fmt = super::store::qgemm_format(callee).expect("matched above");
                let tiles = self.qgemm_operands(fmt, callee, args)?;
                let shape = [self.shape(tiles[0])[0], self.shape(tiles[2])[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQgemm(fmt)), &tiles, ty)))
            }
            _ if super::store::staged_qmma_format(callee).is_some() => {
                let fmt = super::store::staged_qmma_format(callee).expect("matched above");
                let ops = self.staged_qmma_operands(callee, args)?;
                let shape = [self.shape(ops[0])[0], self.shape(ops[2])[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQmmaStaged(fmt)), &ops, ty)))
            }
            "gather" => {
                let [table, idx] = args else {
                    bail!("`gather` expects (table, idx)");
                };
                let table = self.tile_operand(table, "gather")?;
                let idx = self.tile_operand(idx, "gather")?;
                let shape = self.shape(idx);
                let ty = self.shared_ty(self.elem(table), &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::Gather), &[table, idx], ty)))
            }
            "flat" => {
                let [arg] = args else {
                    bail!("`flat` expects one tile argument");
                };
                if let Some(name) = arg.as_var()
                    && let Some(Binding::Tensor(t)) = self.lookup(name)
                {
                    return Ok(Rv::Tile(self.tile_flat(t)?));
                }
                let Rv::Tile(t) = self.emit_expr(arg)? else {
                    bail!("`flat` expects a tile argument");
                };
                Ok(Rv::Tile(self.tile_flat(t)?))
            }
            _ if ElemStep::from_callee(callee).is_some_and(|s| !matches!(s, ElemStep::Cast(_))) => {
                let [arg] = args else {
                    bail!("`{callee}` expects one argument");
                };
                let step = ElemStep::from_callee(callee).expect("a math call");
                let t = match self.emit_expr(arg)? {
                    Rv::Tile(t) => t,
                    Rv::Scalar(v) => return Ok(Rv::Scalar(self.scalar_math(callee, step, v)?)),
                };
                // The emitter may rewrite an owned temp in place. The result
                // type is the source's either way.
                let shape = self.shape(t);
                let in_place = self.owned(t) && !matches!(step, ElemStep::Math(_));
                if !in_place && shape.contains(&DYN) {
                    bail!("`{callee}` needs a static tile shape");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Map(Map::Unary(step)), &[t], ty)))
            }
            "where" => {
                let [cond, a, b] = args else {
                    bail!("`where` expects (condition, a, b)");
                };
                let mut reads_tile = false;
                cond.walk(&mut |e| reads_tile |= self.is_tile_expr(e) || self.is_slice_expr(e));
                if reads_tile {
                    bail!(
                        "a `where` over tiles has to be the value of a tile assignment, \
                         `t = where(..)`, so it can be swept into `t`"
                    );
                }
                let cond = self.emit_scalar(cond)?;
                let cond = self.expect_bool(cond, "where's condition")?;
                let (a, b) = (self.emit_scalar(a)?, self.emit_scalar(b)?);
                let (a, b) = self.unify(a, b)?;
                let ty = Type::Scalar(self.scalar_of(a));
                Ok(Rv::Scalar(self.value(OpKind::Select, &[cond, a, b], ty)))
            }
            "iota" => bail!(
                "`iota(d)` is an element's position in a tile assignment, so it belongs in \
                 the value of one: `t = where(iota(1) < n, t, 0.0)`"
            ),
            "tmax" => {
                let [x, y] = args else {
                    bail!("`tmax` expects two tile arguments");
                };
                let (Rv::Tile(a), Rv::Tile(b)) = (self.emit_expr(x)?, self.emit_expr(y)?) else {
                    bail!("`tmax` expects tile arguments");
                };
                if self.elem(a) != self.elem(b) {
                    bail!("`tmax` operands must share an element type");
                }
                let shape = broadcast_shape(&self.shape(a), &self.shape(b))
                    .ok_or_else(|| anyhow!("`tmax` operands are not broadcast-compatible"))?;
                let ty = self.shared_ty(self.elem(a), &shape);
                Ok(Rv::Tile(self.value(OpKind::Map(Map::Max), &[a, b], ty)))
            }
            "argsel" => self.emit_argsel(args),
            "rowmax" | "rowsum" | "rowargmax" => {
                let how = match callee {
                    "rowmax" => Reduce::RowMax,
                    "rowsum" => Reduce::RowSum,
                    _ => Reduce::RowArgMax,
                };
                if let [Expr { kind: ExprKind::Var(name), .. }] = args
                    && let Some(Binding::Regs(regs)) = self.lookup(name)
                    && let Some(out) = self.regs_reduce(how, regs)
                {
                    return Ok(Rv::Tile(out));
                }
                if let [arg] = args
                    && let Some(out) = self.reduce_fused(how, arg)?
                {
                    return Ok(Rv::Tile(out));
                }
                let t = self.reduce_arg(args, callee)?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("row reduction expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("row reduction needs a static tile shape");
                }
                if !self.elem(t).is_float() && !self.elem(t).is_int() {
                    bail!("row reduction needs a numeric element type");
                }
                let out = if how == Reduce::RowArgMax { Scalar::I32 } else { self.elem(t) };
                let ty = self.shared_ty(out, &[shape[0], 1]);
                Ok(Rv::Tile(self.value(OpKind::Reduce(how), &[t], ty)))
            }
            "cumsum" => {
                let t = self.reduce_arg(args, "cumsum")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("`cumsum` expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("`cumsum` needs a static tile shape");
                }
                if !self.elem(t).is_float() {
                    bail!("`cumsum` needs a float element type");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::CumSum), &[t], ty)))
            }
            "tril" => {
                let t = self.reduce_arg(args, "tril")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("`tril` expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("`tril` needs a static tile shape");
                }
                if !self.elem(t).is_float() {
                    bail!("`tril` needs a float element type");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::Tril), &[t], ty)))
            }
            "transpose" => {
                let t = self.reduce_arg(args, "transpose")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("`transpose` expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("`transpose` needs a static tile shape");
                }
                let ty = self.shared_ty(self.elem(t), &[shape[1], shape[0]]);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::Transpose), &[t], ty)))
            }
            other if AstScalar::from_name(other).is_some_and(|s| s != AstScalar::Bool) => {
                let want = Scalar::from_ast(AstScalar::from_name(other).expect("matched above"));
                let [arg] = args else {
                    bail!("{other} expects one argument to convert");
                };
                match self.emit_expr(arg)? {
                    Rv::Tile(t) => Ok(Rv::Tile(self.tile_cast(t, want)?)),
                    Rv::Scalar(v) => Ok(Rv::Scalar(self.numeric_cast(v, want))),
                }
            }
            // The parser rewrites these with two arguments, so any other
            // count arrives here.
            "min" | "max" | "pow" => bail!("`{callee}` takes two arguments, got {}", args.len()),
            other => {
                let near = crate::diag::similar(other, super::names::builtins().iter().map(String::as_str));
                Err(crate::ir::AtOp::unplaced(format!("unknown function `{other}`"))
                    .label("not a built-in")
                    .instead(other, near)
                    .into())
            }
        }
    }

    /// A math call on one scalar. Only `abs` takes an integer; the rest take
    /// a float and give an f32, the width the target computes in.
    fn scalar_math(&mut self, callee: &str, step: ElemStep, v: ValueId) -> Result<ValueId> {
        let t = self.scalar_of(v);
        let result = match t {
            Scalar::Index if step == ElemStep::Math(MathFn::Abs) => Scalar::Index,
            t if t.is_float() => Scalar::F32,
            t => bail!("`{callee}` needs a float operand, got {}", t.mlir_name()),
        };
        Ok(self.value(OpKind::Math(step), &[v], Type::Scalar(result)))
    }


    /// An operand that has to be a tile: `` "`{what}` expects tile operands" ``.
    pub(crate) fn tile_operand(&mut self, e: &Expr, what: &str) -> Result<ValueId> {
        match self.emit_expr(e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("`{what}` expects tile operands"),
        }
    }

    pub(crate) fn dot_operands(&mut self, args: &[Expr]) -> Result<(ValueId, ValueId)> {
        let [lhs, rhs] = args else {
            bail!("`dot` expects two tile arguments");
        };
        let Rv::Tile(a) = self.emit_expr(lhs)? else {
            bail!("`dot` expects tile operands");
        };
        let Rv::Tile(b) = self.emit_expr(rhs)? else {
            bail!("`dot` expects tile operands");
        };
        if self.has_wmma() {
            // The tensor-core paths stage both operands as f16 themselves.
            return Ok((a, b));
        }
        // a is [m, k]; b is [k, n] for `dot` and [n, k] for `dot_t`.
        let (a_reads, b_reads) = match (&self.shape(a)[..], &self.shape(b)[..]) {
            (&[m, k], &[b0, b1]) => (if b0 == k { b1 } else { b0 }, m),
            _ => (1, 1),
        };
        Ok((self.stage_reused(a, a_reads), self.stage_reused(b, b_reads)))
    }

    /// An operand a contraction reads `reads` times, copied into shared
    /// memory when it is a view of global memory: a slice read once wants to
    /// stay a view, one read many times wants the staging whether or not a
    /// mask forced it.
    fn stage_reused(&mut self, v: ValueId, reads: i64) -> ValueId {
        let Type::Tile(t) = self.ir.ty(v) else {
            return v;
        };
        let shape = self.shape(v);
        if reads < 2 || t.space != Space::Global || self.is_masked(v) || shape.contains(&DYN) {
            return v;
        }
        let elem = self.elem(v);
        let pad = self.should_pad_stage(elem, &shape);
        let ty = if pad { self.padded_ty(elem, &shape) } else { self.shared_ty(elem, &shape) };
        self.value(OpKind::Stage(Stage { pad, sync: true }), &[v], ty)
    }

    pub(crate) fn reduce_arg(&mut self, args: &[Expr], name: &str) -> Result<ValueId> {
        let [arg] = args else {
            bail!("{name} expects one tile argument");
        };
        let Rv::Tile(t) = self.emit_expr(arg)? else {
            bail!("{name} expects a tile argument");
        };
        Ok(t)
    }

    /// `(a, a_scales, w, w_scales)` for the Q8_0 projections: `qmma_t`,
    /// `q50_qmma_t` and `q8_qgemm_t`.
    pub(crate) fn qmma_operands(&mut self, callee: &str, args: &[Expr]) -> Result<[ValueId; 4]> {
        let [a, asc, w, wsc] = args else {
            bail!("`{callee}` expects (a, a_scales, w, w_scales)");
        };
        let a = self.tile_operand(a, callee)?;
        let asc = self.tile_operand(asc, callee)?;
        let w = self.tile_operand(w, callee)?;
        let wsc = self.tile_operand(wsc, callee)?;
        Ok([a, asc, w, wsc])
    }

    pub(crate) fn iq1s_qmma_operands(&mut self, args: &[Expr]) -> Result<[ValueId; 5]> {
        let [a, asc, qb, d, grid] = args else {
            bail!("`iq1s_qmma_t` expects (a, a_scales, qb, d, grid)");
        };
        let a = self.tile_operand(a, "iq1s_qmma_t")?;
        let asc = self.tile_operand(asc, "iq1s_qmma_t")?;
        let qb = self.tile_operand(qb, "iq1s_qmma_t")?;
        let d = self.tile_operand(d, "iq1s_qmma_t")?;
        let grid = self.tile_operand(grid, "iq1s_qmma_t")?;
        Ok([a, asc, qb, d, grid])
    }

    pub(crate) fn qgemm_operands(&mut self, fmt: RawFmt, callee: &str, args: &[Expr]) -> Result<Vec<ValueId>> {
        let want = 4 + qg_tables(fmt);
        // A format that subtracts a minimum may be given the activation's
        // (scale, sum) pairs, after the tables.
        if args.len() != want && !(qg_has_min(fmt) && args.len() == want + 1) {
            if qg_has_min(fmt) {
                bail!("`{callee}` expects {want} or {} operands: (a, a_scales, qb, d, tables.., a_pairs?)", want + 1);
            }
            bail!("`{callee}` expects {want} operands: (a, a_scales, qb, d, tables..)");
        }
        args.iter().map(|e| self.tile_operand(e, callee)).collect()
    }

    /// `(a, a_scales, qb, d, grid, signs)` for the staged IQ2/IQ3 projections.
    pub(crate) fn staged_qmma_operands(&mut self, callee: &str, args: &[Expr]) -> Result<[ValueId; 6]> {
        let [a, asc, qb, d, grid, signs] = args else {
            bail!("`{callee}` expects (a, a_scales, qb, d, grid, signs)");
        };
        let a = self.tile_operand(a, callee)?;
        let asc = self.tile_operand(asc, callee)?;
        let qb = self.tile_operand(qb, callee)?;
        let d = self.tile_operand(d, callee)?;
        let grid = self.tile_operand(grid, callee)?;
        let signs = self.tile_operand(signs, callee)?;
        Ok([a, asc, qb, d, grid, signs])
    }

    fn emit_argsel(&mut self, args: &[Expr]) -> Result<Rv> {
        let [va, vb, ia, ib] = args else {
            bail!("`argsel` expects four tile arguments (value, value, index, index)");
        };
        let (Rv::Tile(va), Rv::Tile(vb), Rv::Tile(ia), Rv::Tile(ib)) = (
            self.emit_expr(va)?,
            self.emit_expr(vb)?,
            self.emit_expr(ia)?,
            self.emit_expr(ib)?,
        ) else {
            bail!("`argsel` expects tile arguments");
        };
        let vshape = broadcast_shape(&self.shape(va), &self.shape(vb))
            .ok_or_else(|| anyhow!("`argsel` value operands are not broadcast-compatible"))?;
        let ishape = broadcast_shape(&self.shape(ia), &self.shape(ib))
            .ok_or_else(|| anyhow!("`argsel` index operands are not broadcast-compatible"))?;
        ensure!(
            self.elem(va) == self.elem(vb) && self.elem(ia) == self.elem(ib) && vshape == ishape,
            "`argsel`: value and index operands must each share an element type, and both \
             pairs must broadcast to the same shape"
        );
        let ty = self.shared_ty(self.elem(ia), &ishape);
        Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::ArgSel), &[va, vb, ia, ib], ty)))
    }

    /// `flat(t)`: a tile viewed as one row.
    fn tile_flat(&mut self, src: ValueId) -> Result<ValueId> {
        let shape = self.shape(src);
        if shape.len() != 2 {
            bail!("`flat` expects a rank-2 tile");
        }
        // A scratch tensor's flat view: the whole buffer as one row.
        if let Type::Tensor(t) = self.ir.ty(src).clone() {
            let [rows, cols] = [shape[0], shape[1]];
            if rows == DYN || cols == DYN || !self.is_whole_tensor(src) {
                bail!("`flat` of a tensor takes a whole one of a static shape: a parameter or scratch, not a slice");
            }
            let ty = Type::Tensor(crate::ir::TensorType {
                elem: t.elem,
                shape: vec![crate::ir::Extent::Fixed(1), crate::ir::Extent::Fixed(rows * cols)],
                div: vec![1, 1],
            });
            return Ok(self.value(OpKind::Flat, &[src], ty));
        }
        if !self.is_buffer(src) {
            bail!("`flat` expects a declared tile, not a slice of one");
        }
        let t = self.tile(src).clone();
        if t.layout.row_stride.is_some() || t.layout.swizzle.is_some() {
            bail!("a padded or swizzled staging tile has no flat view");
        }
        let [rows, cols] = [shape[0], shape[1]];
        if rows == DYN || cols == DYN {
            bail!("`flat` expects a static tile shape");
        }
        let len = rows * cols;
        let ty = Type::Tile(crate::ir::TileType {
            elem: t.elem,
            shape: vec![crate::ir::Extent::Fixed(1), crate::ir::Extent::Fixed(len)],
            layout: crate::ir::Layout {
                row_stride: None,
                swizzle: None,
                align_div: t.layout.align_div,
            },
            space: t.space,
        });
        Ok(self.value(OpKind::Flat, &[src], ty))
    }

    /// Whether `v` is a whole tensor, a parameter or scratch, as a name binds
    /// it: no slice of one.
    fn is_whole_tensor(&self, v: ValueId) -> bool {
        match self.ir.def_op(v) {
            None => true,
            Some(op) => match self.ir.kind(op) {
                OpKind::Scratch => true,
                OpKind::AssumeAlign => self.is_whole_tensor(self.ir.operand(op, 0)),
                _ => false,
            },
        }
    }

    fn emit_atomic_add(&mut self, args: &[Expr]) -> Result<Rv> {
        let [t, i, v] = args else {
            bail!("`atomic_add` expects (tensor, index, value)");
        };
        let (mem, name) = self.atomic_tensor(t, "atomic_add")?;
        let elem = self.elem(mem);
        if !matches!(elem, Scalar::I32 | Scalar::F32) {
            bail!("`atomic_add` expects an i32 or f32 tensor, but `{name}` holds {}", elem.mlir_name());
        }
        let idx = self.emit_index(i, "atomic_add index")?;
        let val = self.emit_scalar(v)?;
        let val = match (elem, self.scalar_of(val)) {
            (Scalar::I32, Scalar::I32) | (Scalar::F32, Scalar::F32) => val,
            (Scalar::I32, Scalar::Index) => {
                self.value(OpKind::IndexCast(Scalar::I32), &[val], Type::Scalar(Scalar::I32))
            }
            (Scalar::I32, Scalar::I8 | Scalar::I64) => self.numeric_cast(val, Scalar::I32),
            (Scalar::F32, t) if t.is_numeric() => self.numeric_cast(val, Scalar::F32),
            (_, t) => bail!("`atomic_add` into {} cannot take a {} value", elem.mlir_name(), t.mlir_name()),
        };
        let old = self.value(OpKind::AtomicAdd, &[mem, idx, val], Type::Scalar(elem));
        Ok(Rv::Scalar(old))
    }

    fn emit_atomic_cas(&mut self, args: &[Expr]) -> Result<Rv> {
        let [t, i, expected, v] = args else {
            bail!("`atomic_cas` expects (tensor, index, expected, value)");
        };
        let (mem, name) = self.atomic_tensor(t, "atomic_cas")?;
        let elem = self.elem(mem);
        if elem != Scalar::I32 {
            bail!("`atomic_cas` expects an i32 tensor, but `{name}` holds {}", elem.mlir_name());
        }
        let idx = self.emit_index(i, "atomic_cas index")?;
        let mut operands = vec![mem, idx];
        for e in [expected, v] {
            let s = self.emit_scalar(e)?;
            operands.push(match self.scalar_of(s) {
                Scalar::I32 => s,
                Scalar::Index => self.value(OpKind::IndexCast(Scalar::I32), &[s], Type::Scalar(Scalar::I32)),
                Scalar::I8 | Scalar::I64 => self.numeric_cast(s, Scalar::I32),
                t => bail!("`atomic_cas` into i32 cannot take a {} value", t.mlir_name()),
            });
        }
        let old = self.value(OpKind::AtomicCas, &operands, Type::Scalar(Scalar::I32));
        Ok(Rv::Scalar(old))
    }

    /// `load_acquire(t, i)` or `store_release(t, i, v)` on an `i32` tensor.
    fn emit_system_scope(&mut self, callee: &str, args: &[Expr]) -> Result<Rv> {
        let store = callee == "store_release";
        let (t, i, v) = match (store, args) {
            (false, [t, i]) => (t, i, None),
            (true, [t, i, v]) => (t, i, Some(v)),
            (false, _) => bail!("`load_acquire` expects (tensor, index)"),
            (true, _) => bail!("`store_release` expects (tensor, index, value)"),
        };
        let (mem, name) = self.atomic_tensor(t, callee)?;
        if self.elem(mem) != Scalar::I32 {
            bail!("`{callee}` expects an i32 tensor, but `{name}` holds {}", self.elem(mem).mlir_name());
        }
        let mut operands = vec![mem, self.emit_index(i, callee)?];
        let Some(v) = v else {
            return Ok(Rv::Scalar(self.value(OpKind::LoadAcquire, &operands, Type::Scalar(Scalar::I32))));
        };
        let v = self.emit_scalar(v)?;
        operands.push(self.coerce(v, Scalar::I32)?);
        Ok(Rv::Scalar(self.value(OpKind::StoreRelease, &operands, Type::INDEX)))
    }

    /// An atomic's tensor operand: a named tensor parameter.
    fn atomic_tensor<'e>(&self, t: &'e Expr, what: &str) -> Result<(ValueId, &'e str)> {
        let ExprKind::Var(name) = &t.kind else {
            bail!("`{what}` expects a named tensor parameter");
        };
        let Some(Binding::Tensor(mem)) = self.lookup(name) else {
            bail!("`{what}` expects a tensor parameter, but `{name}` is not one");
        };
        Ok((mem, name))
    }

    /// The atomic-state operand: a named `i32` tensor parameter.
    fn barrier_tensor(&self, e: &Expr, what: &str) -> Result<ValueId> {
        let ExprKind::Var(name) = &e.kind else {
            bail!("`{what}` expects a named i32 tensor parameter");
        };
        let Some(Binding::Tensor(mem)) = self.lookup(name) else {
            bail!("`{what}` expects a tensor parameter, but `{name}` is not one");
        };
        if self.elem(mem) != Scalar::I32 {
            bail!(
                "`{what}` expects an i32 tensor, but `{name}` holds {}",
                self.elem(mem).mlir_name()
            );
        }
        Ok(mem)
    }

    fn emit_grid_barrier(&mut self, args: &[Expr]) -> Result<Rv> {
        let mem = match args {
            [] => match self.lookup(super::GRID_BARRIER_NAME) {
                Some(Binding::Tensor(mem)) => mem,
                _ => bail!("`grid_barrier()` uses the barrier a `@persistent` launch supplies; this kernel is not `@persistent`"),
            },
            [bar] => self.barrier_tensor(bar, "grid_barrier")?,
            _ => bail!("`grid_barrier` takes the barrier tensor, or nothing in a `@persistent` kernel"),
        };
        Ok(Rv::Scalar(self.value(OpKind::GridBarrier, &[mem], Type::INDEX)))
    }

}

/// Table operands after `(a, a_scales, qb, d)` of the grouped raw formats.
pub(crate) fn qg_tables(fmt: RawFmt) -> usize {
    match fmt {
        RawFmt::Iq1s | RawFmt::Iq1m => 1,
        RawFmt::Iq2xxs | RawFmt::Iq2xs | RawFmt::Iq2s | RawFmt::Iq3xxs | RawFmt::Iq3s => 2,
        RawFmt::Iq4xs | RawFmt::Q2k | RawFmt::Q3k | RawFmt::Q4k | RawFmt::Q5k | RawFmt::Q6k | RawFmt::Q80 => 0,
        RawFmt::Ptq1 => 0,
    }
}

/// Whether a format's runs subtract a minimum, so its `qgemm_t` needs the
/// activation's group sums and may take them, paired with the scales, as an
/// operand.
pub(crate) fn qg_has_min(fmt: RawFmt) -> bool {
    matches!(fmt, RawFmt::Q4k | RawFmt::Q5k)
}

