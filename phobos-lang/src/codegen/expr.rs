use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn emit_expr(&mut self, block: &Block<'c>, expr: &Expr) -> Result<Rv<'c>> {
        match expr {
            Expr::Int(n) => Ok(Rv::Scalar(self.const_index(block, *n)?)),
            Expr::Float(v) => Ok(Rv::Scalar(self.push(
                block,
                arith::constant(
                    self.ctx,
                    // The type is coerced to f16 later when necessary.
                    // We can get away with this for now.
                    FloatAttribute::new(self.ctx, self.f32_t, *v).into(),
                    self.loc,
                ),
            )?)),
            Expr::Bool(b) => Ok(Rv::Scalar(self.const_bool(block, *b)?)),
            Expr::Var(name) => match self.lookup(name) {
                Some(Binding::Let { value, .. }) => Ok(Rv::Scalar(value)),
                Some(Binding::Var { slot, .. }) => Ok(Rv::Scalar(
                    self.push(block, memref::load(slot, &[], self.loc))?,
                )),
                Some(Binding::Tensor(_)) => {
                    bail!("tensor '{name}' used as a value; index or slice it")
                }
                Some(Binding::View(t) | Binding::Tile(t)) => Ok(Rv::Tile(t)),
                Some(Binding::Frags(_)) => bail!(
                    "fragment accumulator '{name}' can only be scaled, dot-accumulated, or stored"
                ),
                None => match self.shape_env.get(name) {
                    Some(&v) => Ok(Rv::Scalar(self.const_index(block, v)?)),
                    None => bail!("unknown identifier '{name}'"),
                },
            },
            Expr::Unary { op, rhs } => {
                let v = self.emit_scalar(block, rhs)?;
                let t = v.r#type();
                let v = match op {
                    UnOp::Neg if self.is_float(t) => self.push(block, arith::negf(v, self.loc))?,
                    UnOp::Neg if t == self.index_t => {
                        let zero = self.const_index(block, 0)?;
                        self.subi(block, zero, v)?
                    }
                    UnOp::Not if t == self.bool_t => {
                        let one = self.const_bool(block, true)?;
                        self.push(block, arith::xori(v, one, self.loc))?
                    }
                    UnOp::Neg => bail!("`-` needs a numeric operand, got {t}"),
                    UnOp::Not => bail!("`!` needs a bool operand, got {t}"),
                };
                Ok(Rv::Scalar(v))
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = self.emit_expr(block, lhs)?;
                let r = self.emit_expr(block, rhs)?;
                match (l, r) {
                    (Rv::Scalar(a), Rv::Scalar(b)) => {
                        Ok(Rv::Scalar(self.emit_binop(block, *op, a, b)?))
                    }
                    (Rv::Tile(a), Rv::Tile(b)) => {
                        Ok(Rv::Tile(self.emit_tile_binary(block, *op, &a, &b)?))
                    }
                    (Rv::Tile(a), Rv::Scalar(b)) => {
                        // tile * scalar broadcasts the scalar over the tile.
                        Ok(Rv::Tile(self.emit_tile_scalar(block, *op, &a, b, false)?))
                    }
                    (Rv::Scalar(a), Rv::Tile(b)) => {
                        Ok(Rv::Tile(self.emit_tile_scalar(block, *op, &b, a, true)?))
                    }
                }
            }
            Expr::Index { base, subs } => {
                let (mv, binding) = self.mem_base(base)?;
                if subs.iter().all(|s| matches!(s, Sub::Point(_))) {
                    let indices = self.emit_indices(block, subs, mv.shape.len())?;
                    Ok(Rv::Scalar(self.load_scalar(block, &mv, &indices)?))
                } else {
                    if !matches!(binding, Binding::Tensor(_)) {
                        bail!("only tensors can be sliced");
                    }

                    let view = self.emit_subview(block, &mv, subs)?;

                    if view.is_masked() {
                        Ok(Rv::Tile(self.materialize_masked(block, &view)?))
                    } else {
                        Ok(Rv::Tile(view))
                    }
                }
            }
            Expr::Call { callee, args } => self.emit_call(block, callee, args),
        }
    }

    pub(super) fn emit_scalar(&mut self, block: &Block<'c>, expr: &Expr) -> Result<Value<'c, 'c>> {
        match self.emit_expr(block, expr)? {
            Rv::Scalar(v) => Ok(v),
            Rv::Tile(_) => bail!("expected a scalar value, got a tile"),
        }
    }

    pub(super) fn emit_index(
        &mut self,
        block: &Block<'c>,
        expr: &Expr,
        what: &str,
    ) -> Result<Value<'c, 'c>> {
        let v = self.emit_scalar(block, expr)?;
        self.expect_index(v, what)
    }

    pub(super) fn loop_bounds(
        &mut self,
        block: &Block<'c>,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>, i64)> {
        let lo = self.emit_index(block, start, "loop start")?;
        let hi = self.emit_index(block, end, "loop end")?;
        let st = match step {
            Some(e) => self.emit_index(block, e, "loop step")?,
            None => self.const_index(block, 1)?,
        };
        let iv_div = gcd(self.expr_div(start), step.map_or(1, |e| self.expr_div(e)));
        Ok((lo, hi, st, iv_div))
    }

    pub(super) fn emit_call(
        &mut self,
        block: &Block<'c>,
        callee: &str,
        args: &[Expr],
    ) -> Result<Rv<'c>> {
        match callee {
            "program_id" => {
                let [Expr::Int(d @ 0..=2)] = args else {
                    bail!("program_id expects one literal dimension argument 0..=2");
                };
                let dim = ["x", "y", "z"][*d as usize];
                Ok(Rv::Scalar(self.gpu_index(block, "gpu.block_id", dim)?))
            }
            // dot outside an assignment: materialize into a fresh buffer.
            // acc += dot(a, b) is handled in store_tile.
            "dot" => {
                let (a, b) = self.dot_operands(block, args)?;
                let shape = [a.shape[0], b.shape[1]];
                if shape.contains(&DYN) {
                    bail!("dot result shape must be static; assign to a tile-typed var instead");
                }
                let out = self.alloc_tile_shaped(block, a.elem, &shape)?;
                self.check_matmul_shapes(&a, &b, &out)?;
                if !self.wmma_dot(block, &a, &b, &out, false, false)? {
                    let zero = self.zero_scalar(block, out.elem)?;
                    self.tile_fill(block, zero, &out)?;
                    self.tile_matmul(block, &a, &b, &out)?;
                }
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            // dot_t(a, b) = a * b^T: materialize into a fresh buffer.
            // acc += dot(a, b) has no transposed analogue.
            "dot_t" => {
                let (a, b) = self.dot_operands(block, args)?;
                if a.shape.len() != 2 || b.shape.len() != 2 {
                    bail!("dot_t expects rank-2 tiles");
                }
                self.check_shapes(&[a.shape[1]], &[b.shape[1]], "dot_t contraction dim")?;
                let shape = [a.shape[0], b.shape[0]];
                if shape.contains(&DYN) {
                    bail!("dot_t result shape must be static");
                }
                let out = self.alloc_tile_shaped(block, a.elem, &shape)?;
                if !self.wmma_dot(block, &a, &b, &out, true, false)? {
                    self.tile_matmul_t(block, &a, &b, &out)?;
                }
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            // exp(t): element-wise e^x over a tile.
            "exp" => {
                let [arg] = args else {
                    bail!("exp expects one tile argument");
                };
                let Rv::Tile(t) = self.emit_expr(block, arg)? else {
                    bail!("exp expects a tile argument");
                };
                Ok(Rv::Tile(self.tile_exp(block, &t)?))
            }
            // tmax(a, b): element-wise maximum (broadcasting).
            "tmax" => {
                let [x, y] = args else {
                    bail!("tmax expects two tile arguments");
                };
                let (Rv::Tile(a), Rv::Tile(b)) =
                    (self.emit_expr(block, x)?, self.emit_expr(block, y)?)
                else {
                    bail!("tmax expects tile arguments");
                };
                if a.elem != b.elem {
                    bail!("tmax operands must share an element type");
                }
                let shape = broadcast_shape(&a.shape, &b.shape)
                    .ok_or_else(|| anyhow!("tmax operands are not broadcast-compatible"))?;
                let out = self.alloc_tile_shaped(block, a.elem, &shape)?;
                self.tile_max_bc(block, &a, &b, &out)?;
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            // rowmax(t) / rowsum(t): reduce a rank-2 tile over its last
            // column dim, producing a [rows, 1] column vector.
            "rowmax" => {
                let t = self.reduce_arg(block, args, "rowmax")?;
                let out = self.tile_rowreduce(block, &t, Reduce::Max)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            "rowsum" => {
                let t = self.reduce_arg(block, args, "rowsum")?;
                let out = self.tile_rowreduce(block, &t, Reduce::Sum)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            // cumsum(t): inclusive prefix sum down the rows (the sequence
            // axis), the running gate cumulant chunkwise linear attention
            // needs.
            "cumsum" => {
                let t = self.reduce_arg(block, args, "cumsum")?;
                let out = self.tile_cumsum(block, &t)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            // tril(t): causal lower-triangular mask (zero the strict upper
            // triangle). May rewrite t in place, so t is not released here.
            "tril" => {
                let t = self.reduce_arg(block, args, "tril")?;
                let out = self.tile_tril(block, &t)?;
                Ok(Rv::Tile(out))
            }
            // transpose(t): rank-2 tile transpose, for contracting over the
            // sequence axis (K.T @ V) in the chunk recurrence.
            "transpose" => {
                let t = self.reduce_arg(block, args, "transpose")?;
                let out = self.tile_transpose(block, &t)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            other => bail!("unknown function '{other}'"),
        }
    }

    pub(super) fn dot_operands(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
    ) -> Result<(MemVal<'c>, MemVal<'c>)> {
        let [lhs, rhs] = args else {
            bail!("dot expects two tile arguments");
        };
        let Rv::Tile(a) = self.emit_expr(block, lhs)? else {
            bail!("dot expects tile operands");
        };
        let Rv::Tile(b) = self.emit_expr(block, rhs)? else {
            bail!("dot expects tile operands");
        };
        Ok((a, b))
    }

    pub(super) fn reduce_arg(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
        name: &str,
    ) -> Result<MemVal<'c>> {
        let [arg] = args else {
            bail!("{name} expects one tile argument");
        };
        let Rv::Tile(t) = self.emit_expr(block, arg)? else {
            bail!("{name} expects a tile argument");
        };
        Ok(t)
    }

    pub(super) fn emit_binop(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let (lhs, rhs) = self.unify(block, lhs, rhs)?;
        let t = lhs.r#type();
        let loc = self.loc;
        let op = if self.is_float(t) {
            match op {
                BinOp::Add => arith::addf(lhs, rhs, loc),
                BinOp::Sub => arith::subf(lhs, rhs, loc),
                BinOp::Mul => arith::mulf(lhs, rhs, loc),
                BinOp::Div => arith::divf(lhs, rhs, loc),
                BinOp::Rem => arith::remf(lhs, rhs, loc),
                BinOp::Eq => arith::cmpf(self.ctx, arith::CmpfPredicate::Oeq, lhs, rhs, loc),
                BinOp::Ne => arith::cmpf(self.ctx, arith::CmpfPredicate::One, lhs, rhs, loc),
                BinOp::Lt => arith::cmpf(self.ctx, arith::CmpfPredicate::Olt, lhs, rhs, loc),
                BinOp::Le => arith::cmpf(self.ctx, arith::CmpfPredicate::Ole, lhs, rhs, loc),
                BinOp::Gt => arith::cmpf(self.ctx, arith::CmpfPredicate::Ogt, lhs, rhs, loc),
                BinOp::Ge => arith::cmpf(self.ctx, arith::CmpfPredicate::Oge, lhs, rhs, loc),
            }
        } else if t == self.index_t {
            match op {
                BinOp::Add => arith::addi(lhs, rhs, loc),
                BinOp::Sub => arith::subi(lhs, rhs, loc),
                BinOp::Mul => arith::muli(lhs, rhs, loc),
                BinOp::Div => arith::divsi(lhs, rhs, loc),
                BinOp::Rem => arith::remsi(lhs, rhs, loc),
                BinOp::Eq => arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lhs, rhs, loc),
                BinOp::Ne => arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, lhs, rhs, loc),
                BinOp::Lt => arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, lhs, rhs, loc),
                BinOp::Le => arith::cmpi(self.ctx, arith::CmpiPredicate::Sle, lhs, rhs, loc),
                BinOp::Gt => arith::cmpi(self.ctx, arith::CmpiPredicate::Sgt, lhs, rhs, loc),
                BinOp::Ge => arith::cmpi(self.ctx, arith::CmpiPredicate::Sge, lhs, rhs, loc),
            }
        } else if t == self.bool_t {
            match op {
                BinOp::Eq => arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lhs, rhs, loc),
                BinOp::Ne => arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, lhs, rhs, loc),
                _ => bail!("operator not supported for bool operands"),
            }
        } else {
            bail!("operator not supported for operands of type {t}");
        };
        self.push(block, op)
    }

    /// Element-wise tile arithmetic into a fresh buffer.
    pub(super) fn emit_tile_binary(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let shape = broadcast_shape(&a.shape, &b.shape).ok_or_else(|| {
            anyhow!(
                "elementwise tile op: shapes {} and {} are not broadcast-compatible",
                fmt_shape(&a.shape),
                fmt_shape(&b.shape)
            )
        })?;
        if shape.contains(&DYN) {
            bail!("elementwise tile result shape must be static");
        }
        let out = self.alloc_tile_shaped(block, a.elem, &shape)?;
        self.tile_binary_dispatch(block, op, a, b, &out)?;
        self.release(a);
        self.release(b);
        Ok(out)
    }

    /// Element-wise tile*scalar (or scalar*tile) into a fresh buffer.
    pub(super) fn emit_tile_scalar(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        tile: &MemVal<'c>,
        scalar: Value<'c, 'c>,
        scalar_left: bool,
    ) -> Result<MemVal<'c>> {
        let out = self.alloc_tile_shaped(block, tile.elem, &tile.shape)?;
        self.tile_scalar_into(block, op, tile, scalar, scalar_left, &out)?;
        self.release(tile);
        Ok(out)
    }

    pub(super) fn unify(
        &mut self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let (lt, rt) = (lhs.r#type(), rhs.r#type());
        if lt == rt {
            return Ok((lhs, rhs));
        }
        // mixed float widths widen to the wider type (f16 < f32 < f64)
        if let (Some(lr), Some(rr)) = (self.float_rank(lt), self.float_rank(rt)) {
            let want = if lr >= rr { lt } else { rt };
            return Ok((
                self.float_cast(block, lhs, want)?,
                self.float_cast(block, rhs, want)?,
            ));
        }
        bail!("mismatched operand types: {lt} vs {rt}")
    }

    /// Coerces value to want for a store
    pub(super) fn coerce(
        &mut self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let t = value.r#type();
        if t == want {
            Ok(value)
        } else if self.float_rank(t).is_some() && self.float_rank(want).is_some() {
            // any float-to-float store rounds/widens to the target type
            // (e.g. an fp32 literal stored into an fp16 tile or tensor).
            self.float_cast(block, value, want)
        } else if t == self.index_t && self.is_int(want) {
            self.push(block, arith::index_cast(value, want, self.loc))
        } else {
            bail!("type mismatch: cannot store {t} where {want} is expected")
        }
    }
}

// memrefs
impl<'p, 'c> Codegen<'p, 'c> {
    /// Resolves the base of an A[...] expression to a memref binding.
    pub(super) fn mem_base(&self, base: &Expr) -> Result<(MemVal<'c>, Binding<'c>)> {
        let Expr::Var(name) = base else {
            bail!("only named tensors and tiles can be indexed");
        };
        match self.lookup(name) {
            Some(binding) => match &binding {
                Binding::Tensor(mv) | Binding::View(mv) | Binding::Tile(mv) => {
                    Ok((mv.clone(), binding.clone()))
                }
                _ => bail!("'{name}' is not a tensor or tile"),
            },
            None => bail!("unknown identifier '{name}'"),
        }
    }

    pub(super) fn emit_indices(
        &mut self,
        block: &Block<'c>,
        subs: &[Sub],
        rank: usize,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if subs.len() != rank {
            bail!("expected {rank} subscripts, got {}", subs.len());
        }
        subs.iter()
            .map(|sub| match sub {
                Sub::Point(e) => self.emit_index(block, e, "subscript"),
                _ => bail!("mixing point and slice subscripts is not supported yet"),
            })
            .collect()
    }

    /// Loads a scalar element; integer elements are widened to index.
    pub(super) fn load_scalar(
        &mut self,
        block: &Block<'c>,
        mv: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<Value<'c, 'c>> {
        let v = self.push(block, memref::load(mv.mem, indices, self.loc))?;
        if self.is_int(mv.elem) {
            self.push(block, arith::index_cast(v, self.index_t, self.loc))
        } else {
            Ok(v)
        }
    }

    /// Lowers slice subscripts to a memref.subview of a tensor.
    ///
    /// Offsets are always passed as dynamic operands; sizes are static when
    /// they fold to constants; strides are always 1, so the result keeps the
    /// source's layout with a dynamic offset.
    pub(super) fn emit_subview(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        subs: &[Sub],
    ) -> Result<MemVal<'c>> {
        let rank = src.shape.len();
        if subs.len() != rank {
            bail!("expected {rank} subscripts, got {}", subs.len());
        }

        let mut offsets = Vec::with_capacity(rank);
        let mut off_divs = Vec::with_capacity(rank);
        let mut dyn_sizes = Vec::new();
        let mut static_sizes = Vec::with_capacity(rank);
        // Dims whose offset rides the ragged remainder chunk's induction
        // variable, and so may run past a dynamic extent (see ragged_iv).
        let mut ragged = vec![false; rank];
        let rides_ragged = |cg: &Self, start: &Expr| {
            cg.ragged_iv
                .as_deref()
                .is_some_and(|iv| start.uses_name(iv))
        };
        for (i, sub) in subs.iter().enumerate() {
            match sub {
                Sub::Point(_) => {
                    bail!("mixing point and slice subscripts is not supported yet")
                }
                // A[start :+ len]
                Sub::Span { start, len } => {
                    ragged[i] = rides_ragged(self, start);
                    offsets.push(self.emit_index(block, start, "slice start")?);
                    off_divs.push(self.expr_div(start));
                    match self.const_fold(len) {
                        Some(n) => static_sizes.push(n),
                        None => {
                            dyn_sizes.push(self.emit_index(block, len, "slice length")?);
                            static_sizes.push(DYN);
                        }
                    }
                }
                // A[start : end]: size is end - start.
                Sub::Range { start, end } => {
                    ragged[i] = rides_ragged(self, start);
                    let off = self.emit_index(block, start, "slice start")?;
                    offsets.push(off);
                    off_divs.push(self.expr_div(start));
                    match (self.const_fold(start), self.const_fold(end)) {
                        (Some(a), Some(b)) => static_sizes.push(b - a),
                        _ => {
                            let end_v = self.emit_index(block, end, "slice end")?;
                            dyn_sizes.push(self.subi(block, end_v, off)?);
                            static_sizes.push(DYN);
                        }
                    }
                }
                // A[:]: the whole dimension.
                Sub::Full => {
                    offsets.push(self.const_index(block, 0)?);
                    off_divs.push(0);
                    if src.shape[i] != DYN {
                        static_sizes.push(src.shape[i]);
                    } else {
                        let pos = self.const_index(block, i as i64)?;
                        dyn_sizes.push(self.push(block, memref::dim(src.mem, pos, self.loc))?);
                        static_sizes.push(DYN);
                    }
                }
            }
        }

        // 4-element alignment proof: the flat offset is sum off_i*stride_i, so
        // its divisibility is the gcd of the per-dim terms. Dynamic strides
        // count as multiples of 4 elements (the row-pitch ABI; see module
        // docs). Every row of the slice is aligned iff the base is and all
        // outer strides are multiples of 4 elements. A multiple-of-4-elements
        // boundary is exactly the byte alignment a 4-element vector of the
        // element type needs (16B for f32, 8B for f16), so the proof is
        // element-type-agnostic; each consumer pairs it with its own element
        // check before building f32-specific vectors.
        //
        // ..thanks Claude!
        let strides = row_major_strides(&src.shape);
        let stride_div = |s: i64| if s == DYN { 4 } else { s.abs().max(1) };
        let base_div = off_divs.iter().zip(&strides).fold(0i64, |acc, (&o, &s)| {
            let term = if o == 0 {
                0
            } else {
                o.saturating_mul(stride_div(s)).min(1 << 20)
            };
            gcd(acc, term)
        });
        let aligned = mult4(base_div) && strides[..rank - 1].iter().all(|&s| mult4(stride_div(s)));

        // Bounds mask: a dim whose (statically known) source extent an aligned
        // tile cannot tile evenly may reach past the end on the last tile, so
        // record its offset and extent for the masked load/store epilogue.
        //
        // A dynamic extent carries no static proof, so it masks only inside a
        // ragged remainder chunk, and then against the tensor's runtime
        // memref.dim. The trimmed main loop stays unmasked and keeps the
        // vector / WMMA / cp.async fast paths (see Codegen::emit_split_for).
        let mut mask = vec![None; rank];
        for i in 0..rank {
            let extent = if !dim_in_bounds(src.shape[i], static_sizes[i], off_divs[i]) {
                self.const_index(block, src.shape[i])?
            } else if src.shape[i] == DYN && ragged[i] && static_sizes[i] != DYN {
                let pos = self.const_index(block, i as i64)?;
                self.push(block, memref::dim(src.mem, pos, self.loc))?
            } else {
                continue;
            };
            mask[i] = Some((offsets[i], extent));
        }

        let result_type = self.subview_type(src, &static_sizes)?;
        let mut operands = vec![src.mem];
        operands.extend_from_slice(&offsets);
        operands.extend_from_slice(&dyn_sizes);

        let op = OperationBuilder::new("memref.subview", self.loc)
            .add_operands(&operands)
            .add_attributes(&[
                (self.id("static_offsets"), self.i64_array(&vec![DYN; rank])?),
                (self.id("static_sizes"), self.i64_array(&static_sizes)?),
                (self.id("static_strides"), self.i64_array(&vec![1; rank])?),
                (
                    self.id("operandSegmentSizes"),
                    self.i32_array(&[1, rank as i32, dyn_sizes.len() as i32, 0])?,
                ),
            ])
            .add_results(&[result_type])
            .build()?;
        Ok(MemVal {
            mem: self.push(block, op)?,
            elem: src.elem,
            shape: static_sizes,
            row_stride: None,
            aligned,
            // subviews are never taken of swizzled staging buffers (ldmatrix reads those directly),
            // so the swizzle does not propagate here.
            swizzle: None,
            global: None,
            owned: false,
            mask,
        })
    }

    /// The subview result type MLIR will infer: the slice's shape over the
    /// source's row-major strides, with a dynamic offset.
    pub(super) fn subview_type(&self, src: &MemVal<'c>, sizes: &[i64]) -> Result<Type<'c>> {
        let strides = row_major_strides(&src.shape);
        let dims: String = sizes.iter().map(|&d| format!("{}x", fmt_dim(d))).collect();
        let strides: Vec<String> = strides.iter().map(|&s| fmt_dim(s)).collect();
        let text = format!(
            "memref<{dims}{}, strided<[{}], offset: ?>, {MEM_GLOBAL}>",
            src.elem,
            strides.join(", ")
        );
        Type::parse(self.ctx, &text).ok_or_else(|| anyhow!("failed to parse type '{text}'"))
    }
}

// expression classifiers
impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn as_scale_mul<'a>(&self, expr: &'a Expr) -> Option<(&'a Expr, &'a Expr)> {
        let Expr::Binary {
            op: BinOp::Mul,
            lhs,
            rhs,
        } = expr
        else {
            return None;
        };
        if self.is_scalar_expr(lhs) && self.is_tile_expr(rhs) {
            Some((lhs, rhs))
        } else if self.is_tile_expr(lhs) && self.is_scalar_expr(rhs) {
            Some((rhs, lhs))
        } else {
            None
        }
    }

    pub(super) fn is_scalar_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Float(_) | Expr::Int(_) => true,
            Expr::Var(name) => matches!(
                self.lookup(name),
                Some(Binding::Let { .. }) | Some(Binding::Var { .. })
            ),
            _ => false,
        }
    }

    pub(super) fn is_tile_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Var(name) => matches!(
                self.lookup(name),
                Some(Binding::Tile(_)) | Some(Binding::View(_))
            ),
            _ => false,
        }
    }
}
