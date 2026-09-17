use super::*;

/// Per-element unary chains, fused into one sweep of the target. Every
/// elementwise call otherwise materializes a shared tile of its own, so a
/// nested chain like `i8(i32(round(q)))` -- how quantizing kernels write
/// their Q8_0 bytes -- costs a tile per step, in kernels whose live set
/// bounds their occupancy.
///
/// This recognizes such a chain on the right of a store and emits a single
/// `distribute`: load the operand once, apply the conversions in registers,
/// store the result.
impl<'c> Codegen<'c> {

    /// The sweep of a chain over `src` into `target`, outermost step last.
    /// The operand may be the target: each thread reads and writes the same
    /// element, so the rewrite in place is race-free.
    pub(super) fn elem_chain_sweep(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        target: &MemVal<'c>,
        steps: &[ElemStep<'c>],
    ) -> Result<()> {
        self.distribute(block, target, 1, true, |cg, blk, idx| {
            let mut v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            for step in steps.iter().rev() {
                v = cg.apply_elem_step(blk, v, *step)?;
            }
            let v = cg.numeric_cast(blk, v, target.elem)?;
            blk.append_operation(memref::store(v, target.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// One step on one element. The transcendentals want f32, and a chain can
    /// have converted away from it, so each checks what it actually has.
    fn apply_elem_step(
        &mut self,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        step: ElemStep<'c>,
    ) -> Result<Value<'c, 'c>> {
        let float_arg = |cg: &mut Self, what: &str| -> Result<Value<'c, 'c>> {
            if !cg.is_float(v.r#type()) {
                bail!("{what} needs a float operand, got {}", v.r#type());
            }
            cg.float_cast(block, v, cg.f32_t)
        };
        match step {
            ElemStep::Cast(want) => self.numeric_cast(block, v, want),
            ElemStep::Round => {
                let f = float_arg(self, "round")?;
                self.round_even(block, f)
            }
            ElemStep::Sqrt => {
                let f = float_arg(self, "sqrt")?;
                self.approx_sqrt(block, f)
            }
            ElemStep::Exp => {
                let f = float_arg(self, "exp")?;
                self.approx_exp(block, f)
            }
            ElemStep::Log => {
                let f = float_arg(self, "log")?;
                self.approx_log(block, f)
            }
            ElemStep::Tanh => {
                let f = float_arg(self, "tanh")?;
                self.approx_tanh(block, f)
            }
        }
    }
}

/// A per-element unary operation, in the order a chain applies them innermost
/// first. See [`Codegen::store_elem_chain`].
#[derive(Clone, Copy)]
pub(in crate::codegen) enum ElemStep<'c> {
    Cast(Type<'c>),
    Round,
    Sqrt,
    Exp,
    Log,
    Tanh,
}

/// A per-element expression tree, evaluated in registers by a single sweep of
/// the target rather than a shared tile and a barrier per node.
pub(super) enum Fused<'c> {
    /// A materialized operand, read with broadcasting.
    Tile(MemVal<'c>),
    /// One value for every element: a literal, a scalar binding, a call that
    /// returned a scalar.
    Scalar(Value<'c, 'c>),
    Unary(ElemStep<'c>, Box<Fused<'c>>),
    Binary(BinOp, Box<Fused<'c>>, Box<Fused<'c>>),
    /// `tmax`, which arith spells as a compare and a select rather than as an
    /// op of its own.
    Max(Box<Fused<'c>>, Box<Fused<'c>>),
}

impl Fused<'_> {
    /// Whether the tree carries a step only the scalar path can take. The
    /// transcendental approximations work on one f32 at a time, so a tree
    /// holding one cannot be swept four elements to the thread.
    fn has_unary(&self) -> bool {
        match self {
            Fused::Tile(_) | Fused::Scalar(_) => false,
            Fused::Unary(..) => true,
            Fused::Binary(_, a, b) | Fused::Max(a, b) => a.has_unary() || b.has_unary(),
        }
    }
}

/// Fusing a whole per-element expression into one sweep of its target.
/// Materializing a shared buffer per operator otherwise costs a barrier
/// each, which is most of a statement's cost on the decode path, where the
/// tiles are a row or two.
///
/// This recognizes the whole tree at once: the operands that have to be
/// materialized anyway (a `dot`, a row reduction) are emitted first and
/// become leaves, and everything above them is evaluated per element in
/// registers.
impl<'c> Codegen<'c> {

    /// One sweep of `target` evaluating `tree`, whose tile leaves are
    /// `leaves`. Checks the leaves broadcast to the target and are unmasked.
    pub(super) fn fused_sweep(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        tree: &Fused<'c>,
        leaves: &[MemVal<'c>],
    ) -> Result<()> {
        for leaf in leaves {
            let fits = leaf.shape.len() == target.shape.len()
                && leaf
                    .shape
                    .iter()
                    .zip(&target.shape)
                    .all(|(&s, &t)| s == t || s == 1);
            ensure!(
                fits,
                "fused elementwise op: operand shape {} does not broadcast to {}",
                fmt_shape(&leaf.shape),
                fmt_shape(&target.shape)
            );
            // Named operands are screened by `fusable_nodes`; this catches an
            // emitted one, whose mask the target-indexed sweep cannot honour.
            ensure!(
                !leaf.is_masked(),
                "fused elementwise op: a partially out-of-bounds operand must be \
                 materialized before it can be read against the target's extent"
            );
        }

        // Four elements to the thread only when nothing broadcasts (a
        // stretched operand has a non-contiguous innermost access) and nothing
        // needs the scalar transcendentals.
        let dense = leaves.iter().all(|l| l.shape == target.shape) && !tree.has_unary();
        let mut operands: Vec<&MemVal<'c>> = leaves.iter().collect();
        operands.push(target);
        let width = if dense {
            self.elementwise_width(&operands)
        } else {
            1
        };
        let vec_t = Type::vector(&[4], target.elem);

        self.distribute(block, target, width, true, |cg, blk, idx| {
            let v = cg.eval_fused(blk, tree, idx, &target.shape, width, vec_t)?;
            let v = if width > 1 {
                v
            } else {
                cg.coerce(blk, v, target.elem)?
            };
            cg.elem_store(blk, v, target.mem, idx, width)
        })
    }

    /// One element of the tree, in registers.
    fn eval_fused(
        &mut self,
        block: &Block<'c>,
        node: &Fused<'c>,
        idx: &[Value<'c, 'c>],
        out_shape: &[i64],
        width: i64,
        vec_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        match node {
            Fused::Tile(t) => {
                if width > 1 {
                    self.vec_load(block, t.mem, idx, vec_t)
                } else {
                    let at = self.bc_index(block, idx, out_shape, &t.shape)?;
                    self.push(block, memref::load(t.mem, &at, self.loc))
                }
            }
            // Already the target's element type, from `plan_fused`. The
            // splat is loop-invariant and sinks to the preheader.
            Fused::Scalar(v) => {
                if width > 1 {
                    self.vec_broadcast(block, *v, vec_t)
                } else {
                    Ok(*v)
                }
            }
            Fused::Unary(step, x) => {
                let v = self.eval_fused(block, x, idx, out_shape, width, vec_t)?;
                self.apply_elem_step(block, v, *step)
            }
            Fused::Binary(op, a, b) => {
                let x = self.eval_fused(block, a, idx, out_shape, width, vec_t)?;
                let y = self.eval_fused(block, b, idx, out_shape, width, vec_t)?;
                let (x, y, elem) = self.fuse_pair(block, x, y, width)?;
                self.push(block, self.elem_arith(*op, elem, x, y)?)
            }
            Fused::Max(a, b) => {
                let x = self.eval_fused(block, a, idx, out_shape, width, vec_t)?;
                let y = self.eval_fused(block, b, idx, out_shape, width, vec_t)?;
                let (x, y, _) = self.fuse_pair(block, x, y, width)?;
                self.fmax(block, x, y)
            }
        }
    }

    /// Two operands of a node brought to a common type, with the element type
    /// the arithmetic is chosen by. A vectorized sweep carries `vector<4xf32>`
    /// values whose element type is f32, and `elem_arith` wants the latter.
    fn fuse_pair(
        &mut self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
        y: Value<'c, 'c>,
        width: i64,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Type<'c>)> {
        let (xt, yt) = (x.r#type(), y.r#type());
        if width > 1 {
            // Vectorization is gated on every operand being f32 already.
            return Ok((x, y, self.f32_t));
        }
        if xt == yt {
            return Ok((x, y, xt));
        }
        let want = self
            .numeric_join(xt, yt)
            .ok_or_else(|| anyhow!("fused elementwise op: no common type for {xt} and {yt}"))?;
        Ok((
            self.numeric_cast(block, x, want)?,
            self.numeric_cast(block, y, want)?,
            want,
        ))
    }
}
