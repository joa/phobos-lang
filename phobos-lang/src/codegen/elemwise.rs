use super::*;

/// Per-element unary chains, fused into one sweep of the target.
///
/// Without this, every elementwise call would materialize its own shared
/// tile, so a chain like `i8(i32(round(q)))` would cost a tile per step. A
/// chain on the right of a store becomes a single `distribute` instead: load
/// the operand once, convert in registers, store the result.
impl<'c> Codegen<'c> {

    /// Sweeps a chain over `src` into `target`, with the outermost step last
    /// in `steps`, converting the result to the target's element type, so an
    /// empty chain is a converting copy. `src` may be `target`, since each
    /// thread reads and writes the same element.
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

    /// Applies one step to one element. The transcendentals need a float,
    /// which they widen to f32. An earlier cast may have left a non-float.
    pub(in crate::codegen) fn apply_elem_step(
        &mut self,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        step: ElemStep<'c>,
    ) -> Result<Value<'c, 'c>> {
        let float_arg = |cg: &mut Self, what: &str| -> Result<Value<'c, 'c>> {
            if !cg.is_float(v.r#type()) {
                bail!("`{what}` needs a float operand, got {}", v.r#type());
            }
            cg.float_cast(block, v, cg.f32_t)
        };
        match step {
            ElemStep::Cast(want) => self.numeric_cast(block, v, want),
            ElemStep::CastUnsigned(want) => self.unsigned_cast(block, v, want),
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
            // max(x, -x) on an integer, which has no instruction of its own here.
            ElemStep::Math(crate::ir::MathFn::Abs) if !self.is_float(v.r#type()) => {
                let zero = self.push(
                    block,
                    arith::constant(self.ctx, IntegerAttribute::new(v.r#type(), 0).into(), self.loc),
                )?;
                let neg = self.push(block, arith::subi(zero, v, self.loc))?;
                self.push(block, self.minmax(BinOp::Max, "si", v, neg)?)
            }
            ElemStep::Math(m) => {
                let f = float_arg(self, m.name())?;
                self.math_f32(block, m, f)
            }
        }
    }
}

/// A per-element unary operation, one step of a chain. See
/// [`Codegen::store_elem_chain`].
#[derive(Clone, Copy)]
pub(in crate::codegen) enum ElemStep<'c> {
    Cast(Type<'c>),
    /// A cast reading an unsigned element, which zero-extends.
    CastUnsigned(Type<'c>),
    Round,
    Sqrt,
    Exp,
    Log,
    Tanh,
    Math(crate::ir::MathFn),
}

/// A per-element expression tree, evaluated in registers by a single sweep of
/// the target.
#[derive(Clone)]
pub(super) enum Fused<'c> {
    /// A materialized operand, read with broadcasting.
    Tile(MemVal<'c>),
    /// One value for every element, such as a literal or a scalar binding.
    Scalar(Value<'c, 'c>),
    Unary(ElemStep<'c>, Box<Fused<'c>>),
    Binary(BinOp, Box<Fused<'c>>, Box<Fused<'c>>),
    /// `tmax`, emitted as a compare and a select.
    Max(Box<Fused<'c>>, Box<Fused<'c>>),
    /// `where(cond, a, b)`.
    Select(Box<Fused<'c>>, Box<Fused<'c>>, Box<Fused<'c>>),
    /// The element's position along an axis of the target, as an i32.
    Iota(usize),
    /// A fused dot's product, which its write-out fills in per element.
    Hole,
}

impl<'c> Fused<'c> {
    /// The tree with every hole replaced by `with`.
    pub(super) fn fill_hole(&self, with: &Fused<'c>) -> Fused<'c> {
        let go = |x: &Fused<'c>| Box::new(x.fill_hole(with));
        match self {
            Fused::Hole => with.clone(),
            Fused::Tile(_) | Fused::Scalar(_) | Fused::Iota(_) => self.clone(),
            Fused::Unary(step, x) => Fused::Unary(*step, go(x)),
            Fused::Binary(op, a, b) => Fused::Binary(*op, go(a), go(b)),
            Fused::Max(a, b) => Fused::Max(go(a), go(b)),
            Fused::Select(c, a, b) => Fused::Select(go(c), go(a), go(b)),
        }
    }

    /// Whether the tree has a unary step or a select. Those work on one
    /// element at a time, so such a tree cannot be swept four elements per
    /// thread.
    fn scalar_only(&self) -> bool {
        match self {
            Fused::Tile(_) | Fused::Scalar(_) | Fused::Hole => false,
            Fused::Unary(..) | Fused::Select(..) | Fused::Iota(_) => true,
            Fused::Binary(_, a, b) | Fused::Max(a, b) => a.scalar_only() || b.scalar_only(),
        }
    }
}

/// Fusing a whole per-element expression into one sweep of its target, which
/// avoids a shared buffer and a barrier per operator.
///
/// Operands that must be materialized anyway, such as a `dot` or a row
/// reduction, are emitted first and become leaves. Everything above them is
/// evaluated per element in registers.
impl<'c> Codegen<'c> {

    /// One sweep of `target` evaluating `tree`, whose tile leaves are
    /// `leaves`. Checks that the leaves broadcast to the target and are
    /// unmasked.
    pub(super) fn fused_sweep(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        tree: &Fused<'c>,
        leaves: &[MemVal<'c>],
    ) -> Result<()> {
        Self::check_sweep_leaves(target, leaves)?;

        // Four elements per thread only when nothing broadcasts and nothing
        // needs the scalar transcendentals. A broadcast operand's innermost
        // access is not contiguous.
        let dense = leaves.iter().all(|l| l.shape == target.shape) && !tree.scalar_only();
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

    /// Every tile leaf of a tree evaluated against `target` broadcasts to it
    /// and is unmasked.
    pub(super) fn check_sweep_leaves(target: &MemVal<'c>, leaves: &[MemVal<'c>]) -> Result<()> {
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
            // `fusable_nodes` screens named operands. This catches an emitted
            // one, since the sweep indexes by the target and ignores masks.
            ensure!(
                !leaf.is_masked(),
                "fused elementwise op: a partially out-of-bounds operand must be \
                 materialized before it can be read against the target's extent"
            );
        }
        Ok(())
    }

    /// One element of the tree, in registers.
    pub(in crate::codegen) fn eval_fused(
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
            // `plan_fused` already cast it to the target's element type. The
            // splat is loop-invariant and gets hoisted.
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
                match op {
                    // Both sides are already i1.
                    BinOp::And => self.push(block, arith::andi(x, y, self.loc)),
                    BinOp::Or => self.push(block, arith::ori(x, y, self.loc)),
                    _ => {
                        let (x, y, elem) = self.fuse_pair(block, x, y, width)?;
                        self.push(block, self.elem_arith(*op, elem, x, y)?)
                    }
                }
            }
            Fused::Iota(d) => self.push(block, arith::index_cast(idx[*d], self.i32_t, self.loc)),
            Fused::Hole => bail!("a fused dot's product read outside its write-out"),
            Fused::Select(c, a, b) => {
                let cond = self.eval_fused(block, c, idx, out_shape, width, vec_t)?;
                let x = self.eval_fused(block, a, idx, out_shape, width, vec_t)?;
                let y = self.eval_fused(block, b, idx, out_shape, width, vec_t)?;
                let (x, y, _) = self.fuse_pair(block, x, y, width)?;
                self.select(block, cond, x, y)
            }
            Fused::Max(a, b) => {
                let x = self.eval_fused(block, a, idx, out_shape, width, vec_t)?;
                let y = self.eval_fused(block, b, idx, out_shape, width, vec_t)?;
                let (x, y, _) = self.fuse_pair(block, x, y, width)?;
                self.fmax(block, x, y)
            }
        }
    }

    /// Brings a node's two operands to a common type. Also returns the
    /// element type `elem_arith` needs, which is f32 for a vectorized sweep.
    fn fuse_pair(
        &mut self,
        block: &Block<'c>,
        x: Value<'c, 'c>,
        y: Value<'c, 'c>,
        width: i64,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Type<'c>)> {
        let (xt, yt) = (x.r#type(), y.r#type());
        if width > 1 {
            // A vectorized sweep only runs when every operand is f32.
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
