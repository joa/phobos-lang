use super::*;

/// Per-element unary chains, fused into one sweep of the target.
///
/// Every elementwise call materializes a shared tile of its own and sweeps it,
/// so a nested one pays per call. `i8(i32(round(q)))`, which is how every
/// quantizing kernel writes its Q8_0 bytes, costs three tiles besides the
/// operand: an f32 for the rounding, an i32 for the widening conversion, and the
/// i8 itself. On the decode path that is 4 KB of shared memory spent on an i32
/// nobody reads, in kernels whose whole live set is what bounds their occupancy.
///
/// This recognizes such a chain on the right of a store and emits a single
/// `distribute`: load the operand once, apply the conversions in registers, store
/// the result.
impl<'p, 'c> Codegen<'p, 'c> {
    /// One step of a chain, as peeled from the outside in.
    fn elem_step(&self, callee: &str) -> Option<ElemStep<'c>> {
        Some(match callee {
            "round" => ElemStep::Round,
            "sqrt" => ElemStep::Sqrt,
            "exp" => ElemStep::Exp,
            "log" => ElemStep::Log,
            "tanh" => ElemStep::Tanh,
            other => {
                let scalar = Scalar::from_name(other)?;
                if scalar == Scalar::Bool {
                    return None;
                }
                ElemStep::Cast(self.scalar_type(scalar))
            }
        })
    }

    /// Peels the per-element unary calls off the outside of `value`, outermost
    /// first, and returns them with whatever they all apply to.
    fn peel_elem_chain<'e>(&self, value: &'e Expr) -> (Vec<ElemStep<'c>>, &'e Expr) {
        let mut steps = Vec::new();
        let mut inner = value;
        while let Expr::Call { callee, args } = inner {
            let [arg] = &args[..] else { break };
            let Some(step) = self.elem_step(callee) else {
                break;
            };
            steps.push(step);
            inner = arg;
        }
        (steps, inner)
    }

    /// `target = f(g(...(t)))` in one sweep, for per-element unary f, g, ...
    ///
    /// Returns false without emitting anything when `value` is not such a chain
    /// over a named, unmasked, statically shaped tile. The gate is deliberately
    /// narrow: the operand is resolved from the symbol table rather than by
    /// emitting it, so a shape this does not handle leaves every other path in
    /// `store_tile` exactly as it was. A masked operand is excluded because the
    /// sweep is indexed by the target and would read the operand out of bounds.
    pub(super) fn store_elem_chain(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        value: &Expr,
    ) -> Result<bool> {
        let (steps, base) = self.peel_elem_chain(value);
        if steps.is_empty() {
            return Ok(false);
        }
        let Expr::Var(name) = base else {
            return Ok(false);
        };
        let src = match self.lookup(name) {
            Some(Binding::Tile(t) | Binding::View(t)) => t,
            _ => return Ok(false),
        };
        if src.is_masked() || src.shape.contains(&DYN) || src.shape != target.shape {
            return Ok(false);
        }
        // The operand may be the target: each thread reads and writes the same
        // element, so the rewrite in place is race-free, exactly as the single
        // op `*_into` helpers rely on.
        self.distribute(block, target, 1, true, |cg, blk, idx| {
            let mut v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            for step in steps.iter().rev() {
                v = cg.apply_elem_step(blk, v, *step)?;
            }
            let v = cg.numeric_cast(blk, v, target.elem)?;
            blk.append_operation(memref::store(v, target.mem, idx, cg.loc));
            Ok(())
        })?;
        self.release(&src);
        Ok(true)
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
pub(super) enum ElemStep<'c> {
    Cast(Type<'c>),
    Round,
    Sqrt,
    Exp,
    Log,
    Tanh,
}

/// A per-element expression tree, evaluated in registers by a single sweep of
/// the target rather than a shared tile and a barrier per node.
enum Fused<'c> {
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

/// Whether a binary operator is arithmetic, so its result is an element rather
/// than a predicate.
fn is_arith(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
    )
}

/// Fusing a whole per-element expression into one sweep of its target.
///
/// Every tile-valued node used to materialize a shared buffer and sweep it, so
/// a statement cost a barrier per operator: `acc = acc * corr + dot(sc, v)`
/// spent one on the product, one on the sum, and two tiles nobody reads again.
/// A chain of them is what a softmax rescale and a GEMM epilogue both are, and
/// on the decode path, where the tiles are a row or two and the CTA is 256
/// threads wide, the barrier is most of what the statement costs.
///
/// This recognizes the whole tree at once: the operands that have to be
/// materialized anyway (a `dot`, a row reduction) are emitted first and become
/// leaves, and everything above them is evaluated per element in registers.
impl<'p, 'c> Codegen<'p, 'c> {
    /// Interior nodes of a fusable tree, or None when `value` is not one.
    ///
    /// Leaves are named tiles, scalars, and the calls whose result has to be
    /// materialized anyway; interior nodes are arithmetic, `tmax`, and the
    /// per-element math and conversion calls. A subscript is deliberately not
    /// a leaf: a slice can carry a bounds mask, and the sweep is indexed by
    /// the target, so reading one would run past the source extent.
    pub(super) fn fusable_nodes(&self, value: &Expr) -> Option<usize> {
        match value {
            Expr::Int(_) | Expr::Float(_) => Some(0),
            Expr::Var(name) => match self.lookup(name)? {
                Binding::Let { .. } | Binding::Var { .. } => Some(0),
                Binding::Tile(t) | Binding::View(t) => {
                    (!t.is_masked() && !t.shape.contains(&DYN)).then_some(0)
                }
                _ => None,
            },
            Expr::Binary { op, lhs, rhs } if is_arith(*op) => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            Expr::Call { callee, args } => match &args[..] {
                [arg] if self.elem_step(callee).is_some() => Some(1 + self.fusable_nodes(arg)?),
                [a, b] if callee == "tmax" => {
                    Some(1 + self.fusable_nodes(a)? + self.fusable_nodes(b)?)
                }
                // Everything else is opaque: a leaf this materializes.
                _ => Some(0),
            },
            _ => None,
        }
    }

    /// Builds the tree, emitting the operands that cannot be fused and
    /// recording every tile leaf for release afterwards.
    ///
    /// A scalar leaf is coerced to `elem` here rather than joined with its
    /// sibling at each use: that is what the single-operand paths have always
    /// done, and it is what keeps an integer literal off the index type.
    fn plan_fused(
        &mut self,
        block: &Block<'c>,
        value: &Expr,
        elem: Type<'c>,
        leaves: &mut Vec<MemVal<'c>>,
    ) -> Result<Fused<'c>> {
        Ok(match value {
            Expr::Binary { op, lhs, rhs } if is_arith(*op) => {
                let a = self.plan_fused(block, lhs, elem, leaves)?;
                let b = self.plan_fused(block, rhs, elem, leaves)?;
                Fused::Binary(*op, Box::new(a), Box::new(b))
            }
            Expr::Call { callee, args } if args.len() == 1 && self.elem_step(callee).is_some() => {
                let step = self.elem_step(callee).expect("checked above");
                let inner = self.plan_fused(block, &args[0], elem, leaves)?;
                Fused::Unary(step, Box::new(inner))
            }
            Expr::Call { callee, args } if args.len() == 2 && callee == "tmax" => {
                let a = self.plan_fused(block, &args[0], elem, leaves)?;
                let b = self.plan_fused(block, &args[1], elem, leaves)?;
                Fused::Max(Box::new(a), Box::new(b))
            }
            other => match self.emit_expr(block, other)? {
                Rv::Scalar(v) => Fused::Scalar(self.coerce(block, v, elem)?),
                Rv::Tile(t) => {
                    leaves.push(t.clone());
                    Fused::Tile(t)
                }
            },
        })
    }

    /// `target op= <per-element expression>` in one sweep, or false when the
    /// value is not such a tree.
    ///
    /// The gate wants at least one operator, since a bare tile is a copy, and
    /// at least one tile operand, since a tree of scalars is a fill.
    pub(super) fn store_fused(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        op: AssignOp,
        value: &Expr,
    ) -> Result<bool> {
        if target.shape.contains(&DYN) || self.fusable_nodes(value).unwrap_or(0) < 1 {
            return Ok(false);
        }

        let mut leaves = Vec::new();
        let mut tree = self.plan_fused(block, value, target.elem, &mut leaves)?;
        if leaves.is_empty() {
            return Ok(false);
        }
        // `+=` is the same sweep with the target as one more addend: a thread
        // reads and writes the element it owns, so it stays race-free.
        if op == AssignOp::Add {
            tree = Fused::Binary(
                BinOp::Add,
                Box::new(Fused::Tile(target.clone())),
                Box::new(tree),
            );
        }

        for leaf in &leaves {
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
            let v = cg.eval_fused(blk, &tree, idx, &target.shape, width, vec_t)?;
            let v = if width > 1 {
                v
            } else {
                cg.coerce(blk, v, target.elem)?
            };
            cg.elem_store(blk, v, target.mem, idx, width)
        })?;

        for leaf in &leaves {
            self.release(leaf);
        }
        Ok(true)
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
