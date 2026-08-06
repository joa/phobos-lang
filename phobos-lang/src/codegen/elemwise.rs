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
/// the result. See `docs/megakernel.md` for why the footprint matters.
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
