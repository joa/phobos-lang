use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    /// Whether a slice of `size` elements starting at `start` provably stays
    /// inside a *dynamic* tensor extent whose known divisor is `extent_div`, so
    /// it needs no runtime bounds mask.
    ///
    /// Three offsets qualify. One riding a trimmed main loop's induction
    /// variable is covered by that loop's rounded-down trip count (see
    /// [`Codegen::emit_split_for`]). A single element at a literal zero is
    /// inside any tensor that has elements at all, and a kernel launched over
    /// an empty one has nothing to do; this is what keeps the leading
    /// `[0 :+ 1]` of a row-vector kernel off the masked path. Last, a
    /// tile-aligned offset into an extent the host declared a whole number of
    /// tiles: every program id then addresses a whole tile, since the grid runs
    /// to the extent (see `@aligned` in [`Codegen::declared_divs`]).
    ///
    /// Anything else, an undeclared program id above all, is unbounded from
    /// inside the kernel: the grid is the host's business, and a kernel that
    /// assumed otherwise would write through the end of a row and into the next
    /// one.
    /// `pending` names loop variables whose loop has not been emitted yet but
    /// will be trimmed once it is, which is what lets a prescan reason about a
    /// slice inside a loop body (see [`Codegen::slice_is_partial_within`]).
    pub(super) fn dyn_in_bounds(
        &self,
        start: &Expr,
        size: i64,
        extent_div: i64,
        pending: &[&str],
    ) -> bool {
        if self
            .trimmed_ivs
            .iter()
            .map(String::as_str)
            .chain(pending.iter().copied())
            .any(|iv| start.uses_name(iv))
        {
            return true;
        }
        if size <= 1 && self.const_fold(start) == Some(0) {
            return true;
        }
        size > 0 && extent_div % size == 0 && self.expr_div(start) % size == 0
    }

    pub(super) fn expr_div(&self, expr: &Expr) -> i64 {
        const CAP: i64 = 1 << 20;
        match expr {
            Expr::Int(n) => n.abs().min(CAP),
            Expr::Var(name) => match self.lookup(name) {
                Some(Binding::Let { div, .. }) => div,
                Some(_) => 1,
                None => self.shape_env.get(name).map_or(1, |v| v.abs().min(CAP)),
            },
            Expr::Unary { op: UnOp::Neg, rhs } => self.expr_div(rhs),
            Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                let (a, b) = (self.expr_div(lhs), self.expr_div(rhs));
                if a == 0 || b == 0 {
                    0
                } else {
                    a.saturating_mul(b).min(CAP)
                }
            }
            Expr::Binary {
                op: BinOp::Add | BinOp::Sub,
                lhs,
                rhs,
            } => gcd(self.expr_div(lhs), self.expr_div(rhs)),
            _ => 1,
        }
    }

    /// Folds an expression to a compile-time constant.
    ///
    /// Scans @autotune symbols. Used for affine bounds and static slice checks.
    pub(super) fn const_fold(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Int(n) => Some(*n),
            // shadowing
            Expr::Var(name) if self.lookup(name).is_none() => self.shape_env.get(name).copied(),
            Expr::Unary { op: UnOp::Neg, rhs } => self.const_fold(rhs)?.checked_neg(),
            Expr::Binary { op, lhs, rhs } => {
                let (a, b) = (self.const_fold(lhs)?, self.const_fold(rhs)?);
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => a.checked_div(b),
                    BinOp::Rem => a.checked_rem(b),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn push(&self, block: &Block<'c>, op: Operation<'c>) -> Result<Value<'c, 'c>> {
        let result = block.append_operation(op).result(0)?;
        Ok(detach(result.into()))
    }

    pub(super) fn barrier(&self, block: &Block<'c>) -> Result<()> {
        block.append_operation(OperationBuilder::new("gpu.barrier", self.loc).build()?);
        Ok(())
    }

    /// Open a cp.async group.
    ///
    /// The token must be used with [`Self::async_wait`].
    pub(super) fn async_create_group(&self, block: &Block<'c>) -> Result<Value<'c, 'c>> {
        let token_t = self.parse_type("!nvgpu.device.async.token")?;
        self.push(
            block,
            OperationBuilder::new("nvgpu.device_async_create_group", self.loc)
                .add_results(&[token_t])
                .build()?,
        )
    }

    pub(super) fn async_wait(&self, block: &Block<'c>, group: Value<'c, 'c>) -> Result<()> {
        block.append_operation(
            OperationBuilder::new("nvgpu.device_async_wait", self.loc)
                .add_operands(&[group])
                .build()?,
        );
        Ok(())
    }

    pub(super) fn assume_align(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        align: i32,
    ) -> Result<Value<'c, 'c>> {
        let t = mem.r#type();
        self.push(
            block,
            OperationBuilder::new("memref.assume_alignment", self.loc)
                .add_operands(&[mem])
                .add_attributes(&[(
                    self.id("alignment"),
                    IntegerAttribute::new(IntegerType::new(self.ctx, 32).into(), align as i64)
                        .into(),
                )])
                .add_results(&[t])
                .build()?,
        )
    }

    pub(super) fn addi(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::addi(lhs, rhs, self.loc))
    }

    pub(super) fn subi(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::subi(lhs, rhs, self.loc))
    }

    pub(super) fn muli(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::muli(lhs, rhs, self.loc))
    }

    pub(super) fn divui(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::divui(lhs, rhs, self.loc))
    }

    pub(super) fn remui(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::remui(lhs, rhs, self.loc))
    }

    pub(super) fn minsi(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::minsi(lhs, rhs, self.loc))
    }

    pub(super) fn const_index(&self, block: &Block<'c>, value: i64) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            arith::constant(
                self.ctx,
                IntegerAttribute::new(self.index_t, value).into(),
                self.loc,
            ),
        )
    }

    pub(super) fn const_bool(&self, block: &Block<'c>, value: bool) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            arith::constant(
                self.ctx,
                IntegerAttribute::new(self.bool_t, value as i64).into(),
                self.loc,
            ),
        )
    }

    pub(super) fn zero_scalar(&self, block: &Block<'c>, elem: Type<'c>) -> Result<Value<'c, 'c>> {
        let attr: Attribute = if self.is_float(elem) {
            FloatAttribute::new(self.ctx, elem, 0.0).into()
        } else if self.is_int(elem) {
            IntegerAttribute::new(elem, 0).into()
        } else {
            bail!("cannot zero-initialize element type {elem}")
        };
        self.push(block, arith::constant(self.ctx, attr, self.loc))
    }

    pub(super) fn i64_array(&self, values: &[i64]) -> Result<Attribute<'c>> {
        self.parse_attr(&format!("array<i64: {}>", int_list(values)))
    }

    pub(super) fn i32_array(&self, values: &[i32]) -> Result<Attribute<'c>> {
        self.parse_attr(&format!("array<i32: {}>", int_list(values)))
    }

    pub(super) fn parse_attr(&self, text: &str) -> Result<Attribute<'c>> {
        Attribute::parse(self.ctx, text)
            .ok_or_else(|| anyhow!("failed to parse attribute '{text}'"))
    }

    pub(super) fn parse_type(&self, text: &str) -> Result<Type<'c>> {
        Type::parse(self.ctx, text).ok_or_else(|| anyhow!("failed to parse type '{text}'"))
    }

    pub(super) fn expect_index(&self, v: Value<'c, 'c>, what: &str) -> Result<Value<'c, 'c>> {
        if v.r#type() == self.index_t {
            Ok(v)
        } else {
            bail!("{what} must be an integer, got {}", v.r#type())
        }
    }

    pub(super) fn expect_bool(&self, v: Value<'c, 'c>, what: &str) -> Result<Value<'c, 'c>> {
        if v.r#type() == self.bool_t {
            Ok(v)
        } else {
            bail!("{what} must be a bool, got {}", v.r#type())
        }
    }

    pub(super) fn is_float(&self, t: Type<'c>) -> bool {
        t == self.f16_t || t == self.bf16_t || t == self.f32_t || t == self.f64_t
    }

    /// The tensor-core operand element types: f16, or f32 rounded down on stage.
    pub(super) fn is_f16_or_f32(&self, t: Type<'c>) -> bool {
        t == self.f16_t || t == self.f32_t
    }

    /// Width in bits of a float type, which orders the widening conversions.
    ///
    /// f16 and bf16 are both 16 bits and neither contains the other: bf16 has
    /// f32's exponent range with 8 fewer mantissa bits. Width alone therefore
    /// does not decide a conversion between them; see [`Self::float_join`].
    pub(super) fn float_bits(&self, t: Type<'c>) -> Option<u32> {
        if t == self.f16_t || t == self.bf16_t {
            Some(16)
        } else if t == self.f32_t {
            Some(32)
        } else if t == self.f64_t {
            Some(64)
        } else {
            None
        }
    }

    /// The narrowest float type both operands convert into without loss.
    ///
    /// Equal types join to themselves and a wider type absorbs a narrower one,
    /// but f16 and bf16 join to f32: each has bits the other cannot hold, so
    /// f32 is their only common supertype.
    pub(super) fn float_join(&self, a: Type<'c>, b: Type<'c>) -> Option<Type<'c>> {
        if a == b {
            return self.is_float(a).then_some(a);
        }
        let (ab, bb) = (self.float_bits(a)?, self.float_bits(b)?);
        Some(match ab.cmp(&bb) {
            std::cmp::Ordering::Greater => a,
            std::cmp::Ordering::Less => b,
            // same width, different types: f16 vs bf16.
            std::cmp::Ordering::Equal => self.f32_t,
        })
    }

    /// The element type a mixed-type pair computes in: [`Self::float_join`]
    /// between floats, the wider of two integers, and the float side when an
    /// integer meets a float.
    pub(super) fn numeric_join(&self, a: Type<'c>, b: Type<'c>) -> Option<Type<'c>> {
        match (self.is_float(a), self.is_float(b)) {
            (true, true) => self.float_join(a, b),
            (true, false) if self.is_int(b) => Some(a),
            (false, true) if self.is_int(a) => Some(b),
            (false, false) if self.is_int(a) && self.is_int(b) => {
                Some(if self.int_bits(a).ok()? >= self.int_bits(b).ok()? {
                    a
                } else {
                    b
                })
            }
            _ => None,
        }
    }

    /// The element type a contraction of `a` and `b` accumulates in.
    ///
    /// Floats accumulate in their join. Integers accumulate in i32 rather than
    /// the operand type: a dot product of bytes overflows i8 almost at once,
    /// and i32 is what the hardware's integer dot product accumulates in
    /// anyway.
    pub(super) fn accumulator_elem(&self, a: Type<'c>, b: Type<'c>) -> Result<Type<'c>> {
        let join = self
            .numeric_join(a, b)
            .ok_or_else(|| anyhow!("no common type for a contraction of {a} and {b}"))?;
        Ok(if self.is_int(join) && join != self.i64_t {
            self.i32_t
        } else {
            join
        })
    }

    pub(super) fn float_cast(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let from = value.r#type();
        if from == want {
            return Ok(value);
        }
        let (lo, hi) = (
            self.float_bits(from)
                .ok_or_else(|| anyhow!("float_cast from non-float {from}"))?,
            self.float_bits(want)
                .ok_or_else(|| anyhow!("float_cast to non-float {want}"))?,
        );
        // f16 <-> bf16 is neither a widening nor a narrowing, and arith has no
        // op for it. Round-trip through f32, which holds either exactly.
        if lo == hi {
            let wide = self.float_cast(block, value, self.f32_t)?;
            return self.float_cast(block, wide, want);
        }
        let op = if hi > lo {
            "arith.extf"
        } else {
            "arith.truncf"
        };
        self.push(
            block,
            OperationBuilder::new(op, self.loc)
                .add_operands(&[value])
                .add_results(&[want])
                .build()?,
        )
    }

    pub(super) fn is_int(&self, t: Type<'c>) -> bool {
        t == self.i8_t || t == self.i32_t || t == self.i64_t
    }

    /// Convert between any two numeric element types, float or signed integer.
    ///
    /// Integers are signed throughout, so the widening and float conversions
    /// are the sign-extending ones.
    pub(super) fn numeric_cast(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let from = value.r#type();
        if from == want {
            return Ok(value);
        }
        let (from_float, want_float) = (self.is_float(from), self.is_float(want));
        if from_float && want_float {
            return self.float_cast(block, value, want);
        }
        // index is its own world; route it through i32/i64 on the way in.
        if from == self.index_t {
            let as_int = self.push(block, arith::index_cast(value, self.i64_t, self.loc))?;
            return self.numeric_cast(block, as_int, want);
        }
        if want == self.index_t {
            let as_int = self.numeric_cast(block, value, self.i64_t)?;
            return self.push(block, arith::index_cast(as_int, self.index_t, self.loc));
        }
        let op = match (from_float, want_float) {
            (true, false) => "arith.fptosi",
            (false, true) => "arith.sitofp",
            (false, false) => {
                let (lo, hi) = (self.int_bits(from)?, self.int_bits(want)?);
                if hi > lo {
                    "arith.extsi"
                } else {
                    "arith.trunci"
                }
            }
            (true, true) => unreachable!("handled above"),
        };
        self.push(
            block,
            OperationBuilder::new(op, self.loc)
                .add_operands(&[value])
                .add_results(&[want])
                .build()?,
        )
    }

    /// Size in bytes of a numeric element type.
    pub(super) fn elem_bytes(&self, t: Type<'c>) -> Option<u32> {
        let bits = match self.float_bits(t) {
            Some(bits) => bits,
            None => self.int_bits(t).ok()?,
        };
        (bits >= 8).then_some(bits / 8)
    }

    fn int_bits(&self, t: Type<'c>) -> Result<u32> {
        if t == self.bool_t {
            Ok(1)
        } else if t == self.i8_t {
            Ok(8)
        } else if t == self.i32_t {
            Ok(32)
        } else if t == self.i64_t {
            Ok(64)
        } else {
            bail!("not an integer type: {t}")
        }
    }

    pub(super) fn id(&self, name: &str) -> Identifier<'c> {
        Identifier::new(self.ctx, name)
    }

    pub(super) fn lookup(&self, name: &str) -> Option<Binding<'c>> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    pub(super) fn bind(&mut self, name: &str, mut binding: Binding<'c>) {
        // A named buffer is never released to the pool: it may be read for
        // the rest of its scope (including the next iteration of an
        // enclosing loop), which the per-statement release sites cannot see.
        if let Binding::View(mv) | Binding::Tile(mv) = &mut binding {
            mv.owned = false;
        }
        self.scopes
            .last_mut()
            .expect("a scope is always open while emitting")
            .insert(name.to_string(), binding);
    }

    /// Replaces the binding of an already-bound name in the innermost scope
    /// containing it (bind would shadow it in the current scope instead).
    pub(super) fn update_binding(&mut self, name: &str, binding: Binding<'c>) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                *slot = binding;
                return;
            }
        }
        panic!("update_binding of unbound name '{name}'");
    }
}
