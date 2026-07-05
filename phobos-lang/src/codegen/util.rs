use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
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
        t == self.f16_t || t == self.f32_t || t == self.f64_t
    }

    /// The tensor-core operand element types: f16, or f32 rounded down on stage.
    pub(super) fn is_f16_or_f32(&self, t: Type<'c>) -> bool {
        t == self.f16_t || t == self.f32_t
    }

    pub(super) fn float_rank(&self, t: Type<'c>) -> Option<u8> {
        if t == self.f16_t {
            Some(0)
        } else if t == self.f32_t {
            Some(1)
        } else if t == self.f64_t {
            Some(2)
        } else {
            None
        }
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
            self.float_rank(from)
                .ok_or_else(|| anyhow!("float_cast from non-float {from}"))?,
            self.float_rank(want)
                .ok_or_else(|| anyhow!("float_cast to non-float {want}"))?,
        );
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
        t == self.i32_t || t == self.i64_t
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
