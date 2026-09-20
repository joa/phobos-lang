use super::*;

impl<'c> Codegen<'c> {

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

        // mixed float types widen to their join (f16 and bf16 meet at f32)
        if let Some(want) = self.float_join(lt, rt) {
            return Ok((
                self.float_cast(block, lhs, want)?,
                self.float_cast(block, rhs, want)?,
            ));
        }

        bail!("mismatched operand types: {lt} vs {rt}")
    }

    /// Coerces value to want for a store.
    ///
    /// Only the conversions a store needs: float to float, and index to either an
    /// integer or a float. Anything else is a type error.
    pub(super) fn coerce(
        &mut self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let t = value.r#type();
        if t == want {
            Ok(value)
        } else if self.is_float(t) && self.is_float(want) {
            // rounds or widens, e.g. an f32 literal into an f16 tile
            self.float_cast(block, value, want)
        } else if t == self.index_t && self.is_int(want) {
            self.push(block, arith::index_cast(value, want, self.loc))
        } else if t == self.index_t && self.is_float(want) {
            self.numeric_cast(block, value, want)
        } else {
            bail!("type mismatch: cannot store {t} where {want} is expected")
        }
    }
}

// memrefs
impl<'c> Codegen<'c> {

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

    /// The `memref.subview` itself, over resolved offsets, sizes and mask.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn subview_raw(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        offsets: &[Value<'c, 'c>],
        dyn_sizes: &[Value<'c, 'c>],
        static_sizes: Vec<i64>,
        mask: Vec<Option<(Value<'c, 'c>, Value<'c, 'c>)>>,
        align_div: i64,
    ) -> Result<MemVal<'c>> {
        let rank = static_sizes.len();
        let result_type = self.subview_type(src, &static_sizes)?;
        let mut operands = vec![src.mem];
        operands.extend_from_slice(offsets);
        operands.extend_from_slice(dyn_sizes);

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
            align_div,
            // never taken of swizzled staging buffers (ldmatrix reads those directly)
            swizzle: None,
            global: None,
            shared: src.shared,
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
            "memref<{dims}{}, strided<[{}], offset: ?>, {}>",
            src.elem,
            strides.join(", "),
            self.mem_space(src.shared)
        );
        Type::parse(self.ctx, &text).ok_or_else(|| anyhow!("failed to parse type '{text}'"))
    }
}

impl<'c> Codegen<'c> {

}
