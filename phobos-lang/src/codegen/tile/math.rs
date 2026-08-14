// Unary math over a tile, and the approximate intrinsics behind it.

use super::*;

impl<'c> Codegen<'c> {
    /// max(a, b) on a float scalar, via cmpf ogt plus select.
    pub(in crate::codegen) fn fmax(
        &self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let cond = self.push(
            block,
            arith::cmpf(self.ctx, arith::CmpfPredicate::Ogt, a, b, self.loc),
        )?;
        self.push(
            block,
            OperationBuilder::new("arith.select", self.loc)
                .add_operands(&[cond, a, b])
                .add_results(&[a.r#type()])
                .build()?,
        )
    }

    /// out[...] = exp(src[...]) elementwise. The hardware ex2.approx is f32, so
    /// f16 tiles round-trip through f32 (load, widen, exp, narrow).
    pub(in crate::codegen) fn tile_exp(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        // An owned temp is rewritten in place, saving a buffer and letting
        // var p = exp(...) elide every copy. Swizzled staging never flows
        // here.
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("exp needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };
        self.tile_exp_into(block, src, &out)?;
        Ok(out)
    }

    /// out = sqrt(src), element-wise. Mirrors [`Self::tile_exp`]: an owned temp
    /// is rewritten in place, otherwise a fresh tile is allocated.
    pub(in crate::codegen) fn tile_sqrt(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("sqrt needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };

        self.tile_sqrt_into(block, src, &out)?;

        Ok(out)
    }

    /// out = convert(src) elementwise, to a tile of `want` element type.
    ///
    /// Always a fresh tile: unlike exp or sqrt, the result has a different
    /// element type (and so a different physical size) than the source, so
    /// there is nothing to rewrite in place.
    pub(in crate::codegen) fn tile_cast(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        want: Type<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.contains(&DYN) {
            bail!("a tile conversion needs a static tile shape");
        }

        let out = self.alloc_tile_shaped(block, want, &src.shape)?;

        self.distribute(block, &out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let c = cg.numeric_cast(blk, v, want)?;
            blk.append_operation(memref::store(c, out.mem, idx, cg.loc));
            Ok(())
        })?;

        Ok(out)
    }

    /// out[...] = sqrt(src[...]); out may be src itself (each thread reads and
    /// writes the same element, so the in-place rewrite is race-free).
    pub(in crate::codegen) fn tile_sqrt_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("sqrt needs a float tile, got {}", src.elem);
        }

        if src.shape.contains(&DYN) {
            bail!("sqrt needs a static tile shape");
        }

        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let vf = cg.float_cast(blk, v, cg.f32_t)?;
            let e = cg.approx_sqrt(blk, vf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = exp(src[...]); out may be src itself (each thread reads and
    /// writes the same element, so the in-place rewrite is race-free).
    pub(in crate::codegen) fn tile_exp_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("exp needs a float tile, got {}", src.elem);
        }
        if src.shape.contains(&DYN) {
            bail!("exp needs a static tile shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let vf = cg.float_cast(blk, v, cg.f32_t)?;
            let e = cg.approx_exp(blk, vf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out = log(src), element-wise. Mirrors [`Self::tile_exp`].
    pub(in crate::codegen) fn tile_log(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("log needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };

        self.tile_log_into(block, src, &out)?;

        Ok(out)
    }

    /// out[...] = log(src[...]); out may be src itself (race-free per element).
    pub(in crate::codegen) fn tile_log_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("log needs a float tile, got {}", src.elem);
        }

        if src.shape.contains(&DYN) {
            bail!("log needs a static tile shape");
        }

        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let vf = cg.float_cast(blk, v, cg.f32_t)?;
            let e = cg.approx_log(blk, vf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out = round(src), element-wise. Mirrors [`Self::tile_exp`].
    pub(in crate::codegen) fn tile_round(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("round needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };
        self.tile_round_into(block, src, &out)?;
        Ok(out)
    }

    /// out[...] = round(src[...]); out may be src itself (race-free per element).
    pub(in crate::codegen) fn tile_round_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("round needs a float tile, got {}", src.elem);
        }
        if src.shape.contains(&DYN) {
            bail!("round needs a static tile shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let vf = cg.float_cast(blk, v, cg.f32_t)?;
            let e = cg.round_even(blk, vf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out = tanh(src), element-wise. Mirrors [`Self::tile_sqrt`].
    pub(in crate::codegen) fn tile_tanh(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("tanh needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };
        self.tile_tanh_into(block, src, &out)?;
        Ok(out)
    }

    /// out[...] = tanh(src[...]); out may be src itself (race-free per element).
    pub(in crate::codegen) fn tile_tanh_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("tanh needs a float tile, got {}", src.elem);
        }
        if src.shape.contains(&DYN) {
            bail!("tanh needs a static tile shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let vf = cg.float_cast(blk, v, cg.f32_t)?;
            let e = cg.approx_tanh(blk, vf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }
}
