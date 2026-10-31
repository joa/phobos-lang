// Unary math over a tile.

use super::*;
use crate::ir;

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

    /// out = step(src) elementwise, for `exp`, `log`, `round`, `sqrt` and
    /// `tanh`, which widen to f32 and narrow back. An owned temp is
    /// rewritten in place, which saves a buffer and lets `var p = exp(...)`
    /// skip every copy.
    pub(in crate::codegen) fn tile_unary(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        step: ir::ElemStep,
    ) -> Result<MemVal<'c>> {
        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            if src.shape.contains(&DYN) {
                bail!("`{step}` needs a static tile shape");
            }
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };
        self.tile_unary_into(block, src, &out, step)?;
        Ok(out)
    }

    /// out[...] = step(src[...]). out may be src itself, since each thread
    /// reads and writes the same element.
    pub(in crate::codegen) fn tile_unary_into(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
        step: ir::ElemStep,
    ) -> Result<()> {
        if !self.is_float(src.elem) {
            bail!("`{step}` needs a float tile, got {}", src.elem);
        }
        if src.shape.contains(&DYN) {
            bail!("`{step}` needs a static tile shape");
        }
        let step = self.ir_elem_step(step);
        self.elem_chain_sweep(block, src, out, &[step])
    }

    /// out = convert(src) elementwise, to a tile of `want` element type.
    ///
    /// Always allocates a fresh tile, since the element type changes.
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
        self.elem_chain_sweep(block, src, &out, &[])?;
        Ok(out)
    }
}
