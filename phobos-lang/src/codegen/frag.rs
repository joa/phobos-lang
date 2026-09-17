use super::*;

impl<'c> Codegen<'c> {

}

// emission
impl<'c> Codegen<'c> {

    /// The rescaled fragments of `fa`, without rebinding anything.
    pub(super) fn frag_scale_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        op: BinOp,
        col: &MemVal<'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let zero = self.const_index(block, 0)?;
        let mut frags = fa.frags.clone();
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], [row, _]) in elems {
                let c = cg.load_as(block, col.mem, &[*row, zero], cg.f32_t)?;
                let e = cg.vec_extract(block, frags[i], &[*di, *dj], cg.f32_t)?;
                let r = cg.push(block, cg.elem_arith(op, cg.f32_t, e, c)?)?;
                frags[i] = cg.vec_insert(block, r, frags[i], &[*di, *dj])?;
            }
            Ok(())
        })?;
        Ok(frags)
    }

    /// The accumulated fragments of `fa += dot(a, b)`, without rebinding.
    pub(super) fn frag_dot_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let kk = a.shape[1];
        self.check_shapes(&a.shape, &[fa.m, kk], "fragment dot lhs")?;
        self.check_shapes(&b.shape, &[kk, fa.n], "fragment dot rhs")?;
        let (fm, fnn) = fa.warp_frags();

        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[fa.m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let finals = self.mma_sync_mac(
            block,
            &a_buf,
            &b_buf,
            (kk, fm, fnn),
            m0,
            n0,
            &fa.frags,
            false,
        )?;

        // No shared output to publish, but the barrier still orders the
        // ldmatrix reads before the released staging is reused. Hoisted
        // buffers outlive the loop.
        self.barrier(block)?;
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }
        Ok(finals)
    }

    /// <tensor slice> = acc: scatters each lane's fragment elements straight
    /// to the target, rounding f32 to f16 when the tensor is f16. The
    /// epilogue analogue of mma_sync_store_frags, minus the shared hop.
    pub(super) fn frag_store(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        fa: &FragAcc<'c>,
    ) -> Result<()> {
        self.check_shapes(&[fa.m, fa.n], &target.shape, "fragment store")?;
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], addr) in elems {
                let e = cg.vec_extract(block, fa.frags[i], &[*di, *dj], cg.f32_t)?;
                let e = cg.coerce(block, e, target.elem)?;
                block.append_operation(memref::store(e, target.mem, addr, cg.loc));
            }
            Ok(())
        })
    }

}
