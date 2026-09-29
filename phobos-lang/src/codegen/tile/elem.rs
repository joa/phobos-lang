// Filling, copying and converting tiles, and the binary elementwise ops.

use super::*;

impl<'c> Codegen<'c> {
    /// out[i] = alpha * a[i] + beta * b[i], with no intermediate tiles. b may
    /// be a global-memory view.
    pub(in crate::codegen) fn tile_scaled_add_into(
        &mut self,
        block: &Block<'c>,
        alpha: Value<'c, 'c>,
        a: &MemVal<'c>,
        beta: Value<'c, 'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if !(self.is_float(a.elem) && self.is_float(b.elem) && self.is_float(out.elem)) {
            bail!("scaled add needs float operands");
        }

        let work = self.f32_t;
        let alpha = self.coerce(block, alpha, work)?;
        let beta = self.coerce(block, beta, work)?;
        let width = self.elementwise_width(&[a, b, out]);
        let vec_t = Type::vector(&[4], work);
        let (alpha_splat, beta_splat) = if width > 1 {
            (
                Some(self.vec_broadcast(block, alpha, vec_t)?),
                Some(self.vec_broadcast(block, beta, vec_t)?),
            )
        } else {
            (None, None)
        };

        self.distribute(block, out, width, true, |cg, blk, idx| {
            let (av, bv) = if width > 1 {
                (
                    cg.vec_load(blk, a.mem, idx, vec_t)?,
                    cg.vec_load(blk, b.mem, idx, vec_t)?,
                )
            } else {
                (
                    cg.load_as(blk, a.mem, idx, work)?,
                    cg.load_as(blk, b.mem, idx, work)?,
                )
            };

            let alpha_v = alpha_splat.unwrap_or(alpha);
            let beta_v = beta_splat.unwrap_or(beta);
            let alpha_a = cg.push(blk, cg.elem_arith(BinOp::Mul, work, alpha_v, av)?)?;
            let beta_b = cg.push(blk, cg.elem_arith(BinOp::Mul, work, beta_v, bv)?)?;
            let r = cg.push(blk, cg.elem_arith(BinOp::Add, work, alpha_a, beta_b)?)?;

            if width > 1 {
                cg.elem_store(blk, r, out.mem, idx, width)
            } else {
                let r = cg.coerce(blk, r, out.elem)?;
                blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
                Ok(())
            }
        })
    }

    /// out[...] = scalar for every element. scalar must have out's element
    /// type.
    pub(in crate::codegen) fn tile_fill(
        &mut self,
        block: &Block<'c>,
        scalar: Value<'c, 'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        let width = self.elementwise_width(&[out]);
        let splat = if width > 1 {
            Some(self.vec_broadcast(block, scalar, Type::vector(&[4], out.elem))?)
        } else {
            None
        };

        self.distribute(block, out, width, true, |cg, blk, idx| {
            cg.elem_store(blk, splat.unwrap_or(scalar), out.mem, idx, width)
        })
    }

    /// dst[...] = src[...] for every element.
    ///
    /// With async_copy (pipelined prefetches on sm_80+), global to shared
    /// transfers use cp.async so the thread does not stall on the load. The
    /// caller owns the async group and the wait.
    pub(in crate::codegen) fn tile_copy(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
        sync: bool,
        async_copy: bool,
    ) -> Result<()> {
        if src.elem != dst.elem {
            bail!(
                "tile copy with mismatched element types ({} vs {})",
                src.elem,
                dst.elem
            );
        }

        // An aligned copy moves 16 bytes per lane: four f32 or eight f16.
        // Staging is bound by load count, not bytes, so width goes by bytes.
        //
        // Only f16 goes to eight. Four elements of any type is always legal,
        // while a wider access needs the divisibility proof `@aligned` gives.
        let elem_bytes = self.elem_bytes(dst.elem);
        let last = *dst.shape.last().expect("tile values are not rank-0");
        let vec_ok = !src.is_masked()
            && !dst.is_masked()
            && last != DYN
            && elem_bytes.is_some()
            && src.vectorizes(4)
            && dst.vectorizes(4)
            && last % 4 == 0;
        // Both sides share an element type, checked above.
        let wide =
            dst.elem == self.f16_t && src.vectorizes(8) && dst.vectorizes(8) && last % 8 == 0;
        let width = match (vec_ok, wide) {
            (true, true) => 8,
            (true, false) => 4,
            _ => 1,
        };
        // At least four elements even for a scalar copy, since the f32
        // cp.async path reads that width.
        let align = i64::from(elem_bytes.unwrap_or(4)) * width.max(4);

        // cp.async needs a 4, 8 or 16-byte transfer. f32 qualifies at any
        // width. Narrower types qualify only when vectorized.
        let use_async = async_copy
            && !dst.is_masked()
            && (dst.elem == self.f32_t || (width > 1 && matches!(align, 4 | 8 | 16)));
        let vec_t = Type::vector(&[width.max(4) as u64], dst.elem);

        self.distribute(block, dst, width, sync, |cg, blk, idx| {
            let didx = cg.swizzled_index(blk, dst, idx)?;

            if use_async {
                cg.async_copy(blk, src, idx, dst, &didx, width)
            } else if width > 1 {
                let v = cg.vec_load_al(blk, src.mem, idx, vec_t, align)?;
                cg.vec_store_al(blk, v, dst.mem, &didx, align)
            } else {
                let v = cg.elem_load(blk, src.mem, idx, 1, vec_t)?;
                cg.elem_store(blk, v, dst.mem, &didx, 1)
            }
        })
    }

    /// dst[...] = cast(src[...]) for every element, e.g. an f32 accumulator
    /// stored into an f16 or i8 output tensor. Not vectorized.
    pub(in crate::codegen) fn tile_convert(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
    ) -> Result<()> {
        let numeric = |t| self.is_float(t) || self.is_int(t);
        if !(numeric(src.elem) && numeric(dst.elem)) {
            bail!(
                "tile copy with mismatched element types ({} vs {})",
                src.elem,
                dst.elem
            );
        }

        self.distribute(block, dst, 1, true, |cg, blk, idx| {
            let v = cg.load_as(blk, src.mem, idx, dst.elem)?;
            blk.append_operation(memref::store(v, dst.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// dst[k, m] = src[m, k]: stages a tile k-major so a row of dst holds one
    /// k-slice and fragment loads vectorize.
    ///
    /// Iterates the source, so each thread reads a coalesced row and scatters
    /// it into 4 scalar column writes, or 4-byte cp.async transfers with
    /// async_copy. Never emits a barrier, the caller owns synchronization.
    pub(in crate::codegen) fn tile_copy_transposed(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        if src.elem != dst.elem {
            bail!(
                "tile copy with mismatched element types ({} vs {})",
                src.elem,
                dst.elem
            );
        }
        let width = self.elementwise_width(&[src]);
        let vec_t = Type::vector(&[4], src.elem);
        self.distribute(block, src, width, false, |cg, blk, idx| {
            let (mi, k0) = (idx[0], idx[1]);
            if async_copy {
                for j in 0..width {
                    let c = cg.const_index(blk, j)?;
                    let kj = cg.addi(blk, k0, c)?;
                    cg.async_copy(blk, src, &[mi, kj], dst, &[kj, mi], 1)?;
                }
            } else if width > 1 {
                let v = cg.vec_load(blk, src.mem, idx, vec_t)?;
                for j in 0..4 {
                    let e = cg.vec_extract(blk, v, &[j], src.elem)?;
                    let c = cg.const_index(blk, j)?;
                    let kj = cg.addi(blk, k0, c)?;
                    blk.append_operation(memref::store(e, dst.mem, &[kj, mi], cg.loc));
                }
            } else {
                let e = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
                blk.append_operation(memref::store(e, dst.mem, &[k0, mi], cg.loc));
            }
            Ok(())
        })
    }

    /// out[...] = a[...] * b[...] for every element.
    pub(in crate::codegen) fn tile_binary(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if a.elem != out.elem || b.elem != out.elem {
            bail!("elementwise tile op with mismatched element types");
        }
        let width = self.elementwise_width(&[a, b, out]);
        let vec_t = Type::vector(&[4], out.elem);
        self.distribute(block, out, width, true, |cg, blk, idx| {
            // Arith ops work on vectors too, so this serves both widths.
            let x = cg.elem_load(blk, a.mem, idx, width, vec_t)?;
            let y = cg.elem_load(blk, b.mem, idx, width, vec_t)?;
            let r = cg.push(blk, cg.elem_arith(op, out.elem, x, y)?)?;
            cg.elem_store(blk, r, out.mem, idx, width)
        })
    }

    /// Elementwise out = a * b. Uses [`Self::tile_binary`] when nothing needs
    /// broadcasting, otherwise [`Self::tile_binary_bc`].
    pub(in crate::codegen) fn tile_binary_dispatch(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        let numeric = |t| self.is_float(t) || self.is_int(t);

        if !(numeric(a.elem) && numeric(b.elem) && numeric(out.elem)) {
            bail!(
                "elementwise tile op with non-numeric element types ({}, {} into {})",
                a.elem,
                b.elem,
                out.elem
            );
        }

        if a.elem == out.elem && b.elem == out.elem && a.shape == out.shape && b.shape == out.shape
        {
            self.tile_binary(block, op, a, b, out)
        } else {
            self.tile_binary_bc(block, op, a, b, out)
        }
    }

    /// out[...] = a[...] * b[...] with broadcasting. An operand dim of extent
    /// 1 always reads index 0, so an [R, 1] column stretches across an [R, C]
    /// output. Not vectorized.
    pub(in crate::codegen) fn tile_binary_bc(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if out.shape.contains(&DYN) {
            bail!("broadcast elementwise op needs a static result shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let ai = cg.bc_index(blk, idx, &out.shape, &a.shape)?;
            let bi = cg.bc_index(blk, idx, &out.shape, &b.shape)?;
            let x = cg.load_as(blk, a.mem, &ai, out.elem)?;
            let y = cg.load_as(blk, b.mem, &bi, out.elem)?;
            let r = cg.elem_arith(op, out.elem, x, y)?;
            let r = cg.push(blk, r)?;
            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = max(a[...], b[...]) with broadcasting, like
    /// [`Self::tile_binary_bc`].
    pub(in crate::codegen) fn tile_max_bc(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if a.elem != out.elem || b.elem != out.elem {
            bail!("tmax with mismatched element types");
        }
        if out.shape.contains(&DYN) {
            bail!("tmax needs a static result shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let ai = cg.bc_index(blk, idx, &out.shape, &a.shape)?;
            let bi = cg.bc_index(blk, idx, &out.shape, &b.shape)?;
            let x = cg.push(blk, memref::load(a.mem, &ai, cg.loc))?;
            let y = cg.push(blk, memref::load(b.mem, &bi, cg.loc))?;
            let r = cg.fmax(blk, x, y)?;
            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = select(va[...] >= vb[...], ia[...], ib[...]), broadcasting.
    /// This is the index side of a (value, index) fold.
    ///
    /// It uses `>=`, so a caller that always passes the later candidate as
    /// `(va, ia)` breaks exact ties reproducibly.
    pub(in crate::codegen) fn tile_argsel_bc(
        &mut self,
        block: &Block<'c>,
        va: &MemVal<'c>,
        vb: &MemVal<'c>,
        ia: &MemVal<'c>,
        ib: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if va.elem != vb.elem {
            bail!("argsel value operands must share an element type");
        }
        if ia.elem != out.elem || ib.elem != out.elem {
            bail!("argsel index operands must match the result element type");
        }
        if out.shape.contains(&DYN) {
            bail!("argsel needs a static result shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let vai = cg.bc_index(blk, idx, &out.shape, &va.shape)?;
            let vbi = cg.bc_index(blk, idx, &out.shape, &vb.shape)?;
            let iai = cg.bc_index(blk, idx, &out.shape, &ia.shape)?;
            let ibi = cg.bc_index(blk, idx, &out.shape, &ib.shape)?;
            let x = cg.push(blk, memref::load(va.mem, &vai, cg.loc))?;
            let y = cg.push(blk, memref::load(vb.mem, &vbi, cg.loc))?;
            let cond = cg.push(
                blk,
                arith::cmpf(cg.ctx, arith::CmpfPredicate::Oge, x, y, cg.loc),
            )?;
            let m = cg.push(blk, memref::load(ia.mem, &iai, cg.loc))?;
            let n = cg.push(blk, memref::load(ib.mem, &ibi, cg.loc))?;
            let r = cg.push(
                blk,
                OperationBuilder::new("arith.select", cg.loc)
                    .add_operands(&[cond, m, n])
                    .add_results(&[m.r#type()])
                    .build()?,
            )?;
            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = tile[...] * scalar, or scalar * tile[...]. The scalar is
    /// coerced to the output element type.
    pub(in crate::codegen) fn tile_scalar_into(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        tile: &MemVal<'c>,
        scalar: Value<'c, 'c>,
        scalar_left: bool,
        out: &MemVal<'c>,
    ) -> Result<()> {
        let numeric = |t| self.is_float(t) || self.is_int(t);

        if !(numeric(tile.elem) && numeric(out.elem)) {
            bail!(
                "tile * scalar with non-numeric element types ({} into {})",
                tile.elem,
                out.elem
            );
        }

        self.check_shapes(&tile.shape, &out.shape, "tile * scalar")?;

        let scalar = self.coerce(block, scalar, out.elem)?;

        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.load_as(blk, tile.mem, idx, out.elem)?;
            let (x, y) = if scalar_left {
                (scalar, v)
            } else {
                (v, scalar)
            };
            let r = cg.elem_arith(op, out.elem, x, y)?;
            let r = cg.push(blk, r)?;
            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
            Ok(())
        })
    }
}
