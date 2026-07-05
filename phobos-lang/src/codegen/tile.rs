use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn alloc_tile(
        &mut self,
        block: &Block<'c>,
        scalar: Scalar,
        dims: &[Dim],
    ) -> Result<MemVal<'c>> {
        let shape = self.tile_shape(dims)?;
        self.alloc_tile_shaped(block, self.scalar_type(scalar), &shape)
    }

    /// alloc a tile buffer in SM
    pub(super) fn alloc_tile_shaped(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        shape: &[i64],
    ) -> Result<MemVal<'c>> {
        if shape.contains(&DYN) {
            bail!("tile buffers must have a static shape");
        }
        let space: Attribute = IntegerAttribute::new(self.i64_t, MEM_SHARED).into();
        let t = MemRefType::new(elem, shape, None, Some(space));

        // Reuse a released buffer of the same type when one is free (see
        // release); otherwise mint a new global.
        let key = (elem.to_string(), shape.to_vec());
        let name = match self.tile_pool.get_mut(&key).and_then(Vec::pop) {
            Some(name) => name,
            None => {
                let name = format!("__{}_tile{}", self.kernel_name, self.tile_count);
                self.tile_count += 1;
                self.shared_globals.push(memref::global(
                    self.ctx,
                    &name,
                    Some("private"),
                    t,
                    None, // uninitialized
                    false,
                    Some(IntegerAttribute::new(self.i64_t, 16)), // 128-bit vector access
                    self.loc,
                ));
                name
            }
        };

        let mem = self.push(block, memref::get_global(self.ctx, &name, t, self.loc))?;

        let mem = self.assume_align(block, mem, 16)?; // be safe

        // the buffer is always 16-byte aligned! the rows are if every outer
        // stride is a multiple of 4 elements, which is exactly the byte alignment
        // a 4-element vector of the element type needs (16b for f32, 8b for f16).
        let aligned = row_major_strides(shape)[..shape.len() - 1]
            .iter()
            .all(|&s| mult4(s));

        Ok(MemVal {
            mem,
            elem,
            shape: shape.to_vec(),
            row_stride: None,
            aligned,
            swizzle: None,
            global: Some(name),
            owned: true,
        })
    }

    /// Returns an owned temp's shared buffer to the pool so a later
    /// allocation of the same element type and physical shape reuses it
    /// instead of growing the CTA's static shared footprint (each distinct
    /// global costs occupancy). No-op for views, params, and named tiles
    /// (bind clears owned).
    ///
    /// Only call after every op reading the buffer has been emitted. Reuse
    /// is race-free because each tile op ends in a CTA barrier: the reusing
    /// op's writes are ordered after the previous consumer's reads. Reused
    /// buffers hold garbage, which is fine since every producing op fully
    /// writes its output before it is read.
    pub(super) fn release(&mut self, mv: &MemVal<'c>) {
        if !mv.owned {
            return;
        }
        let Some(name) = &mv.global else {
            return;
        };
        // Pool by the physical allocation shape (padded buffers carry a
        // logical shape narrower than the backing global).
        let mut shape = mv.shape.clone();
        if let (Some(stride), Some(last)) = (mv.row_stride, shape.last_mut()) {
            *last = stride;
        }
        let names = self
            .tile_pool
            .entry((mv.elem.to_string(), shape))
            .or_default();
        if !names.contains(name) {
            names.push(name.clone());
        }
    }

    /// Allocates a plain (unpadded) shared staging tile with an XOR column
    /// swizzle (see [`Swizzle`]) so ldmatrix reads avoid bank conflicts without
    /// paying for the WMMA path's padding. The swizzle permutes 8-element column
    /// blocks (elem_log = 3, the ldmatrix row granule) by the low row bits, and
    /// the block count caps the permutation at the bank period. The buffer keeps
    /// its logical shape and contiguous layout; only the column index each access
    /// uses gets transformed, the same way on store and load.
    pub(super) fn alloc_tile_swizzled(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        shape: &[i64],
    ) -> Result<MemVal<'c>> {
        let mut mv = self.alloc_tile_shaped(block, elem, shape)?;
        let width = *shape.last().expect("tile values are not rank-0");

        // 8-f16 blocks per row; permute the block index by up to the bank period
        // (32 banks / 4 banks per 16B granule = 8 phases, so at most 3 bits).
        let blocks = (width / 8).max(1);
        let bits = blocks.trailing_zeros().min(3);

        mv.swizzle = (bits > 0).then_some(Swizzle {
            bits,
            shift: 0,
            elem_log: 3,
        });

        Ok(mv)
    }

    /// Permutes a column index through a buffer's [`Swizzle`], or returns it
    /// unchanged when the buffer is unswizzled: col ^ (((row >> shift) &
    /// ((1<<bits)-1)) << elem_log). Every staging store and ldmatrix load goes
    /// through the same call, so the data round-trips whatever the params are.
    pub(super) fn swizzle_col(
        &self,
        block: &Block<'c>,
        mv: &MemVal<'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let Some(sw) = mv.swizzle else {
            return Ok(col);
        };
        let mut r = row;

        if sw.shift > 0 {
            let s = self.const_index(block, sw.shift as i64)?;
            r = self.push(block, arith::shrui(r, s, self.loc))?;
        }

        let mask = self.const_index(block, (1i64 << sw.bits) - 1)?;
        let masked = self.push(block, arith::andi(r, mask, self.loc))?;
        let e = self.const_index(block, sw.elem_log as i64)?;
        let perm = self.push(block, arith::shli(masked, e, self.loc))?;

        self.push(block, arith::xori(col, perm, self.loc))
    }

    /// A copy of idx with its last (column) component swizzled through mv's
    /// layout, or idx unchanged for an unswizzled buffer. Lets a staging store
    /// land at the same permuted column the ldmatrix load reads.
    pub(super) fn swizzled_index(
        &self,
        block: &Block<'c>,
        mv: &MemVal<'c>,
        idx: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        if mv.swizzle.is_none() || idx.len() < 2 {
            return Ok(idx.to_vec());
        }
        let mut out = idx.to_vec();
        let last = idx.len() - 1;
        out[last] = self.swizzle_col(block, mv, idx[last - 1], idx[last])?;
        Ok(out)
    }

    /// Allocates a shared WMMA staging tile whose innermost dimension is padded
    /// by [`WMMA_SMEM_PAD`] elements, spreading consecutive rows across distinct
    /// shared-memory banks (see that constant). The returned tile keeps its
    /// logical shape (so iteration and fragment indexing are unchanged); the
    /// padding lives only in the physical allocation and the row_stride the WMMA
    /// leadDimension reads.
    pub(super) fn alloc_tile_padded(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        shape: &[i64],
    ) -> Result<MemVal<'c>> {
        let mut phys = shape.to_vec();
        let last = phys.len() - 1;

        phys[last] += WMMA_SMEM_PAD;

        // alloc_tile_shaped sizes the buffer and proves alignment off the
        // padded physical shape; restore the logical view afterwards.
        let mut mv = self.alloc_tile_shaped(block, elem, &phys)?;

        mv.row_stride = Some(phys[last]);
        mv.shape = shape.to_vec();

        Ok(mv)
    }

    pub(super) fn check_matmul_shapes(
        &self,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if a.shape.len() != 2 || b.shape.len() != 2 || out.shape.len() != 2 {
            bail!("dot expects rank-2 tiles");
        }
        self.check_shapes(&[a.shape[1]], &[b.shape[0]], "dot contraction dim")?;
        self.check_shapes(&[a.shape[0], b.shape[1]], &out.shape, "dot result")
    }

    pub(super) fn check_matmul_elems(
        &self,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        what: &str,
    ) -> Result<()> {
        if a.elem == b.elem && a.elem == out.elem {
            return Ok(());
        }

        match (
            self.float_rank(a.elem),
            self.float_rank(b.elem),
            self.float_rank(out.elem),
        ) {
            (Some(ra), Some(rb), Some(ro)) if ro >= ra && ro >= rb => Ok(()),
            _ => bail!(
                "{what}: cannot accumulate {} and {} operands into {}",
                a.elem,
                b.elem,
                out.elem
            ),
        }
    }

    pub(super) fn check_shapes(&self, a: &[i64], b: &[i64], what: &str) -> Result<()> {
        let ok = a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| x == DYN || y == DYN || x == y);

        if !ok {
            bail!(
                "{what}: shape mismatch ({} vs {})",
                fmt_shape(a),
                fmt_shape(b)
            );
        }

        Ok(())
    }
}

// thread-distributed loops
impl<'p, 'c> Codegen<'p, 'c> {
    /// Emits body once per element of out, or once per width-element innermost
    /// segment when width > 1 (the caller guarantees a static, width-divisible
    /// innermost extent), distributed across the CTA.
    ///
    /// sync=false skips the trailing barrier, for pipelined prefetch copies that
    /// get synced by their iteration's closing barrier instead.
    pub(super) fn distribute(
        &mut self,
        block: &Block<'c>,
        out: &MemVal<'c>,
        width: i64,
        sync: bool,
        body: impl FnOnce(&mut Self, &Block<'c>, &[Value<'c, 'c>]) -> Result<()>,
    ) -> Result<()> {
        let mut sizes = self.tile_sizes(block, out)?;
        let rank = sizes.len();

        if width > 1 {
            sizes[rank - 1] = self.const_index(block, out.shape[rank - 1] / width)?;
        }

        let mut total = sizes[0];

        for &s in &sizes[1..] {
            total = self.muli(block, total, s)?;
        }

        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;

        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body_block.argument(0)?.into());

        // last dim varies fastest, so adjacent threads touch adjacent elements for coalescing
        let mut idx = vec![li; rank];

        if rank > 1 {
            let mut rem = li;

            for i in (1..rank).rev() {
                idx[i] = self.push(&body_block, arith::remui(rem, sizes[i], self.loc))?;
                rem = self.push(&body_block, arith::divui(rem, sizes[i], self.loc))?;
            }

            idx[0] = rem;
        }

        if width > 1 {
            let w = self.const_index(&body_block, width)?;
            idx[rank - 1] = self.push(&body_block, arith::muli(idx[rank - 1], w, self.loc))?;
        }

        body(self, &body_block, &idx)?;

        body_block.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body_block);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));

        // barriers can't deadlock here: the language never exposes thread
        // ids, so every scalar value, and therefore all control flow, is
        // uniform across the CTA.
        if sync {
            self.barrier(block)?;
        }

        Ok(())
    }

    pub(super) fn tile_sizes(
        &mut self,
        block: &Block<'c>,
        mv: &MemVal<'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        mv.shape
            .clone()
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d == DYN {
                    let pos = self.const_index(block, i as i64)?;
                    self.push(block, memref::dim(mv.mem, pos, self.loc))
                } else {
                    self.const_index(block, d)
                }
            })
            .collect()
    }

    /// Vector width for an elementwise op over these buffers: 4 when every
    /// buffer has f32 elements, provably 16B-aligned rows, and a static
    /// innermost extent divisible by 4; otherwise 1 (scalar).
    pub(super) fn elementwise_width(&self, mvs: &[&MemVal<'c>]) -> i64 {
        let ok = mvs.iter().all(|m| {
            let last = *m.shape.last().expect("tile values are not rank-0");
            m.elem == self.f32_t && m.aligned && last != DYN && last % 4 == 0
        });
        if ok { 4 } else { 1 }
    }

    /// out[i] = alpha * a[i] + beta * b[i]
    ///
    /// No intermediate tile allocations. b may be a global-memory view.
    pub(super) fn tile_scaled_add_into(
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

    /// out[...] = scalar for every element (scalar must be elem-typed)
    pub(super) fn tile_fill(
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
    /// With async_copy (pipelined prefetches on sm_80+), the global->shared
    /// transfers go out as cp.async so the issuing thread doesn't stall on the
    /// global load; the caller owns the async group and wait.
    pub(super) fn tile_copy(
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

        // Vectorize aligned copies at 4 elements: 16B/4xf32 or 8B/4xf16 (the
        // src and dst element types match, checked above). The row-pitch ABI's
        // multiple-of-4-elements guarantee is exactly the byte alignment a
        // 4-element vector of either type needs.
        let last = *dst.shape.last().expect("tile values are not rank-0");
        let vec_ok = src.aligned
            && dst.aligned
            && last != DYN
            && last % 4 == 0
            && (dst.elem == self.f32_t || dst.elem == self.f16_t);
        let width = if vec_ok { 4 } else { 1 };

        // cp.async needs a 4/8/16-byte transfer and can't convert (so the
        // src/dst element types must match, as checked above): f32 qualifies at
        // any width (4B scalar or 16B vector), f16 only vectorized (4 elems =
        // 8B; a scalar 2B f16 is below cp.async's minimum).
        let use_async =
            async_copy && (dst.elem == self.f32_t || (dst.elem == self.f16_t && width == 4));
        let vec_t = Type::vector(&[4], dst.elem);
        let align = if dst.elem == self.f16_t { 8 } else { 16 };

        self.distribute(block, dst, width, sync, |cg, blk, idx| {
            // Read from the (unswizzled) source, store to the swizzled column.
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

    /// dst[...] = cast(src[...]) for every element.
    ///
    /// Float-casts between the source and destination element types, e.g. an
    /// f32 accumulator stored into an f16 output tensor. Not vectorized.
    pub(super) fn tile_convert(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
    ) -> Result<()> {
        if !(self.is_float(src.elem) && self.is_float(dst.elem)) {
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

    /// One cp.async transfer of width elements via nvgpu.device_async_copy
    /// (src[src_idx] -> dst[dst_idx]). The resulting token is dropped: the
    /// enclosing pipeline stage commits all pending copies with one
    /// device_async_create_group and waits on that.
    pub(super) fn async_copy(
        &self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        src_idx: &[Value<'c, 'c>],
        dst: &MemVal<'c>,
        dst_idx: &[Value<'c, 'c>],
        width: i64,
    ) -> Result<()> {
        let token_t = self.parse_type("!nvgpu.device.async.token")?;
        let mut operands = vec![dst.mem];

        operands.extend_from_slice(dst_idx);
        operands.push(src.mem);
        operands.extend_from_slice(src_idx);

        let mut attributes = vec![
            (
                self.id("dstElements"),
                IntegerAttribute::new(self.index_t, width).into(),
            ),
            (
                self.id("operandSegmentSizes"),
                self.i32_array(&[1, dst_idx.len() as i32, 1, src_idx.len() as i32, 0])?,
            ),
        ];

        // Only 16-byte copies can skip L1 (cp.async.cg): staged tiles are
        // consumed from shared memory, not re-read through L1. A vectorized f32
        // copy is 16B (width 4), but a vectorized f16 copy is only 8B, so gate
        // on the byte size rather than the element count.
        let elem_bytes = if dst.elem == self.f16_t { 2 } else { 4 };
        if width * elem_bytes == 16 {
            attributes.push((self.id("bypassL1"), Attribute::unit(self.ctx)));
        }
        block.append_operation(
            OperationBuilder::new("nvgpu.device_async_copy", self.loc)
                .add_operands(&operands)
                .add_attributes(&attributes)
                .add_results(&[token_t])
                .build()?,
        );
        Ok(())
    }

    /// dst[k, m] = src[m, k]: stages a tile k-major (transposed), so a row of
    /// dst holds one k-slice and fragment loads vectorize. The distribution
    /// iterates the source (the map is a bijection, so ownership still partitions
    /// the output): each thread reads a vector row segment (coalesced) and
    /// scatters 4 scalar column writes. With async_copy the elements move as
    /// 4-byte cp.async transfers instead, which can't vectorize since the
    /// destination is strided, but don't stall. Never emits a barrier; the caller
    /// owns synchronization.
    pub(super) fn tile_copy_transposed(
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
    pub(super) fn tile_binary(
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
            // The arith ops apply elementwise to vectors, so the same code
            // serves both widths.
            let x = cg.elem_load(blk, a.mem, idx, width, vec_t)?;
            let y = cg.elem_load(blk, b.mem, idx, width, vec_t)?;
            let r = cg.push(blk, cg.elem_arith(op, out.elem, x, y)?)?;
            cg.elem_store(blk, r, out.mem, idx, width)
        })
    }

    /// Elementwise out = a * b. Takes the vectorized equal-shape path
    /// ([`Self::tile_binary`]) when nothing needs broadcasting, otherwise the
    /// scalar broadcast path ([`Self::tile_binary_bc`]).
    pub(super) fn tile_binary_dispatch(
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
        if a.shape == out.shape && b.shape == out.shape {
            self.tile_binary(block, op, a, b, out)
        } else {
            self.tile_binary_bc(block, op, a, b, out)
        }
    }

    /// out[...] = a[...] * b[...] with broadcasting: an operand dim of extent 1
    /// reads index 0 in that axis (so a [R, 1] column vector stretches across the
    /// [R, C] output). Scalar, not vectorized, since the broadcast operands have
    /// a non-contiguous innermost access.
    pub(super) fn tile_binary_bc(
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
            let x = cg.push(blk, memref::load(a.mem, &ai, cg.loc))?;
            let y = cg.push(blk, memref::load(b.mem, &bi, cg.loc))?;
            let r = cg.elem_arith(op, out.elem, x, y)?;
            let r = cg.push(blk, r)?;
            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = exp(a[...] op b[...]) with broadcasting: the softmax
    /// probability form t = exp(t - mnew) in a single sweep and barrier,
    /// instead of one for the subtract and one for the exp. out may be an
    /// operand (each thread reads and writes the same element).
    pub(super) fn tile_exp_binary_bc(
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
        if out.shape.contains(&DYN) {
            bail!("broadcast elementwise op needs a static result shape");
        }
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let ai = cg.bc_index(blk, idx, &out.shape, &a.shape)?;
            let bi = cg.bc_index(blk, idx, &out.shape, &b.shape)?;
            let x = cg.push(blk, memref::load(a.mem, &ai, cg.loc))?;
            let y = cg.push(blk, memref::load(b.mem, &bi, cg.loc))?;
            let r = cg.push(blk, cg.elem_arith(op, out.elem, x, y)?)?;
            let rf = cg.float_cast(blk, r, cg.f32_t)?;
            let e = cg.approx_exp(blk, rf)?;
            let e = cg.float_cast(blk, e, out.elem)?;
            blk.append_operation(memref::store(e, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[...] = max(a[...], b[...]) with broadcasting (like
    /// [`Self::tile_binary_bc`], but arith has no float-max BinOp so it lowers to
    /// cmpf plus select).
    pub(super) fn tile_max_bc(
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

    /// out[...] = tile[...] * scalar (or scalar * tile[...]). The scalar is
    /// coerced to the output element type and broadcast over every element.
    pub(super) fn tile_scalar_into(
        &mut self,
        block: &Block<'c>,
        op: BinOp,
        tile: &MemVal<'c>,
        scalar: Value<'c, 'c>,
        scalar_left: bool,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if tile.elem != out.elem {
            bail!("tile * scalar with mismatched element types");
        }
        self.check_shapes(&tile.shape, &out.shape, "tile * scalar")?;
        let scalar = self.coerce(block, scalar, out.elem)?;
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let v = cg.push(blk, memref::load(tile.mem, idx, cg.loc))?;
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

    /// Per-axis index for reading a broadcast operand: axes where the operand
    /// has extent 1 but the output is wider read index 0.
    pub(super) fn bc_index(
        &self,
        block: &Block<'c>,
        idx: &[Value<'c, 'c>],
        out_shape: &[i64],
        src_shape: &[i64],
    ) -> Result<Vec<Value<'c, 'c>>> {
        idx.iter()
            .enumerate()
            .map(|(d, &ix)| {
                if src_shape[d] == 1 && out_shape[d] != 1 {
                    self.const_index(block, 0)
                } else {
                    Ok(ix)
                }
            })
            .collect()
    }

    /// max(a, b) on a float scalar, via cmpf ogt plus select.
    pub(super) fn fmax(
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

    /// The value of v on the lane whose id differs in the given xor mask
    /// bits, via gpu.shuffle xor (shfl.sync.bfly after convert-gpu-to-nvvm).
    /// Every lane of the executing warp must reach this op.
    pub(super) fn shfl_xor_f32(
        &self,
        block: &Block<'c>,
        v: Value<'c, 'c>,
        mask: i64,
    ) -> Result<Value<'c, 'c>> {
        let c_i32 = |val: i64| {
            arith::constant(
                self.ctx,
                IntegerAttribute::new(self.i32_t, val).into(),
                self.loc,
            )
        };
        let offset = self.push(block, c_i32(mask))?;
        let width = self.push(block, c_i32(32))?;
        self.push(
            block,
            OperationBuilder::new("gpu.shuffle", self.loc)
                .add_operands(&[v, offset, width])
                .add_attributes(&[(self.id("mode"), self.parse_attr("#gpu<shuffle_mode xor>")?)])
                .add_results(&[self.f32_t, self.bool_t])
                .build()?,
        )
    }

    /// out[...] = exp(src[...]) elementwise. The hardware ex2.approx is f32, so
    /// f16 tiles round-trip through f32 (load, widen, exp, narrow).
    pub(super) fn tile_exp(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
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

    /// out[...] = exp(src[...]); out may be src itself (each thread reads and
    /// writes the same element, so the in-place rewrite is race-free).
    pub(super) fn tile_exp_into(
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

    /// e^x for an f32 scalar as ex2(x * log2e), emitting the PTX ex2.approx.ftz.f32
    /// directly via llvm.inline_asm. The MLIR math.exp route doesn't work here:
    /// convert-math-to-llvm (which runs before the gpu->nvvm libdevice patterns,
    /// so that math.fma becomes the fma.rn intrinsic) would rewrite it to
    /// llvm.intr.exp, which the NVPTX backend can't select. The hardware
    /// approximation is also what Triton emits for softmax.
    pub(super) fn approx_exp(&self, block: &Block<'c>, x: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let log2e = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, std::f64::consts::LOG2_E).into(),
                self.loc,
            ),
        )?;
        let t = self.push(block, arith::mulf(x, log2e, self.loc))?;
        self.push(
            block,
            OperationBuilder::new("llvm.inline_asm", self.loc)
                .add_operands(&[t])
                .add_attributes(&[
                    (
                        self.id("asm_string"),
                        StringAttribute::new(self.ctx, "ex2.approx.ftz.f32 $0, $1;").into(),
                    ),
                    (
                        self.id("constraints"),
                        StringAttribute::new(self.ctx, "=f,f").into(),
                    ),
                    (self.id("has_side_effects"), Attribute::unit(self.ctx)),
                ])
                .add_results(&[self.f32_t])
                .build()?,
        )
    }

    /// Reduces a rank-2 tile over its last (column) dim into a [rows, 1]
    /// column vector. When the CTA has threads to spare, lanes of a warp
    /// cooperate on each row ([`Self::rowreduce_warp`]); otherwise one thread
    /// sweeps each row serially ([`Self::rowreduce_serial`]).
    pub(super) fn tile_rowreduce(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        kind: Reduce,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("row reduction expects a rank-2 tile");
        }
        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("row reduction needs a static tile shape");
        }
        let elem = src.elem;
        if !self.is_float(elem) {
            bail!("row reduction needs a float element type");
        }
        let out = self.alloc_tile_shaped(block, elem, &[rows, 1])?;
        match self.reduce_lanes(rows) {
            // gpu.shuffle carries a 32-bit payload; f16 keeps the serial path.
            Some(lanes) if elem == self.f32_t => {
                self.rowreduce_warp(block, src, &out, kind, lanes)?
            }
            _ => self.rowreduce_serial(block, src, &out, kind)?,
        }
        Ok(out)
    }

    /// The fold identity: 0 for sum; for max, smaller than any finite input
    /// (the first column overwrites it). f16 saturates at -65504, so use a
    /// representable floor.
    fn reduce_identity(
        &self,
        block: &Block<'c>,
        elem: Type<'c>,
        kind: Reduce,
    ) -> Result<Value<'c, 'c>> {
        match kind {
            Reduce::Sum => self.zero_scalar(block, elem),
            Reduce::Max => {
                let floor = if elem == self.f16_t { -65504.0 } else { -3.0e38 };
                self.push(
                    block,
                    arith::constant(
                        self.ctx,
                        FloatAttribute::new(self.ctx, elem, floor).into(),
                        self.loc,
                    ),
                )
            }
        }
    }

    /// Lanes cooperating on each row of a warp-shuffled row reduction: the
    /// largest power of two the CTA can spend per row, capped at the warp
    /// width, such that rows * lanes covers whole warps (shfl.sync stalls
    /// unless every lane of a participating warp reaches it). None when only
    /// a single lane per row fits; the serial path covers that.
    fn reduce_lanes(&self, rows: i64) -> Option<i64> {
        let per_row = self.cta_threads / rows.max(1);
        if per_row < 2 {
            return None;
        }
        let mut lanes = 1i64 << per_row.min(32).ilog2();
        while lanes >= 2 {
            if (rows * lanes) % 32 == 0 {
                return Some(lanes);
            }
            lanes /= 2;
        }
        None
    }

    /// Each output row is owned by one thread, which sweeps the columns with
    /// a scalar accumulator (a thread-private rank-0 alloca).
    fn rowreduce_serial(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
        kind: Reduce,
    ) -> Result<()> {
        let cols = src.shape[1];
        let elem = src.elem;
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let i = idx[0];
            // Thread-private scalar accumulator.
            let slot_t = MemRefType::new(elem, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let init = cg.reduce_identity(blk, elem, kind)?;
            blk.append_operation(memref::store(init, slot, &[], cg.loc));

            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, cols)?;
            let st = cg.const_index(blk, 1)?;
            let jb = Block::new(&[(cg.index_t, cg.loc)]);
            let j = detach(jb.argument(0)?.into());
            let v = cg.push(&jb, memref::load(src.mem, &[i, j], cg.loc))?;
            let cur = cg.push(&jb, memref::load(slot, &[], cg.loc))?;
            let nv = match kind {
                Reduce::Sum => cg.push(&jb, arith::addf(cur, v, cg.loc))?,
                Reduce::Max => cg.fmax(&jb, cur, v)?,
            };
            jb.append_operation(memref::store(nv, slot, &[], cg.loc));
            jb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(jb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// Warp-cooperative row reduction: lanes consecutive lanes fold one row
    /// (each folds a strided slice of the columns as an scf.for iter_arg),
    /// then a gpu.shuffle xor butterfly combines the partials and lane 0 of
    /// the group stores the row result.
    ///
    /// Safety of the shuffle: rows * lanes covers whole warps and the block
    /// dim is a warp multiple, so any warp reaching the shuffle has all 32
    /// lanes present. The xor masks stay below lanes, so a lanes-aligned
    /// group never exchanges outside itself.
    fn rowreduce_warp(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
        kind: Reduce,
        lanes: i64,
    ) -> Result<()> {
        let (rows, cols) = (src.shape[0], src.shape[1]);
        let total = self.const_index(block, rows * lanes)?;
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let lanes_v = self.const_index(&body, lanes)?;
        let row = self.divui(&body, li, lanes_v)?;
        let lane = self.remui(&body, li, lanes_v)?;

        // This lane's strided slice of the row, folded in a register.
        let init = self.reduce_identity(&body, self.f32_t, kind)?;
        let hi = self.const_index(&body, cols)?;
        let jb = Block::new(&[(self.index_t, self.loc), (self.f32_t, self.loc)]);
        let j = detach(jb.argument(0)?.into());
        let cur = detach(jb.argument(1)?.into());
        let v = self.push(&jb, memref::load(src.mem, &[row, j], self.loc))?;
        let nv = match kind {
            Reduce::Sum => self.push(&jb, arith::addf(cur, v, self.loc))?,
            Reduce::Max => self.fmax(&jb, cur, v)?,
        };
        jb.append_operation(scf::r#yield(&[nv], self.loc));
        let jr = Region::new();
        jr.append_block(jb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lane, hi, lanes_v, init])
                .add_results(&[self.f32_t])
                .add_regions([jr])
                .build()?,
        )?;

        let mut mask = lanes / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = match kind {
                Reduce::Sum => self.push(&body, arith::addf(acc, other, self.loc))?,
                Reduce::Max => self.fmax(&body, acc, other)?,
            };
            mask /= 2;
        }

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let then = Block::new(&[]);
        then.append_operation(memref::store(acc, out.mem, &[row, zero], self.loc));
        then.append_operation(scf::r#yield(&[], self.loc));
        let tr = Region::new();
        tr.append_block(then);
        body.append_operation(scf::r#if(is_lead, &[], tr, Region::new(), self.loc));

        body.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }

    /// out[i, j] = sum_k(a[i, k] * b[j, k]), the transposed matmul behind dot_t
    /// (contracts the last dim of both operands). A plain thread-per-output
    /// scalar reduction; the heavily-tuned [`tile_matmul`] has
    /// no transposed-b variant, and the attention S = Q @ K.T tile is small
    /// relative to the kernel's other costs.
    pub(super) fn tile_matmul_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        self.check_matmul_elems(a, b, out, "dot_t")?;
        let kd = a.shape[1];
        if kd == DYN {
            bail!("dot_t needs a static contraction dim");
        }
        let elem = out.elem;
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let slot_t = MemRefType::new(elem, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let zero = cg.zero_scalar(blk, elem)?;
            blk.append_operation(memref::store(zero, slot, &[], cg.loc));

            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, kd)?;
            let st = cg.const_index(blk, 1)?;
            let kb = Block::new(&[(cg.index_t, cg.loc)]);
            let k = detach(kb.argument(0)?.into());
            // operands widen to the accumulator type (f16 inputs, f32 acc).
            let va = cg.load_as(&kb, a.mem, &[i, k], elem)?;
            let vb = cg.load_as(&kb, b.mem, &[j, k], elem)?;
            let cur = cg.push(&kb, memref::load(slot, &[], cg.loc))?;
            let nv = cg.elem_mac(&kb, elem, va, vb, cur)?;
            kb.append_operation(memref::store(nv, slot, &[], cg.loc));
            kb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(kb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[m, n] += sum_k(a[m, k] * b[k, n]): accumulates into out.
    ///
    /// Register-blocked when out has a static shape: the CTA's threads stride over
    /// TMxTN sub-tiles of the output ([`Self::sub_tile`]), carrying the
    /// accumulator through the k-loop as one vector<TMxTN> iter_arg, fed by
    /// vector.contract over k-chunks. That loads TM + TN operand elements per
    /// k-step for TM*TN MACs, instead of two loads per MAC in the element-wise
    /// scheme.
    ///
    /// Warp-tiled when a factorization of the 32 lanes divides the sub-tile grid
    /// ([`Self::lane_grid`]): warps stride over WMxWN warp tiles (WM = lm*TM,
    /// WN = ln*TN) with lanes laid out lmxln row-major inside, so the warp's
    /// per-k-step shared reads collapse to WM + WN distinct elements. Lanes in a
    /// row broadcast the same a fragment, lanes in a column the same b fragment,
    /// and the b row segments the lanes read are contiguous (conflict-free). The
    /// flat fallback scatters the warp across a thin full-width strip and reads up
    /// to twice as much.
    pub(super) fn tile_matmul(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        self.check_matmul_elems(a, b, out, "dot")?;
        let k_extent = if a.shape[1] == DYN {
            let pos = self.const_index(block, 1)?;
            self.push(block, memref::dim(a.mem, pos, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };
        let (m, n) = (out.shape[0], out.shape[1]);
        if m == DYN || n == DYN {
            return self.tile_matmul_dynamic(block, a, b, out, k_extent);
        }
        let elem = out.elem;
        let (tm, tn) = self.sub_tile(m, n);
        let (tiles_m, tiles_n) = (m / tm, n / tn);
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;

        // Warp decomposition, hoisted out of the warp-tile loop: which warp
        // this thread belongs to, how many warps the CTA has (the launch ABI
        // requires a multiple of 32 threads), and the lane's element offset
        // within its warp tile. Unsigned div/rem (non-negative operands) by
        // constants strength-reduce to shift/mask.
        let warp = match Self::lane_grid(tiles_m, tiles_n, tm, tn) {
            Some((lm, ln)) => {
                let w = self.const_index(block, 32)?;
                let warp_id = self.divui(block, tid, w)?;
                let lane = self.remui(block, tid, w)?;
                let nwarps = self.divui(block, bdim, w)?;
                let ln_v = self.const_index(block, ln)?;
                let lane_m = self.divui(block, lane, ln_v)?;
                let lane_n = self.remui(block, lane, ln_v)?;
                let tm_v = self.const_index(block, tm)?;
                let tn_v = self.const_index(block, tn)?;
                let off_m = self.muli(block, lane_m, tm_v)?;
                let off_n = self.muli(block, lane_n, tn_v)?;
                Some((lm, ln, warp_id, nwarps, off_m, off_n))
            }
            None => None,
        };
        let (lo, total, step) = match warp {
            Some((lm, ln, warp_id, nwarps, ..)) => {
                let total = self.const_index(block, (tiles_m / lm) * (tiles_n / ln))?;
                (warp_id, total, nwarps)
            }
            None => (tid, self.const_index(block, tiles_m * tiles_n)?, bdim),
        };

        let body = Block::new(&[(self.index_t, self.loc)]);
        let st = detach(body.argument(0)?.into());

        let (m0, n0) = if let Some((lm, ln, _, _, off_m, off_n)) = warp {
            // Warp-tile origin: wt -> (wt / wtiles_n * WM, wt % wtiles_n * WN),
            // plus the lane's offset within the warp tile.
            let wtiles_n_v = self.const_index(&body, tiles_n / ln)?;
            let q = self.push(&body, arith::divui(st, wtiles_n_v, self.loc))?;
            let r = self.push(&body, arith::remui(st, wtiles_n_v, self.loc))?;
            let wm_v = self.const_index(&body, lm * tm)?;
            let wn_v = self.const_index(&body, ln * tn)?;
            let wm0 = self.push(&body, arith::muli(q, wm_v, self.loc))?;
            let wn0 = self.push(&body, arith::muli(r, wn_v, self.loc))?;
            (
                self.push(&body, arith::addi(wm0, off_m, self.loc))?,
                self.push(&body, arith::addi(wn0, off_n, self.loc))?,
            )
        } else {
            // Flat sub-tile origin: st -> (st / tiles_n * TM, st % tiles_n * TN).
            // tiles_n is a constant, so the div/rem strength-reduce.
            let tiles_n_v = self.const_index(&body, tiles_n)?;
            let q = self.push(&body, arith::divui(st, tiles_n_v, self.loc))?;
            let r = self.push(&body, arith::remui(st, tiles_n_v, self.loc))?;
            let tm_v = self.const_index(&body, tm)?;
            let tn_v = self.const_index(&body, tn)?;
            (
                self.push(&body, arith::muli(q, tm_v, self.loc))?,
                self.push(&body, arith::muli(r, tn_v, self.loc))?,
            )
        };
        let mut ms = Vec::with_capacity(tm as usize);
        for i in 0..tm {
            let c = self.const_index(&body, i)?;
            ms.push(self.push(&body, arith::addi(m0, c, self.loc))?);
        }
        let mut ns = Vec::with_capacity(tn as usize);
        for j in 0..tn {
            let c = self.const_index(&body, j)?;
            ns.push(self.push(&body, arith::addi(n0, c, self.loc))?);
        }

        // Row segments become single vector accesses when the buffer's base
        // and row starts are provably 16-byte aligned (MemVal::aligned; n0 is
        // a multiple of tn, k chunks are multiples of 4). A vector access per
        // thread is also free of shared-bank conflicts, where stride-4 scalar
        // accesses conflict 4-way. Unvectorizable operands get assembled
        // element-wise instead; the MAC grid is a vector.contract either way
        // (see register_mac for the scheme; here a is m-major, so lhs chunks
        // are (m, k) and the contract's lhs transpose folds at constant
        // positions).
        let kk = a.shape[1];
        let chunk = if kk == DYN {
            1
        } else {
            [4, 2, 1]
                .into_iter()
                .find(|c| kk % c == 0)
                .expect("1 divides everything")
        };
        let vec_a = chunk == 4 && elem == self.f32_t && a.aligned && a.elem == elem;
        let vec_b = tn % 4 == 0 && elem == self.f32_t && b.aligned && b.elem == elem;
        let vec_out = tn % 4 == 0 && elem == self.f32_t && out.aligned;
        let acc_t = Type::vector(&[tm as u64, tn as u64], elem);
        let row_t = Type::vector(&[tn as u64], elem);
        let a_row_t = Type::vector(&[chunk as u64], elem);
        let lhs_t = Type::vector(&[tm as u64, chunk as u64], elem);
        let rhs_t = Type::vector(&[chunk as u64, tn as u64], elem);

        // The accumulator starts from the current output values (the +=),
        // assembled into one TMxTN vector (zero seeds are fully overwritten
        // and fold away in lowering).
        let zero = self.zero_scalar(&body, elem)?;
        let mut acc = self.vec_broadcast(&body, zero, acc_t)?;
        for (i, mi) in ms.iter().enumerate() {
            if vec_out {
                let v = self.vec_load(&body, out.mem, &[*mi, n0], row_t)?;
                acc = self.vec_insert(&body, v, acc, &[i as i64])?;
            } else {
                for (j, nj) in ns.iter().enumerate() {
                    let e = self.push(&body, memref::load(out.mem, &[*mi, *nj], self.loc))?;
                    acc = self.vec_insert(&body, e, acc, &[i as i64, j as i64])?;
                }
            }
        }

        let k_lo = self.const_index(&body, 0)?;
        let k_st = self.const_index(&body, chunk)?;
        let finals =
            self.carry_loop(&body, k_lo, k_extent, k_st, &[acc], |cg, lblk, k, accs| {
                let zero = cg.zero_scalar(lblk, elem)?;
                let mut lhs = cg.vec_broadcast(lblk, zero, lhs_t)?;
                for (i, mi) in ms.iter().enumerate() {
                    if vec_a {
                        let v = cg.vec_load(lblk, a.mem, &[*mi, k], a_row_t)?;
                        lhs = cg.vec_insert(lblk, v, lhs, &[i as i64])?;
                    } else {
                        for j in 0..chunk {
                            let c = cg.const_index(lblk, j)?;
                            let kj = cg.addi(lblk, k, c)?;
                            let e = cg.load_as(lblk, a.mem, &[*mi, kj], elem)?;
                            lhs = cg.vec_insert(lblk, e, lhs, &[i as i64, j])?;
                        }
                    }
                }
                let mut rhs = cg.vec_broadcast(lblk, zero, rhs_t)?;
                for j in 0..chunk {
                    let c = cg.const_index(lblk, j)?;
                    let kj = cg.addi(lblk, k, c)?;
                    if vec_b {
                        let v = cg.vec_load(lblk, b.mem, &[kj, n0], row_t)?;
                        rhs = cg.vec_insert(lblk, v, rhs, &[j])?;
                    } else {
                        for (l, nl) in ns.iter().enumerate() {
                            let e = cg.load_as(lblk, b.mem, &[kj, *nl], elem)?;
                            rhs = cg.vec_insert(lblk, e, rhs, &[j, l as i64])?;
                        }
                    }
                }
                Ok(vec![cg.vec_contract(lblk, lhs, rhs, accs[0], false)?])
            })?;

        for (i, mi) in ms.iter().enumerate() {
            let row = self.vec_extract(&body, finals[0], &[i as i64], row_t)?;
            if vec_out {
                self.vec_store(&body, row, out.mem, &[*mi, n0])?;
            } else {
                for (j, nj) in ns.iter().enumerate() {
                    let e = self.vec_extract(&body, row, &[j as i64], elem)?;
                    body.append_operation(memref::store(e, out.mem, &[*mi, *nj], self.loc));
                }
            }
        }
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(lo, total, step, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }

    /// Largest register sub-tile extent that divides d.
    pub(super) fn sub_extent(d: i64) -> i64 {
        [4, 2].into_iter().find(|c| d % c == 0).unwrap_or(1)
    }

    /// Register sub-tile extents (TM, TN) for an mxn matmul output: the
    /// largest of 8x8 (4 MACs per shared element loaded) and 8x4 (2.67)
    /// whose sub-tile grid keeps at least one sub-tile per CTA thread (the
    /// @launch thread count); bigger lane tiles below that would idle
    /// threads instead of adding work per lane, else the legacy <=4 extents
    /// (2.0). (Every shape that selects 8x8 needs an m*n >= 128x128 output,
    /// whose shared acc tile exceeds the 48KB CTA budget on the unfused path
    /// -- those configs only ever launch through the register-accumulator
    /// fusion.)
    pub(super) fn sub_tile(&self, m: i64, n: i64) -> (i64, i64) {
        for (tm, tn) in [(8, 8), (8, 4)] {
            if m % tm == 0 && n % tn == 0 && (m / tm) * (n / tn) >= self.cta_threads {
                return (tm, tn);
            }
        }
        (Self::sub_extent(m), Self::sub_extent(n))
    }

    /// Lane grid (lm x ln, lm*ln = 32) for warp tiling: each warp owns an
    /// (lm*TM)x(ln*TN) warp tile with lanes laid out row-major inside it.
    /// Picks the factorization minimizing the warp's distinct shared reads
    /// per k-step (WM + WN, for WM*WN MACs), i.e. the most square warp tile.
    /// Ties break toward wider WN so the warp's b reads span one contiguous
    /// row segment. None when no factorization divides the sub-tile grid, in
    /// which case the flat per-thread distribution is used instead.
    pub(super) fn lane_grid(tiles_m: i64, tiles_n: i64, tm: i64, tn: i64) -> Option<(i64, i64)> {
        [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
            .into_iter()
            .filter(|&(lm, ln)| tiles_m % lm == 0 && tiles_n % ln == 0)
            .min_by_key(|&(lm, ln)| (lm * tm + ln * tn, lm))
    }

    /// Element-wise matmul fallback for dynamically-shaped outputs.
    pub(super) fn tile_matmul_dynamic(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        k_extent: Value<'c, 'c>,
    ) -> Result<()> {
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (m, n) = (idx[0], idx[1]);
            let init = cg.push(blk, memref::load(out.mem, &[m, n], cg.loc))?;
            let sums = cg.reduce_loop_multi(blk, k_extent, &[init], |cg, lblk, k, accs| {
                let x = cg.load_as(lblk, a.mem, &[m, k], out.elem)?;
                let y = cg.load_as(lblk, b.mem, &[k, n], out.elem)?;
                Ok(vec![cg.elem_mac(lblk, out.elem, x, y, accs[0])?])
            })?;
            blk.append_operation(memref::store(sums[0], out.mem, &[m, n], cg.loc));
            Ok(())
        })
    }

    /// scf.for k = 0..ub carrying inits as iter_args; returns the finals.
    pub(super) fn reduce_loop_multi(
        &mut self,
        block: &Block<'c>,
        ub: Value<'c, 'c>,
        inits: &[Value<'c, 'c>],
        body: impl FnOnce(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let zero = self.const_index(block, 0)?;
        let one = self.const_index(block, 1)?;
        self.carry_loop(block, zero, ub, one, inits, body)
    }

    /// scf.for iv = lo to hi step st carrying inits as iter_args.
    pub(super) fn carry_loop(
        &mut self,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        inits: &[Value<'c, 'c>],
        body: impl FnOnce(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let types: Vec<Type<'c>> = inits.iter().map(|v| v.r#type()).collect();
        let mut block_args = vec![(self.index_t, self.loc)];
        block_args.extend(types.iter().map(|&t| (t, self.loc)));
        let body_block = Block::new(&block_args);
        let iv = detach(body_block.argument(0)?.into());
        let mut accs = Vec::with_capacity(inits.len());
        for i in 0..inits.len() {
            accs.push(detach(body_block.argument(i + 1)?.into()));
        }
        let next = body(self, &body_block, iv, &accs)?;
        body_block.append_operation(scf::r#yield(&next, self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(inits);
        let op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&types)
                .add_regions([region])
                .build()?,
        );
        let mut finals = Vec::with_capacity(inits.len());
        for i in 0..inits.len() {
            finals.push(detach(op.result(i)?.into()));
        }
        Ok(finals)
    }

    /// One multiply-accumulate (acc + a*b) on the element type. Floats use
    /// math.fma, a single rounding that lowers to PTX fma.rn, because a
    /// separate mul/add pair emits explicitly-rounded mul.rn/add.rn, which
    /// ptxas is not allowed to contract into an FMA: the matmul would spend
    /// two instructions per MAC and halve its FLOP ceiling.
    pub(super) fn elem_mac(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        if self.is_float(elem) {
            self.push(
                block,
                OperationBuilder::new("math.fma", self.loc)
                    .add_operands(&[a, b, acc])
                    .add_results(&[elem])
                    .build()?,
            )
        } else {
            let prod = self.elem_arith(BinOp::Mul, elem, a, b)?;
            let prod = self.push(block, prod)?;
            let sum = self.elem_arith(BinOp::Add, elem, acc, prod)?;
            self.push(block, sum)
        }
    }

    /// The arith op for a tile loop body, on the element type.
    pub(super) fn elem_arith(
        &self,
        op: BinOp,
        elem: Type<'c>,
        a: Value<'c, '_>,
        b: Value<'c, '_>,
    ) -> Result<Operation<'c>> {
        let loc = self.loc;
        Ok(if self.is_float(elem) {
            match op {
                BinOp::Add => arith::addf(a, b, loc),
                BinOp::Sub => arith::subf(a, b, loc),
                BinOp::Mul => arith::mulf(a, b, loc),
                BinOp::Div => arith::divf(a, b, loc),
                BinOp::Rem => arith::remf(a, b, loc),
                _ => bail!("operator not supported for tile operands"),
            }
        } else if self.is_int(elem) {
            match op {
                BinOp::Add => arith::addi(a, b, loc),
                BinOp::Sub => arith::subi(a, b, loc),
                BinOp::Mul => arith::muli(a, b, loc),
                BinOp::Div => arith::divsi(a, b, loc),
                BinOp::Rem => arith::remsi(a, b, loc),
                _ => bail!("operator not supported for tile operands"),
            }
        } else {
            bail!("tile ops need a numeric element type, got {elem}")
        })
    }
}

// vector accesses (128-bit, for provably 16B-aligned buffers)
impl<'p, 'c> Codegen<'p, 'c> {
    /// vector.load mem[indices] : vector<4 x elem>, declared align-16.
    /// (Without the explicit attribute the lowering uses the element
    /// alignment, and the backend splits the access into scalars.)
    pub(super) fn vec_load(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        vec_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.vec_load_al(block, mem, indices, vec_t, 16)
    }

    /// vector.load with an explicit alignment (16 for 4xf32 accesses, 8 for
    /// the f16 staging's 4xf16 loads).
    pub(super) fn vec_load_al(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        vec_t: Type<'c>,
        align: i64,
    ) -> Result<Value<'c, 'c>> {
        let mut operands = vec![mem];
        operands.extend_from_slice(indices);
        self.push(
            block,
            OperationBuilder::new("vector.load", self.loc)
                .add_operands(&operands)
                .add_attributes(&[(
                    self.id("alignment"),
                    IntegerAttribute::new(self.i64_t, align).into(),
                )])
                .add_results(&[vec_t])
                .build()?,
        )
    }

    pub(super) fn vec_store(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<()> {
        self.vec_store_al(block, value, mem, indices, 16)
    }

    /// vector.store with an explicit alignment (16 for 4xf32 accesses,
    /// 8 for the f16 staging's 4xf16 stores).
    pub(super) fn vec_store_al(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
        align: i64,
    ) -> Result<()> {
        let mut operands = vec![value, mem];
        operands.extend_from_slice(indices);
        block.append_operation(
            OperationBuilder::new("vector.store", self.loc)
                .add_operands(&operands)
                .add_attributes(&[(
                    self.id("alignment"),
                    IntegerAttribute::new(self.i64_t, align).into(),
                )])
                .build()?,
        );
        Ok(())
    }

    /// Loads a scalar element and float-casts it to want (used by the
    /// mixed-precision matmul fallbacks: f16 operands accumulated in f32).
    pub(super) fn load_as(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        idx: &[Value<'c, 'c>],
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let v = self.push(block, memref::load(mem, idx, self.loc))?;
        self.float_cast(block, v, want)
    }

    /// Loads a scalar (width == 1) or a width-vector from mem[idx], matching
    /// the vectorization width the surrounding tile op chose.
    pub(super) fn elem_load(
        &self,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        idx: &[Value<'c, 'c>],
        width: i64,
        vec_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        if width > 1 {
            self.vec_load(block, mem, idx, vec_t)
        } else {
            self.push(block, memref::load(mem, idx, self.loc))
        }
    }

    /// The store counterpart of [`Self::elem_load`].
    pub(super) fn elem_store(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        mem: Value<'c, 'c>,
        idx: &[Value<'c, 'c>],
        width: i64,
    ) -> Result<()> {
        if width > 1 {
            self.vec_store(block, value, mem, idx)?;
        } else {
            block.append_operation(memref::store(value, mem, idx, self.loc));
        }
        Ok(())
    }

    pub(super) fn vec_broadcast(
        &self,
        block: &Block<'c>,
        scalar: Value<'c, 'c>,
        vec_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("vector.broadcast", self.loc)
                .add_operands(&[scalar])
                .add_results(&[vec_t])
                .build()?,
        )
    }

    pub(super) fn vec_extract(
        &self,
        block: &Block<'c>,
        vector: Value<'c, 'c>,
        positions: &[i64],
        result: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[vector])
                .add_attributes(&[(self.id("static_position"), self.i64_array(positions)?)])
                .add_results(&[result])
                .build()?,
        )
    }

    pub(super) fn vec_insert(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        dest: Value<'c, 'c>,
        positions: &[i64],
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("vector.insert", self.loc)
                .add_operands(&[value, dest])
                .add_attributes(&[(self.id("static_position"), self.i64_array(positions)?)])
                .add_results(&[dest.r#type()])
                .build()?,
        )
    }

    /// Widens a float vector to want (a wider-element vector of the same
    /// shape) with elementwise arith.extf.
    pub(super) fn vec_extf(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("arith.extf", self.loc)
                .add_operands(&[value])
                .add_results(&[want])
                .build()?,
        )
    }

    /// Rounds a float vector down to want (a narrower-element vector of the
    /// same shape) with elementwise arith.truncf.
    pub(super) fn vec_truncf(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("arith.truncf", self.loc)
                .add_operands(&[value])
                .add_results(&[want])
                .build()?,
        )
    }

    /// vector.contract accumulating lhs * rhs into acc over the chunk
    /// dimension k. With lhs_k_major the maps are {(k, m), (k, n) -> (m, n)},
    /// the fused path's layout (k-major a staging), whose outer-product
    /// lowering needs no transposes; otherwise {(m, k), (k, n) -> (m, n)} (the
    /// unfused path's m-major a; the lowering's lhs transpose folds away at
    /// constant positions).
    pub(super) fn vec_contract(
        &self,
        block: &Block<'c>,
        lhs: Value<'c, 'c>,
        rhs: Value<'c, 'c>,
        acc: Value<'c, 'c>,
        lhs_k_major: bool,
    ) -> Result<Value<'c, 'c>> {
        let lhs_map = if lhs_k_major { "(d2, d0)" } else { "(d0, d2)" };
        let maps = self.parse_attr(&format!(
            "[affine_map<(d0, d1, d2) -> {lhs_map}>, \
              affine_map<(d0, d1, d2) -> (d2, d1)>, \
              affine_map<(d0, d1, d2) -> (d0, d1)>]",
        ))?;
        let iters = self.parse_attr(
            "[#vector.iterator_type<parallel>, #vector.iterator_type<parallel>, \
              #vector.iterator_type<reduction>]",
        )?;
        let kind = self.parse_attr("#vector.kind<add>")?;
        self.push(
            block,
            OperationBuilder::new("vector.contract", self.loc)
                .add_operands(&[lhs, rhs, acc])
                .add_attributes(&[
                    (self.id("indexing_maps"), maps),
                    (self.id("iterator_types"), iters),
                    (self.id("kind"), kind),
                ])
                .add_results(&[acc.r#type()])
                .build()?,
        )
    }

    /// A gpu index op (gpu.thread_id / gpu.block_id / gpu.block_dim).
    pub(super) fn gpu_index(
        &self,
        block: &Block<'c>,
        op: &str,
        dim: &str,
    ) -> Result<Value<'c, 'c>> {
        // Bracket form required for gpu::DimensionAttr.
        let attr = self.parse_attr(&format!("#gpu<dim {dim}>"))?;
        self.push(
            block,
            OperationBuilder::new(op, self.loc)
                .add_attributes(&[(self.id("dimension"), attr)])
                .add_results(&[self.index_t])
                .build()?,
        )
    }
}
