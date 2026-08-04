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

        // a dynamic buffer names its address space symbolically and a view of
        // it has to agree; a static global uses the integer form.
        let space: Attribute = if self.dynamic_shared {
            Attribute::parse(self.ctx, MEM_SHARED_SYM).context("workgroup address space")?
        } else {
            IntegerAttribute::new(self.i64_t, MEM_SHARED).into()
        };

        let t = MemRefType::new(elem, shape, None, Some(space));

        // Reuse a released buffer of the same type when one is free (see
        // release); otherwise mint a new one.
        let key = (elem.to_string(), shape.to_vec());
        let name = match self.tile_pool.get_mut(&key).and_then(Vec::pop) {
            Some(name) => name,
            None => {
                let name = format!("__{}_tile{}", self.kernel_name, self.tile_count);
                self.tile_count += 1;

                if self.dynamic_shared {
                    // each tile is a window of the one allocation and 16-byte
                    // aligned so a four-element vector access stays legal.
                    let width = self
                        .elem_bytes(elem)
                        .with_context(|| format!("tile element {elem} has no known width"))?;
                    
                    let bytes = i64::from(width) * shape.iter().product::<i64>();
                    
                    self.tile_offsets.insert(name.clone(), self.shared_bytes);

                    self.shared_bytes += (bytes + 15) & !15;
                } else {
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
                }
                name
            }
        };

        let mem = if self.dynamic_shared {
            let offset = self.tile_offsets[&name];
            let byte_t = MemRefType::new(self.i8_t, &[DYN], None, Some(space));
            let base = self.push(
                block,
                OperationBuilder::new("gpu.dynamic_shared_memory", self.loc)
                    .add_results(&[byte_t.into()])
                    .build()?,
            )?;
            let at = self.const_index(block, offset)?;

            self.push(
                block,
                OperationBuilder::new("memref.view", self.loc)
                    .add_operands(&[base, at])
                    .add_results(&[t.into()])
                    .build()?,
            )?
        } else {
            self.push(block, memref::get_global(self.ctx, &name, t, self.loc))?
        };

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
            mask: Vec::new(),
            dim_div: Vec::new(),
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

        // the acc has to hold both operands. specifically their join, or wider of
        // the same kind.
        //
        // equal width is not enough since f16 and bf16 are both 16b and neither holds the other.
        // 
        // int8 contraction accumulates into i32.
        let holds = self.numeric_join(a.elem, b.elem).is_some_and(|join| {
            if join == out.elem {
                return true;
            }
            match (self.is_float(join), self.is_float(out.elem)) {
                (true, true) => self.float_bits(out.elem) > self.float_bits(join),
                (false, false) => self.elem_bytes(out.elem) > self.elem_bytes(join),
                // nn integer contraction into a float accumulator, or the
                // reverse, would silently change what the sum means
                _ => false,
            }
        });

        if holds {
            Ok(())
        } else {
            bail!(
                "{what}: cannot accumulate {} and {} operands into {}",
                a.elem,
                b.elem,
                out.elem
            )
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
    /// Conjunction of `offset + idx[d] < extent` over the masked dims of a
    /// tensor slice, or None when the mask is empty (every dim in bounds).
    /// The offset and extent values were materialized where the slice was
    /// taken, so they dominate the distributed loop body that calls this.
    pub(super) fn bounds_pred(
        &self,
        block: &Block<'c>,
        mask: &[Option<(Value<'c, 'c>, Value<'c, 'c>)>],
        idx: &[Value<'c, 'c>],
    ) -> Result<Option<Value<'c, 'c>>> {
        let mut pred: Option<Value<'c, 'c>> = None;

        for (d, entry) in mask.iter().enumerate() {
            let Some((off, extent)) = entry else { continue };
            let global = self.addi(block, *off, idx[d])?;
            let in_bounds = self.push(
                block,
                arith::cmpi(
                    self.ctx,
                    arith::CmpiPredicate::Ult,
                    global,
                    *extent,
                    self.loc,
                ),
            )?;

            pred = Some(match pred {
                Some(p) => self.push(block, arith::andi(p, in_bounds, self.loc))?,
                None => in_bounds,
            });
        }

        Ok(pred)
    }

    /// Stages a partially out-of-bounds slice into a fresh, fully in-bounds
    /// tile: in-bounds elements are copied, out-of-bounds ones read as zero.
    /// The load index is clamped to 0 on any masked dim that overflows, so no
    /// access ever leaves the tensor, then a select substitutes zero for the
    /// clamped reads. Downstream ops then treat the result as an ordinary
    /// dense tile.
    pub(super) fn materialize_masked(
        &mut self,
        block: &Block<'c>,
        view: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if view.shape.contains(&DYN) {
            bail!("a masked tensor slice needs a static shape");
        }

        let dst = self.alloc_tile_shaped(block, view.elem, &view.shape)?;
        let mask = view.mask.clone();
        let src = view.mem;
        let dst_mem = dst.mem;
        let elem = view.elem;

        self.distribute(block, &dst, 1, true, |cg, blk, idx| {
            let zero_idx = cg.const_index(blk, 0)?;
            let mut safe = idx.to_vec();
            let mut pred: Option<Value<'c, 'c>> = None;

            for (d, entry) in mask.iter().enumerate() {
                let Some((off, extent)) = entry else { continue };
                let global = cg.addi(blk, *off, idx[d])?;
                let in_bounds = cg.push(
                    blk,
                    arith::cmpi(cg.ctx, arith::CmpiPredicate::Ult, global, *extent, cg.loc),
                )?;

                safe[d] = cg.select(blk, in_bounds, idx[d], zero_idx)?;
                pred = Some(match pred {
                    Some(p) => cg.push(blk, arith::andi(p, in_bounds, cg.loc))?,
                    None => in_bounds,
                });
            }

            let loaded = cg.push(blk, memref::load(src, &safe, cg.loc))?;
            let val = match pred {
                Some(p) => {
                    let zero = cg.zero_scalar(blk, elem)?;
                    cg.select(blk, p, loaded, zero)?
                }
                None => loaded,
            };

            blk.append_operation(memref::store(val, dst_mem, idx, cg.loc));

            Ok(())
        })?;

        Ok(dst)
    }

    /// arith.select(cond, a, b): a when cond is true, else b.
    pub(super) fn select(
        &self,
        block: &Block<'c>,
        cond: Value<'c, 'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::select(cond, a, b, self.loc))
    }

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
                idx[i] = self.remui(&body_block, rem, sizes[i])?;
                rem = self.divui(&body_block, rem, sizes[i])?;
            }

            idx[0] = rem;
        }

        if width > 1 {
            let w = self.const_index(&body_block, width)?;
            idx[rank - 1] = self.muli(&body_block, idx[rank - 1], w)?;
        }

        // A masked output writes only the in-bounds elements: the whole body
        // (its loads and its store) runs under an scf.if guarding offset +
        // local index < extent. Callers scalarize masked writes
        // (elementwise_width and tile_copy return width 1), so the guard is
        // exact per element. The trailing barrier stays outside the guard: it
        // is CTA-uniform, while the guard is a per-element (non-uniform) test.
        if let Some(pred) = self.bounds_pred(&body_block, &out.mask, &idx)? {
            let then = Block::new(&[]);
            body(self, &then, &idx)?;
            then.append_operation(scf::r#yield(&[], self.loc));
            let then_region = Region::new();
            then_region.append_block(then);
            body_block.append_operation(scf::r#if(pred, &[], then_region, Region::new(), self.loc));
        } else {
            body(self, &body_block, &idx)?;
        }

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
        // A masked buffer is scalarized so the per-element store guard is
        // exact (a partial vector could straddle the bounds).
        if mvs.iter().any(|m| m.is_masked()) {
            return 1;
        }
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

        // Vectorize aligned copies at 4 elements: 16B/4xf32, 8B/4xf16 or
        // 4xbf16, 4B/4xi8 (the src and dst element types match, checked
        // above). The row-pitch ABI's multiple-of-4-elements guarantee is
        // exactly the byte alignment a 4-element vector of any of them needs.
        let elem_bytes = self.elem_bytes(dst.elem);
        let last = *dst.shape.last().expect("tile values are not rank-0");
        let vec_ok = src.aligned
            && dst.aligned
            && !src.is_masked()
            && !dst.is_masked()
            && last != DYN
            && last % 4 == 0
            && elem_bytes.is_some();
        let width = if vec_ok { 4 } else { 1 };
        let align = i64::from(elem_bytes.unwrap_or(4)) * 4;

        // cp.async needs a 4/8/16-byte transfer and can't convert (so the
        // src/dst element types must match, as checked above): f32 qualifies
        // at any width (4B scalar or 16B vector), the narrower types only
        // vectorized, since a scalar 1B or 2B element is below cp.async's
        // minimum.
        let use_async = async_copy
            && !dst.is_masked()
            && (dst.elem == self.f32_t || (width == 4 && matches!(align, 4 | 8 | 16)));
        let vec_t = Type::vector(&[4], dst.elem);

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
    /// Converts between the source and destination element types, e.g. an f32
    /// accumulator stored into an f16 or i8 output tensor. Not vectorized.
    pub(super) fn tile_convert(
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
            let x = cg.load_as(blk, a.mem, &ai, out.elem)?;
            let y = cg.load_as(blk, b.mem, &bi, out.elem)?;
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

    /// out = sqrt(src), element-wise. Mirrors [`Self::tile_exp`]: an owned temp
    /// is rewritten in place, otherwise a fresh tile is allocated.
    pub(super) fn tile_sqrt(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
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
    pub(super) fn tile_cast(
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
    pub(super) fn tile_sqrt_into(
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

    /// out = log(src), element-wise. Mirrors [`Self::tile_exp`].
    pub(super) fn tile_log(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
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
    pub(super) fn tile_log_into(
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

    /// ln(x) for an f32 scalar as lg2(x) * ln2, the mirror of
    /// [`Self::approx_exp`]: the hardware primitive is base two, so the change
    /// of base rides on the outside.
    pub(super) fn approx_log(&self, block: &Block<'c>, x: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let lg2 = self.push(
            block,
            OperationBuilder::new("llvm.inline_asm", self.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        self.id("asm_string"),
                        StringAttribute::new(self.ctx, "lg2.approx.ftz.f32 $0, $1;").into(),
                    ),
                    (
                        self.id("constraints"),
                        StringAttribute::new(self.ctx, "=f,f").into(),
                    ),
                    (self.id("has_side_effects"), Attribute::unit(self.ctx)),
                ])
                .add_results(&[self.f32_t])
                .build()?,
        )?;

        let ln2 = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, std::f64::consts::LN_2).into(),
                self.loc,
            ),
        )?;
        
        self.push(block, arith::mulf(lg2, ln2, self.loc))
    }

    /// out = round(src), element-wise. Mirrors [`Self::tile_exp`].
    pub(super) fn tile_round(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
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
    pub(super) fn tile_round_into(
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

    /// The nearest integer to an f32, ties to even, as an f32.
    ///
    /// This is the hardware's own rounding, so unlike biasing into a positive
    /// range and truncating it loses nothing: adding a bias large enough to
    /// cover the range costs the low mantissa bits, which at the top of an
    /// int8 quantization range is enough to move a value across a rounding
    /// boundary.
    pub(super) fn round_even(&self, block: &Block<'c>, x: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("llvm.inline_asm", self.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        self.id("asm_string"),
                        StringAttribute::new(self.ctx, "cvt.rni.f32.f32 $0, $1;").into(),
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

    pub(super) fn approx_sqrt(&self, block: &Block<'c>, x: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("llvm.inline_asm", self.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        self.id("asm_string"),
                        StringAttribute::new(self.ctx, "sqrt.approx.f32 $0, $1;").into(),
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

    /// out = tanh(src), element-wise. Mirrors [`Self::tile_sqrt`].
    pub(super) fn tile_tanh(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
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
    pub(super) fn tile_tanh_into(
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

    /// tanh(x) for an f32 scalar via the PTX tanh.approx.f32 intrinsic (sm_75+),
    /// same inline-asm approach as [`Self::approx_exp`] / [`Self::approx_sqrt`].
    pub(super) fn approx_tanh(&self, block: &Block<'c>, x: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("llvm.inline_asm", self.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        self.id("asm_string"),
                        StringAttribute::new(self.ctx, "tanh.approx.f32 $0, $1;").into(),
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
                let floor = if elem == self.f16_t {
                    -65504.0
                } else {
                    -3.0e38
                };
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

    /// `dot_t` over int8 operands on the integer tensor cores.
    ///
    /// Unlike the f16 tensor-core path there is no staging buffer and no
    /// ldmatrix: the m8n8k16 fragment layout is already what `dot_t` holds in
    /// memory. A lane's A register is four contiguous bytes of row `lane / 4`,
    /// its B register is four contiguous bytes of row `lane / 4` of the [n, k]
    /// operand, and both are exactly the four bytes `dp4a` would have read.
    /// One `mma.sync` folds sixteen products per lane where `dp4a` folds four.
    ///
    /// Returns false when it does not apply, leaving the caller on the dp4a
    /// path: the tensor core issues whole 8x8 output tiles over a k that is a
    /// multiple of 16, and there are no integer tensor cores before Turing. A
    /// masked output would need the store guarded per element, which is what
    /// the generic paths already do.
    fn tile_matmul_t_imma(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        kd: i64,
    ) -> Result<bool> {
        let (md, nd) = (out.shape[0], out.shape[1]);
        let applies = a.elem == self.i8_t
            && b.elem == self.i8_t
            && out.elem == self.i32_t
            // Positive as well as divisible: DYN is i64::MIN, which every one
            // of these divides.
            && md > 0
            && nd > 0
            && md % 8 == 0
            && nd % 8 == 0
            && kd % 16 == 0
            && a.swizzle.is_none()
            && b.swizzle.is_none()
            && !a.is_masked()
            && !b.is_masked()
            && !out.is_masked()
            && self.base.gpu_config.supports_int8_mma();
        if !applies {
            return Ok(false);
        }

        let i32_t = self.i32_t;
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);

        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;
        let warp_size = self.const_index(block, 32)?;
        let warp = self.divui(block, tid, warp_size)?;
        let warps = self.divui(block, bdim, warp_size)?;
        let lane = self.remui(block, tid, warp_size)?;

        // The lane's place in the fragments: both operands are read from row
        // lane / 4, four bytes starting at column 4 * (lane % 4), and the two
        // accumulator elements land in columns 2 * (lane % 4) and one past.
        let four = self.const_index(block, 4)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let two = self.const_index(block, 2)?;
        let d_col = self.muli(block, in_quad, two)?;

        // One 8x8 output tile per warp, row-major so neighbouring warps share
        // the A rows they read.
        let eight = self.const_index(block, 8)?;
        let n_tiles = self.const_index(block, nd / 8)?;
        let total = self.const_index(block, (md / 8) * (nd / 8))?;

        let tb = Block::new(&[(self.index_t, self.loc)]);
        let t = detach(tb.argument(0)?.into());
        let ti = self.divui(&tb, t, n_tiles)?;
        let tj = self.remui(&tb, t, n_tiles)?;
        let i0 = self.muli(&tb, ti, eight)?;
        let j0 = self.muli(&tb, tj, eight)?;
        let a_row = self.addi(&tb, i0, quad)?;
        let b_row = self.addi(&tb, j0, quad)?;

        let zero = self.zero_scalar(&tb, i32_t)?;
        let init = self.vec_broadcast(&tb, zero, acc_t)?;
        let lo = self.const_index(&tb, 0)?;
        let hi = self.const_index(&tb, kd)?;
        let st = self.const_index(&tb, 16)?;

        let kb = Block::new(&[(self.index_t, self.loc), (acc_t, self.loc)]);
        let k = detach(kb.argument(0)?.into());
        let acc = detach(kb.argument(1)?.into());
        let k_col = self.addi(&kb, k, k_off)?;
        let va = self.vec_load_al(&kb, a.mem, &[a_row, k_col], vec4_i8, 4)?;
        let vb = self.vec_load_al(&kb, b.mem, &[b_row, k_col], vec4_i8, 4)?;
        let va = self.vec_shape_cast(&kb, va, frag_t)?;
        let vb = self.vec_shape_cast(&kb, vb, frag_t)?;
        let shape = self.parse_attr("[8, 8, 16]")?;
        let next = self.push(
            &kb,
            OperationBuilder::new("nvgpu.mma.sync", self.loc)
                .add_operands(&[va, vb, acc])
                .add_attributes(&[(self.id("mmaShape"), shape)])
                .add_results(&[acc_t])
                .build()?,
        )?;
        kb.append_operation(scf::r#yield(&[next], self.loc));
        let k_region = Region::new();
        k_region.append_block(kb);
        let fin = self.push(
            &tb,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lo, hi, st, init])
                .add_results(&[acc_t])
                .add_regions([k_region])
                .build()?,
        )?;

        let out_col = self.addi(&tb, j0, d_col)?;
        for dj in 0..2 {
            let e = self.vec_extract(&tb, fin, &[0, dj], i32_t)?;
            let off = self.const_index(&tb, dj)?;
            let col = self.addi(&tb, out_col, off)?;
            tb.append_operation(memref::store(e, out.mem, &[a_row, col], self.loc));
        }
        tb.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(tb);
        block.append_operation(scf::r#for(warp, total, warps, region, self.loc));
        self.barrier(block)?;
        Ok(true)
    }

    /// `dot_t` over int8 operands using the hardware four-way byte dot product.
    ///
    /// `dp4a` multiplies four int8 pairs and accumulates into an i32 in one
    /// instruction, so this replaces four loads, four multiplies and four adds
    /// per step with one vector load per operand and one instruction. It needs
    /// the four bytes of each operand contiguous, which is why it lands on
    /// `dot_t` and not `dot`: `dot_t` contracts the last axis of both operands,
    /// so both walk memory contiguously.
    ///
    /// Returns false when it does not apply, leaving the caller on the generic
    /// path: below Pascal there is no `dp4a`, and a contraction that is not a
    /// multiple of four bytes has a remainder this does not handle.
    /// out[i, j] = sum_b (sum_{k in block b} a[i, k] * w[j, k]) * asc[i, b] * wsc[j, b]:
    /// the whole Q8_0 contraction, block scales included, as one operation.
    ///
    /// This exists because `dot_t` cannot be given enough of `k` at a time. A
    /// Q8_0 block carries its own scale, so a plain dot has to stop every 32
    /// elements to apply it, and `dot_t` puts one thread on each output and
    /// walks `k` in that thread. A warp then reads 32 rows four bytes apart,
    /// which is 32 sectors fetched to use 128 bytes of them, and the block
    /// pays five barriers per 32 elements of `k`. 
    /// 
    /// Folding the scales in is what lets the mapping turn around: a warp owns
    /// one output and its lanes divide `k`, so the 32 lanes read 512
    /// contiguous bytes of one weight row. Nothing is staged, the accumulator
    /// is a register, and the only synchronization is the closing butterfly
    /// shuffle. Each lane takes 16 bytes, which is four `dp4a` under one scale
    /// pair, since 16 divides the 32-element block.
    ///
    /// The scales are indexed `[row, block]` so a lane's scale load is
    /// contiguous with its neighbours'. Reading them `[block, row]`, the
    /// layout the tensor-core kernel wants, would cost one sector per lane and
    /// double the traffic.
    pub(super) fn tile_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "qdot_t a"),
            (asc, "qdot_t a scales"),
            (w, "qdot_t w"),
            (wsc, "qdot_t w scales"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.i8_t || w.elem != self.i8_t {
            bail!("qdot_t contracts int8 operands");
        }
        if asc.elem != self.f32_t || wsc.elem != self.f32_t {
            bail!("qdot_t scales must be f32");
        }
        if !self.base.gpu_config.supports_dp4a() {
            bail!("qdot_t needs dp4a (sm_61 or later)");
        }
        let (rows, cols) = (a.shape[0], w.shape[0]);
        if rows == DYN || cols == DYN {
            bail!("qdot_t needs a static output shape");
        }
        self.check_shapes(&[rows], &[asc.shape[0]], "qdot_t a scale rows")?;
        self.check_shapes(&[cols], &[wsc.shape[0]], "qdot_t w scale rows")?;
        if self.cta_threads % WARP != 0 {
            bail!("qdot_t needs a CTA that is a whole number of warps");
        }

        let out = self.alloc_tile_shaped(block, self.f32_t, &[rows, cols])?;
        let (i32_t, f32_t, vec4_i8) = (self.i32_t, self.f32_t, Type::vector(&[4], self.i8_t));

        // The contraction length: static when the slice pinned it, otherwise
        // the operand's own extent.
        let one = self.const_index(block, 1)?;
        let kd = if a.shape[1] == DYN {
            self.push(block, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };

        let lane_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, rows * cols * WARP)?;
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;

        // A warp per output element: the CTA size is a warp multiple and so is
        // `total`, so a warp is either wholly inside this loop or wholly
        // outside it and every lane reaches the shuffle.
        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let unit = self.divui(&body, li, lane_w)?;
        let lane = self.remui(&body, li, lane_w)?;
        let ncols = self.const_index(&body, cols)?;
        let i = self.divui(&body, unit, ncols)?;
        let j = self.remui(&body, unit, ncols)?;

        let step = self.const_index(&body, QDOT_STEP)?;
        let lane_bytes = self.const_index(&body, QDOT_LANE)?;
        let lane_off = self.muli(&body, lane, lane_bytes)?;
        let zero_k = self.const_index(&body, 0)?;
        let init = self.zero_scalar(&body, f32_t)?;

        let kb = Block::new(&[(self.index_t, self.loc), (f32_t, self.loc)]);
        let base = detach(kb.argument(0)?.into());
        let carry = detach(kb.argument(1)?.into());
        let koff = self.addi(&kb, base, lane_off)?;
        // A lane's chunk divides the Q8_0 block, so it is wholly in or wholly
        // out and one predicate covers it.
        let live = self.push(
            &kb,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, koff, kd, self.loc),
        )?;

        let then = Block::new(&[]);
        let mut dots = self.zero_scalar(&then, i32_t)?;
        let signed = self.parse_attr("#nvvm.dot_accumulate_type<signed>")?;
        for c in 0..QDOT_LANE / 4 {
            let at = self.const_index(&then, c * 4)?;
            let k = self.addi(&then, koff, at)?;
            let va = self.vec_load_al(&then, a.mem, &[i, k], vec4_i8, 4)?;
            let vb = self.vec_load_al(&then, w.mem, &[j, k], vec4_i8, 4)?;
            dots = self.push(
                &then,
                OperationBuilder::new("nvvm.dot.accumulate.4way", self.loc)
                    .add_operands(&[va, vb, dots])
                    .add_attributes(&[(self.id("a_type"), signed), (self.id("b_type"), signed)])
                    .add_results(&[i32_t])
                    .build()?,
            )?;
        }
        let blk_w = self.const_index(&then, Q8_BLOCK)?;
        let b = self.divui(&then, koff, blk_w)?;
        let sa = self.push(&then, memref::load(asc.mem, &[i, b], self.loc))?;
        let sw = self.push(&then, memref::load(wsc.mem, &[j, b], self.loc))?;
        let as_f = self.numeric_cast(&then, dots, f32_t)?;
        let scaled = self.push(&then, arith::mulf(as_f, sa, self.loc))?;
        let scaled = self.push(&then, arith::mulf(scaled, sw, self.loc))?;
        let summed = self.push(&then, arith::addf(carry, scaled, self.loc))?;
        then.append_operation(scf::r#yield(&[summed], self.loc));

        let otherwise = Block::new(&[]);
        otherwise.append_operation(scf::r#yield(&[carry], self.loc));

        let (tr, er) = (Region::new(), Region::new());
        tr.append_block(then);
        er.append_block(otherwise);
        let next = self.push(&kb, scf::r#if(live, &[f32_t], tr, er, self.loc))?;
        kb.append_operation(scf::r#yield(&[next], self.loc));

        let kr = Region::new();
        kr.append_block(kb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[zero_k, kd, step, init])
                .add_results(&[f32_t])
                .add_regions([kr])
                .build()?,
        )?;

        let mut mask = WARP / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = self.push(&body, arith::addf(acc, other, self.loc))?;
            mask /= 2;
        }

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, out.mem, &[i, j], self.loc));
        store.append_operation(scf::r#yield(&[], self.loc));
        let sr = Region::new();
        sr.append_block(store);
        body.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(out)
    }

    /// A small signed integer as an f32, without the conversion instruction.
    ///
    /// Adding 1.5 * 2^23 to `value` as an integer lands it in the mantissa of
    /// that float, so the bits are already the f32 of `1.5 * 2^23 + value` and
    /// subtracting the constant back off leaves the value exactly. It holds for
    /// `|value| < 2^22`, which a Q8_0 block guarantees: 32 products of two
    /// int8s cannot exceed 32 * 127 * 127, about an eighth of the room.
    ///
    /// This is worth doing rather than a `cvt` because on Turing conversions
    /// issue at a quarter of the arithmetic rate, one per eight cycles against
    /// one per cycle, and the quantized matmul does one per accumulator per
    /// block: at a patch's size that is 128 of them against the same block's
    /// 128 tensor instructions, which run one per four cycles.
    fn small_int_to_f32(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        const MAGIC_BITS: i64 = 0x4B40_0000;
        const MAGIC: f64 = 12_582_912.0;
        let bias = self.push(
            block,
            arith::constant(
                self.ctx,
                IntegerAttribute::new(self.i32_t, MAGIC_BITS).into(),
                self.loc,
            ),
        )?;
        let shifted = self.addi(block, value, bias)?;
        let bits = self.push(block, arith::bitcast(shifted, self.f32_t, self.loc))?;
        let magic = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, MAGIC).into(),
                self.loc,
            ),
        )?;
        self.push(block, arith::subf(bits, magic, self.loc))
    }

    /// The warp's patch of `m8n8k16` tiles: how many down, how many across.
    ///
    /// Grown from one tile a side, alternating so the patch stays square, and
    /// stopped when the patch would leave warps of the CTA with nothing to do.
    /// A square patch is what makes the operand loads pay: `rm` by `rn` tiles
    /// issue `2 * rm * rn` tensor instructions against `2 * (rm + rn)` loads.
    fn qmma_patch(&self, rt: i64, ct: i64) -> (i64, i64) {
        let warps = self.cta_threads / WARP;
        let (mut rm, mut rn) = (1, 1);
        while rm * rn < QMMA_TILES {
            let grow_rows = rm < rn;
            let (try_m, try_n) = if grow_rows {
                (rm * 2, rn)
            } else {
                (rm, rn * 2)
            };
            let fits = rt % try_m == 0 && ct % try_n == 0;
            if !fits || (rt / try_m) * (ct / try_n) < warps {
                // The other direction may still have room.
                let (alt_m, alt_n) = if grow_rows {
                    (rm, rn * 2)
                } else {
                    (rm * 2, rn)
                };
                if alt_m * alt_n <= QMMA_TILES
                    && rt % alt_m == 0
                    && ct % alt_n == 0
                    && (rt / alt_m) * (ct / alt_n) >= warps
                {
                    (rm, rn) = (alt_m, alt_n);
                    continue;
                }
                break;
            }
            (rm, rn) = (try_m, try_n);
        }
        (rm, rn)
    }

    /// out[i, j] = sum_b (sum_{k in block b} a[i, k] * w[j, k]) * asc[i, b] * wsc[b, j]:
    /// the batched Q8_0 contraction on the integer tensor cores, block scales
    /// included, as one operation.
    ///
    /// This is to a prompt pass what [`Self::tile_qdot_t`] is to a decode step,
    /// and it exists for the same reason. Written in the tile language the
    /// contraction has to stop every 32 elements of `k` to apply the scales,
    /// which puts the accumulator in shared memory and stages both operands
    /// there per block: for a `[64, 64]` tile the accumulator alone is 16 KB,
    /// so the tile cannot even be built.
    ///
    /// Folding the scales in lets the whole of `k` stay inside one operation,
    /// so the accumulators are registers and live across it, both operands are
    /// read straight from global memory in the layout the `m8n8k16` fragments
    /// already want, and there is no barrier in the loop at all.
    ///
    /// The weight scales are indexed `[block, out]` here and `[out, block]` in
    /// `qdot_t`, which is not an inconsistency: a lane of this kernel holds two
    /// neighbouring output columns of one block, so `[block, out]` puts its two
    /// scales next to each other and a warp's eight columns in one sector.
    ///
    /// Like `qdot_t` this assumes `k` is a whole number of Q8_0 blocks, which
    /// is what the format guarantees.
    pub(super) fn tile_qmma_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let (md, nd) = (a.shape[0], w.shape[0]);
        if md == DYN || nd == DYN {
            bail!("qmma_t needs a static output shape");
        }
        let out = self.alloc_tile_shaped(block, self.f32_t, &[md, nd])?;
        self.qmma_t_into(block, a, asc, w, wsc, &out)?;
        Ok(out)
    }

    /// [`Self::tile_qmma_t`] writing an existing destination rather than a
    /// fresh tile.
    ///
    /// The destination is normally a slice of the output tensor, which is what
    /// makes this worth having: the accumulators are already in registers, and
    /// going through a shared tile on the way out costs a `[128, 64]` f32
    /// buffer, which is 32 KB and holds the kernel to one CTA per
    /// multiprocessor. Writing global directly leaves the occupancy to the
    /// register file.
    pub(super) fn qmma_t_into(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if out.elem != self.f32_t {
            bail!("qmma_t produces f32");
        }
        if out.is_masked() {
            bail!("qmma_t needs a fully in-bounds destination");
        }
        for (v, what) in [
            (a, "qmma_t a"),
            (asc, "qmma_t a scales"),
            (w, "qmma_t w"),
            (wsc, "qmma_t w scales"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
            if v.swizzle.is_some() {
                bail!("{what} must not be swizzled");
            }
        }
        if a.elem != self.i8_t || w.elem != self.i8_t {
            bail!("qmma_t contracts int8 operands");
        }
        if asc.elem != self.f32_t || wsc.elem != self.f32_t {
            bail!("qmma_t scales must be f32");
        }
        if !self.base.gpu_config.supports_int8_mma() {
            bail!("qmma_t needs the integer tensor cores (sm_75 or later)");
        }
        let (md, nd) = (a.shape[0], w.shape[0]);
        if md == DYN || nd == DYN {
            bail!("qmma_t needs a static output shape");
        }
        self.check_shapes(&[md, nd], &out.shape, "qmma_t destination")?;
        if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
            bail!("qmma_t needs an output tile that is a multiple of {IMMA_TILE} both ways");
        }
        self.check_shapes(&[md], &[asc.shape[0]], "qmma_t a scale rows")?;
        self.check_shapes(&[nd], &[wsc.shape[1]], "qmma_t w scale columns")?;
        if self.cta_threads % WARP != 0 {
            bail!("qmma_t needs a CTA that is a whole number of warps");
        }

        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);

        let one = self.const_index(block, 1)?;
        let kd = if a.shape[1] == DYN {
            self.push(block, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };

        // The lane's place in the fragments, as in [`Self::tile_matmul_t_imma`]:
        // both operands are read from row lane / 4 four bytes on, and the two
        // accumulator elements land in columns 2 * (lane % 4) and one past.
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let two = self.const_index(block, 2)?;
        let d_col = self.muli(block, in_quad, two)?;

        let (rt, ct) = (md / IMMA_TILE, nd / IMMA_TILE);
        let (rm, rn) = self.qmma_patch(rt, ct);
        let patches = (rt / rm) * (ct / rn);
        let warps = self.const_index(block, self.cta_threads / WARP)?;
        let total = self.const_index(block, patches)?;
        let across = self.const_index(block, ct / rn)?;

        let tb = Block::new(&[(self.index_t, self.loc)]);
        let u = detach(tb.argument(0)?.into());
        let ui = self.divui(&tb, u, across)?;
        let uj = self.remui(&tb, u, across)?;
        let patch_m = self.const_index(&tb, rm * IMMA_TILE)?;
        let patch_n = self.const_index(&tb, rn * IMMA_TILE)?;
        let i0 = self.muli(&tb, ui, patch_m)?;
        let j0 = self.muli(&tb, uj, patch_n)?;
        let row_base = self.addi(&tb, i0, quad)?;
        let col_base = self.addi(&tb, j0, d_col)?;
        let frag_base = self.addi(&tb, j0, quad)?;

        // Row of A and of the accumulator, row of W, and output column, one per
        // tile of the patch. All are loop-invariant, so they are built once.
        let mut a_rows = Vec::with_capacity(rm as usize);
        for r in 0..rm {
            let off = self.const_index(&tb, r * IMMA_TILE)?;
            a_rows.push(self.addi(&tb, row_base, off)?);
        }
        let (mut w_rows, mut out_cols) = (Vec::new(), Vec::new());
        for c in 0..rn {
            let off = self.const_index(&tb, c * IMMA_TILE)?;
            w_rows.push(self.addi(&tb, frag_base, off)?);
            out_cols.push(self.addi(&tb, col_base, off)?);
        }

        let lanes = (rm * rn * 2) as usize;
        let mut args = vec![(self.index_t, self.loc)];
        args.extend(std::iter::repeat_n((f32_t, self.loc), lanes));
        let kb = Block::new(&args);
        let k = detach(kb.argument(0)?.into());
        let mut accs = Vec::with_capacity(lanes);
        for slot in 0..lanes {
            accs.push(detach(kb.argument(slot + 1)?.into()));
        }

        let blk_w = self.const_index(&kb, Q8_BLOCK)?;
        let b = self.divui(&kb, k, blk_w)?;
        let k_from = self.addi(&kb, k, k_off)?;
        let halves = Q8_BLOCK / IMMA_K;
        let mut k_cols = Vec::with_capacity(halves as usize);
        for h in 0..halves {
            let off = self.const_index(&kb, h * IMMA_K)?;
            k_cols.push(self.addi(&kb, k_from, off)?);
        }

        let mut a_frags = Vec::new();
        for row in &a_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, a.mem, &[*row, *col], vec4_i8, 4)?;
                a_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }
        let mut w_frags = Vec::new();
        for row in &w_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, w.mem, &[*row, *col], vec4_i8, 4)?;
                w_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        let zero_i = self.zero_scalar(&kb, i32_t)?;
        let empty = self.vec_broadcast(&kb, zero_i, acc_t)?;
        let shape = self.parse_attr("[8, 8, 16]")?;

        // A weight scale belongs to an output column, not to a row of the
        // patch, so loading it inside the row loop would fetch each one `rm`
        // times. The patch is square and square is where the tensor work pays,
        // so that is eight redundant loads out of every nine.
        let mut w_scales = Vec::with_capacity(rn as usize * 2);
        for out_col in &out_cols {
            for dj in 0..2 {
                let off = self.const_index(&kb, dj)?;
                let col = self.addi(&kb, *out_col, off)?;
                w_scales.push(self.push(&kb, memref::load(wsc.mem, &[b, col], self.loc))?);
            }
        }

        let mut next = Vec::with_capacity(lanes);
        for r in 0..rm as usize {
            let sa = self.push(&kb, memref::load(asc.mem, &[a_rows[r], b], self.loc))?;
            for c in 0..rn as usize {
                let mut sum = empty;
                for h in 0..halves as usize {
                    sum = self.push(
                        &kb,
                        OperationBuilder::new("nvgpu.mma.sync", self.loc)
                            .add_operands(&[
                                a_frags[r * halves as usize + h],
                                w_frags[c * halves as usize + h],
                                sum,
                            ])
                            .add_attributes(&[(self.id("mmaShape"), shape)])
                            .add_results(&[acc_t])
                            .build()?,
                    )?;
                }
                for dj in 0..2 {
                    let sw = w_scales[c * 2 + dj as usize];
                    let scale = self.push(&kb, arith::mulf(sa, sw, self.loc))?;
                    let raw = self.vec_extract(&kb, sum, &[0, dj], i32_t)?;
                    let as_f = self.small_int_to_f32(&kb, raw)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    // A mul and an add here are mul.rn and add.rn, which ptxas
                    // may not contract; at a patch's size that is 128 wasted
                    // instructions against the same block's 128 tensor ones.
                    next.push(self.elem_mac(&kb, f32_t, as_f, scale, accs[slot])?);
                }
            }
        }
        kb.append_operation(scf::r#yield(&next, self.loc));

        let zero_k = self.const_index(&tb, 0)?;
        let step = self.const_index(&tb, Q8_BLOCK)?;
        let init = self.zero_scalar(&tb, f32_t)?;
        let mut operands = vec![zero_k, kd, step];
        operands.extend(std::iter::repeat_n(init, lanes));
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = tb.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&vec![f32_t; lanes])
                .add_regions([kr])
                .build()?,
        );

        for (r, row) in a_rows.iter().enumerate() {
            for (c, out_col) in out_cols.iter().enumerate() {
                for dj in 0..2 {
                    let off = self.const_index(&tb, dj)?;
                    let col = self.addi(&tb, *out_col, off)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    let value = detach(loop_op.result(slot)?.into());
                    tb.append_operation(memref::store(value, out.mem, &[*row, col], self.loc));
                }
            }
        }
        tb.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(tb);
        block.append_operation(scf::r#for(warp, total, warps, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }

    fn tile_matmul_t_dp4a(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        kd: i64,
    ) -> Result<bool> {
        let applies = a.elem == self.i8_t
            && b.elem == self.i8_t
            && out.elem == self.i32_t
            && kd % 4 == 0
            && a.swizzle.is_none()
            && b.swizzle.is_none()
            && !a.is_masked()
            && !b.is_masked()
            && self.base.gpu_config.supports_dp4a();
        if !applies {
            return Ok(false);
        }

        let i32_t = self.i32_t;
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let slot_t = MemRefType::new(i32_t, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let zero = cg.zero_scalar(blk, i32_t)?;
            blk.append_operation(memref::store(zero, slot, &[], cg.loc));

            // One iteration per group of four bytes.
            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, kd / 4)?;
            let st = cg.const_index(blk, 1)?;
            let kb = Block::new(&[(cg.index_t, cg.loc)]);
            let group = detach(kb.argument(0)?.into());
            let four = cg.const_index(&kb, 4)?;
            let k = cg.push(&kb, arith::muli(group, four, cg.loc))?;

            let va = cg.vec_load_al(&kb, a.mem, &[i, k], vec4_i8, 4)?;
            let vb = cg.vec_load_al(&kb, b.mem, &[j, k], vec4_i8, 4)?;
            let cur = cg.push(&kb, memref::load(slot, &[], cg.loc))?;
            let signed = cg.parse_attr("#nvvm.dot_accumulate_type<signed>")?;
            let acc = cg.push(
                &kb,
                OperationBuilder::new("nvvm.dot.accumulate.4way", cg.loc)
                    .add_operands(&[va, vb, cur])
                    .add_attributes(&[(cg.id("a_type"), signed), (cg.id("b_type"), signed)])
                    .add_results(&[i32_t])
                    .build()?,
            )?;
            kb.append_operation(memref::store(acc, slot, &[], cg.loc));
            kb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(kb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })?;
        Ok(true)
    }

    /// out[i, j] = sum_k(a[i, k] * b[j, k]), the transposed matmul behind
    /// `dot_t` (contracts the last dim of both operands).
    ///
    /// The int8 operands take the tensor cores first and `dp4a` second. The
    /// generic fallback is a plain thread-per-output scalar reduction: the
    /// heavily-tuned [`Self::tile_matmul`] has no transposed-b variant, and the
    /// attention S = Q @ K.T tile is small relative to the kernel's other
    /// costs.
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
        
        if self.tile_matmul_t_imma(block, a, b, out, kd)?
            || self.tile_matmul_t_dp4a(block, a, b, out, kd)?
        {
            return Ok(());
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

    /// out[i, j] = sum_{r <= i} src[r, j]: an inclusive prefix sum down the
    /// rows (the sequence axis) of a rank-2 tile, the running gate cumulant
    /// gated linear attention needs. One thread owns each column and sweeps
    /// its rows in order, carrying the partial in a register (an scf.for
    /// iter_arg); the sequential row dependence keeps this off the warp
    /// path.
    pub(super) fn tile_cumsum(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("cumsum expects a rank-2 tile");
        }

        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("cumsum needs a static tile shape");
        }

        let elem = src.elem;
        if !self.is_float(elem) {
            bail!("cumsum needs a float element type");
        }

        let out = self.alloc_tile_shaped(block, elem, &[rows, cols])?;

        let total = self.const_index(block, cols)?;
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let bdim = self.gpu_index(block, "gpu.block_dim", "x")?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let j = detach(body.argument(0)?.into());

        let init = self.zero_scalar(&body, elem)?;
        let lo = self.const_index(&body, 0)?;
        let hi = self.const_index(&body, rows)?;
        let st = self.const_index(&body, 1)?;
        let rb = Block::new(&[(self.index_t, self.loc), (elem, self.loc)]);
        let i = detach(rb.argument(0)?.into());
        let acc = detach(rb.argument(1)?.into());
        let v = self.push(&rb, memref::load(src.mem, &[i, j], self.loc))?;
        let nacc = self.push(&rb, arith::addf(acc, v, self.loc))?;

        rb.append_operation(memref::store(nacc, out.mem, &[i, j], self.loc));
        rb.append_operation(scf::r#yield(&[nacc], self.loc));

        let rr = Region::new();
        rr.append_block(rb);

        body.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lo, hi, st, init])
                .add_results(&[elem])
                .add_regions([rr])
                .build()?,
        );

        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);

        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;

        Ok(out)
    }

    /// out[i, j] = src[i, j] when j <= i, else 0: the causal (lower-
    /// triangular) mask for intra-chunk attention. Rewrites src in place
    /// when it owns an unswizzled buffer (each thread reads and writes one
    /// element), otherwise writes a fresh tile.
    pub(super) fn tile_tril(&mut self, block: &Block<'c>, src: &MemVal<'c>) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("tril expects a rank-2 tile");
        }

        if src.shape.contains(&DYN) {
            bail!("tril needs a static tile shape");
        }

        if !self.is_float(src.elem) {
            bail!("tril needs a float element type");
        }

        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };

        let zero = self.zero_scalar(block, src.elem)?;

        self.distribute(block, &out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let keep = cg.push(
                blk,
                arith::cmpi(cg.ctx, arith::CmpiPredicate::Sle, j, i, cg.loc),
            )?;

            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let r = cg.push(
                blk,
                OperationBuilder::new("arith.select", cg.loc)
                    .add_operands(&[keep, v, zero])
                    .add_results(&[src.elem])
                    .build()?,
            )?;

            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));

            Ok(())
        })?;

        Ok(out)
    }

    /// out[i, j] = src[j, i]: the rank-2 tile transpose. Each output element
    /// is owned by one thread that reads the mirrored source element. Needed
    /// to contract over the sequence axis (the K.T @ V state update) since
    /// dot / dot_t only contract the last axes.
    pub(super) fn tile_transpose(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("transpose expects a rank-2 tile");
        }

        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("transpose needs a static tile shape");
        }

        let out = self.alloc_tile_shaped(block, src.elem, &[cols, rows])?;

        self.distribute(block, &out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let v = cg.push(blk, memref::load(src.mem, &[j, i], cg.loc))?;
            blk.append_operation(memref::store(v, out.mem, idx, cg.loc));
            Ok(())
        })?;

        Ok(out)
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
            let q = self.divui(&body, st, wtiles_n_v)?;
            let r = self.remui(&body, st, wtiles_n_v)?;
            let wm_v = self.const_index(&body, lm * tm)?;
            let wn_v = self.const_index(&body, ln * tn)?;
            let wm0 = self.muli(&body, q, wm_v)?;
            let wn0 = self.muli(&body, r, wn_v)?;
            (self.addi(&body, wm0, off_m)?, self.addi(&body, wn0, off_n)?)
        } else {
            // Flat sub-tile origin: st -> (st / tiles_n * TM, st % tiles_n * TN).
            // tiles_n is a constant, so the div/rem strength-reduce.
            let tiles_n_v = self.const_index(&body, tiles_n)?;
            let q = self.divui(&body, st, tiles_n_v)?;
            let r = self.remui(&body, st, tiles_n_v)?;
            let tm_v = self.const_index(&body, tm)?;
            let tn_v = self.const_index(&body, tn)?;
            (self.muli(&body, q, tm_v)?, self.muli(&body, r, tn_v)?)
        };
        let mut ms = Vec::with_capacity(tm as usize);
        for i in 0..tm {
            let c = self.const_index(&body, i)?;
            ms.push(self.addi(&body, m0, c)?);
        }
        let mut ns = Vec::with_capacity(tn as usize);
        for j in 0..tn {
            let c = self.const_index(&body, j)?;
            ns.push(self.addi(&body, n0, c)?);
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
        self.numeric_cast(block, v, want)
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

    pub(super) fn vec_shape_cast(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("vector.shape_cast", self.loc)
                .add_operands(&[value])
                .add_results(&[want])
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
