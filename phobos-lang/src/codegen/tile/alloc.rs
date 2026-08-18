// Tile buffers: allocating them in shared memory, swizzling and
// padding them, and returning them to the pool.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn alloc_tile(
        &mut self,
        block: &Block<'c>,
        scalar: Scalar,
        dims: &[Dim],
    ) -> Result<MemVal<'c>> {
        let shape = self.tile_shape(dims)?;
        self.alloc_tile_shaped(block, self.scalar_type(scalar), &shape)
    }

    /// alloc a tile buffer in SM
    pub(in crate::codegen) fn alloc_tile_shaped(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        shape: &[i64],
    ) -> Result<MemVal<'c>> {
        if shape.contains(&DYN) {
            bail!("tile buffers must have a static shape");
        }

        let space = self.shared_space()?;
        let t = MemRefType::new(elem, shape, None, Some(space));

        // Reuse a released buffer of the same type when one is free (see
        // release); otherwise mint a new one.
        let key = (elem.to_string(), shape.to_vec());
        let name = match self.tile_pool.get_mut(&key).and_then(Vec::pop) {
            Some(name) => name,
            None => {
                // Nothing from an earlier phase is still live: a kernel
                // whose phases are separated by a barrier (see
                // attention_persist_src) fully drains one phase's tiles
                // before the next phase declares its own shapes, so the
                // allocation can restart at offset 0 instead of growing to
                // fit both phases' tiles at once. shared_bytes_peak already
                // has the high-water mark, so this only ever shrinks what
                // gets requested from the driver, never grows it: every
                // live reference to a name cleared here has already gone
                // through release() (or is still counted live and this
                // branch is unreached), so nothing dangles.
                if self.dynamic_shared && self.dynamic_live == 0 && self.shared_bytes > 0 {
                    self.tile_pool.clear();
                    self.tile_offsets.clear();
                    self.shared_bytes = 0;
                }

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
                    self.shared_bytes_peak = self.shared_bytes_peak.max(self.shared_bytes);
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
            let base = self.dynamic_shared_base(block, byte_t.into())?;
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

        let mem = self.assume_align(block, mem, 16)?;

        let align_div = row_major_strides(shape)[..shape.len() - 1]
            .iter()
            .fold(0i64, |acc, &s| gcd(acc, s.abs().max(1)));

        if self.dynamic_shared {
            self.dynamic_live += 1;
        }

        Ok(MemVal {
            mem,
            elem,
            shape: shape.to_vec(),
            row_stride: None,
            align_div,
            swizzle: None,
            global: Some(name),
            shared: true,
            owned: true,
            mask: Vec::new(),
            dim_div: Vec::new(),
        })
    }

    /// Returns an owned temp's shared buffer to the pool, so a later allocation
    /// of the same element type and physical shape reuses it instead of growing
    /// the CTA's static shared footprint. No-op for views, params and named
    /// tiles (bind clears owned).
    ///
    /// Only call after every op reading the buffer has been emitted. Reuse is
    /// race-free because each tile op ends in a CTA barrier, so the reusing op's
    /// writes are ordered after the previous consumer's reads; the garbage a
    /// reused buffer holds is fine, every producing op fully writes its output.
    pub(in crate::codegen) fn release(&mut self, mv: &MemVal<'c>) {
        if !mv.owned {
            return;
        }

        let Some(name) = &mv.global else {
            return;
        };

        if self.aliased.contains(name) {
            return;
        }

        if self.dynamic_shared {
            self.dynamic_live -= 1;
        }

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

    /// Allocates an unpadded shared staging tile with an XOR column swizzle (see
    /// [`Swizzle`]), so ldmatrix reads avoid bank conflicts without paying for
    /// the WMMA path's padding. Shape and layout are unchanged; only the column
    /// index each access uses is permuted, the same way on store and load.
    pub(in crate::codegen) fn alloc_tile_swizzled(
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
    /// unchanged when the buffer is unswizzled. Every staging store and ldmatrix
    /// load goes through here, so the data round-trips whatever the params are.
    pub(in crate::codegen) fn swizzle_col(
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
    pub(in crate::codegen) fn swizzled_index(
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
    /// banks. The tile keeps its logical shape, so iteration and fragment
    /// indexing are unchanged; the padding lives only in the physical allocation
    /// and the row_stride the WMMA leadDimension reads.
    pub(in crate::codegen) fn alloc_tile_padded(
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

    /// Whether a `var x = <tensor slice>` staging tile should allocate through
    /// [`Self::alloc_tile_padded`] rather than [`Self::alloc_tile_shaped`].
    ///
    /// Gated by `@padstage`, and even then only for a row pitch that is an
    /// exact multiple of [`SHARED_BANK_BYTES`], the one layout where every row
    /// collides on the same bank at a fixed column. Padding any other tile
    /// would grow the CTA's shared footprint for nothing.
    pub(in crate::codegen) fn should_pad_stage(&self, elem: Type<'c>, shape: &[i64]) -> bool {
        if !self.pad_stage {
            return false;
        }
        let Some(&cols) = shape.last() else {
            return false;
        };
        if cols == DYN {
            return false;
        }
        let Some(width) = self.elem_bytes(elem) else {
            return false;
        };
        let pitch_bytes = cols * i64::from(width);
        pitch_bytes > 0 && pitch_bytes % SHARED_BANK_BYTES == 0
    }

    /// tile_flat aliases a tile as one row [1, rows * cols].
    ///
    /// Nothing is allocated and no data moves!
    pub(in crate::codegen) fn tile_flat(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("flat expects a rank-2 tile");
        }

        if src.global.is_none() {
            bail!("flat expects a declared tile, not a slice of one");
        }

        if src.row_stride.is_some() || src.swizzle.is_some() {
            bail!("a padded or swizzled staging tile has no flat view");
        }

        let [rows, cols] = [src.shape[0], src.shape[1]];
        if rows == DYN || cols == DYN {
            bail!("flat expects a static tile shape");
        }

        let len = rows * cols;

        if let Some(name) = &src.global {
            self.aliased.insert(name.clone());
        }

        let text = format!("memref<1x{len}x{}, {}>", src.elem, self.mem_space(true));
        let result =
            Type::parse(self.ctx, &text).ok_or_else(|| anyhow!("failed to parse type '{text}'"))?;

        let op = OperationBuilder::new("memref.reinterpret_cast", self.loc)
            .add_operands(&[src.mem])
            .add_attributes(&[
                (self.id("static_offsets"), self.i64_array(&[0])?),
                (self.id("static_sizes"), self.i64_array(&[1, len])?),
                (self.id("static_strides"), self.i64_array(&[len, 1])?),
                (
                    self.id("operandSegmentSizes"),
                    self.i32_array(&[1, 0, 0, 0])?,
                ),
            ])
            .add_results(&[result])
            .build()?;

        Ok(MemVal {
            mem: self.push(block, op)?,
            elem: src.elem,
            shape: vec![1, len],
            row_stride: None,
            align_div: src.align_div,
            swizzle: None,
            global: None, // it's a view; we don't own the memory and must not release it
            shared: true,
            owned: false,
            mask: Vec::new(),
            dim_div: Vec::new(),
        })
    }

    pub(in crate::codegen) fn tile_sizes(
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
}
