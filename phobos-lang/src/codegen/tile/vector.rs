// The vector and memref primitives every other file here builds on.
//
// The loads and stores are 128-bit, so they are only valid on a buffer
// provably aligned to 16 bytes; the `_al` variants are the ones that carry
// the alignment attribute through to the lowering.

use super::*;

impl<'c> Codegen<'c> {
    /// vector.load mem[indices] : vector<4 x elem>, declared align-16.
    /// (Without the explicit attribute the lowering uses the element
    /// alignment, and the backend splits the access into scalars.)
    pub(in crate::codegen) fn vec_load(
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
    pub(in crate::codegen) fn vec_load_al(
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

    pub(in crate::codegen) fn vec_store(
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
    pub(in crate::codegen) fn vec_store_al(
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
    pub(in crate::codegen) fn load_as(
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
    pub(in crate::codegen) fn elem_load(
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
    pub(in crate::codegen) fn elem_store(
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

    /// Copies a lookup table into shared memory, cooperatively; `None` when
    /// the table is dynamic or too wide to be worth it.
    ///
    /// A decode gather is uncoalesced by nature -- a lane picks a random entry,
    /// so a warp touches as many sectors as it has lanes. Shared memory has no
    /// sector granularity, and every CTA reads the whole table anyway, so one
    /// copy and a barrier pay for themselves.
    pub(in crate::codegen) fn stage_table(
        &mut self,
        block: &Block<'c>,
        table: &MemVal<'c>,
    ) -> Result<Option<MemVal<'c>>> {
        let width = table.shape[1];
        if width == DYN || width > MAX_STAGED_TABLE || table.is_masked() {
            return Ok(None);
        }
        let tile = self.alloc_tile_shaped(block, table.elem, &[1, width])?;
        let zero = self.const_index(block, 0)?;
        let len = self.const_index(block, width)?;
        let from = self.thread_id(block)?;
        let by = self.block_dim(block)?;
        let cb = Block::new(&[(self.index_t, self.loc)]);
        let i = detach(cb.argument(0)?.into());
        let v = self.push(&cb, memref::load(table.mem, &[zero, i], self.loc))?;
        cb.append_operation(memref::store(v, tile.mem, &[zero, i], self.loc));
        cb.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(cb);
        block.append_operation(scf::r#for(from, len, by, region, self.loc));
        self.barrier(block)?;
        Ok(Some(tile))
    }

    pub(in crate::codegen) fn vec_broadcast(
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

    pub(in crate::codegen) fn vec_shape_cast(
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

    /// `mask` picking elements out of `a` and `b` laid end to end, which is
    /// what joins two four-wide grid entries into the eight-wide lane the
    /// staged projection stores: see `qmma_signed.rs`.
    pub(in crate::codegen) fn vec_shuffle(
        &self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        mask: &[i64],
        want: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("vector.shuffle", self.loc)
                .add_operands(&[a, b])
                .add_attributes(&[(self.id("mask"), self.i64_array(mask)?)])
                .add_results(&[want])
                .build()?,
        )
    }

    pub(in crate::codegen) fn vec_extract(
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

    pub(in crate::codegen) fn vec_insert(
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
    pub(in crate::codegen) fn vec_extf(
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
    pub(in crate::codegen) fn vec_truncf(
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
    pub(in crate::codegen) fn vec_contract(
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

}
