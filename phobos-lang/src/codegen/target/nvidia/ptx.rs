// The PTX-level helpers [`Nvidia`]'s `Isa` methods build on: element and
// shared addresses, inline assembly and the index-typed GPU queries.

use super::*;

impl Nvidia {
    /// The global address of `mem[indices]` as an `!llvm.ptr`, from the
    /// memref's aligned base, offset and strides. Formed in `index`, so it
    /// is only whole at 64-bit indices.
    pub(super) fn element_ptr<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<Value<'c, 'c>> {
        if self.index_bits < 64 {
            bail!("a system-scope access forms a 64-bit address and needs 64-bit indices");
        }
        let addr = self.byte_address(cg, block, mem.mem, indices)?;
        let i64_t: Type<'c> = IntegerType::new(cg.ctx, 64).into();
        let addr = cg.push(block, arith::index_cast(addr, i64_t, cg.loc))?;
        let ptr_t = cg.parse_type("!llvm.ptr")?;
        cg.push(
            block,
            OperationBuilder::new("llvm.inttoptr", cg.loc)
                .add_operands(&[addr])
                .add_results(&[ptr_t])
                .build()?,
        )
    }

    /// The shared-memory address of `mem[indices]` as an `!llvm.ptr<3>`. A
    /// shared address fits 32 bits, so this is whole at any index width.
    pub(super) fn shared_ptr<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<Value<'c, 'c>> {
        let addr = self.byte_address(cg, block, mem, indices)?;
        let i32_t: Type<'c> = IntegerType::new(cg.ctx, 32).into();
        let addr = cg.push(block, arith::index_cast(addr, i32_t, cg.loc))?;
        let ptr_t = cg.parse_type("!llvm.ptr<3>")?;
        cg.push(
            block,
            OperationBuilder::new("llvm.inttoptr", cg.loc)
                .add_operands(&[addr])
                .add_results(&[ptr_t])
                .build()?,
        )
    }

    /// The byte address of `mem[indices]` as an `index`, from the memref's
    /// aligned base, offset and strides.
    pub(super) fn byte_address<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: &[Value<'c, 'c>],
    ) -> Result<Value<'c, 'c>> {
        let rank = indices.len();
        let mem_t = MemRefType::try_from(mem.r#type()).map_err(|_| anyhow!("an address of a non-memref"))?;
        let elem = melior::ir::r#type::ShapedTypeLike::element(&mem_t);
        let bytes = cg
            .elem_bytes(elem)
            .ok_or_else(|| anyhow!("an address of an element of unknown size"))?;
        let index_t = cg.index_t;
        let base_t: Type<'c> = MemRefType::new(elem, &[], None, mem_t.memory_space()).into();
        let mut results = vec![base_t, index_t];
        results.extend(std::iter::repeat_n(index_t, 2 * rank));
        let meta = block.append_operation(
            OperationBuilder::new("memref.extract_strided_metadata", cg.loc)
                .add_operands(&[mem])
                .add_results(&results)
                .build()?,
        );
        let result = |k: usize| -> Result<Value<'c, 'c>> { Ok(detach(meta.result(k)?.into())) };
        let mut elem_off = result(1)?;
        for (k, &idx) in indices.iter().enumerate() {
            let stride = result(2 + rank + k)?;
            let step = cg.muli(block, idx, stride)?;
            elem_off = cg.addi(block, elem_off, step)?;
        }
        let byte_off = if bytes == 1 {
            elem_off
        } else {
            let scale = cg.const_index(block, i64::from(bytes))?;
            cg.muli(block, elem_off, scale)?
        };
        let base = cg.push(
            block,
            OperationBuilder::new("memref.extract_aligned_pointer_as_index", cg.loc)
                .add_operands(&[mem])
                .add_results(&[index_t])
                .build()?,
        )?;
        cg.addi(block, base, byte_off)
    }

    /// `ldmatrix` at 32-bit indices, where nvgpu's lowering does not reach:
    /// nvvm's own op on the shared address, its 32-bit registers bitcast to
    /// the fragment's rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ldmatrix_narrow<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        mem: Value<'c, 'c>,
        indices: [Value<'c, 'c>; 2],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let ptr = self.shared_ptr(cg, block, mem, &indices)?;
        let words = (0..num_tiles).map(|_| "i32").collect::<Vec<_>>().join(", ");
        let result_t = if num_tiles == 1 {
            cg.i32_t
        } else {
            cg.parse_type(&format!("!llvm.struct<({words})>"))?
        };
        let layout = if transpose { "col" } else { "row" };
        let attrs = [
            (cg.id("num"), IntegerAttribute::new(cg.i32_t, num_tiles).into()),
            (cg.id("layout"), cg.parse_attr(&format!("#nvvm.mma_layout<{layout}>"))?),
            (cg.id("shape"), cg.parse_attr("#nvvm.ld_st_matrix_shape<m = 8, n = 8>")?),
            (cg.id("eltType"), cg.parse_attr("#nvvm.ld_st_matrix_elt_type<b16>")?),
        ];
        let loaded = cg.push(
            block,
            OperationBuilder::new("nvvm.ldmatrix", cg.loc)
                .add_operands(&[ptr])
                .add_attributes(&attrs)
                .add_results(&[result_t])
                .build()?,
        )?;
        // A fragment is `vector<num x k x elem>`, a row the 32 bits of one
        // matrix.
        let text = frag_t.to_string();
        let dims = text
            .strip_prefix("vector<")
            .and_then(|t| t.strip_suffix('>'))
            .ok_or_else(|| anyhow!("an ldmatrix fragment that is not a vector: {text}"))?;
        let (_, row) = dims.split_once('x').ok_or_else(|| anyhow!("an ldmatrix fragment of rank 1: {text}"))?;
        let row_t = cg.parse_type(&format!("vector<{row}>"))?;
        let elem = cg.parse_type(row.rsplit('x').next().expect("split yields one"))?;
        let mut out = cg.vec_broadcast(block, cg.zero_scalar(block, elem)?, frag_t)?;
        for t in 0..num_tiles {
            let word = if num_tiles == 1 {
                loaded
            } else {
                cg.push(
                    block,
                    OperationBuilder::new("llvm.extractvalue", cg.loc)
                        .add_operands(&[loaded])
                        .add_attributes(&[(cg.id("position"), cg.parse_attr(&format!("array<i64: {t}>"))?)])
                        .add_results(&[cg.i32_t])
                        .build()?,
                )?
            };
            let one = cg.vec_broadcast(block, word, Type::vector(&[1], cg.i32_t))?;
            let row = cg.vec_bitcast(block, one, row_t)?;
            out = cg.vec_insert(block, row, out, &[t])?;
        }
        Ok(out)
    }

    /// One side-effecting inline PTX statement, with a result when `result`
    /// is given.
    pub(super) fn asm<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        operands: &[Value<'c, 'c>],
        text: &str,
        constraints: &str,
        result: Option<Type<'c>>,
    ) -> Result<Option<Value<'c, 'c>>> {
        let op = OperationBuilder::new("llvm.inline_asm", cg.loc)
            .add_operands(operands)
            .add_attributes(&[
                (cg.id("asm_string"), cg.parse_attr(&format!("\"{text}\""))?),
                (cg.id("constraints"), cg.parse_attr(&format!("\"{constraints}\""))?),
                (cg.id("has_side_effects"), cg.parse_attr("unit")?),
            ])
            .add_results(result.as_slice())
            .build()?;
        match result {
            Some(_) => Ok(Some(cg.push(block, op)?)),
            None => {
                block.append_operation(op);
                Ok(None)
            }
        }
    }

    /// One PTX instruction over an f32, through llvm.inline_asm.
    ///
    /// The math dialect cannot be used here. convert-math-to-llvm runs first
    /// and turns math.exp and friends into llvm.intr.*, which NVPTX cannot
    /// select. The instruction is pure, so loads around it still merge.
    pub(super) fn ptx_f32<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        asm: &str,
        x: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        cg.push(
            block,
            OperationBuilder::new("llvm.inline_asm", cg.loc)
                .add_operands(&[x])
                .add_attributes(&[
                    (
                        cg.id("asm_string"),
                        StringAttribute::new(cg.ctx, asm).into(),
                    ),
                    (
                        cg.id("constraints"),
                        StringAttribute::new(cg.ctx, "=f,f").into(),
                    ),
                ])
                .add_results(&[cg.f32_t])
                .build()?,
        )
    }

    /// A gpu index op (gpu.thread_id / gpu.block_id / gpu.block_dim /
    /// gpu.grid_dim) along one dimension.
    pub(super) fn gpu_index<'c>(
        &self,
        cg: &Codegen<'c>,
        block: &Block<'c>,
        op: &str,
        dim: &str,
    ) -> Result<Value<'c, 'c>> {
        // Bracket form required for gpu::DimensionAttr.
        let attr = cg.parse_attr(&format!("#gpu<dim {dim}>"))?;
        cg.push(
            block,
            OperationBuilder::new(op, cg.loc)
                .add_attributes(&[(cg.id("dimension"), attr)])
                .add_results(&[cg.index_t])
                .build()?,
        )
    }

    pub(super) fn async_token_type<'c>(&self, cg: &Codegen<'c>) -> Result<Type<'c>> {
        cg.parse_type("!nvgpu.device.async.token")
    }
}
