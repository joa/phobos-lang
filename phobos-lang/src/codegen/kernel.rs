use super::*;

impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn emit_kernel(&mut self, kernel: &Kernel) -> Result<Operation<'c>> {
        let param_types = kernel
            .params
            .iter()
            .map(|p| self.param_type(&p.ty))
            .collect::<Result<Vec<_>>>()?;

        let block_args: Vec<_> = param_types.iter().map(|&t| (t, self.loc)).collect();
        let entry = Block::new(&block_args);

        self.scopes.push(HashMap::new());

        for (i, param) in kernel.params.iter().enumerate() {
            let arg = detach(entry.argument(i)?.into());
            let binding = match &param.ty {
                // integer params join the index-type world on entry
                AstType::Scalar(Scalar::I32 | Scalar::I64) => Binding::Let {
                    value: self.push(&entry, arith::index_cast(arg, self.index_t, self.loc))?,
                    div: 1,
                },
                AstType::Scalar(_) => Binding::Let { value: arg, div: 1 },
                AstType::Tensor(scalar, dims) => {
                    let elem = self.scalar_type(*scalar);
                    // float tensor base pointers are assumed 16-byte aligned
                    // (the host allocator returns aligned buffers).
                    let mem = if matches!(scalar, Scalar::F32 | Scalar::F16) {
                        self.assume_align(&entry, arg, 16)?
                    } else {
                        arg
                    };
                    Binding::Tensor(MemVal {
                        mem,
                        elem,
                        shape: self.tensor_shape(dims),
                        row_stride: None,
                        aligned: false,
                        swizzle: None,
                        global: None,
                        owned: false,
                        mask: Vec::new(),
                    })
                }
                AstType::Tile(..) => bail!(
                    "kernel '{}': tile-typed parameters are not supported (use tensor instead)",
                    kernel.name
                ),
            };
            self.bind(&param.name, binding);

            // bind the symbolic tensor dims such as M, N, K that aren't autotuned
            // constants to their runtime size so range(0, K) et al work.
            if let AstType::Tensor(_, dims) = &param.ty {
                for (d, dim) in dims.iter().enumerate() {
                    if let Dim::Sym(name) = dim
                        && !self.shape_env.contains_key(name)
                        && self.lookup(name).is_none()
                    {
                        let pos = self.const_index(&entry, d as i64)?;
                        let size = self.push(&entry, memref::dim(arg, pos, self.loc))?;

                        // dynamic dims are assumed multiples of 4 elements (the 16-byte row-pitch ABI; see module memref docs).
                        // TODO(joa): check fp64
                        self.bind(
                            name,
                            Binding::Let {
                                value: size,
                                div: 4,
                            },
                        );
                    }
                }
            }
        }

        self.emit_stmts(&entry, &kernel.body)?;

        self.scopes.pop();

        // phobos has no return :)
        entry.append_operation(OperationBuilder::new("gpu.return", self.loc).build()?);

        let fn_region = Region::new();
        fn_region.append_block(entry);

        let fn_type: Type = FunctionType::new(self.ctx, &param_types, &[]).into();
        let mut attrs = vec![
            (
                self.id("sym_name"),
                StringAttribute::new(self.ctx, &kernel.name).into(),
            ),
            (self.id("function_type"), TypeAttribute::new(fn_type).into()),
            (self.id("gpu.kernel"), Attribute::unit(self.ctx)),
        ];

        // @launch
        if let Some(launch) = self.launch {
            attrs.push((
                self.id("nvvm.maxntid"),
                DenseI32ArrayAttribute::new(self.ctx, &[launch.max_threads as i32]).into(),
            ));
            if let Some(min_blocks) = launch.min_blocks {
                attrs.push((
                    self.id("nvvm.minctasm"),
                    IntegerAttribute::new(self.i32_t, min_blocks).into(),
                ));
            }
            if let Some(max_nreg) = launch.max_nreg {
                attrs.push((
                    self.id("nvvm.maxnreg"),
                    IntegerAttribute::new(self.i32_t, max_nreg).into(),
                ));
            }
        }

        Ok(OperationBuilder::new("gpu.func", self.loc)
            .add_attributes(&attrs)
            .add_regions([fn_region])
            .build()?)
    }
}

// types
impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn scalar_type(&self, scalar: Scalar) -> Type<'c> {
        match scalar {
            Scalar::F16 => self.f16_t,
            Scalar::F32 => self.f32_t,
            Scalar::F64 => self.f64_t,
            Scalar::I32 => self.i32_t,
            Scalar::I64 => self.i64_t,
            Scalar::Bool => self.bool_t,
        }
    }

    /// Literal and @autotune dims become static; the rest is DYNamic.
    pub(super) fn tensor_shape(&self, dims: &[Dim]) -> Vec<i64> {
        dims.iter()
            .map(|d| match d {
                Dim::Int(n) => *n,
                Dim::Sym(name) => self.shape_env.get(name).copied().unwrap_or(DYN),
            })
            .collect()
    }

    pub(super) fn param_type(&self, ty: &AstType) -> Result<Type<'c>> {
        Ok(match ty {
            AstType::Scalar(s) => self.scalar_type(*s),
            AstType::Tensor(s, dims) => {
                let mem_space: Attribute = IntegerAttribute::new(self.i64_t, MEM_GLOBAL).into();
                MemRefType::new(
                    self.scalar_type(*s),
                    &self.tensor_shape(dims),
                    None,
                    Some(mem_space),
                )
                .into()
            }
            AstType::Tile(..) => {
                bail!("tile-typed parameters are not supported (use tensor instead)")
            }
        })
    }

    /// A tile's shape must be fully static (literals or @autotune symbols).
    pub(super) fn tile_shape(&self, dims: &[Dim]) -> Result<Vec<i64>> {
        dims.iter()
            .map(|d| match d {
                Dim::Int(n) => Ok(*n),
                Dim::Sym(name) => self.shape_env.get(name).copied().ok_or_else(|| {
                    anyhow!("tile dim '{name}' must be a constant (literal or @autotune symbol)")
                }),
            })
            .collect()
    }
}
