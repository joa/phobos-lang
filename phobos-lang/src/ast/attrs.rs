// What a kernel's attributes ask for, each read where the build or the
// emitter needs it.

use super::*;

impl Kernel {
    pub fn wants_dynamic_shared(&self) -> bool {
        self.attrs.iter().any(|a| a.name == "dynshared")
    }

    /// Whether `@padstage` was written on this kernel: pad a `var x = <tensor
    /// slice>` staging tile against shared-memory bank conflicts. See
    /// `Build::should_pad_stage` for which tiles that reaches.
    pub fn wants_padded_stage(&self) -> bool {
        self.attrs.iter().any(|a| a.name == "padstage")
    }

    /// Whether the kernel calls a `_qgemm_t` built-in, which loads its
    /// operands with `ldmatrix` and so wants a module indexed at 64 bits.
    pub fn wants_ldmatrix(&self) -> bool {
        self.calls(|callee| callee.ends_with("_qgemm_t"))
    }

    /// What `@swizzle` asks of the shared tiles the compiler would swizzle:
    /// None to decide alone, Some(true) for `@swizzle` or `@swizzle(on)`,
    /// which fails where it cannot, Some(false) for `@swizzle(off)`.
    pub fn swizzle_mode(&self) -> Result<Option<bool>, String> {
        self.switch("swizzle")
    }

    /// What `@pipeline` asks of the loops the compiler would double-buffer:
    /// None to decide alone, Some(true) for `@pipeline` or `@pipeline(on)`,
    /// which fails where nothing pipelines, Some(false) for `@pipeline(off)`.
    pub fn pipeline_mode(&self) -> Result<Option<bool>, String> {
        self.switch("pipeline")
    }

    /// `@name` as a switch: None when absent, Some(true) for `@name` or
    /// `@name(on)`, Some(false) for `@name(off)`.
    fn switch(&self, name: &str) -> Result<Option<bool>, String> {
        let Some(attr) = self.attr(name) else {
            return Ok(None);
        };
        match &attr.args[..] {
            [] => Ok(Some(true)),
            [AttrArg::Positional(Literal::Ident(v))] if v == "on" => Ok(Some(true)),
            [AttrArg::Positional(Literal::Ident(v))] if v == "off" => Ok(Some(false)),
            _ => Err(format!("`@{name}` takes `on` or `off`")),
        }
    }

    /// `@persistent`: None when absent, `Some(None)` written bare, and
    /// `Some(Some(dim))` with the grid's block count, a literal or an
    /// `@autotune` name, which `ctx.blocks.x` then is.
    pub fn persistent(&self) -> Result<Option<Option<Dim>>, String> {
        let Some(attr) = self.attr("persistent") else {
            return Ok(None);
        };
        match &attr.args[..] {
            [] => Ok(Some(None)),
            [AttrArg::Positional(Literal::Int(n))] if *n > 0 => Ok(Some(Some(Dim::Int(*n)))),
            [AttrArg::Positional(Literal::Ident(name))] => Ok(Some(Some(Dim::Sym(name.clone())))),
            _ => Err("`@persistent` takes the grid's block count, as `@persistent(BLOCKS)`, or nothing".to_string()),
        }
    }

    /// The tensors the body declares, `var HQ: tensor<i8>[1, FF]`: scratch
    /// the launch owns, in declaration order, with where each is declared.
    pub fn scratch(&self) -> Vec<(&str, &Type, crate::diag::Span)> {
        fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<(&'a str, &'a Type, crate::diag::Span)>) {
            for stmt in stmts {
                match stmt {
                    Stmt::Var {
                        name,
                        ty: Some(ty @ Type::Tensor(..)),
                        value: None,
                        span,
                        ..
                    } => out.push((name, ty, *span)),
                    Stmt::For { body, .. } | Stmt::While { body, .. } => walk(body, out),
                    Stmt::If { then, r#else, .. } => {
                        walk(then, out);
                        walk(r#else.as_deref().unwrap_or_default(), out);
                    }
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.body, &mut out);
        out
    }

    /// Whether the body calls `grid_barrier()` with no barrier named, which
    /// takes the one a `@persistent` launch supplies.
    pub fn implicit_grid_barrier(&self) -> bool {
        let mut found = false;
        for stmt in &self.body {
            stmt.walk_exprs(&mut |e| {
                found |= matches!(&e.kind, ExprKind::Call { callee, args } if callee == "grid_barrier" && args.is_empty());
            });
        }
        found
    }

    /// The names an attribute lists, as `@shared(a, b)` does.
    pub fn named_by(&self, attr: &str) -> Vec<String> {
        self.attrs
            .iter()
            .filter(|a| a.name == attr)
            .flat_map(|a| a.args.iter())
            .filter_map(|arg| match arg {
                AttrArg::Positional(Literal::Ident(name)) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// `@grid_barriers(n)`: the grid barriers the kernel has, written and
    /// placed together.
    pub fn grid_barriers(&self) -> Result<Option<usize>, String> {
        let Some(attr) = self.attr("grid_barriers") else {
            return Ok(None);
        };
        match &attr.args[..] {
            [AttrArg::Positional(Literal::Int(n))] if *n >= 0 => Ok(Some(*n as usize)),
            _ => Err("`@grid_barriers` takes the kernel's barrier count, as `@grid_barriers(1)`".to_string()),
        }
    }

    /// `@index_bits(n)`: the index width the kernel asks for, 32 or 64, with
    /// `n` a literal or an `@autotune` constant taken from `env`.
    pub fn index_bits_in(&self, env: &BTreeMap<String, i64>) -> Result<Option<u32>, String> {
        let Some(attr) = self.attr("index_bits") else {
            return Ok(None);
        };
        let bits = match &attr.args[..] {
            [AttrArg::Positional(Literal::Int(n))] => *n,
            [AttrArg::Positional(Literal::Ident(name))] => *env
                .get(name)
                .ok_or_else(|| format!("`@index_bits({name})`: `{name}` is not an `@autotune` constant"))?,
            _ => {
                return Err("`@index_bits` takes 32 or 64, or an `@autotune` constant choosing between them"
                    .to_string());
            }
        };
        match bits {
            32 | 64 => Ok(Some(bits as u32)),
            _ => Err(format!("`@index_bits` takes 32 or 64 (got {bits})")),
        }
    }

    /// Why the kernel needs 64-bit indices on every chip, if it does.
    pub fn wide_index_reason(&self) -> Option<&'static str> {
        if self.wants_mma_sync() {
            Some("`@tensorcore` takes `mma.sync`")
        } else if self.wants_ldmatrix() {
            Some("a `_qgemm_t` built-in loads its operands with `ldmatrix`")
        } else if self.wants_system_scope() {
            Some("`load_acquire` and `store_release` form full 64-bit addresses")
        } else {
            None
        }
    }

    /// True when the kernel reads or writes at system scope, which forms
    /// full 64-bit addresses.
    pub fn wants_system_scope(&self) -> bool {
        self.calls(|callee| matches!(callee, "load_acquire" | "store_release"))
    }

    /// Whether the body calls a built-in `pick` accepts.
    fn calls(&self, pick: impl Fn(&str) -> bool) -> bool {
        let mut found = false;
        for stmt in &self.body {
            stmt.walk_exprs(&mut |e| {
                if let ExprKind::Call { callee, .. } = &e.kind
                    && pick(callee)
                {
                    found = true;
                }
            });
        }
        found
    }

    /// True for `@tensorcore` without the `wmma` argument, which forces the
    /// legacy WMMA path.
    pub fn wants_mma_sync(&self) -> bool {
        self.attrs.iter().any(|a| {
            a.name == "tensorcore"
                && !a
                    .args
                    .iter()
                    .any(|arg| matches!(arg, AttrArg::Positional(Literal::Ident(f)) if f == "wmma"))
        })
    }
}
