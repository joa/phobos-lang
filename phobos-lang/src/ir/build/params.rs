// The kernel's parameters as the body sees them: the declared ones, the
// scratch tensors the body declares, and the barrier word a `@persistent`
// launch appends, all bound before the body is built; and the `@aligned`
// promises checked against them at entry.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};

use super::{Binding, Build, Decl, extent};
use crate::ast::{AttrArg, BinOp, Dim, Kernel, Literal as AstLiteral, Span, Type as AstType};
use crate::ir::{OpKind, Scalar, TensorType, Type};

/// The barrier word a `@persistent` launch appends for `grid_barrier()`,
/// the `[2, 1]` column `grid_barrier(bar)` takes as a named parameter.
pub(super) fn grid_barrier_type() -> AstType {
    AstType::Tensor(crate::ast::Scalar::I32, vec![Dim::Int(2), Dim::Int(1)])
}

/// The name the appended barrier word is bound to, which no source spells.
pub(crate) const GRID_BARRIER_NAME: &str = "#grid_barrier";

impl Build {
    /// Binds the parameters and the symbolic tensor dims, then builds the
    /// body.
    pub(super) fn emit_kernel(&mut self, kernel: &Kernel) -> Result<()> {
        self.scopes.push(HashMap::new());
        let entry = self.ir.entry();
        if self.ir.kernel.grid_barrier {
            let arg = self.ir.args(entry)[kernel.params.len()];
            self.ir.set_name(arg, "grid_barrier");
            self.bind(GRID_BARRIER_NAME, Binding::Tensor(arg));
        }
        for (i, param) in kernel.params.iter().enumerate() {
            let arg = self.ir.args(entry)[i];
            self.ir.set_name(arg, &param.name);
            let binding = match &param.ty {
                // Integer params convert to index on entry.
                AstType::Scalar(crate::ast::Scalar::I32 | crate::ast::Scalar::I64) => {
                    let value = self.value(OpKind::IndexCast(Scalar::Index), &[arg], Type::INDEX);
                    Binding::Let { value, div: 1 }
                }
                AstType::Scalar(_) => Binding::Let { value: arg, div: 1 },
                // These tensors' base pointers are assumed 16-byte aligned,
                // as the host allocator guarantees, so their rows can
                // vectorize.
                AstType::Tensor(
                    crate::ast::Scalar::F32
                    | crate::ast::Scalar::F16
                    | crate::ast::Scalar::BF16
                    | crate::ast::Scalar::I8,
                    _,
                ) => {
                    let ty = self.ir.ty(arg).clone();
                    Binding::Tensor(self.value(OpKind::AssumeAlign, &[arg], ty))
                }
                AstType::Tensor(..) => Binding::Tensor(arg),
                AstType::Tile(..) => bail!("a kernel parameter cannot be a tile; declare it as a `tensor`"),
            };
            let decl = Decl {
                at: param.span,
                ty: Decl::held(Some(&param.ty), param.ty_span),
                value: Span::default(),
            };
            self.declaring = Some((param.name.clone(), decl));
            self.bind(&param.name, binding);

            if let AstType::Tensor(_, dims) = &param.ty {
                for (d, dim) in dims.iter().enumerate() {
                    if let Dim::Sym(name) = dim
                        && !self.shape_env.contains_key(name)
                        && self.lookup(name).is_none()
                    {
                        let size = self.value(OpKind::Dim(d), &[arg], Type::INDEX);
                        self.ir.set_name(size, name);
                        // Dynamic dims are assumed multiples of 4 elements, the
                        // 16-byte row-pitch ABI.
                        self.bind(name, Binding::Let { value: size, div: 4 });
                    }
                }
            }
        }
        self.check_aligned(kernel)?;
        self.emit_stmts(&kernel.body)?;
        // A declaration that failed was reported where it was written.
        let missing = |n: &&String| !self.regs_declared.contains(n) && !self.poisoned.contains(*n);
        if let Some(name) = self.registers.iter().find(missing) {
            bail!("`@registers({name})` names no `var {name}: tile<..>[..]` of the kernel");
        }
        self.scopes.pop();
        Ok(())
    }

    /// Traps at entry when a launch breaks an `@aligned(DIM = tile)` promise
    /// on a dynamic dimension, before anything is written past a tensor.
    fn check_aligned(&mut self, kernel: &Kernel) -> Result<()> {
        for (name, div) in self.aligned_promises(kernel)? {
            let Some(Binding::Let { value: size, .. }) = self.lookup(&name) else {
                continue;
            };
            if div <= 1 {
                continue;
            }
            let div = self.const_index(div);
            let rem = self.value(OpKind::Binary(BinOp::Rem), &[size, div], Type::INDEX);
            let zero = self.const_index(0);
            let broken = self.value(OpKind::Binary(BinOp::Ne), &[rem, zero], Type::Scalar(Scalar::Bool));
            let then = self.ir.new_block(&[]);
            self.in_block(then, |b| {
                b.stmt(OpKind::Trap, &[]);
                b.stmt(OpKind::Yield, &[]);
            });
            self.op(OpKind::If, &[broken], Vec::new(), vec![then]);
        }
        Ok(())
    }

    /// Every `@aligned(NAME = tile)` pair, the tile resolved to its value.
    fn aligned_promises(&self, kernel: &Kernel) -> Result<Vec<(String, i64)>> {
        let Some(attr) = kernel.attrs.iter().find(|a| a.name == "aligned") else {
            return Ok(Vec::new());
        };
        attr.args
            .iter()
            .filter_map(|a| match a {
                AttrArg::KeyValue { key, value } => Some((key, value)),
                _ => None,
            })
            .map(|(name, value)| {
                let div = match value {
                    AstLiteral::Int(n) => *n,
                    AstLiteral::Ident(sym) => *self
                        .shape_env
                        .get(sym)
                        .ok_or_else(|| anyhow!("`@aligned({name} = {sym})`: unknown constant"))?,
                    other => bail!("`@aligned({name})`: expected an integer, found {other:?}"),
                };
                anyhow::ensure!(div > 0, "`@aligned({name})`: the divisor must be positive");
                Ok((name.clone(), div))
            })
            .collect()
    }

    pub(super) fn param_type(&self, kernel: &Kernel, ty: &AstType) -> Result<Type> {
        Ok(match ty {
            AstType::Scalar(s) => Type::Scalar(Scalar::from_ast(*s)),
            AstType::Tensor(s, dims) => Type::Tensor(TensorType {
                elem: Scalar::from_ast(*s),
                shape: self.tensor_shape(dims).iter().map(|&d| extent(d)).collect(),
                div: self.declared_divs(kernel, dims)?,
            }),
            AstType::Tile(..) => bail!("a kernel parameter cannot be a tile; declare it as a `tensor`"),
        })
    }

    /// Per-dimension divisors promised by `@aligned(NAME = tile)`.
    fn declared_divs(&self, kernel: &Kernel, dims: &[Dim]) -> Result<Vec<i64>> {
        let promises = self.aligned_promises(kernel)?;
        Ok(dims
            .iter()
            .map(|dim| match dim {
                Dim::Sym(name) => promises.iter().find(|(n, _)| n == name).map_or(1, |&(_, div)| div),
                Dim::Int(_) => 1,
            })
            .collect())
    }

}
