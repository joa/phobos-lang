// Register tiles: a `var` the kernel names in `@registers` lives in each
// thread's registers instead of shared memory, as long as every update reads
// it only at the element it writes. Any other read stores it to a tile first.

use anyhow::{Result, bail, ensure};

use super::{Binding, Build, Rv};
use crate::ast::{AssignOp, BinOp, Dim, Expr, Scalar as AstScalar, Stmt};
use crate::ir::{OpKind, Reduce, RegsType, Scalar, Tree, Type, ValueId};

/// Elements one thread may hold of one register tile.
const MAX_REGS_PER_THREAD: i64 = 64;

impl Build {
    /// `var name: tile<T>[dims] = value` for a name in `@registers`.
    pub(crate) fn emit_regs_decl(
        &mut self,
        name: &str,
        scalar: AstScalar,
        dims: &[Dim],
        value: Option<&Expr>,
    ) -> Result<()> {
        let shape = self.tile_shape(dims)?;
        let elem = Scalar::from_ast(scalar);
        let elems: i64 = shape.iter().product();
        let threads = self.cta_threads;
        ensure!(
            elems % threads == 0,
            "@registers({name}): {elems} elements do not split evenly over {threads} threads"
        );
        ensure!(
            elems / threads <= MAX_REGS_PER_THREAD,
            "@registers({name}): {} elements a thread, over the {MAX_REGS_PER_THREAD} a register tile may hold",
            elems / threads
        );
        let Some(value) = value else {
            bail!("@registers({name}) needs an initial value");
        };
        let ty = Type::Regs(RegsType { elem, shape: shape.clone() });
        let regs = match self.emit_expr(value)? {
            Rv::Scalar(v) => {
                let v = self.coerce(v, elem)?;
                self.value(OpKind::RegsFill, &[v], ty)
            }
            Rv::Tile(t) => {
                self.check_leaves(&[t], &shape)?;
                self.value(OpKind::RegsFused(Tree::Leaf(0)), &[t], ty)
            }
        };
        self.regs_declared.push(name.to_string());
        self.bind(name, Binding::Regs(regs));
        Ok(())
    }

    /// `name = value` or `name += value` on a register tile.
    pub(crate) fn emit_regs_assign(&mut self, name: &str, regs: ValueId, op: AssignOp, value: &Expr) -> Result<()> {
        let Type::Regs(ty) = self.ir.ty(regs).clone() else {
            bail!("'{name}' is not a register tile");
        };
        let Some((mut tree, mut operands)) = self.plan_regs(value, ty.elem, &ty.shape)? else {
            bail!(
                "register tile '{name}' takes only updates that read it at the element they write; \
                 store it to a tile for anything else"
            );
        };
        if op == AssignOp::Add {
            operands.push(regs);
            tree = Tree::Binary(BinOp::Add, Box::new(Tree::Leaf(operands.len() - 1)), Box::new(tree));
        }
        let next = self.value(OpKind::RegsFused(tree), &operands, Type::Regs(ty));
        self.update_binding(name, Binding::Regs(next));
        Ok(())
    }

    /// A register tile written out to a fresh shared tile, for a read that
    /// needs one.
    pub(crate) fn materialize_regs(&mut self, regs: ValueId) -> Result<ValueId> {
        let Type::Regs(ty) = self.ir.ty(regs).clone() else {
            bail!("not a register tile");
        };
        let tile_ty = self.shared_ty(ty.elem, &ty.shape);
        let tile = self.value(OpKind::Alloc, &[], tile_ty);
        self.stmt(OpKind::RegsStore, &[regs, tile]);
        Ok(tile)
    }

    /// `rowsum` or `rowmax` of a register tile, folded with shuffles where
    /// its elements live, or None when a row is not a run of lanes in one
    /// warp. Thread `t` holds row-major positions `t + e * threads`, so a
    /// row of a power-of-two width up to a warp is such a run. A wider row
    /// spans warps, and folding their parts through shared memory costs a
    /// barrier more than storing the tile and reducing that.
    pub(crate) fn regs_reduce(&mut self, how: Reduce, regs: ValueId) -> Option<ValueId> {
        let Type::Regs(ty) = self.ir.ty(regs).clone() else {
            return None;
        };
        let [rows, cols] = ty.shape[..] else {
            return None;
        };
        let lanes = self.target.lanes;
        let fits = matches!(how, Reduce::RowSum | Reduce::RowMax)
            && matches!(ty.elem, Scalar::F32 | Scalar::I32)
            && (cols as u64).is_power_of_two()
            && cols <= lanes
            && self.cta_threads % lanes == 0;
        if !fits {
            return None;
        }
        let out = self.shared_ty(ty.elem, &[rows, 1]);
        Some(self.value(OpKind::RegsReduce(how), &[regs], out))
    }

    /// Fails when a branch or a `while` would update a register tile: the
    /// update could not leave it as a carried value.
    pub(crate) fn check_regs_unassigned(&self, body: &[Stmt], what: &str) -> Result<()> {
        for stmt in body {
            match stmt {
                Stmt::Assign {
                    target: Expr::Var(n),
                    ..
                } if matches!(self.lookup(n), Some(Binding::Regs(_))) => {
                    bail!("register tile '{n}' is updated inside {what}; only straight-line code and `for` loops may update one")
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => self.check_regs_unassigned(body, what)?,
                Stmt::If { then, r#else, .. } => {
                    self.check_regs_unassigned(then, what)?;
                    self.check_regs_unassigned(r#else.as_deref().unwrap_or_default(), what)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
