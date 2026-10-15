// Register tiles: a `var` the kernel names in `@registers` lives in each
// thread's registers instead of shared memory, as long as every update reads
// it only at the element it writes. Any other read stores it to a tile first.

use anyhow::{Result, bail, ensure};

use super::{Binding, Build, Rv};
use crate::ast::{AssignOp, BinOp, Dim, Expr, ExprKind, Scalar as AstScalar, Stmt};
use crate::ir::{Bounds, DistType, ForInfo, OpKind, Reduce, RegLayout, Scalar, Tree, Type, ValueId};

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
            "`@registers({name})`: {elems} elements do not split evenly over {threads} threads"
        );
        ensure!(
            elems / threads <= MAX_REGS_PER_THREAD,
            "`@registers({name})`: {} elements a thread, over the {MAX_REGS_PER_THREAD} a register tile may hold",
            elems / threads
        );
        let Some(value) = value else {
            bail!("`@registers({name})` needs an initial value");
        };
        let ty = Type::Dist(DistType {
            elem,
            shape: shape.clone(),
            layout: RegLayout::Blocked,
        });
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
        let Some(ty) = self.ir.ty(regs).regs().cloned() else {
            bail!("`{name}` is not a register tile");
        };
        let Some((mut tree, mut operands)) = self.plan_regs(value, ty.elem, &ty.shape)? else {
            bail!(
                "register tile `{name}` takes only updates that read it at the element they write; \
                 store it to a tile for anything else"
            );
        };
        if op == AssignOp::Add {
            operands.push(regs);
            tree = Tree::Binary(BinOp::Add, Box::new(Tree::Leaf(operands.len() - 1)), Box::new(tree));
        }
        let next = self.value(OpKind::RegsFused(tree), &operands, Type::Dist(ty));
        self.update_binding(name, Binding::Regs(next));
        Ok(())
    }

    /// A register tile written out to a fresh shared tile, for a read that
    /// needs one.
    pub(crate) fn materialize_regs(&mut self, regs: ValueId) -> Result<ValueId> {
        let Some(ty) = self.ir.ty(regs).regs().cloned() else {
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
        let ty = self.ir.ty(regs).regs().cloned()?;
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
                    target: Expr { kind: ExprKind::Var(n), .. },
                    ..
                } if matches!(self.lookup(n), Some(Binding::Regs(_))) => {
                    bail!("register tile `{n}` is updated inside {what}; only straight-line code and `for` loops may update one")
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

    /// The register tiles a loop body assigns, in the order it first does.
    pub(crate) fn regs_carried(&self, body: &[Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        self.regs_carried_into(body, &mut names);
        names
    }

    fn regs_carried_into(&self, stmts: &[Stmt], names: &mut Vec<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign {
                    target: Expr { kind: ExprKind::Var(n), .. },
                    ..
                } if !names.contains(n) && matches!(self.lookup(n), Some(Binding::Regs(_))) => names.push(n.clone()),
                Stmt::For { body, .. } => self.regs_carried_into(body, names),
                _ => {}
            }
        }
    }

    /// A loop threading the register tiles the body assigns as carried
    /// values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_regs_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        names: &[String],
        hoisted: &[ValueId],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(start, end, step)?;
        let mut inits = Vec::new();
        for name in names {
            let Some(Binding::Regs(r)) = self.lookup(name) else {
                bail!("`{name}` is not register-bound");
            };
            inits.push(r);
        }
        let types: Vec<Type> = inits.iter().map(|&v| self.ir.ty(v).clone()).collect();
        let mut arg_types = vec![Type::INDEX];
        arg_types.extend(types.iter().cloned());
        let block = self.ir.new_block(&arg_types);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        let args: Vec<ValueId> = self.ir.args(block)[1..].to_vec();
        self.in_block(block, |b| {
            b.push_scope();
            b.bind(var, Binding::Let { value: iv, div: iv_div });
            for (name, &arg) in names.iter().zip(&args) {
                b.bind(name, Binding::Regs(arg));
            }
            let finals = b.emit_stmts(body).and_then(|()| {
                names
                    .iter()
                    .map(|name| match b.lookup(name) {
                        Some(Binding::Regs(r)) => Ok(r),
                        _ => bail!("`{name}` lost its register binding in the loop body"),
                    })
                    .collect::<Result<Vec<_>>>()
            });
            b.pop_scope();
            b.stmt(OpKind::Yield, &finals?);
            Ok::<(), anyhow::Error>(())
        })?;
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(&inits);
        operands.extend_from_slice(hoisted);
        let op = self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: inits.len(),
                hoisted: hoisted.len(),
                pipeline: None,
            }),
            &operands,
            types,
            vec![block],
        );
        let results = self.ir.results(op).to_vec();
        for (name, result) in names.iter().zip(results) {
            self.update_binding(name, Binding::Regs(result));
        }
        Ok(())
    }
}
