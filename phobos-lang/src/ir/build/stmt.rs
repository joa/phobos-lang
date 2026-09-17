use anyhow::{Result, bail};

use super::{Binding, Build, DYN, Rv, gcd};
use crate::ast::{AssignOp, BinOp, Dim, Expr, Scalar as AstScalar, Stmt, Sub, Type as AstType};
use crate::ir::{
    BlockId, Bounds, ForInfo, Layout, OpKind, Scalar, Space, Stage, TileType, Type, ValueId,
};

/// Shared-memory bank period.
const SHARED_BANK_BYTES: i64 = 128;

impl Build {
    pub(crate) fn emit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { name, ty, value } => {
                if let Some(AstType::Tile(scalar, dims)) = ty {
                    let tile = self.emit_tile_decl(*scalar, dims, value)?;
                    self.bind(name, Binding::View(tile));
                } else {
                    let div = self.expr_div(value);
                    match self.emit_expr(value)? {
                        Rv::Scalar(v) => self.bind(name, Binding::Let { value: v, div }),
                        Rv::Tile(t) => self.bind(name, Binding::View(t)),
                    }
                }
            }
            Stmt::Var { name, ty, value } => {
                if let Some(AstType::Tile(scalar, dims)) = ty {
                    let tile = match value {
                        Some(value) => self.emit_tile_decl(*scalar, dims, value)?,
                        None => self.alloc_tile(*scalar, dims)?,
                    };
                    self.bind(name, Binding::Tile(tile));
                } else {
                    let Some(value) = value else {
                        bail!("'var {name}' without an initializer needs a tile type");
                    };
                    match self.emit_expr(value)? {
                        Rv::Scalar(v) => {
                            let elem = self.scalar_of(v);
                            let slot_t = Type::Tile(TileType {
                                elem,
                                shape: Vec::new(),
                                layout: Layout::CONTIGUOUS,
                                space: Space::Private,
                            });
                            let slot = self.value(OpKind::Alloc, &[], slot_t);
                            self.stmt(OpKind::Store, &[v, slot]);
                            self.bind(name, Binding::Var { slot, elem });
                        }
                        Rv::Tile(src) => self.bind_staged_tile(name, src, true)?,
                    }
                }
            }
            Stmt::Assign { target, op, value } => self.emit_assign(target, *op, value)?,
            Stmt::For {
                var,
                start,
                end,
                step,
                body,
            } => self.emit_for(var, start, end, step.as_ref(), body)?,
            Stmt::While { cond, body } => self.emit_while(cond, body)?,
            Stmt::If { cond, then, r#else } => self.emit_if(cond, then, r#else.as_deref())?,
            Stmt::Expr(e) => {
                self.emit_expr(e)?;
            }
        }
        Ok(())
    }

    /// The `Rv::Tile(src)` half of `var name = <tile expr>`: adopt a fresh
    /// temp's buffer outright, or stage a borrowed view into a fresh tile.
    fn bind_staged_tile(&mut self, name: &str, src: ValueId, sync: bool) -> Result<()> {
        if self.owned(src) {
            self.bind(name, Binding::Tile(src));
        } else {
            let (elem, shape) = (self.elem(src), self.shape(src));
            let pad = self.should_pad_stage(elem, &shape);
            if shape.contains(&DYN) {
                bail!("tile buffers must have a static shape");
            }
            let ty = if pad {
                self.padded_ty(elem, &shape)
            } else {
                self.shared_ty(elem, &shape)
            };
            let tile = self.value(OpKind::Stage(Stage { pad, sync }), &[src], ty);
            self.bind(name, Binding::Tile(tile));
        }
        Ok(())
    }

    /// Whether a staging tile's row pitch lands on a bank boundary, and so
    /// wants the WMMA pad.
    pub(crate) fn should_pad_stage(&self, elem: Scalar, shape: &[i64]) -> bool {
        if !self.pad_stage {
            return false;
        }
        let Some(&cols) = shape.last() else {
            return false;
        };
        if cols == DYN {
            return false;
        }
        let Some(width) = elem.bytes() else {
            return false;
        };
        let pitch_bytes = cols * width;
        pitch_bytes > 0 && pitch_bytes % SHARED_BANK_BYTES == 0
    }

    /// Length of the maximal run at the front of `stmts` of `var name =
    /// <tensor slice>` statements.
    fn stage_run(&self, stmts: &[Stmt]) -> usize {
        stmts
            .iter()
            .take_while(|s| {
                matches!(s, Stmt::Var { ty: None, value: Some(v), .. }
                    if self.slice_static_shape(v).is_some() && !self.slice_is_partial(v))
            })
            .count()
    }

    /// A staging run, every barrier but the last deferred.
    fn emit_staging_run(&mut self, stmts: &[Stmt]) -> Result<()> {
        let last = stmts.len() - 1;
        for (j, s) in stmts.iter().enumerate() {
            let Stmt::Var {
                name,
                value: Some(value),
                ..
            } = s
            else {
                bail!("stage_run matched a statement emit_staging_run cannot emit");
            };
            match self.emit_expr(value)? {
                Rv::Tile(src) => self.bind_staged_tile(name, src, j == last)?,
                Rv::Scalar(_) => bail!("stage_run matched a scalar-valued statement"),
            }
        }
        Ok(())
    }

    pub(crate) fn alloc_tile(&mut self, scalar: AstScalar, dims: &[Dim]) -> Result<ValueId> {
        let shape = self.tile_shape(dims)?;
        let ty = self.shared_ty(Scalar::from_ast(scalar), &shape);
        Ok(self.value(OpKind::Alloc, &[], ty))
    }

    /// A tile-typed let/var initializer.
    fn emit_tile_decl(&mut self, scalar: AstScalar, dims: &[Dim], value: &Expr) -> Result<ValueId> {
        let unfused = matches!(
            value,
            Expr::Call { callee, .. } if matches!(callee.as_str(),
                "exp" | "log" | "round" | "sqrt" | "tanh" | "tmax" | "rowmax" | "rowsum" | "cumsum"
                | "tril" | "transpose")
        ) && self.fusable_nodes(value).unwrap_or(0) < 1;
        if !unfused {
            let tile = self.alloc_tile(scalar, dims)?;
            self.store_tile(tile, AssignOp::Set, value)?;
            return Ok(tile);
        }

        let shape = self.tile_shape(dims)?;
        let elem = Scalar::from_ast(scalar);
        let Rv::Tile(t) = self.emit_expr(value)? else {
            bail!("expected a tile initializer");
        };
        if self.owned(t) && self.elem(t) == elem && self.shape(t) == shape {
            return Ok(t);
        }
        self.check_shapes(&self.shape(t), &shape, "tile init")?;
        let ty = self.shared_ty(elem, &shape);
        let tile = self.value(OpKind::Alloc, &[], ty);
        if self.elem(t) != elem {
            self.stmt(OpKind::Convert, &[t, tile]);
        } else {
            self.stmt(OpKind::Copy { sync: true }, &[t, tile]);
        }
        Ok(tile)
    }

    pub(crate) fn check_shapes(&self, a: &[i64], b: &[i64], what: &str) -> Result<()> {
        let ok = a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| x == DYN || y == DYN || x == y);
        if !ok {
            bail!(
                "{what}: shape mismatch ({} vs {})",
                super::fmt_shape(a),
                super::fmt_shape(b)
            );
        }
        Ok(())
    }

    pub(crate) fn emit_assign(&mut self, target: &Expr, op: AssignOp, value: &Expr) -> Result<()> {
        match target {
            Expr::Var(name) => match self.lookup(name) {
                Some(Binding::Var { slot, elem }) => {
                    let rhs = self.emit_scalar(value)?;
                    let rhs = match op {
                        AssignOp::Set => rhs,
                        AssignOp::Add => {
                            let cur = self.value(OpKind::Load, &[slot], Type::Scalar(elem));
                            self.emit_binop(BinOp::Add, cur, rhs)?
                        }
                    };
                    let rhs = self.coerce(rhs, elem)?;
                    self.stmt(OpKind::Store, &[rhs, slot]);
                }
                Some(Binding::Tile(tile)) => self.store_tile(tile, op, value)?,
                Some(Binding::Frags(fa)) => self.emit_frag_assign(name, fa, op, value)?,
                Some(Binding::View(_)) => {
                    bail!("'{name}' is a read-only view; declare it with `var` to assign")
                }
                Some(_) => bail!("'{name}' is not assignable (declare it with `var`)"),
                None => bail!("unknown identifier '{name}'"),
            },
            Expr::Index { base, subs } => {
                let (mv, binding) = self.mem_base(base)?;
                if subs.iter().all(|s| matches!(s, Sub::Point(_))) {
                    if matches!(binding, Binding::View(_)) {
                        bail!("cannot assign through a read-only view");
                    }
                    let indices = self.emit_indices(subs, self.shape(mv).len())?;
                    let rhs = self.emit_scalar(value)?;
                    let rhs = match op {
                        AssignOp::Set => rhs,
                        AssignOp::Add => {
                            let cur = self.load_scalar(mv, &indices);
                            self.emit_binop(BinOp::Add, cur, rhs)?
                        }
                    };
                    let rhs = self.coerce(rhs, self.elem(mv))?;
                    let mut operands = vec![rhs, mv];
                    operands.extend_from_slice(&indices);
                    self.stmt(OpKind::Store, &operands);
                } else {
                    if matches!(binding, Binding::View(_)) {
                        bail!("cannot assign through a read-only view");
                    }
                    self.check_sliceable(mv, &binding)?;
                    let view = self.emit_subview(mv, subs)?;
                    self.store_tile(view, op, value)?;
                }
            }
            _ => bail!("invalid assignment target"),
        }
        Ok(())
    }

    pub(crate) fn emit_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
    ) -> Result<()> {
        // Loop-invariant dot operands staged into the preheader once; the
        // loop carries the buffers as operands so they live through it.
        let frame = self.hoist_dot_staging(body)?;
        let bufs: Vec<ValueId> = frame.iter().map(|&(_, buf)| buf).collect();
        self.hoisted.push(frame);
        let result = self.emit_for_inner(var, start, end, step, body, &bufs);
        self.hoisted.pop();
        result
    }

    fn emit_for_inner(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        hoisted: &[ValueId],
    ) -> Result<()> {
        let carried = self.frag_carried(body);
        if !carried.is_empty() {
            return self.emit_frag_for(var, start, end, step, body, &carried, hoisted);
        }

        match self.pipeline_candidate(body) {
            Ok((staged, rest, info)) => {
                return self.emit_pipelined_for(var, start, end, step, &staged, rest, info, hoisted);
            }
            Err(decline) => {
                if self.pipeline_assert {
                    self.report
                        .pipeline_declines
                        .push(format!("loop `{var}`: {decline}"));
                }
            }
        }

        let const_step = match step {
            None => Some(1),
            Some(e) => self.const_fold(e),
        };
        if let (Some(lo), Some(hi), Some(st)) =
            (self.const_fold(start), self.const_fold(end), const_step)
            && st > 0
        {
            return self.emit_affine_for(var, lo, hi, st, body, hoisted);
        }

        let (lo, hi, st, iv_div) = self.loop_bounds(start, end, step)?;

        if self.needs_ragged_epilogue(body, var) {
            return self.emit_split_for(var, lo, hi, st, iv_div, body, hoisted);
        }

        let block = self.body_block(var, iv_div, body)?;
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(hoisted);
        self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: 0,
                hoisted: hoisted.len(),
                pipeline: None,
            }),
            &operands,
            Vec::new(),
            vec![block],
        );
        Ok(())
    }

    /// A loop body block: one `index` argument bound to `var`, the body
    /// built in a fresh scope, a closing yield.
    pub(crate) fn body_block(&mut self, var: &str, iv_div: i64, body: &[Stmt]) -> Result<BlockId> {
        let block = self.ir.new_block(&[Type::INDEX]);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        self.in_block(block, |b| {
            b.emit_scope(&[(var, Binding::Let { value: iv, div: iv_div })], body)?;
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(block)
    }

    /// Whether some slice in this loop body rides `var` into a dynamic
    /// tensor extent.
    fn needs_ragged_epilogue(&self, body: &[Stmt], var: &str) -> bool {
        let mut found = false;
        for stmt in body {
            stmt.walk_exprs(&mut |e| {
                let Expr::Index { base, subs } = e else {
                    return;
                };
                let Expr::Var(name) = &**base else {
                    return;
                };
                let Some(Binding::Tensor(src)) = self.lookup(name) else {
                    return;
                };
                let shape = self.shape(src);
                if subs.len() != shape.len() {
                    return;
                }
                found |= subs.iter().enumerate().any(|(d, sub)| {
                    shape[d] == DYN
                        && matches!(sub, Sub::Span { start, len }
                            if start.uses_name(var) && self.const_fold(len).is_some())
                });
            });
        }
        found
    }

    /// The trimmed main loop plus one masked replay of the body.
    #[allow(clippy::too_many_arguments)]
    fn emit_split_for(
        &mut self,
        var: &str,
        lo: ValueId,
        hi: ValueId,
        st: ValueId,
        iv_div: i64,
        body: &[Stmt],
        hoisted: &[ValueId],
    ) -> Result<()> {
        let span = self.value(OpKind::Binary(BinOp::Sub), &[hi, lo], Type::INDEX);
        let chunks = self.value(OpKind::IndexOp(crate::ir::IndexOp::DivU), &[span, st], Type::INDEX);
        let covered = self.value(OpKind::Binary(BinOp::Mul), &[chunks, st], Type::INDEX);
        let full = self.value(OpKind::Binary(BinOp::Add), &[lo, covered], Type::INDEX);

        self.trimmed_ivs.push(var.to_string());
        let block = self.body_block(var, iv_div, body);
        self.trimmed_ivs.pop();
        let block = block?;

        let replay = self.ir.new_block(&[]);
        let outer = self.ragged_iv.replace(var.to_string());
        let built = self.in_block(replay, |b| {
            b.emit_scope(&[(var, Binding::Let { value: full, div: iv_div })], body)?;
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        });
        self.ragged_iv = outer;
        built?;

        let mut operands = vec![lo, full, st, hi];
        operands.extend_from_slice(hoisted);
        self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: true,
                carried: 0,
                hoisted: hoisted.len(),
                pipeline: None,
            }),
            &operands,
            Vec::new(),
            vec![block, replay],
        );
        Ok(())
    }

    fn emit_affine_for(
        &mut self,
        var: &str,
        lo: i64,
        hi: i64,
        step: i64,
        body: &[Stmt],
        hoisted: &[ValueId],
    ) -> Result<()> {
        let block = self.body_block(var, gcd(lo, step), body)?;
        self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Affine { lo, hi, step },
                ragged: false,
                carried: 0,
                hoisted: hoisted.len(),
                pipeline: None,
            }),
            hoisted,
            Vec::new(),
            vec![block],
        );
        Ok(())
    }

    fn emit_while(&mut self, cond: &Expr, body: &[Stmt]) -> Result<()> {
        let before = self.ir.new_block(&[]);
        self.in_block(before, |b| {
            let c = b.emit_scalar(cond)?;
            let c = b.expect_bool(c, "while condition")?;
            b.stmt(OpKind::Condition, &[c]);
            Ok::<(), anyhow::Error>(())
        })?;
        let after = self.ir.new_block(&[]);
        self.in_block(after, |b| {
            b.emit_scope(&[], body)?;
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        })?;
        self.op(OpKind::While, &[], Vec::new(), vec![before, after]);
        Ok(())
    }

    fn emit_if(&mut self, cond: &Expr, then: &[Stmt], els: Option<&[Stmt]>) -> Result<()> {
        let c = self.emit_scalar(cond)?;
        let c = self.expect_bool(c, "if condition")?;
        let then_block = self.ir.new_block(&[]);
        self.in_block(then_block, |b| {
            b.emit_scope(&[], then)?;
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        })?;
        let mut blocks = vec![then_block];
        if let Some(els) = els {
            let else_block = self.ir.new_block(&[]);
            self.in_block(else_block, |b| {
                b.emit_scope(&[], els)?;
                b.stmt(OpKind::Yield, &[]);
                Ok::<(), anyhow::Error>(())
            })?;
            blocks.push(else_block);
        }
        self.op(OpKind::If, &[c], Vec::new(), blocks);
        Ok(())
    }

    /// Emits stmts in a fresh scope, pre-populated with given bindings.
    pub(crate) fn emit_scope(&mut self, bindings: &[(&str, Binding)], stmts: &[Stmt]) -> Result<()> {
        self.push_scope();
        for (name, binding) in bindings {
            self.bind(name, binding.clone());
        }
        let result = self.emit_stmts(stmts);
        self.pop_scope();
        result
    }

    pub(crate) fn emit_stmts(&mut self, stmts: &[Stmt]) -> Result<()> {
        let mut i = 0;
        while i < stmts.len() {
            let consumed = if let Some(consumed) = self.matmul_candidate(&stmts[i..])? {
                consumed
            } else if let Some(plan) = self.frag_acc_candidate(&stmts[i..]) {
                self.bind_frag_acc(&plan);
                1
            } else {
                let run = self.stage_run(&stmts[i..]);
                if run >= 2 {
                    self.emit_staging_run(&stmts[i..i + run])?;
                    run
                } else {
                    self.emit_stmt(&stmts[i])?;
                    1
                }
            };
            i += consumed;
        }
        Ok(())
    }
}

