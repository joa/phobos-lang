// `for<unroll n> i in range(lo, hi, step)`: the body built `n` times a turn,
// then the iterations a whole turn does not cover as a plain loop.

use anyhow::{Result, bail, ensure};

use super::{Binding, Build};
use crate::ast::{BinOp, Expr, Stmt};
use crate::ir::{Bounds, ForInfo, OpKind, Type, ValueId};

/// The most copies `for<unroll n>` builds. Past a few the copies cost
/// registers and instruction cache rather than buy anything, and without a
/// bound a large n builds until memory runs out.
pub(crate) const MAX_UNROLL: i64 = 64;

impl Build {
    /// `n` of `for<unroll n>`, when it is a constant within [`MAX_UNROLL`].
    pub(crate) fn unroll_factor(&self, unroll: &Expr) -> Option<i64> {
        self.const_fold(unroll).filter(|n| (1..=MAX_UNROLL).contains(n))
    }

    /// The main loop steps `n * step` from `lo` to the last whole turn, each
    /// copy of the body in its own scope with `i` at `iv + k * step`; the rest
    /// runs one iteration at a time.
    pub(crate) fn emit_unrolled_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        unroll: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        let Some(n) = self.unroll_factor(unroll) else {
            bail!("`for<unroll n>` takes a constant n from 1 to {MAX_UNROLL}");
        };
        if n == 1 {
            return self.emit_for(var, start, end, step, body);
        }
        if self.pipeline == Some(true) {
            self.report
                .pipeline_declines
                .push(format!("loop `{var}`: an unrolled loop pipelines only as a register matmul"));
        }
        let Some(st) = step.map_or(Some(1), |e| self.const_fold(e)).filter(|&s| s > 0) else {
            bail!("`for<unroll n>` needs a constant positive step");
        };
        ensure!(
            self.regs_carried(body).is_empty(),
            "`for<unroll n>` does not apply to a loop that updates a register tile"
        );
        let (lo, hi, _, iv_div) = self.loop_bounds(start, end, step)?;

        // The end of the last whole turn: `lo + max(hi - lo, 0) / turn * turn`.
        let turn = self.const_index(n * st);
        let zero = self.const_index(0);
        let span = self.index_op(BinOp::Sub, hi, lo);
        let span = self.index_op(BinOp::Max, span, zero);
        let turns = self.index_op(BinOp::Div, span, turn);
        let whole = self.index_op(BinOp::Mul, turns, turn);
        let mid = self.index_op(BinOp::Add, lo, whole);

        let block = self.ir.new_block(&[Type::INDEX]);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        self.in_block(block, |b| {
            for k in 0..n {
                let at = match k {
                    0 => iv,
                    k => {
                        let off = b.const_index(k * st);
                        b.index_op(BinOp::Add, iv, off)
                    }
                };
                b.emit_scope(&[(var, Binding::Let { value: at, div: iv_div })], body)?;
            }
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        })?;
        self.plain_for(&[lo, mid, turn], block);

        let rest = self.body_block(var, iv_div, body)?;
        let st = self.const_index(st);
        self.plain_for(&[mid, hi, st], rest);
        Ok(())
    }

    fn index_op(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        self.value(OpKind::Binary(op), &[a, b], Type::INDEX)
    }

    /// A loop over `[lo, hi, step]` with `block` as its body.
    fn plain_for(&mut self, bounds: &[ValueId; 3], block: crate::ir::BlockId) {
        self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: 0,
                hoisted: 0,
                pipeline: None,
                unroll: 1,
            }),
            bounds,
            Vec::new(),
            vec![block],
        );
    }
}
