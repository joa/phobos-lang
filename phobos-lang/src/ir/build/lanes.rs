// Collectives and shuffles, built from one shuffle op and ordinary scalar
// ops: a lane collective is an xor butterfly, and a CTA collective adds one
// exchange of every warp's result through shared memory.

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};

use super::{Binding, Build, Rv};
use crate::ast::lanes::{self, COMBINE_A, COMBINE_B, Collective, Fold, Group, Mode};
use crate::ast::{BinOp, Expr, ExprKind};
use crate::ir::{OpKind, Scalar, ThreadIndex, Type, ValueId};

impl Build {
    /// A per-thread builtin: a collective, a shuffle, a thread's run, or a
    /// word op. None for any other callee.
    pub(crate) fn emit_lane_builtin(&mut self, callee: &str, args: &[Expr]) -> Result<Option<Rv>> {
        if let Some(c) = lanes::as_collective(callee, args) {
            if c.fold == Fold::Reduce && matches!(c.combine, Some(Expr { kind: ExprKind::Var(_), .. })) {
                bail!("`reduce<..>(x, f)` combines with a `fn f(a, b) = ..`, and none is defined");
            }
            return self.emit_collective(c).map(Some);
        }
        if let Some((mode, value, operand, width)) = lanes::as_shuffle(callee, args) {
            return self.emit_shuffle(mode, value, operand, width).map(Some);
        }
        Ok(Some(match callee {
            "sum" if args.len() == 1 => self.emit_lane_sum(&args[0])?,
            "elem" => {
                let [slice, at] = args else {
                    bail!("elem takes a thread's run and which element of it, as `elem(X[r, k :+ 4], 2)`");
                };
                self.emit_elem(slice, at)?
            }
            // Statements in practice; the zero is what the source sees.
            "store" => {
                let [slice, values @ ..] = args else {
                    bail!("store takes a thread's run and its values, as `store(Y[r, k :+ 2], a, b)`");
                };
                self.emit_lane_store(slice, values)?
            }
            // A statement in practice; the zero is what the source sees.
            "barrier" => {
                ensure!(args.is_empty(), "barrier takes no arguments");
                self.stmt(OpKind::Barrier, &[]);
                Rv::Scalar(self.const_index(0))
            }
            "word" => match args {
                [slice] => self.emit_word(slice, &Expr::int(0))?,
                [slice, at] => self.emit_word(slice, at)?,
                _ => bail!("word takes a thread's bytes and which word of them, as `word(B[r, k :+ 16], 2)`"),
            },
            "hword" => match args {
                [slice] => self.emit_hword(slice, &Expr::int(0))?,
                [slice, at] => self.emit_hword(slice, at)?,
                _ => bail!("hword takes a thread's bytes and which halfword of them, as `hword(B[r, k :+ 4], 1)`"),
            },
            "dp4a" => {
                let [x, y, acc] = args else {
                    bail!("dp4a takes two words and an accumulator");
                };
                let operands = [self.emit_index(x, "dp4a")?, self.emit_index(y, "dp4a")?, self.emit_index(acc, "dp4a")?];
                ensure!(self.target.has_dp4a, "dp4a: this target has no dp4a");
                Rv::Scalar(self.value(OpKind::Dp4a, &operands, Type::INDEX))
            }
            "popcount" => {
                let [x] = args else {
                    bail!("popcount takes one integer");
                };
                let x = self.emit_index(x, "popcount")?;
                Rv::Scalar(self.value(OpKind::PopCount, &[x], Type::INDEX))
            }
            "byteperm" => {
                let [x, y, sel] = args else {
                    bail!("byteperm takes two words and a selector");
                };
                let operands = [
                    self.emit_index(x, "byteperm")?,
                    self.emit_index(y, "byteperm")?,
                    self.emit_index(sel, "byteperm")?,
                ];
                Rv::Scalar(self.value(OpKind::BytePerm, &operands, Type::INDEX))
            }
            "f16bits" => {
                let [bits] = args else {
                    bail!("f16bits takes one integer, the half's bits");
                };
                let bits = self.emit_index(bits, "f16bits")?;
                Rv::Scalar(self.value(OpKind::F16Bits, &[bits], Type::Scalar(Scalar::F16)))
            }
            _ => return Ok(None),
        }))
    }

    pub(crate) fn emit_collective(&mut self, c: Collective<'_>) -> Result<Rv> {
        let v = self.emit_scalar(c.value)?;
        let elem = self.scalar_of(v);
        match c.fold {
            Fold::Any | Fold::All => ensure!(
                elem == Scalar::Bool,
                "`{}` folds conditions, not {}",
                c.fold.name(),
                elem.mlir_name()
            ),
            Fold::Sum | Fold::Max | Fold::Min | Fold::ArgMax => ensure!(
                elem != Scalar::Bool,
                "`{}` folds numbers, not conditions",
                c.fold.name()
            ),
            Fold::Reduce => {}
        }
        let width = self.lane_width(c.width, c.fold.name())?;
        let combine = c.combine;
        let lanes = self.target.lanes;
        let index = match c.group {
            Group::Lanes => ThreadIndex::Lane,
            Group::Cta => ThreadIndex::Thread,
        };
        let idx = (c.fold == Fold::ArgMax).then(|| self.value(OpKind::Thread(index), &[], Type::INDEX));
        let (mut v, mut idx) = self.butterfly(c.fold, v, idx, width, combine)?;
        if width == lanes {
            v = self.value(OpKind::Converged(Group::Lanes), &[v], Type::Scalar(elem));
            idx = idx.map(|i| self.value(OpKind::Converged(Group::Lanes), &[i], Type::INDEX));
        }
        if c.group == Group::Cta {
            let parts = self.exchange(v)?;
            let ids = idx.map(|i| self.exchange(i)).transpose()?;
            v = parts[0];
            idx = ids.as_ref().map(|ids| ids[0]);
            for w in 1..parts.len() {
                let other_idx = ids.as_ref().map(|ids| ids[w]);
                (v, idx) = self.fold_pair(c.fold, (v, idx), (parts[w], other_idx), combine)?;
            }
        }
        Ok(Rv::Scalar(match c.fold {
            Fold::ArgMax => idx.expect("argmax carries its index"),
            _ => v,
        }))
    }

    pub(crate) fn emit_shuffle(&mut self, mode: Mode, value: &Expr, operand: &Expr, width: &Expr) -> Result<Rv> {
        let v = self.emit_scalar(value)?;
        let width = self.lane_width(width, "shfl")?;
        let operand = self.emit_index(operand, "shuffle lane")?;
        Ok(Rv::Scalar(self.shuffle(mode, v, operand, width)))
    }

    /// `width` folded, 0 meaning the whole warp.
    fn lane_width(&self, width: &Expr, what: &str) -> Result<i64> {
        let lanes = self.target.lanes;
        match self.const_fold(width) {
            Some(0) => Ok(lanes),
            Some(w) if w > 0 && (w as u64).is_power_of_two() && lanes % w == 0 => Ok(w),
            Some(w) => bail!("`{what}<width {w}>`: a width is a power of two dividing {lanes}"),
            None => bail!("`{what}<width ..>` must be a constant"),
        }
    }

    fn shuffle(&mut self, mode: Mode, v: ValueId, operand: ValueId, width: i64) -> ValueId {
        let ty = Type::Scalar(self.scalar_of(v));
        self.value(OpKind::Shuffle { mode, width }, &[v, operand], ty)
    }

    /// An xor butterfly over `width` lanes, carrying an index for argmax.
    fn butterfly(
        &mut self,
        fold: Fold,
        mut v: ValueId,
        mut idx: Option<ValueId>,
        width: i64,
        combine: Option<&Expr>,
    ) -> Result<(ValueId, Option<ValueId>)> {
        let mut mask = width / 2;
        while mask >= 1 {
            let m = self.const_index(mask);
            let ov = self.shuffle(Mode::Xor, v, m, width);
            let oi = idx.map(|i| self.shuffle(Mode::Xor, i, m, width));
            (v, idx) = self.fold_pair(fold, (v, idx), (ov, oi), combine)?;
            mask /= 2;
        }
        Ok((v, idx))
    }

    /// Two partial results folded into one.
    fn fold_pair(
        &mut self,
        fold: Fold,
        (a, ai): (ValueId, Option<ValueId>),
        (b, bi): (ValueId, Option<ValueId>),
        combine: Option<&Expr>,
    ) -> Result<(ValueId, Option<ValueId>)> {
        let op = match fold {
            Fold::Sum => BinOp::Add,
            Fold::Max => BinOp::Max,
            Fold::Min => BinOp::Min,
            Fold::Any => BinOp::Or,
            Fold::All => BinOp::And,
            Fold::ArgMax => {
                let (ai, bi) = (ai.expect("argmax index"), bi.expect("argmax index"));
                // The larger value wins, and the smaller index on a tie.
                let gt = self.emit_binop(BinOp::Gt, b, a)?;
                let eq = self.emit_binop(BinOp::Eq, b, a)?;
                let lt = self.emit_binop(BinOp::Lt, bi, ai)?;
                let tie = self.emit_binop(BinOp::And, eq, lt)?;
                let take = self.emit_binop(BinOp::Or, gt, tie)?;
                let v = self.select(take, b, a)?;
                let i = self.select(take, bi, ai)?;
                return Ok((v, Some(i)));
            }
            Fold::Reduce => {
                let body = combine.expect("the parser gives reduce its combine");
                let elem = self.scalar_of(a);
                let mut scope = HashMap::new();
                scope.insert(COMBINE_A.to_string(), Binding::Let { value: a, div: 1 });
                scope.insert(COMBINE_B.to_string(), Binding::Let { value: b, div: 1 });
                self.scopes.push(scope);
                let out = self.emit_scalar(body);
                self.scopes.pop();
                return Ok((self.coerce(out?, elem)?, None));
            }
        };
        Ok((self.emit_binop(op, a, b)?, None))
    }

    fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> Result<ValueId> {
        let ty = Type::Scalar(self.scalar_of(a));
        Ok(self.value(OpKind::Select, &[cond, a, b], ty))
    }

    /// Every warp's copy of a warp-uniform value, in warp order, in every
    /// thread.
    fn exchange(&mut self, v: ValueId) -> Result<Vec<ValueId>> {
        let lanes = self.target.lanes;
        ensure!(
            self.cta_threads % lanes == 0,
            "a CTA collective needs whole warps; {} threads are not",
            self.cta_threads
        );
        let warps = (self.cta_threads / lanes) as usize;
        let ty = Type::Scalar(self.scalar_of(v));
        let op = self.op(OpKind::Exchange, &[v], vec![ty; warps], Vec::new());
        Ok(self.ir.results(op).to_vec())
    }
}
