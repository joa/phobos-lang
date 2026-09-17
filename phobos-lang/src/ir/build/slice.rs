use anyhow::{Result, bail};

use super::{Binding, Build, DYN, dim_in_bounds, extent, gcd};
use crate::ast::{BinOp, Expr, Sub, UnOp};
use crate::ir::{Layout, OpKind, Slice, Space, TileType, Type, ValueId};

impl Build {
    /// Whether a slice of `size` elements starting at `start` provably stays
    /// inside a dynamic tensor extent whose known divisor is `extent_div`.
    pub(crate) fn dyn_in_bounds(
        &self,
        start: &Expr,
        size: i64,
        extent_div: i64,
        pending: &[&str],
    ) -> bool {
        if self
            .trimmed_ivs
            .iter()
            .map(String::as_str)
            .chain(pending.iter().copied())
            .any(|iv| start.uses_name(iv))
        {
            return true;
        }
        if size <= 1 && self.const_fold(start) == Some(0) {
            return true;
        }
        size > 0 && extent_div % size == 0 && self.expr_div(start) % size == 0
    }

    pub(crate) fn expr_div(&self, expr: &Expr) -> i64 {
        const CAP: i64 = 1 << 20;
        match expr {
            Expr::Int(n) => n.abs().min(CAP),
            Expr::Var(name) => match self.lookup(name) {
                Some(Binding::Let { div, .. }) => div,
                Some(_) => 1,
                None => self.shape_env.get(name).map_or(1, |v| v.abs().min(CAP)),
            },
            Expr::Unary { op: UnOp::Neg, rhs } => self.expr_div(rhs),
            Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                let (a, b) = (self.expr_div(lhs), self.expr_div(rhs));
                if a == 0 || b == 0 { 0 } else { a.saturating_mul(b).min(CAP) }
            }
            Expr::Binary {
                op: BinOp::Add | BinOp::Sub,
                lhs,
                rhs,
            } => gcd(self.expr_div(lhs), self.expr_div(rhs)),
            _ => 1,
        }
    }

    /// Folds an expression to a compile-time constant, through `@autotune`
    /// symbols.
    pub(crate) fn const_fold(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Int(n) => Some(*n),
            Expr::Var(name) if self.lookup(name).is_none() => self.shape_env.get(name).copied(),
            Expr::Unary { op: UnOp::Neg, rhs } => self.const_fold(rhs)?.checked_neg(),
            Expr::Binary { op, lhs, rhs } => {
                let (a, b) = (self.const_fold(lhs)?, self.const_fold(rhs)?);
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => a.checked_div(b),
                    BinOp::Rem => a.checked_rem(b),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The runtime extent of a dynamic dimension.
    pub(crate) fn dim_of(&mut self, src: ValueId, d: usize) -> ValueId {
        self.value(OpKind::Dim(d), &[src], Type::INDEX)
    }

    /// Lowers slice subscripts to a `Slice`. Offsets are always operands;
    /// sizes are static when they fold; a dimension that may run past the
    /// source carries its extent as a mask operand.
    pub(crate) fn emit_subview(&mut self, src: ValueId, subs: &[Sub]) -> Result<ValueId> {
        let src_shape = self.shape(src);
        let rank = src_shape.len();
        if subs.len() != rank {
            bail!("expected {rank} subscripts, got {}", subs.len());
        }

        let mut offsets = Vec::with_capacity(rank);
        let mut off_divs = Vec::with_capacity(rank);
        let mut dyn_sizes = Vec::new();
        let mut static_sizes = Vec::with_capacity(rank);
        let mut ragged = vec![false; rank];
        let mut proven = vec![false; rank];
        let rides_ragged = |cg: &Self, start: &Expr| {
            cg.ragged_iv
                .as_deref()
                .is_some_and(|iv| start.uses_name(iv))
        };
        for (i, sub) in subs.iter().enumerate() {
            match sub {
                Sub::Point(_) => {
                    bail!("mixing point and slice subscripts is not supported yet")
                }
                Sub::Span { start, len } => {
                    ragged[i] = rides_ragged(self, start);
                    offsets.push(self.emit_index(start, "slice start")?);
                    off_divs.push(self.expr_div(start));
                    match self.const_fold(len) {
                        Some(n) => {
                            proven[i] = self.dyn_in_bounds(start, n, self.div_of(src, i), &[]);
                            static_sizes.push(n)
                        }
                        None => {
                            dyn_sizes.push(self.emit_index(len, "slice length")?);
                            static_sizes.push(DYN);
                        }
                    }
                }
                Sub::Range { start, end } => {
                    ragged[i] = rides_ragged(self, start);
                    let off = self.emit_index(start, "slice start")?;
                    offsets.push(off);
                    off_divs.push(self.expr_div(start));
                    match (self.const_fold(start), self.const_fold(end)) {
                        (Some(a), Some(b)) => {
                            proven[i] = self.dyn_in_bounds(start, b - a, self.div_of(src, i), &[]);
                            static_sizes.push(b - a)
                        }
                        _ => {
                            let end_v = self.emit_index(end, "slice end")?;
                            let size = self.value(OpKind::Binary(BinOp::Sub), &[end_v, off], Type::INDEX);
                            dyn_sizes.push(size);
                            static_sizes.push(DYN);
                        }
                    }
                }
                Sub::Full => {
                    offsets.push(self.const_index(0));
                    off_divs.push(0);
                    if src_shape[i] != DYN {
                        static_sizes.push(src_shape[i]);
                    } else {
                        dyn_sizes.push(self.dim_of(src, i));
                        static_sizes.push(DYN);
                    }
                }
            }
        }

        let extent_div = |d: usize| {
            if src_shape[d] == DYN {
                self.div_of(src, d).max(4)
            } else {
                src_shape[d].abs().max(1)
            }
        };
        let stride_div = |i: usize| {
            (i + 1..rank)
                .map(extent_div)
                .try_fold(1i64, |acc: i64, d| acc.checked_mul(d))
                .map_or(1 << 20, |d| d.min(1 << 20))
        };
        let base_div = off_divs.iter().enumerate().fold(0i64, |acc, (i, &o)| {
            let term = if o == 0 { 0 } else { o.saturating_mul(stride_div(i)).min(1 << 20) };
            gcd(acc, term)
        });
        let align_div = (0..rank - 1).map(stride_div).fold(base_div, gcd);

        let mut masked = vec![false; rank];
        let mut extents = Vec::new();
        for i in 0..rank {
            if src_shape[i] != DYN {
                if dim_in_bounds(src_shape[i], static_sizes[i], off_divs[i]) {
                    continue;
                }
                extents.push(self.const_index(src_shape[i]));
            } else {
                if static_sizes[i] == DYN || (proven[i] && !ragged[i]) {
                    continue;
                }
                extents.push(self.dim_of(src, i));
            }
            masked[i] = true;
        }

        let space = match self.ir.ty(src) {
            Type::Tensor(_) => Space::Global,
            Type::Tile(t) => t.space,
            _ => unreachable!("mem_base only yields tensors and tiles"),
        };
        let ty = Type::Tile(TileType {
            elem: self.elem(src),
            shape: static_sizes.iter().map(|&d| extent(d)).collect(),
            layout: Layout {
                row_stride: None,
                swizzle: None,
                align_div,
            },
            space,
        });
        let divs: Vec<i64> = subs
            .iter()
            .enumerate()
            .map(|(i, s)| match s {
                Sub::Full => self.div_of(src, i),
                _ => 1,
            })
            .collect();
        let slice = Slice {
            sizes: static_sizes.iter().map(|&d| extent(d)).collect(),
            masked,
            divs: divs.clone(),
        };
        let mut operands = vec![src];
        operands.extend_from_slice(&offsets);
        operands.extend_from_slice(&dyn_sizes);
        operands.extend_from_slice(&extents);
        let view = self.value(OpKind::Slice(slice), &operands, ty);
        self.set_view_divs(view, divs);
        Ok(view)
    }

    /// A partially out-of-bounds slice staged into a fresh, in-bounds tile.
    pub(crate) fn materialize_masked(&mut self, view: ValueId) -> Result<ValueId> {
        let shape = self.shape(view);
        if shape.contains(&DYN) {
            bail!("a masked tensor slice needs a static shape");
        }
        let ty = self.like(view);
        Ok(self.value(OpKind::Materialize, &[view], ty))
    }

    /// Whether a tensor-slice expression can reach past its source on the
    /// last tile.
    pub(crate) fn slice_is_partial(&self, expr: &Expr) -> bool {
        self.slice_is_partial_within(expr, &[])
    }

    /// [`Build::slice_is_partial`] for a slice inside loops whose induction
    /// variables are `ivs` and whose bodies are not built yet.
    pub(crate) fn slice_is_partial_within(&self, expr: &Expr, ivs: &[&str]) -> bool {
        let Expr::Index { base, subs } = expr else {
            return false;
        };
        let Expr::Var(name) = &**base else {
            return false;
        };
        let mv = match self.lookup(name) {
            Some(Binding::Tensor(mv) | Binding::View(mv) | Binding::Tile(mv)) => mv,
            _ => return false,
        };
        let shape = self.shape(mv);
        if subs.len() != shape.len() {
            return false;
        }
        subs.iter().enumerate().any(|(d, s)| {
            let (start, size) = match s {
                Sub::Full | Sub::Point(_) => return false,
                Sub::Span { start, len } => (start, self.const_fold(len).unwrap_or(DYN)),
                Sub::Range { start, end } => {
                    let size = match (self.const_fold(start), self.const_fold(end)) {
                        (Some(a), Some(b)) => b - a,
                        _ => DYN,
                    };
                    (start, size)
                }
            };
            if shape[d] == DYN {
                return size != DYN && !self.dyn_in_bounds(start, size, self.div_of(mv, d), ivs);
            }
            !dim_in_bounds(shape[d], size, self.expr_div(start))
        })
    }

    /// The static shape of a tensor-slice expression, if it has one.
    pub(crate) fn slice_static_shape(&self, expr: &Expr) -> Option<Vec<i64>> {
        let Expr::Index { base, subs } = expr else {
            return None;
        };
        let Expr::Var(name) = &**base else {
            return None;
        };
        let Some(Binding::Tensor(src)) = self.lookup(name) else {
            return None;
        };
        let shape = self.shape(src);
        if subs.len() != shape.len() {
            return None;
        }
        subs.iter()
            .enumerate()
            .map(|(d, s)| match s {
                Sub::Span { len, .. } => self.const_fold(len),
                Sub::Range { start, end } => Some(self.const_fold(end)? - self.const_fold(start)?),
                Sub::Full => (shape[d] != DYN).then_some(shape[d]),
                Sub::Point(_) => None,
            })
            .collect()
    }

    /// The element type of the tensor an `A[...]` slice reads from.
    pub(crate) fn slice_tensor_elem(&self, expr: &Expr) -> Option<crate::ir::Scalar> {
        let Expr::Index { base, .. } = expr else {
            return None;
        };
        let Expr::Var(name) = &**base else {
            return None;
        };
        match self.lookup(name)? {
            Binding::Tensor(src) => Some(self.elem(src)),
            _ => None,
        }
    }
}
