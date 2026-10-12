// `sum(e)` with no group: a per-element expression over one thread's own
// slices, summed in the thread.
//
// A tensor slice with one span, `W[col, k :+ 4]`, is read by each thread
// on its own: one vector load when its address is provably aligned to the
// load's size, else a load per element. The expression is evaluated per
// element and the results added. `sum(i32(a) * i32(b))` over `i8` slices a
// multiple of four long is a chain of `dp4a`.

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};

use super::{Binding, Build, Rv, gcd};
use crate::ast::{BinOp, Expr, Sub};
use crate::ir::{Bytes, Extent, OpKind, Scalar, Space, Type, ValueId};

/// The most bytes one vector load moves.
const MAX_VECTOR_BYTES: i64 = 16;

impl Build {
    pub(crate) fn emit_lane_sum(&mut self, e: &Expr) -> Result<Rv> {
        let mut spans = Vec::new();
        collect_spans(e, &mut spans);
        ensure!(
            !spans.is_empty(),
            "`sum(e)` without `over` sums one thread's own slice, as `sum(A[0, k :+ 4])`; \
             a tile's rows are `rowsum`"
        );
        if let Some(sides) = dp4a_sides(e)
            && self.target.has_dp4a
            && sides.iter().flatten().all(|(span, _)| self.span_elem(span) == Some(Scalar::I8))
        {
            let mut runs = Vec::new();
            for (span, how) in sides.iter().flatten() {
                runs.push((self.lane_load(span)?, *how));
            }
            let n = runs[0].0.len();
            ensure!(
                runs.iter().all(|(r, _)| r.len() == n) && n % 4 == 0,
                "`sum(..)` over `i8` slices as `dp4a`: every slice one length, a multiple of four"
            );
            let side = |k: usize| runs.get(k).map_or(Bytes::Ones, |r| r.1);
            let (a, b) = (side(0), side(1));
            let mut acc = self.const_index(0);
            for q in 0..n / 4 {
                let mut operands = Vec::new();
                for (r, _) in &runs {
                    operands.extend_from_slice(&r[4 * q..4 * q + 4]);
                }
                operands.push(acc);
                acc = self.value(OpKind::Dot4 { a, b }, &operands, Type::INDEX);
            }
            return Ok(Rv::Scalar(acc));
        }
        let loaded = spans
            .iter()
            .map(|s| {
                let raw = self.lane_load(s)?;
                Ok(raw.into_iter().map(|v| self.widen(v)).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        let n = loaded[0].len();
        ensure!(
            loaded.iter().all(|l| l.len() == n),
            "`sum(e)`: every slice in it has the same length"
        );
        let names: Vec<String> = (0..spans.len()).map(|j| format!("#lane{j}")).collect();
        let per_element = substitute(e, &spans, &names);
        let mut total: Option<ValueId> = None;
        for i in 0..n {
            let scope = names
                .iter()
                .zip(&loaded)
                .map(|(name, l)| (name.clone(), Binding::Let { value: l[i], div: 1 }))
                .collect::<HashMap<_, _>>();
            self.scopes.push(scope);
            let v = self.emit_scalar(&per_element);
            self.scopes.pop();
            let v = v?;
            total = Some(match total {
                Some(t) => self.emit_binop(BinOp::Add, t, v)?,
                None => v,
            });
        }
        Ok(Rv::Scalar(total.expect("a slice has at least one element")))
    }

    /// `word(T[.., k :+ n], t)`: bytes `4t ..= 4t + 3` of a thread's run of
    /// `i8`, as the word they pack. The run is loaded whole, so the words of
    /// one run share its load once LLVM folds the repeats.
    pub(crate) fn emit_word(&mut self, slice: &Expr, at: &Expr) -> Result<Rv> {
        ensure!(
            self.span_elem(slice) == Some(Scalar::I8),
            "word reads the bytes of an i8 tensor, as `word(B[r, k :+ 4])`"
        );
        let Some(t) = self.const_fold(at).filter(|&t| t >= 0) else {
            bail!("word: which word must be a constant");
        };
        let bytes = self.lane_load(slice)?;
        let from = 4 * t as usize;
        ensure!(
            bytes.len().is_multiple_of(4) && from + 4 <= bytes.len(),
            "word {t} of a run of {} bytes: the run is whole words and holds it",
            bytes.len()
        );
        Ok(Rv::Scalar(self.value(OpKind::Word, &bytes[from..from + 4], Type::INDEX)))
    }

    /// An integer element widened to `index`, as a scalar load gives it.
    fn widen(&mut self, v: ValueId) -> ValueId {
        let elem = self.scalar_of(v);
        if elem.is_int() || elem.is_unsigned() {
            self.value(OpKind::IndexCast(Scalar::Index), &[v], Type::INDEX)
        } else {
            v
        }
    }

    /// The element type of the tensor a slice reads, emitting nothing.
    fn span_elem(&self, span: &Expr) -> Option<Scalar> {
        let Expr::Index { base, .. } = span else {
            return None;
        };
        let Expr::Var(name) = base.as_ref() else {
            return None;
        };
        let mem = match self.lookup(name)? {
            Binding::Tensor(t) | Binding::Tile(t) => t,
            _ => return None,
        };
        self.lane_source(mem).map(|(elem, _)| elem)
    }

    /// What a thread's own run may be read from: a tensor, or a whole
    /// shared tile of a static shape. Its element type and, per dimension,
    /// the extent's known divisor.
    fn lane_source(&self, mem: ValueId) -> Option<(Scalar, Vec<i64>)> {
        let divs = |shape: &[Extent], div: &dyn Fn(usize) -> i64| -> Vec<i64> {
            shape
                .iter()
                .enumerate()
                .map(|(d, e)| match e {
                    Extent::Fixed(n) => *n,
                    Extent::Dyn => div(d),
                })
                .collect()
        };
        match self.ir.ty(mem) {
            Type::Tensor(t) => Some((t.elem, divs(&t.shape, &|d| t.div[d]))),
            Type::Tile(t) if t.space == Space::Shared && t.is_static() => Some((t.elem, divs(&t.shape, &|_| 1))),
            _ => None,
        }
    }

    /// One thread's elements of a tensor slice with one span, unwidened.
    fn lane_load(&mut self, span: &Expr) -> Result<Vec<ValueId>> {
        let Expr::Index { base, subs } = span else {
            unreachable!("collect_spans keeps slices");
        };
        let (mem, binding) = self.mem_base(base)?;
        let source = match binding {
            Binding::Tensor(_) | Binding::Tile(_) => self.lane_source(mem),
            _ => None,
        };
        let Some((elem, extents)) = source else {
            bail!("a thread's own run reads a tensor or a whole shared tile, not a view");
        };
        ensure!(
            subs.len() == extents.len(),
            "expected {} subscripts, got {}",
            extents.len(),
            subs.len()
        );
        let mut indices = Vec::new();
        let mut divs = Vec::new();
        let mut span_at = None;
        for (d, sub) in subs.iter().enumerate() {
            let at = match sub {
                Sub::Point(e) => e,
                Sub::Span { start, len } => {
                    let Some(n) = self.const_fold(len).filter(|&n| n > 0) else {
                        bail!("`sum(e)`: a thread's slice has a constant length");
                    };
                    span_at = Some((d, n));
                    start
                }
                _ => bail!("`sum(e)`: a thread's slice is points and one `:+` span"),
            };
            indices.push(self.emit_index(at, "subscript")?);
            divs.push(self.expr_div(at));
        }
        let (dim, len) = span_at.expect("collect_spans keeps slices with a span");
        let bytes = elem.bytes().expect("a tensor element has a width");
        // The address in elements is a sum of index times stride, so its
        // divisor is the gcd of each index's divisor times its stride's.
        let rank = extents.len();
        let mut addr_div = 0;
        for (d, div) in divs.iter().enumerate() {
            let stride: i64 = extents[d + 1..rank].iter().product();
            addr_div = gcd(addr_div, div.saturating_mul(stride));
        }
        let vector_bytes = len * bytes;
        let aligned = dim == rank - 1
            && vector_bytes <= MAX_VECTOR_BYTES
            && (vector_bytes as u64).is_power_of_two()
            && (addr_div == 0 || (addr_div * bytes) % vector_bytes == 0);
        let align = if aligned { vector_bytes } else { 0 };
        let mut operands = vec![mem];
        operands.extend(indices);
        let op = self.op(
            OpKind::LaneLoad { dim, len, align },
            &operands,
            vec![Type::Scalar(elem); len as usize],
            Vec::new(),
        );
        Ok(self.ir.results(op).to_vec())
    }
}

/// The tensor slices with one `:+` span in `e`, in order.
fn collect_spans<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
    match e {
        Expr::Index { subs, .. }
            if subs.iter().filter(|s| matches!(s, Sub::Span { .. })).count() == 1 =>
        {
            out.push(e)
        }
        Expr::Unary { rhs, .. } => collect_spans(rhs, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_spans(lhs, out);
            collect_spans(rhs, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|a| collect_spans(a, out)),
        _ => {}
    }
}

/// `e` with each slice in `spans` replaced by its name.
fn substitute(e: &Expr, spans: &[&Expr], names: &[String]) -> Expr {
    if let Some(j) = spans.iter().position(|s| std::ptr::eq(*s, e)) {
        return Expr::Var(names[j].clone());
    }
    match e {
        Expr::Unary { op, rhs } => Expr::Unary {
            op: *op,
            rhs: Box::new(substitute(rhs, spans, names)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: *op,
            lhs: Box::new(substitute(lhs, spans, names)),
            rhs: Box::new(substitute(rhs, spans, names)),
        },
        Expr::Call { callee, args } => Expr::Call {
            callee: callee.clone(),
            args: args.iter().map(|a| substitute(a, spans, names)).collect(),
        },
        other => other.clone(),
    }
}

/// The `dp4a` form of `e`: `x * y`, or `x` alone (summed against ones),
/// each side a slice, optionally under `i32(..)`, as loaded or as
/// `s & c` or `(s >> k) & c`. A mask keeps every byte under 128, so it reads
/// the same signed, and a shift stays inside the byte.
fn dp4a_sides(e: &Expr) -> Option<[Option<(&Expr, Bytes)>; 2]> {
    if let Expr::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = e
    {
        return Some([Some(dp4a_side(lhs)?), Some(dp4a_side(rhs)?)]);
    }
    Some([Some(dp4a_side(e)?), None])
}

fn dp4a_side(x: &Expr) -> Option<(&Expr, Bytes)> {
    let x = match x {
        Expr::Call { callee, args } if callee == "i32" && args.len() == 1 => &args[0],
        _ => x,
    };
    let int = |e: &Expr| match e {
        Expr::Int(n) => Some(*n),
        _ => None,
    };
    match x {
        Expr::Index { .. } => Some((x, Bytes::Raw)),
        Expr::Binary {
            op: BinOp::BitAnd,
            lhs,
            rhs,
        } => {
            let mask = int(rhs).filter(|m| (0..128).contains(m))?;
            let (span, shift) = match lhs.as_ref() {
                Expr::Index { .. } => (lhs.as_ref(), 0),
                Expr::Binary {
                    op: BinOp::Shr,
                    lhs: inner,
                    rhs: by,
                } if matches!(inner.as_ref(), Expr::Index { .. }) => {
                    let shift = int(by).filter(|s| (0..8).contains(s))?;
                    // The bits past the byte's top must be masked off.
                    (mask < 1 << (8 - shift)).then_some(())?;
                    (inner.as_ref(), shift)
                }
                _ => return None,
            };
            Some((span, Bytes::Masked { shift: shift as u8, mask: mask as u8 }))
        }
        _ => None,
    }
}
