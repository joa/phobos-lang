use anyhow::{Result, anyhow, bail};

use super::{Binding, Build, DYN, Rv, broadcast_shape, fmt_shape};
use crate::ast::{BinOp, Expr, ExprKind, Sub, UnOp};
use crate::ir::{Map, OpKind, Scalar, Type, ValueId};

impl Build {
    pub(crate) fn emit_expr(&mut self, expr: &Expr) -> Result<Rv> {
        self.at_span(expr.span, |b| b.emit_expr_kind(expr))
    }

    fn emit_expr_kind(&mut self, expr: &Expr) -> Result<Rv> {
        match &expr.kind {
            ExprKind::Generate(_) => {
                bail!("`generate(f)` is the value of a typed tile, as `let w: tile<i8>[r, c] = generate(f)`")
            }
            ExprKind::Int(n) => Ok(Rv::Scalar(self.const_index(*n))),
            ExprKind::Float(v) => Ok(Rv::Scalar(self.const_f32(*v))),
            ExprKind::Bool(b) => Ok(Rv::Scalar(self.const_bool(*b))),
            ExprKind::Var(name) => match self.lookup(name) {
                Some(Binding::Let { value, .. }) => Ok(Rv::Scalar(value)),
                Some(Binding::Var { slot, elem }) => {
                    Ok(Rv::Scalar(self.value(OpKind::Load, &[slot], Type::Scalar(elem))))
                }
                Some(Binding::Tensor(_)) => {
                    bail!("tensor `{name}` used as a value; index or slice it")
                }
                Some(Binding::View(t) | Binding::Tile(t)) => Ok(Rv::Tile(t)),
                Some(Binding::Frags(_)) => bail!(
                    "fragment accumulator `{name}` can only be scaled, dot-accumulated, or stored"
                ),
                Some(Binding::Regs(r)) => Ok(Rv::Tile(self.materialize_regs(r)?)),
                Some(Binding::Generated(_)) => {
                    bail!("generated tile `{name}` is read only as a scaled `dot_t`'s weight here")
                }
                None => match self.shape_env.get(name) {
                    Some(&v) => Ok(Rv::Scalar(self.const_index(v))),
                    None => Err(self.unknown(name)),
                },
            },
            ExprKind::Unary { op, rhs } => {
                let v = match (op, self.emit_expr(rhs)?) {
                    // -t becomes 0 - t
                    (UnOp::Neg, Rv::Tile(t)) => {
                        let zero = self.const_f32(0.0);
                        let out = self.emit_tile_scalar(BinOp::Sub, t, zero, true)?;
                        return Ok(Rv::Tile(out));
                    }
                    // ~t becomes t ^ -1
                    (UnOp::BitNot, Rv::Tile(t)) => {
                        let ones = self.const_index(-1);
                        let out = self.emit_tile_scalar(BinOp::BitXor, t, ones, false)?;
                        return Ok(Rv::Tile(out));
                    }
                    (UnOp::Not, Rv::Tile(_)) => bail!("`!` needs a bool operand, got a tile"),
                    (_, Rv::Scalar(v)) => v,
                };
                let t = self.scalar_of(v);
                let ok = match op {
                    UnOp::Neg => t.is_float() || t == Scalar::Index,
                    UnOp::Not => t == Scalar::Bool,
                    UnOp::BitNot => t == Scalar::Index,
                };
                if !ok {
                    match op {
                        UnOp::Neg => bail!("`-` needs a numeric operand, got {}", t.mlir_name()),
                        UnOp::Not => bail!("`!` needs a bool operand, got {}", t.mlir_name()),
                        UnOp::BitNot => bail!("`~` needs an integer operand, got {}", t.mlir_name()),
                    }
                }
                Ok(Rv::Scalar(self.value(OpKind::Unary(*op), &[v], Type::Scalar(t))))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.emit_expr(lhs)?;
                let r = self.emit_expr(rhs)?;
                match (l, r) {
                    (Rv::Scalar(a), Rv::Scalar(b)) => Ok(Rv::Scalar(self.emit_binop(*op, a, b)?)),
                    (Rv::Tile(a), Rv::Tile(b)) => Ok(Rv::Tile(self.emit_tile_binary(*op, a, b)?)),
                    (Rv::Tile(a), Rv::Scalar(b)) => {
                        Ok(Rv::Tile(self.emit_tile_scalar(*op, a, b, false)?))
                    }
                    (Rv::Scalar(a), Rv::Tile(b)) => {
                        Ok(Rv::Tile(self.emit_tile_scalar(*op, b, a, true)?))
                    }
                }
            }
            ExprKind::Index { base, subs } => {
                let (mv, binding) = self.mem_base(base)?;
                if super::slice::is_point_load(subs) {
                    let indices = self.emit_indices(subs, self.shape(mv).len())?;
                    Ok(Rv::Scalar(self.load_scalar(mv, &indices)))
                } else {
                    self.check_sliceable(mv, &binding)?;
                    let view = self.emit_subview(mv, subs)?;
                    if self.is_masked(view) {
                        Ok(Rv::Tile(self.materialize_masked(view)?))
                    } else {
                        Ok(Rv::Tile(view))
                    }
                }
            }
            ExprKind::Call { callee, args } => self.emit_call(callee, args),
        }
    }

    pub(crate) fn emit_scalar(&mut self, expr: &Expr) -> Result<ValueId> {
        match self.emit_expr(expr)? {
            Rv::Scalar(v) => Ok(v),
            Rv::Tile(_) => bail!("expected a scalar value, got a tile"),
        }
    }

    pub(crate) fn emit_index(&mut self, expr: &Expr, what: &str) -> Result<ValueId> {
        let v = self.emit_scalar(expr)?;
        self.expect_index(v, what)
    }

    pub(crate) fn expect_index(&self, v: ValueId, what: &str) -> Result<ValueId> {
        if self.scalar_of(v) == Scalar::Index {
            Ok(v)
        } else {
            bail!("{what} must be an integer, got {}", self.scalar_of(v).mlir_name())
        }
    }

    pub(crate) fn expect_bool(&self, v: ValueId, what: &str) -> Result<ValueId> {
        if self.scalar_of(v) == Scalar::Bool {
            Ok(v)
        } else {
            bail!("{what} must be a bool, got {}", self.scalar_of(v).mlir_name())
        }
    }

    /// `(lo, hi, step, iv_div)` of a loop with dynamic bounds.
    pub(crate) fn loop_bounds(
        &mut self,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
    ) -> Result<(ValueId, ValueId, ValueId, i64)> {
        let lo = self.emit_index(start, "loop start")?;
        let hi = self.emit_index(end, "loop end")?;
        let st = match step {
            Some(e) => self.emit_index(e, "loop step")?,
            None => self.const_index(1),
        };
        let iv_div = super::gcd(self.expr_div(start), step.map_or(1, |e| self.expr_div(e)));
        Ok((lo, hi, st, iv_div))
    }

    pub(crate) fn emit_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        // A comparison narrows an index to the sized integer on the other
        // side, so `atomic_cas(..) == 0` reads naturally. Never the other
        // way: a sized integer must not become an index, or a loop bound.
        let (lhs, rhs) = match (self.scalar_of(lhs), self.scalar_of(rhs)) {
            (Scalar::Index, t) if op.is_compare() && t.is_int() => (self.coerce(lhs, t)?, rhs),
            (t, Scalar::Index) if op.is_compare() && t.is_int() => (lhs, self.coerce(rhs, t)?),
            _ => (lhs, rhs),
        };
        let (lhs, rhs) = self.unify(lhs, rhs)?;
        let t = self.scalar_of(lhs);
        // A sized integer is an element type; a scalar integer is an index.
        // Only a comparison takes sized integers, which yields a bool.
        let result = op.result(t).filter(|_| !t.is_int() || op.is_compare()).ok_or_else(|| {
            anyhow!("`{}` does not take {} operands", op.symbol(), t.mlir_name())
        })?;
        Ok(self.value(OpKind::Binary(op), &[lhs, rhs], Type::Scalar(result)))
    }

    /// Element-wise tile arithmetic into a fresh buffer.
    pub(crate) fn emit_tile_binary(&mut self, op: BinOp, a: ValueId, b: ValueId) -> Result<ValueId> {
        let (ash, bsh) = (self.shape(a), self.shape(b));
        let shape = broadcast_shape(&ash, &bsh).ok_or_else(|| {
            anyhow!(
                "elementwise tile op: shapes {} and {} are not broadcast-compatible",
                fmt_shape(&ash),
                fmt_shape(&bsh)
            )
        })?;
        if shape.contains(&DYN) {
            bail!("elementwise tile result shape must be static");
        }
        let (wa, wb) = self.widen_pair(a, b)?;
        check_tile_op(op, self.elem(wa))?;
        let ty = self.shared_ty(self.elem(wa), &shape);
        Ok(self.value(OpKind::Map(Map::Binary(op)), &[wa, wb], ty))
    }

    /// Widens two tiles to a common element type.
    fn widen_pair(&mut self, a: ValueId, b: ValueId) -> Result<(ValueId, ValueId)> {
        let (ae, be) = (self.elem(a), self.elem(b));
        if ae == be {
            return Ok((a, b));
        }
        let want = numeric_join(ae, be).ok_or_else(|| {
            anyhow!(
                "elementwise tile op: no common type for {} and {} operands",
                ae.mlir_name(),
                be.mlir_name()
            )
        })?;
        let widen = |cg: &mut Self, t: ValueId| -> Result<ValueId> {
            if cg.elem(t) == want {
                return Ok(t);
            }
            cg.tile_cast(t, want)
        };
        let wa = widen(self, a)?;
        let wb = widen(self, b)?;
        Ok((wa, wb))
    }

    /// A tile converted to another element type, into a fresh buffer.
    pub(crate) fn tile_cast(&mut self, src: ValueId, want: Scalar) -> Result<ValueId> {
        let shape = self.shape(src);
        if shape.contains(&DYN) {
            bail!("a tile conversion needs a static tile shape");
        }
        let ty = self.shared_ty(want, &shape);
        Ok(self.value(OpKind::Map(Map::Unary(crate::ir::ElemStep::Cast(want))), &[src], ty))
    }

    /// Element-wise tile*scalar (or scalar*tile) into a fresh buffer.
    pub(crate) fn emit_tile_scalar(
        &mut self,
        op: BinOp,
        tile: ValueId,
        scalar: ValueId,
        scalar_left: bool,
    ) -> Result<ValueId> {
        check_tile_op(op, self.elem(tile))?;
        let ty = self.like(tile);
        Ok(self.value(OpKind::Map(Map::Scalar { op, scalar_left }), &[tile, scalar], ty))
    }

    pub(crate) fn unify(&mut self, lhs: ValueId, rhs: ValueId) -> Result<(ValueId, ValueId)> {
        let (lt, rt) = (self.scalar_of(lhs), self.scalar_of(rhs));
        if lt == rt {
            return Ok((lhs, rhs));
        }
        if let Some(want) = float_join(lt, rt) {
            return Ok((self.float_cast(lhs, want), self.float_cast(rhs, want)));
        }
        bail!("mismatched operand types: {} vs {}", lt.mlir_name(), rt.mlir_name())
    }

    /// Coerces a value to `want` for a store.
    pub(crate) fn coerce(&mut self, value: ValueId, want: Scalar) -> Result<ValueId> {
        let t = self.scalar_of(value);
        if t == want {
            Ok(value)
        } else if t.is_float() && want.is_float() {
            Ok(self.float_cast(value, want))
        } else if t == Scalar::Index && (want.is_int() || want.is_unsigned()) {
            Ok(self.value(OpKind::IndexCast(want), &[value], Type::Scalar(want)))
        } else if t == Scalar::Index && want.is_float() {
            Ok(self.numeric_cast(value, want))
        } else {
            bail!(
                "type mismatch: cannot store {} where {} is expected",
                t.mlir_name(),
                want.mlir_name()
            )
        }
    }

    pub(crate) fn float_cast(&mut self, value: ValueId, want: Scalar) -> ValueId {
        self.numeric_cast(value, want)
    }

    pub(crate) fn numeric_cast(&mut self, value: ValueId, want: Scalar) -> ValueId {
        if self.scalar_of(value) == want {
            return value;
        }
        self.value(OpKind::Cast(want), &[value], Type::Scalar(want))
    }

    /// Resolves the base of an `A[...]` expression.
    pub(crate) fn mem_base(&self, base: &Expr) -> Result<(ValueId, Binding)> {
        let ExprKind::Var(name) = &base.kind else {
            bail!("only named tensors and tiles can be indexed");
        };
        match self.lookup(name) {
            Some(binding) => match &binding {
                Binding::Tensor(v) | Binding::View(v) | Binding::Tile(v) => Ok((*v, binding.clone())),
                _ => bail!("`{name}` is not a tensor or tile"),
            },
            None => Err(self.unknown(name)),
        }
    }

    pub(crate) fn emit_indices(&mut self, subs: &[Sub], rank: usize) -> Result<Vec<ValueId>> {
        if subs.len() != rank {
            bail!("expected {rank} subscripts, got {}", subs.len());
        }
        subs.iter()
            .map(|sub| match sub {
                Sub::Point(e) => self.emit_index(e, "subscript"),
                _ => bail!("a scalar load takes a point subscript in every dimension"),
            })
            .collect()
    }

    /// Loads a scalar element; integer elements are widened to index.
    pub(crate) fn load_scalar(&mut self, mem: ValueId, indices: &[ValueId]) -> ValueId {
        let elem = self.elem(mem);
        let mut operands = vec![mem];
        operands.extend_from_slice(indices);
        let v = self.value(OpKind::Load, &operands, Type::Scalar(elem));
        if elem.is_int() || elem.is_unsigned() {
            self.value(OpKind::IndexCast(Scalar::Index), &[v], Type::INDEX)
        } else {
            v
        }
    }

    /// Fails unless a slice of this binding is a subview the type system can name.
    pub(crate) fn check_sliceable(&self, mv: ValueId, binding: &Binding) -> Result<()> {
        if !matches!(binding, Binding::Tensor(_) | Binding::Tile(_) | Binding::View(_)) {
            bail!("only tensors, tiles and their views can be sliced");
        }
        if matches!(binding, Binding::View(_)) && self.is_masked(mv) {
            bail!("a partially out-of-bounds view cannot be sliced; slice the tensor it views instead");
        }
        if let Type::Tile(t) = self.ir.ty(mv)
            && t.layout.swizzle.is_some()
        {
            bail!("a swizzled staging tile cannot be sliced");
        }
        Ok(())
    }

    // Expression classifiers.

    pub(crate) fn as_scale_mul<'a>(&self, expr: &'a Expr) -> Option<(&'a Expr, &'a Expr)> {
        let ExprKind::Binary {
            op: BinOp::Mul,
            lhs,
            rhs,
        } = &expr.kind
        else {
            return None;
        };
        if self.is_scalar_expr(lhs) && self.is_tile_expr(rhs) {
            Some((lhs, rhs))
        } else if self.is_tile_expr(lhs) && self.is_scalar_expr(rhs) {
            Some((rhs, lhs))
        } else {
            None
        }
    }

    pub(crate) fn is_scalar_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Float(_) | ExprKind::Int(_) => true,
            ExprKind::Var(name) => matches!(
                self.lookup(name),
                Some(Binding::Let { .. }) | Some(Binding::Var { .. })
            ),
            _ => false,
        }
    }

    /// Whether `expr` slices a tensor or tile rather than loading one element.
    pub(crate) fn is_slice_expr(&self, expr: &Expr) -> bool {
        matches!(&expr.kind, ExprKind::Index { subs, .. } if !super::slice::is_point_load(subs))
    }

    pub(crate) fn is_tile_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Var(name) => matches!(
                self.lookup(name),
                Some(Binding::Tile(_)) | Some(Binding::View(_))
            ),
            _ => false,
        }
    }
}

/// Fails unless a tile of `elem` takes `op` element by element.
pub(crate) fn check_tile_op(op: BinOp, elem: Scalar) -> Result<()> {
    if elem.is_unsigned() {
        bail!(
            "a {elem} tile is storage: convert it first, as in `i32(t) {} ...`",
            op.symbol()
        );
    }
    if op.is_compare() || matches!(op, BinOp::And | BinOp::Or) {
        bail!(
            "`{}` on tiles has no bool tile to produce; select with \
             `where(a {} b, x, y)` instead",
            op.symbol(),
            op.symbol()
        );
    }
    if op.is_bitwise() && !elem.is_int() {
        bail!("`{}` needs integer tile elements, got {}", op.symbol(), elem.mlir_name());
    }
    Ok(())
}

/// The narrowest float type both operands convert into without loss; f16
/// and bf16 join to f32.
pub(crate) fn float_join(a: Scalar, b: Scalar) -> Option<Scalar> {
    if a == b {
        return a.is_float().then_some(a);
    }
    let (ab, bb) = (a.float_bits()?, b.float_bits()?);
    Some(match ab.cmp(&bb) {
        std::cmp::Ordering::Greater => a,
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => Scalar::F32,
    })
}

/// The element type a mixed-type pair computes in.
pub(crate) fn numeric_join(a: Scalar, b: Scalar) -> Option<Scalar> {
    match (a.is_float(), b.is_float()) {
        (true, true) => float_join(a, b),
        (true, false) if b.is_int() => Some(a),
        (false, true) if a.is_int() => Some(b),
        (false, false) if a.is_int() && b.is_int() => {
            Some(if int_bits(a) >= int_bits(b) { a } else { b })
        }
        _ => None,
    }
}

fn int_bits(s: Scalar) -> u32 {
    match s {
        Scalar::Bool => 1,
        Scalar::I8 => 8,
        Scalar::I32 => 32,
        Scalar::I64 => 64,
        _ => 0,
    }
}

/// The element type a contraction of `a` and `b` accumulates in.
pub(crate) fn accumulator_elem(a: Scalar, b: Scalar) -> Result<Scalar> {
    let join = numeric_join(a, b).ok_or_else(|| {
        anyhow!(
            "no common type for a contraction of {} and {}",
            a.mlir_name(),
            b.mlir_name()
        )
    })?;
    Ok(if join.is_int() && join != Scalar::I64 { Scalar::I32 } else { join })
}
