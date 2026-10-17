use anyhow::{Result, bail, ensure};

use super::{Binding, Build, DYN, Rv, fmt_shape};
use crate::ast::{AssignOp, BinOp, Expr, ExprKind, Sub, UnOp};
use crate::ir::{ElemStep, OpKind, Reduce, Scalar, Tree, ValueId};

/// What planning a fused tree collects: the operands in order, and which
/// of them are tiles.
struct Plan {
    /// The target's element type, which scalar leaves are coerced to.
    elem: Scalar,
    /// Whether the planner is inside a comparison, where a scalar keeps its
    /// own type: an `n` compared against `iota` must not round to an f16
    /// target.
    in_compare: bool,
    /// The target's rank, which bounds an `iota` axis.
    rank: usize,
    operands: Vec<ValueId>,
    leaves: Vec<ValueId>,
    /// Each scalar operand's index and its value before the coercion, so an
    /// integer subtree can take it at its own type instead.
    scalars: Vec<(usize, ValueId)>,
    /// Each operand expression emitted so far and its value. An expression
    /// has no side effects, so one the tree repeats, as a function's body
    /// repeats an argument, is emitted once: a tensor slice is staged once.
    emitted: Vec<(Expr, Rv)>,
}

impl Build {
    /// Peels the per-element unary calls off the outside of `value`.
    fn peel_elem_chain<'e>(&self, value: &'e Expr) -> (Vec<ElemStep>, &'e Expr) {
        let mut steps = Vec::new();
        let mut inner = value;
        while let ExprKind::Call { callee, args } = &inner.kind {
            let [arg] = &args[..] else { break };
            let Some(step) = ElemStep::from_callee(callee) else {
                break;
            };
            steps.push(step);
            inner = arg;
        }
        (steps, inner)
    }

    /// `target = f(g(...(t)))` in one sweep, or false when `value` is not
    /// such a chain over a named, unmasked, statically shaped tile.
    pub(crate) fn store_elem_chain(&mut self, target: ValueId, value: &Expr) -> Result<bool> {
        let (steps, base) = self.peel_elem_chain(value);
        if steps.is_empty() {
            return Ok(false);
        }
        let ExprKind::Var(name) = &base.kind else {
            return Ok(false);
        };
        let src = match self.lookup(name) {
            Some(Binding::Tile(t) | Binding::View(t)) => t,
            _ => return Ok(false),
        };
        let shape = self.shape(src);
        if self.is_masked(src) || shape.contains(&DYN) || shape != self.shape(target) {
            return Ok(false);
        }
        self.stmt(OpKind::Chain(steps), &[src, target]);
        Ok(true)
    }

    /// Interior nodes of a fusable tree, or None when `value` is not one.
    pub(crate) fn fusable_nodes(&self, value: &Expr) -> Option<usize> {
        match &value.kind {
            ExprKind::Int(_) | ExprKind::Float(_) => Some(0),
            ExprKind::Var(name) => match self.lookup(name)? {
                Binding::Let { .. } | Binding::Var { .. } => Some(0),
                Binding::Regs(_) => self.fusing_regs.then_some(0),
                // Bound only when every read is an int8 accumulator's update.
                Binding::Generated(_) => Some(0),
                // An unsigned tile is read only through a cast, below.
                Binding::Tile(t) | Binding::View(t) => (!self.is_masked(t)
                    && !self.shape(t).contains(&DYN)
                    && !self.elem(t).is_unsigned())
                .then_some(0),
                _ => None,
            },
            // A statically shaped tensor slice is a leaf too; one that runs
            // past the tensor is materialized, masked, before the sweep.
            ExprKind::Index { .. } => self.slice_leaf(value).map(|_| 0),
            ExprKind::Binary { op, lhs, rhs } if op.is_elementwise() => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            // A negated literal is a constant leaf; anything else is a node.
            ExprKind::Unary {
                op: UnOp::Neg | UnOp::BitNot,
                rhs,
            } if matches!(&rhs.kind, ExprKind::Int(_) | ExprKind::Float(_)) => Some(0),
            ExprKind::Unary {
                op: UnOp::Neg | UnOp::BitNot,
                rhs,
            } => Some(1 + self.fusable_nodes(rhs)?),
            ExprKind::Call { callee, args } => match &args[..] {
                [arg] if self.is_unsigned_cast(callee, arg) => Some(1),
                [arg] if ElemStep::from_callee(callee).is_some() => {
                    Some(1 + self.fusable_nodes(arg)?)
                }
                [a, b] if callee == "tmax" => {
                    Some(1 + self.fusable_nodes(a)? + self.fusable_nodes(b)?)
                }
                [Expr { kind: ExprKind::Int(_), .. }] if callee == "iota" => Some(1),
                [c, a, b] if callee == "where" => Some(
                    1 + self.fusable_cond(c)? + self.fusable_nodes(a)? + self.fusable_nodes(b)?,
                ),
                _ => Some(0),
            },
            _ => None,
        }
    }

    /// The element type and shape an untyped `var name = value` takes when
    /// it fuses, read off the named tiles it computes with. None when there
    /// is no such tile, when they differ in type or rank, or when the tree
    /// casts: there the unfused path's widening decides the type.
    pub(crate) fn fused_decl_type(&self, value: &Expr) -> Option<(Scalar, Vec<i64>)> {
        let mut found = None;
        self.decl_leaves(value, &mut found).then_some(())?;
        found
    }

    fn decl_leaves(&self, value: &Expr, found: &mut Option<(Scalar, Vec<i64>)>) -> bool {
        let mut meet = |elem: Scalar, shape: Vec<i64>| {
            let Some((seen_elem, seen_shape)) = found else {
                *found = Some((elem, shape));
                return true;
            };
            if *seen_elem != elem || seen_shape.len() != shape.len() {
                return false;
            }
            match super::broadcast_shape(seen_shape, &shape) {
                Some(joined) => {
                    *seen_shape = joined;
                    true
                }
                None => false,
            }
        };
        match &value.kind {
            ExprKind::Int(_) | ExprKind::Float(_) => true,
            ExprKind::Index { .. } => self.slice_leaf(value).is_some_and(|(elem, shape)| meet(elem, shape)),
            ExprKind::Var(name) => match self.lookup(name) {
                Some(Binding::Tile(t) | Binding::View(t)) => meet(self.elem(t), self.shape(t)),
                _ => true,
            },
            ExprKind::Binary { op, lhs, rhs } if op.is_elementwise() => {
                self.decl_leaves(lhs, found) && self.decl_leaves(rhs, found)
            }
            ExprKind::Unary {
                op: UnOp::Neg | UnOp::BitNot,
                rhs,
            } => self.decl_leaves(rhs, found),
            ExprKind::Call { callee, args } => match (callee.as_str(), &args[..]) {
                ("iota", _) => true,
                ("tmax", [a, b]) | ("where", [_, a, b]) => {
                    self.decl_leaves(a, found) && self.decl_leaves(b, found)
                }
                (callee, [a]) => {
                    matches!(ElemStep::from_callee(callee), Some(s) if !matches!(s, ElemStep::Cast(_)))
                        && self.decl_leaves(a, found)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// The element type and static shape of a tensor slice a sweep can read
    /// as a leaf; None for a point load, a dynamic shape or unsigned storage.
    fn slice_leaf(&self, value: &Expr) -> Option<(Scalar, Vec<i64>)> {
        let shape = self.slice_static_shape(value)?;
        let elem = self.slice_tensor_elem(value)?;
        (!elem.is_unsigned()).then_some((elem, shape))
    }

    /// Whether `callee(arg)` converts a named, fusable unsigned tile, which
    /// the sweep reads zero-extended.
    fn is_unsigned_cast(&self, callee: &str, arg: &Expr) -> bool {
        let ExprKind::Var(name) = &arg.kind else {
            return false;
        };
        let Some(Binding::Tile(t) | Binding::View(t)) = self.lookup(name) else {
            return false;
        };
        matches!(ElemStep::from_callee(callee), Some(ElemStep::Cast(_)))
            && self.elem(t).is_unsigned()
            && !self.is_masked(t)
            && !self.shape(t).contains(&DYN)
    }

    /// Interior nodes of a `where` condition: comparisons of fusable trees,
    /// joined by `&&` and `||`.
    fn fusable_cond(&self, cond: &Expr) -> Option<usize> {
        match &cond.kind {
            ExprKind::Binary {
                op: BinOp::And | BinOp::Or,
                lhs,
                rhs,
            } => Some(1 + self.fusable_cond(lhs)? + self.fusable_cond(rhs)?),
            ExprKind::Binary { op, lhs, rhs } if op.is_compare() => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            _ => None,
        }
    }

    /// Builds the tree, emitting the operands that cannot be fused. Every
    /// leaf becomes an operand of the fused op; a scalar leaf is coerced to
    /// the target's element type here.
    fn plan_fused(&mut self, value: &Expr, plan: &mut Plan) -> Result<Tree> {
        Ok(match &value.kind {
            ExprKind::Binary { op, lhs, rhs } if op.is_elementwise() => {
                let a = self.plan_fused(lhs, plan)?;
                let b = self.plan_fused(rhs, plan)?;
                self.scalar_to_int_side(&a, &b, plan)?;
                self.scalar_to_int_side(&b, &a, plan)?;
                Tree::Binary(*op, Box::new(a), Box::new(b))
            }
            // -x is 0 - x and ~x is x ^ -1, against a scalar leaf.
            ExprKind::Unary {
                op: op @ (UnOp::Neg | UnOp::BitNot),
                rhs,
            } if !matches!(&rhs.kind, ExprKind::Int(_) | ExprKind::Float(_)) => {
                let x = Box::new(self.plan_fused(rhs, plan)?);
                if *op == UnOp::Neg {
                    let zero = self.const_index(0);
                    let zero = Box::new(self.scalar_leaf(zero, plan)?);
                    Tree::Binary(BinOp::Sub, zero, x)
                } else {
                    let ones = self.const_index(-1);
                    let ones = Box::new(self.scalar_leaf(ones, plan)?);
                    Tree::Binary(BinOp::BitXor, x, ones)
                }
            }
            ExprKind::Call { callee, args } if args.len() == 1 && ElemStep::from_callee(callee).is_some() => {
                let step = ElemStep::from_callee(callee).expect("checked above");
                // A conversion's argument computes in its own type, so its
                // scalars take that type, not the target's.
                let outer = plan.elem;
                if let ElemStep::Cast(_) = step
                    && let Some(elem) = self.operand_elem(&args[0])
                {
                    plan.elem = elem;
                }
                let inner = self.plan_fused(&args[0], plan);
                plan.elem = outer;
                Tree::Unary(step, Box::new(inner?))
            }
            ExprKind::Call { callee, args } if callee == "iota" => {
                let [Expr { kind: ExprKind::Int(d), .. }] = args[..] else {
                    bail!("`iota` expects one literal axis");
                };
                let rank = plan.rank;
                ensure!(
                    usize::try_from(d).is_ok_and(|d| d < rank),
                    "`iota` axis {d} is past the assigned tile's rank {rank}"
                );
                Tree::Iota(d as usize)
            }
            ExprKind::Call { callee, args } if args.len() == 3 && callee == "where" => {
                let c = self.plan_cond(&args[0], plan)?;
                let a = self.plan_fused(&args[1], plan)?;
                let b = self.plan_fused(&args[2], plan)?;
                Tree::Select(Box::new(c), Box::new(a), Box::new(b))
            }
            ExprKind::Call { callee, args } if args.len() == 2 && callee == "tmax" => {
                let a = self.plan_fused(&args[0], plan)?;
                let b = self.plan_fused(&args[1], plan)?;
                Tree::Max(Box::new(a), Box::new(b))
            }
            ExprKind::Var(name) if self.fusing_regs && matches!(self.lookup(name), Some(Binding::Regs(_))) => {
                let Some(Binding::Regs(r)) = self.lookup(name) else {
                    unreachable!("matched above");
                };
                plan.operands.push(r);
                Tree::Leaf(plan.operands.len() - 1)
            }
            _ => match self.emit_operand(value, plan)? {
                Rv::Scalar(v) => self.scalar_leaf(v, plan)?,
                Rv::Tile(t) => {
                    plan.leaves.push(t);
                    plan.operands.push(t);
                    Tree::Leaf(plan.operands.len() - 1)
                }
            },
        })
    }

    /// The element type `value` computes in, where its tiles or a conversion
    /// around it decide it.
    fn operand_elem(&self, value: &Expr) -> Option<Scalar> {
        if let ExprKind::Call { callee, args } = &value.kind
            && let [_] = &args[..]
            && let Some(ElemStep::Cast(to)) = ElemStep::from_callee(callee)
        {
            return Some(to);
        }
        self.fused_decl_type(value)
            .map(|(elem, _)| elem)
            .or_else(|| self.is_iota_arith(value).then_some(Scalar::I32))
    }

    /// Whether `value` is integer arithmetic over `iota` with no tile in it,
    /// as the counter `(r + iota(0)) * C + iota(1)` is. Such a value computes
    /// in `i32`, as `iota` does.
    fn is_iota_arith(&self, value: &Expr) -> bool {
        fn walk(b: &Build, e: &Expr, iota: &mut bool) -> bool {
            match &e.kind {
                ExprKind::Int(_) => true,
                ExprKind::Call { callee, .. } if callee == "iota" => {
                    *iota = true;
                    true
                }
                ExprKind::Var(name) => match b.lookup(name) {
                    Some(Binding::Let { value, .. }) => {
                        let elem = b.scalar_of(value);
                        elem.is_int() || elem == Scalar::Index
                    }
                    Some(Binding::Var { elem, .. }) => elem.is_int() || elem == Scalar::Index,
                    _ => false,
                },
                ExprKind::Binary { op, lhs, rhs } if op.is_elementwise() => walk(b, lhs, iota) && walk(b, rhs, iota),
                ExprKind::Unary {
                    op: UnOp::Neg | UnOp::BitNot,
                    rhs,
                } => walk(b, rhs, iota),
                _ => false,
            }
        }
        let mut iota = false;
        walk(self, value, &mut iota) && iota
    }

    /// `value` emitted, or the value it had where the tree already holds it.
    fn emit_operand(&mut self, value: &Expr, plan: &mut Plan) -> Result<Rv> {
        if let Some(&(_, rv)) = plan.emitted.iter().find(|(e, _)| same_expr(e, value)) {
            return Ok(rv);
        }
        let rv = self.emit_expr(value)?;
        plan.emitted.push((value.clone(), rv));
        Ok(rv)
    }

    /// A scalar operand of the tree, coerced to the target's element type
    /// outside a comparison and kept at its own width inside one.
    fn scalar_leaf(&mut self, v: ValueId, plan: &mut Plan) -> Result<Tree> {
        let raw = v;
        let v = match self.scalar_of(v) {
            Scalar::Index if plan.in_compare => self.coerce(v, Scalar::I32)?,
            _ if plan.in_compare => v,
            _ => self.coerce(v, plan.elem)?,
        };
        plan.scalars.push((plan.operands.len(), raw));
        plan.operands.push(v);
        Ok(Tree::Scalar(plan.operands.len() - 1))
    }

    /// When `scalar` is a scalar leaf and `other` an integer subtree under a
    /// target of another type, as in `f32(i32(q) % 256)`, coerces the scalar
    /// to the subtree's type, so the arithmetic stays integer.
    fn scalar_to_int_side(&mut self, scalar: &Tree, other: &Tree, plan: &mut Plan) -> Result<()> {
        let Tree::Scalar(i) = *scalar else {
            return Ok(());
        };
        let Some(want) = self.tree_elem(other, plan) else {
            return Ok(());
        };
        if plan.in_compare || !want.is_int() || want == plan.elem {
            return Ok(());
        }
        let Some(&(_, raw)) = plan.scalars.iter().find(|(at, _)| *at == i) else {
            return Ok(());
        };
        plan.operands[i] = self.coerce(raw, want)?;
        Ok(())
    }

    /// The element type a planned subtree computes in, or None when only its
    /// scalars decide it.
    fn tree_elem(&self, tree: &Tree, plan: &Plan) -> Option<Scalar> {
        match tree {
            Tree::Leaf(i) => Some(self.elem(plan.operands[*i])),
            Tree::Scalar(_) => None,
            Tree::Iota(_) => Some(Scalar::I32),
            Tree::Unary(ElemStep::Cast(to), _) => Some(*to),
            Tree::Unary(_, x) => self.tree_elem(x, plan),
            Tree::Binary(op, _, _) if op.is_compare() => None,
            Tree::Binary(_, a, b) | Tree::Max(a, b) | Tree::Select(_, a, b) => {
                match (self.tree_elem(a, plan), self.tree_elem(b, plan)) {
                    (Some(x), Some(y)) if x != y => Some(if x.is_float() { x } else { y }),
                    (x, y) => x.or(y),
                }
            }
        }
    }

    /// A `where` condition's tree, which `fusable_cond` accepted.
    fn plan_cond(&mut self, cond: &Expr, plan: &mut Plan) -> Result<Tree> {
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            bail!("a `where` condition must be a comparison");
        };
        let (a, b) = if op.is_compare() {
            let outer = std::mem::replace(&mut plan.in_compare, true);
            let sides = (self.plan_fused(lhs, plan)?, self.plan_fused(rhs, plan)?);
            plan.in_compare = outer;
            sides
        } else {
            (self.plan_cond(lhs, plan)?, self.plan_cond(rhs, plan)?)
        };
        Ok(Tree::Binary(*op, Box::new(a), Box::new(b)))
    }

    /// `target op= <per-element expression>` in one sweep, or false when the
    /// value is not such a tree.
    pub(crate) fn store_fused(&mut self, target: ValueId, op: AssignOp, value: &Expr) -> Result<bool> {
        let tshape = self.shape(target);
        if tshape.contains(&DYN) || self.fusable_nodes(value).unwrap_or(0) < 1 {
            return Ok(false);
        }
        let mut plan = Plan {
            elem: self.elem(target),
            in_compare: false,
            rank: tshape.len(),
            operands: Vec::new(),
            leaves: Vec::new(),
            scalars: Vec::new(),
            emitted: Vec::new(),
        };
        let mut tree = self.plan_fused(value, &mut plan)?;
        let Plan {
            mut operands,
            leaves,
            ..
        } = plan;
        if leaves.is_empty() && !tree.has_iota() {
            return Ok(false);
        }
        if op == AssignOp::Add {
            operands.push(target);
            tree = Tree::Binary(
                BinOp::Add,
                Box::new(Tree::Leaf(operands.len() - 1)),
                Box::new(tree),
            );
        }
        self.check_leaves(&leaves, &tshape)?;
        operands.push(target);
        self.stmt(OpKind::Fused(tree), &operands);
        Ok(true)
    }

    /// `rowmax(e)`, `rowsum(e)` or `rowargmax(e)` with `e` evaluated per element inside the
    /// reduction's loop, never stored. None when `e` is not a fusable tree
    /// over named tiles of one type and rank 2.
    pub(crate) fn reduce_fused(&mut self, how: Reduce, arg: &Expr) -> Result<Option<ValueId>> {
        if self.fusable_nodes(arg).unwrap_or(0) < 1 {
            return Ok(None);
        }
        let Some((elem, shape)) = self.fused_decl_type(arg) else {
            return Ok(None);
        };
        if shape.len() != 2 || shape.contains(&DYN) || !(elem.is_float() || elem.is_int()) {
            return Ok(None);
        }
        let mut plan = Plan {
            elem,
            in_compare: false,
            rank: 2,
            operands: Vec::new(),
            leaves: Vec::new(),
            scalars: Vec::new(),
            emitted: Vec::new(),
        };
        let tree = self.plan_fused(arg, &mut plan)?;
        self.check_leaves(&plan.leaves, &shape)?;
        let out = if how == Reduce::RowArgMax { Scalar::I32 } else { elem };
        let ty = self.shared_ty(out, &[shape[0], 1]);
        let kind = OpKind::ReduceFused {
            how,
            tree,
            cols: shape[1],
            arg: elem,
        };
        Ok(Some(self.value(kind, &plan.operands, ty)))
    }

    /// The tree and operands of a register tile's update, or None when
    /// `value` is not per-element. Only here may a register tile be a leaf.
    pub(crate) fn plan_regs(&mut self, value: &Expr, elem: Scalar, shape: &[i64]) -> Result<Option<(Tree, Vec<ValueId>)>> {
        self.fusing_regs = true;
        let planned = if self.fusable_nodes(value).is_some() {
            let mut plan = Plan {
                elem,
                in_compare: false,
                rank: shape.len(),
                operands: Vec::new(),
                leaves: Vec::new(),
                scalars: Vec::new(),
                emitted: Vec::new(),
            };
            self.plan_fused(value, &mut plan).map(|tree| Some((tree, plan)))
        } else {
            Ok(None)
        };
        self.fusing_regs = false;
        let Some((tree, plan)) = planned? else {
            return Ok(None);
        };
        self.check_leaves(&plan.leaves, shape)?;
        Ok(Some((tree, plan.operands)))
    }

    /// Every tile leaf broadcasts to `shape` and is unmasked: the sweep
    /// indexes by `shape` and ignores masks.
    pub(crate) fn check_leaves(&self, leaves: &[ValueId], shape: &[i64]) -> Result<()> {
        for &leaf in leaves {
            let lshape = self.shape(leaf);
            let fits = lshape.len() == shape.len()
                && lshape.iter().zip(shape).all(|(&s, &t)| s == t || s == 1);
            ensure!(
                fits,
                "fused elementwise op: operand shape {} does not broadcast to {}",
                fmt_shape(&lshape),
                fmt_shape(shape)
            );
            ensure!(
                !self.is_masked(leaf),
                "fused elementwise op: a partially out-of-bounds operand must be \
                 materialized before it can be read against the target's extent"
            );
        }
        Ok(())
    }
}

/// Whether `a` and `b` are the same expression, wherever each was written.
fn same_expr(a: &Expr, b: &Expr) -> bool {
    let all = |a: &[Expr], b: &[Expr]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_expr(a, b));
    match (&a.kind, &b.kind) {
        (ExprKind::Int(a), ExprKind::Int(b)) => a == b,
        (ExprKind::Float(a), ExprKind::Float(b)) => a.to_bits() == b.to_bits(),
        (ExprKind::Bool(a), ExprKind::Bool(b)) => a == b,
        (ExprKind::Var(a), ExprKind::Var(b)) => a == b,
        (ExprKind::Unary { op: p, rhs: a }, ExprKind::Unary { op: q, rhs: b }) => p == q && same_expr(a, b),
        (ExprKind::Binary { op: p, lhs: a, rhs: c }, ExprKind::Binary { op: q, lhs: b, rhs: d }) => {
            p == q && same_expr(a, b) && same_expr(c, d)
        }
        (ExprKind::Call { callee: f, args: a }, ExprKind::Call { callee: g, args: b }) => f == g && all(a, b),
        (ExprKind::Index { base: a, subs: s }, ExprKind::Index { base: b, subs: t }) => {
            same_expr(a, b) && s.len() == t.len() && s.iter().zip(t).all(|(s, t)| same_sub(s, t))
        }
        _ => false,
    }
}

fn same_sub(a: &Sub, b: &Sub) -> bool {
    match (a, b) {
        (Sub::Full, Sub::Full) => true,
        (Sub::Point(a), Sub::Point(b)) | (Sub::From(a), Sub::From(b)) => same_expr(a, b),
        (Sub::Range { start: a, end: c }, Sub::Range { start: b, end: d })
        | (Sub::Span { start: a, len: c }, Sub::Span { start: b, len: d }) => same_expr(a, b) && same_expr(c, d),
        _ => false,
    }
}
