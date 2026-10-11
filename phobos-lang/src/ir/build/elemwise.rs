use anyhow::{Result, bail, ensure};

use super::{Binding, Build, DYN, Rv, fmt_shape};
use crate::ast::{AssignOp, BinOp, Expr, UnOp};
use crate::ir::{ElemStep, OpKind, Scalar, Tree, ValueId};

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
}

impl Build {
    /// Peels the per-element unary calls off the outside of `value`.
    fn peel_elem_chain<'e>(&self, value: &'e Expr) -> (Vec<ElemStep>, &'e Expr) {
        let mut steps = Vec::new();
        let mut inner = value;
        while let Expr::Call { callee, args } = inner {
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
        let Expr::Var(name) = base else {
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
        match value {
            Expr::Int(_) | Expr::Float(_) => Some(0),
            Expr::Var(name) => match self.lookup(name)? {
                Binding::Let { .. } | Binding::Var { .. } => Some(0),
                Binding::Tile(t) | Binding::View(t) => {
                    (!self.is_masked(t) && !self.shape(t).contains(&DYN)).then_some(0)
                }
                _ => None,
            },
            Expr::Binary { op, lhs, rhs } if op.is_elementwise() => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            // A negated literal is a constant leaf; anything else is a node.
            Expr::Unary {
                op: UnOp::Neg | UnOp::BitNot,
                rhs,
            } if matches!(**rhs, Expr::Int(_) | Expr::Float(_)) => Some(0),
            Expr::Unary {
                op: UnOp::Neg | UnOp::BitNot,
                rhs,
            } => Some(1 + self.fusable_nodes(rhs)?),
            Expr::Call { callee, args } => match &args[..] {
                [arg] if ElemStep::from_callee(callee).is_some() => {
                    Some(1 + self.fusable_nodes(arg)?)
                }
                [a, b] if callee == "tmax" => {
                    Some(1 + self.fusable_nodes(a)? + self.fusable_nodes(b)?)
                }
                [Expr::Int(_)] if callee == "iota" => Some(1),
                [c, a, b] if callee == "where" => Some(
                    1 + self.fusable_cond(c)? + self.fusable_nodes(a)? + self.fusable_nodes(b)?,
                ),
                _ => Some(0),
            },
            _ => None,
        }
    }

    /// Interior nodes of a `where` condition: comparisons of fusable trees,
    /// joined by `&&` and `||`.
    fn fusable_cond(&self, cond: &Expr) -> Option<usize> {
        match cond {
            Expr::Binary {
                op: BinOp::And | BinOp::Or,
                lhs,
                rhs,
            } => Some(1 + self.fusable_cond(lhs)? + self.fusable_cond(rhs)?),
            Expr::Binary { op, lhs, rhs } if op.is_compare() => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            _ => None,
        }
    }

    /// Builds the tree, emitting the operands that cannot be fused. Every
    /// leaf becomes an operand of the fused op; a scalar leaf is coerced to
    /// the target's element type here.
    fn plan_fused(&mut self, value: &Expr, plan: &mut Plan) -> Result<Tree> {
        Ok(match value {
            Expr::Binary { op, lhs, rhs } if op.is_elementwise() => {
                let a = self.plan_fused(lhs, plan)?;
                let b = self.plan_fused(rhs, plan)?;
                Tree::Binary(*op, Box::new(a), Box::new(b))
            }
            // -x is 0 - x and ~x is x ^ -1, against a scalar leaf.
            Expr::Unary {
                op: op @ (UnOp::Neg | UnOp::BitNot),
                rhs,
            } if !matches!(**rhs, Expr::Int(_) | Expr::Float(_)) => {
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
            Expr::Call { callee, args } if args.len() == 1 && ElemStep::from_callee(callee).is_some() => {
                let step = ElemStep::from_callee(callee).expect("checked above");
                let inner = self.plan_fused(&args[0], plan)?;
                Tree::Unary(step, Box::new(inner))
            }
            Expr::Call { callee, args } if callee == "iota" => {
                let [Expr::Int(d)] = args[..] else {
                    bail!("iota expects one literal axis");
                };
                let rank = plan.rank;
                ensure!(
                    usize::try_from(d).is_ok_and(|d| d < rank),
                    "iota axis {d} is past the assigned tile's rank {rank}"
                );
                Tree::Iota(d as usize)
            }
            Expr::Call { callee, args } if args.len() == 3 && callee == "where" => {
                let c = self.plan_cond(&args[0], plan)?;
                let a = self.plan_fused(&args[1], plan)?;
                let b = self.plan_fused(&args[2], plan)?;
                Tree::Select(Box::new(c), Box::new(a), Box::new(b))
            }
            Expr::Call { callee, args } if args.len() == 2 && callee == "tmax" => {
                let a = self.plan_fused(&args[0], plan)?;
                let b = self.plan_fused(&args[1], plan)?;
                Tree::Max(Box::new(a), Box::new(b))
            }
            other => match self.emit_expr(other)? {
                Rv::Scalar(v) => self.scalar_leaf(v, plan)?,
                Rv::Tile(t) => {
                    plan.leaves.push(t);
                    plan.operands.push(t);
                    Tree::Leaf(plan.operands.len() - 1)
                }
            },
        })
    }

    /// A scalar operand of the tree, coerced to the target's element type
    /// outside a comparison and kept at its own width inside one.
    fn scalar_leaf(&mut self, v: ValueId, plan: &mut Plan) -> Result<Tree> {
        let v = match self.scalar_of(v) {
            Scalar::Index if plan.in_compare => self.coerce(v, Scalar::I32)?,
            _ if plan.in_compare => v,
            _ => self.coerce(v, plan.elem)?,
        };
        plan.operands.push(v);
        Ok(Tree::Scalar(plan.operands.len() - 1))
    }

    /// A `where` condition's tree, which `fusable_cond` accepted.
    fn plan_cond(&mut self, cond: &Expr, plan: &mut Plan) -> Result<Tree> {
        let Expr::Binary { op, lhs, rhs } = cond else {
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
        for &leaf in &leaves {
            let lshape = self.shape(leaf);
            let fits = lshape.len() == tshape.len()
                && lshape.iter().zip(&tshape).all(|(&s, &t)| s == t || s == 1);
            ensure!(
                fits,
                "fused elementwise op: operand shape {} does not broadcast to {}",
                fmt_shape(&lshape),
                fmt_shape(&tshape)
            );
            ensure!(
                !self.is_masked(leaf),
                "fused elementwise op: a partially out-of-bounds operand must be \
                 materialized before it can be read against the target's extent"
            );
        }
        operands.push(target);
        self.stmt(OpKind::Fused(tree), &operands);
        Ok(true)
    }
}
