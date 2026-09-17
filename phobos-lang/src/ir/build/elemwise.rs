use anyhow::{Result, ensure};

use super::{Binding, Build, DYN, Rv, fmt_shape};
use crate::ast::{AssignOp, BinOp, Expr};
use crate::ir::{ElemStep, OpKind, Tree, ValueId};

fn is_arith(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
    )
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
            Expr::Binary { op, lhs, rhs } if is_arith(*op) => {
                Some(1 + self.fusable_nodes(lhs)? + self.fusable_nodes(rhs)?)
            }
            Expr::Call { callee, args } => match &args[..] {
                [arg] if ElemStep::from_callee(callee).is_some() => {
                    Some(1 + self.fusable_nodes(arg)?)
                }
                [a, b] if callee == "tmax" => {
                    Some(1 + self.fusable_nodes(a)? + self.fusable_nodes(b)?)
                }
                _ => Some(0),
            },
            _ => None,
        }
    }

    /// Builds the tree, emitting the operands that cannot be fused. Every
    /// leaf becomes an operand of the fused op; a scalar leaf is coerced to
    /// `elem` here.
    fn plan_fused(
        &mut self,
        value: &Expr,
        elem: crate::ir::Scalar,
        operands: &mut Vec<ValueId>,
        leaves: &mut Vec<ValueId>,
    ) -> Result<Tree> {
        Ok(match value {
            Expr::Binary { op, lhs, rhs } if is_arith(*op) => {
                let a = self.plan_fused(lhs, elem, operands, leaves)?;
                let b = self.plan_fused(rhs, elem, operands, leaves)?;
                Tree::Binary(*op, Box::new(a), Box::new(b))
            }
            Expr::Call { callee, args } if args.len() == 1 && ElemStep::from_callee(callee).is_some() => {
                let step = ElemStep::from_callee(callee).expect("checked above");
                let inner = self.plan_fused(&args[0], elem, operands, leaves)?;
                Tree::Unary(step, Box::new(inner))
            }
            Expr::Call { callee, args } if args.len() == 2 && callee == "tmax" => {
                let a = self.plan_fused(&args[0], elem, operands, leaves)?;
                let b = self.plan_fused(&args[1], elem, operands, leaves)?;
                Tree::Max(Box::new(a), Box::new(b))
            }
            other => match self.emit_expr(other)? {
                Rv::Scalar(v) => {
                    let v = self.coerce(v, elem)?;
                    operands.push(v);
                    Tree::Scalar(operands.len() - 1)
                }
                Rv::Tile(t) => {
                    leaves.push(t);
                    operands.push(t);
                    Tree::Leaf(operands.len() - 1)
                }
            },
        })
    }

    /// `target op= <per-element expression>` in one sweep, or false when the
    /// value is not such a tree.
    pub(crate) fn store_fused(&mut self, target: ValueId, op: AssignOp, value: &Expr) -> Result<bool> {
        let tshape = self.shape(target);
        if tshape.contains(&DYN) || self.fusable_nodes(value).unwrap_or(0) < 1 {
            return Ok(false);
        }
        let elem = self.elem(target);
        let mut operands = Vec::new();
        let mut leaves = Vec::new();
        let mut tree = self.plan_fused(value, elem, &mut operands, &mut leaves)?;
        if leaves.is_empty() {
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
