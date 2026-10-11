// The per-element vocabulary: the operators' typing rules, the unary steps,
// and the trees a fused sweep evaluates.

use std::fmt;

use super::types::Scalar;
use crate::ast::BinOp;

/// A per-element unary step: what `exp(x)`, `round(x)` and `i8(x)` do to
/// one element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemStep {
    Cast(Scalar),
    Round,
    Sqrt,
    Exp,
    Log,
    Tanh,
    Math(MathFn),
}

/// The per-element math the target supplies one instruction for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFn {
    /// Exact, and on an integer as well as a float.
    Abs,
    /// Exact.
    Floor,
    /// Exact.
    Ceil,
    /// Approximate, like `exp`.
    Rsqrt,
    /// Approximate, and least accurate far from zero.
    Sin,
    Cos,
}

impl MathFn {
    pub fn name(self) -> &'static str {
        match self {
            MathFn::Abs => "abs",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Rsqrt => "rsqrt",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
        }
    }

    fn from_name(name: &str) -> Option<MathFn> {
        [
            MathFn::Abs,
            MathFn::Floor,
            MathFn::Ceil,
            MathFn::Rsqrt,
            MathFn::Sin,
            MathFn::Cos,
        ]
        .into_iter()
        .find(|f| f.name() == name)
    }
}

impl ElemStep {
    pub fn from_callee(callee: &str) -> Option<ElemStep> {
        if let Some(f) = MathFn::from_name(callee) {
            return Some(ElemStep::Math(f));
        }
        Some(match callee {
            "round" => ElemStep::Round,
            "sqrt" => ElemStep::Sqrt,
            "exp" => ElemStep::Exp,
            "log" => ElemStep::Log,
            "tanh" => ElemStep::Tanh,
            other => {
                let scalar = Scalar::from_ast(crate::ast::Scalar::from_name(other)?);
                if scalar == Scalar::Bool {
                    return None;
                }
                ElemStep::Cast(scalar)
            }
        })
    }
}

impl fmt::Display for ElemStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElemStep::Cast(s) => write!(f, "{s}"),
            ElemStep::Round => f.write_str("round"),
            ElemStep::Sqrt => f.write_str("sqrt"),
            ElemStep::Exp => f.write_str("exp"),
            ElemStep::Log => f.write_str("log"),
            ElemStep::Tanh => f.write_str("tanh"),
            ElemStep::Math(m) => f.write_str(m.name()),
        }
    }
}

/// A per-element map into a fresh tile, or into a given one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// Two tiles, broadcasting.
    Binary(BinOp),
    /// A tile and a scalar; `scalar_left` for `s op t`.
    Scalar {
        op: BinOp,
        scalar_left: bool,
    },
    Unary(ElemStep),
    /// `tmax`, broadcasting.
    Max,
}

impl Map {
    /// How many tile operands precede a `MapInto`'s destination.
    pub fn tile_operands(self) -> usize {
        match self {
            Map::Binary(_) | Map::Max => 2,
            Map::Scalar { .. } | Map::Unary(_) => 1,
        }
    }
}

/// A per-element expression tree over a `Fused` op's operands, evaluated
/// in registers by one sweep of the destination. Leaves index the op's
/// operands.
#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    /// A tile operand, read with broadcasting.
    Leaf(usize),
    /// A scalar operand, one value for every element.
    Scalar(usize),
    Unary(ElemStep, Box<Tree>),
    /// An elementwise operator, or a comparison or `&&`/`||` under a
    /// `Select`'s condition.
    Binary(BinOp, Box<Tree>, Box<Tree>),
    Max(Box<Tree>, Box<Tree>),
    /// `where(cond, a, b)`: `a` where the condition holds, else `b`.
    Select(Box<Tree>, Box<Tree>, Box<Tree>),
    /// `iota(d)`: the element's position along the target's axis `d`.
    Iota(usize),
}

impl Tree {
    /// Whether the tree reads an element's position, which makes it worth a
    /// sweep with no tile operand.
    pub fn has_iota(&self) -> bool {
        match self {
            Tree::Iota(_) => true,
            Tree::Leaf(_) | Tree::Scalar(_) => false,
            Tree::Unary(_, x) => x.has_iota(),
            Tree::Binary(_, a, b) | Tree::Max(a, b) => a.has_iota() || b.has_iota(),
            Tree::Select(c, a, b) => c.has_iota() || a.has_iota() || b.has_iota(),
        }
    }

    /// Every operand index the tree reads.
    pub fn operands(&self, out: &mut Vec<usize>) {
        match self {
            Tree::Leaf(i) | Tree::Scalar(i) => out.push(*i),
            Tree::Iota(_) => {}
            Tree::Unary(_, x) => x.operands(out),
            Tree::Binary(_, a, b) | Tree::Max(a, b) => {
                a.operands(out);
                b.operands(out);
            }
            Tree::Select(c, a, b) => {
                c.operands(out);
                a.operands(out);
                b.operands(out);
            }
        }
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tree::Leaf(i) => write!(f, "#{i}"),
            Tree::Scalar(i) => write!(f, "s#{i}"),
            Tree::Unary(step, x) => write!(f, "({step} {x})"),
            Tree::Binary(op, a, b) => write!(f, "({} {a} {b})", binop_name(*op)),
            Tree::Max(a, b) => write!(f, "(max {a} {b})"),
            Tree::Select(c, a, b) => write!(f, "(select {c} {a} {b})"),
            Tree::Iota(d) => write!(f, "(iota {d})"),
        }
    }
}

pub fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Rem => "rem",
        BinOp::Eq => "eq",
        BinOp::Ne => "ne",
        BinOp::Lt => "lt",
        BinOp::Le => "le",
        BinOp::Gt => "gt",
        BinOp::Ge => "ge",
        BinOp::BitAnd => "and",
        BinOp::BitOr => "or",
        BinOp::BitXor => "xor",
        BinOp::Shl => "shl",
        BinOp::Shr => "shr",
        BinOp::And => "land",
        BinOp::Or => "lor",
        BinOp::Min => "min",
        BinOp::Max => "max",
    }
}

impl BinOp {
    pub fn is_compare(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    pub fn is_arith(self) -> bool {
        matches!(
            self,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
        )
    }

    /// Whether a tile takes the operator element by element, giving a tile
    /// of the same element type.
    pub fn is_elementwise(self) -> bool {
        self.is_arith() || self.is_bitwise() || matches!(self, BinOp::Min | BinOp::Max)
    }

    pub fn is_bitwise(self) -> bool {
        matches!(
            self,
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
        )
    }

    /// The operator as written in source.
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::Min => "min",
            BinOp::Max => "max",
        }
    }

    /// The result type of this operator on two scalars of type `t`, or None
    /// when `t` does not take it.
    pub fn result(self, t: Scalar) -> Option<Scalar> {
        use BinOp::*;
        let ok = match self {
            Eq | Ne => t.is_numeric() || t == Scalar::Bool,
            Lt | Le | Gt | Ge | Add | Sub | Mul | Div | Rem | Min | Max => t.is_numeric(),
            BitAnd | BitOr | BitXor | Shl | Shr => t.is_int() || t == Scalar::Index,
            And | Or => t == Scalar::Bool,
        };
        ok.then_some(if self.is_compare() { Scalar::Bool } else { t })
    }
}
