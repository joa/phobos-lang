use std::fmt;

pub use crate::ast::{AssignOp, BinOp, UnOp};

use super::types::{Extent, Scalar};

/// Index arithmetic the language has no operator for, which the loop
/// split and the emitters' own address arithmetic use: the unsigned
/// division and remainder, and the unsigned compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexOp {
    DivU,
    RemU,
    CmpUlt,
}

impl IndexOp {
    pub fn name(self) -> &'static str {
        match self {
            IndexOp::DivU => "divu",
            IndexOp::RemU => "remu",
            IndexOp::CmpUlt => "cmp_ult",
        }
    }
}

/// How a `For` gets its bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bounds {
    /// Constant bounds, folded at build: the loop takes no bound operands
    /// and lowers to `affine.for`.
    Affine { lo: i64, hi: i64, step: i64 },
    /// Three leading `index` operands: `lo`, `hi`, `step`.
    Dynamic,
}

/// A loop the build found shaped for double buffering: its body opens with
/// `staged` stage ops of tensor slices that nothing later writes. Whether
/// the doubled buffers fit is the emitter's call, since the budget counts
/// what the pool holds at that point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    pub staged: usize,
    /// Whether the compute half ends in a tile op, whose own trailing
    /// barrier stands in for the closing one.
    pub ends_with_tile_op: bool,
    /// Bytes the doubled staging buffers take, 16-byte aligned each.
    pub doubled_bytes: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForInfo {
    pub bounds: Bounds,
    /// A loop over a dynamic extent split into whole chunks and a masked
    /// remainder: the bounds are `lo, full, step, hi`, the first block runs
    /// the trimmed main loop and the second replays the body once at `full`
    /// when `full < hi`. Dynamic bounds only.
    pub ragged: bool,
    /// Loop-carried values: operands after the bounds, block arguments
    /// after the induction variable, and the op's results.
    pub carried: usize,
    /// Trailing operands: buffers a `HoistStage` filled in the preheader for
    /// the dots inside, returned to the pool after the loop.
    pub hoisted: usize,
    pub pipeline: Option<Pipeline>,
}

impl ForInfo {
    pub fn bound_operands(&self) -> usize {
        match self.bounds {
            Bounds::Affine { .. } => 0,
            Bounds::Dynamic => 3 + usize::from(self.ragged),
        }
    }

    pub fn operand_count(&self) -> usize {
        self.bound_operands() + self.carried + self.hoisted
    }
}

/// One coefficient of a register matmul's epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coeff {
    Absent,
    /// Applied as one: the epilogue has a prev_load term and this side of
    /// it is unscaled.
    One,
    /// An operand of the op, after the accumulator and the view.
    Given,
}

impl Coeff {
    pub fn operands(self) -> usize {
        usize::from(self == Coeff::Given)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    /// An integer literal, always `index`.
    Int(i64),
    /// A float literal, always `f32` until a store rounds it.
    Float(f64),
    Bool(bool),
}

/// The raw quantized formats the intrinsics decode inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawFmt {
    Iq1s,
    Iq1m,
    Iq2xxs,
    Iq2xs,
    Iq2s,
    Iq3xxs,
    Iq3s,
    Iq4xs,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Ptq1,
}

impl RawFmt {
    pub const ALL: [RawFmt; 14] = [
        RawFmt::Iq1s,
        RawFmt::Iq1m,
        RawFmt::Iq2xxs,
        RawFmt::Iq2xs,
        RawFmt::Iq2s,
        RawFmt::Iq3xxs,
        RawFmt::Iq3s,
        RawFmt::Iq4xs,
        RawFmt::Q2k,
        RawFmt::Q3k,
        RawFmt::Q4k,
        RawFmt::Q5k,
        RawFmt::Q6k,
        RawFmt::Ptq1,
    ];

    pub fn name(self) -> &'static str {
        match self {
            RawFmt::Iq1s => "iq1s",
            RawFmt::Iq1m => "iq1m",
            RawFmt::Iq2xxs => "iq2xxs",
            RawFmt::Iq2xs => "iq2xs",
            RawFmt::Iq2s => "iq2s",
            RawFmt::Iq3xxs => "iq3xxs",
            RawFmt::Iq3s => "iq3s",
            RawFmt::Iq4xs => "iq4xs",
            RawFmt::Q2k => "q2k",
            RawFmt::Q3k => "q3k",
            RawFmt::Q4k => "q4k",
            RawFmt::Q5k => "q5k",
            RawFmt::Q6k => "q6k",
            RawFmt::Ptq1 => "ptq1",
        }
    }

    pub fn from_name(name: &str) -> Option<RawFmt> {
        RawFmt::ALL.into_iter().find(|f| f.name() == name)
    }
}

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
}

impl ElemStep {
    pub fn from_callee(callee: &str) -> Option<ElemStep> {
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
        }
    }
}

/// A whole-tile operation with one tile in and one out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduce {
    /// Reduce the last axis to a `[rows, 1]` column.
    RowMax,
    RowSum,
    /// Inclusive prefix sum down the rows.
    CumSum,
    /// Causal lower-triangular mask.
    Tril,
    Transpose,
}

impl Reduce {
    pub fn name(self) -> &'static str {
        match self {
            Reduce::RowMax => "rowmax",
            Reduce::RowSum => "rowsum",
            Reduce::CumSum => "cumsum",
            Reduce::Tril => "tril",
            Reduce::Transpose => "transpose",
        }
    }
}

/// A per-element map into a fresh tile, or into a given one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// Two tiles, broadcasting.
    Binary(BinOp),
    /// A tile and a scalar; `scalar_left` for `s op t`.
    Scalar { op: BinOp, scalar_left: bool },
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
    Binary(BinOp, Box<Tree>, Box<Tree>),
    Max(Box<Tree>, Box<Tree>),
}

impl Tree {
    /// Every operand index the tree reads.
    pub fn operands(&self, out: &mut Vec<usize>) {
        match self {
            Tree::Leaf(i) | Tree::Scalar(i) => out.push(*i),
            Tree::Unary(_, x) => x.operands(out),
            Tree::Binary(_, a, b) | Tree::Max(a, b) => {
                a.operands(out);
                b.operands(out);
            }
        }
    }

    pub fn has_unary(&self) -> bool {
        match self {
            Tree::Leaf(_) | Tree::Scalar(_) => false,
            Tree::Unary(..) => true,
            Tree::Binary(_, a, b) | Tree::Max(a, b) => a.has_unary() || b.has_unary(),
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
    }
}

impl BinOp {
    pub fn is_compare(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }
}

/// A slice of a tensor or tile: `src[offsets :+ sizes]`. Operands are the
/// source, one offset per dimension, one size per `Extent::Dyn` in `sizes`,
/// then one extent per masked dimension, in dimension order. A masked
/// dimension may run past the source, and its extent is what the load
/// zero-fills and the store skips against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub sizes: Vec<Extent>,
    pub masked: Vec<bool>,
    /// Known divisor of each extent: the source's promise on a dimension
    /// taken whole, 1 elsewhere. Only the build's prescans and its register
    /// matmul match read it.
    pub divs: Vec<i64>,
}

impl Slice {
    pub fn rank(&self) -> usize {
        self.sizes.len()
    }

    pub fn dyn_sizes(&self) -> usize {
        self.sizes.iter().filter(|e| e.is_dyn()).count()
    }

    pub fn masked_dims(&self) -> usize {
        self.masked.iter().filter(|&&m| m).count()
    }

    pub fn operand_count(&self) -> usize {
        1 + self.rank() + self.dyn_sizes() + self.masked_dims()
    }

    /// The operand index of dimension `d`'s offset.
    pub fn offset_operand(&self, d: usize) -> usize {
        1 + d
    }

    /// The operand index of dimension `d`'s mask extent, when it is masked.
    pub fn mask_operand(&self, d: usize) -> Option<usize> {
        if !self.masked[d] {
            return None;
        }
        let before = self.masked[..d].iter().filter(|&&m| m).count();
        Some(1 + self.rank() + self.dyn_sizes() + before)
    }
}

/// A copy of a view into a fresh shared tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// Pad the row pitch against bank conflicts (`@padstage`).
    pub pad: bool,
    /// Close the copy with a CTA barrier. Only a staging run's inner
    /// members skip it.
    pub sync: bool,
}

/// The builtins that take tiles (and the odd scalar) and produce a tile,
/// or write their destination in place in the `Into` form. Each spells
/// its source name, which is what the emitter dispatches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    /// `qdot_t`: the Q8_0 contraction with the block scales folded in.
    QdotT,
    /// `qmma_t`: the same, batched over rows on the integer tensor cores.
    QmmaT,
    /// `<fmt>_qdot_t`: a raw format's matvec with its decode folded in.
    RawQdot(RawFmt),
    /// `<fmt>_qdot_i8_t`: the dp4a decode matvec.
    RawQdotI8(RawFmt),
    /// `<fmt>_qmma_t`: a raw format's batched projection.
    RawQmma(RawFmt),
    /// `<fmt>_qmma_staged_t`: the same with the decoded weight staged.
    RawQmmaStaged(RawFmt),
    /// `<fmt>_qgemm_t`: the tensor-core GEMM over a raw format.
    RawQgemm(RawFmt),
    /// `<fmt>_qdecode_t`: a raw format expanded into a scratch tensor.
    /// Only ever the whole right-hand side of a store, so `Into` only.
    RawQdecode(RawFmt),
    /// `gather(table, idx)`.
    Gather,
    /// `argsel`: a `tmax`-shaped fold's index side.
    ArgSel,
    /// `rms_norm_q_t(x, gain, eps, [out,] q, scales)`: writes its outputs
    /// and yields the f32 inverse rms.
    RmsNormQ,
    /// `warp_partial(q, K, V, WM, WL, WACC, scale, lo, hi, col)`, the
    /// buffers first and the scalars in evaluation order; yields the zero
    /// the source sees.
    WarpPartial,
}

impl Intrinsic {
    pub fn name(self) -> String {
        match self {
            Intrinsic::QdotT => "qdot_t".into(),
            Intrinsic::QmmaT => "qmma_t".into(),
            Intrinsic::RawQdot(f) => format!("{}_qdot_t", f.name()),
            Intrinsic::RawQdotI8(f) => format!("{}_qdot_i8_t", f.name()),
            Intrinsic::RawQmma(f) => format!("{}_qmma_t", f.name()),
            Intrinsic::RawQmmaStaged(f) => format!("{}_qmma_staged_t", f.name()),
            Intrinsic::RawQgemm(f) => format!("{}_qgemm_t", f.name()),
            Intrinsic::RawQdecode(f) => format!("{}_qdecode_t", f.name()),
            Intrinsic::Gather => "gather".into(),
            Intrinsic::ArgSel => "argsel".into(),
            Intrinsic::RmsNormQ => "rms_norm_q_t".into(),
            Intrinsic::WarpPartial => "warp_partial".into(),
        }
    }

    pub fn from_name(callee: &str) -> Option<Intrinsic> {
        match callee {
            "qdot_t" => return Some(Intrinsic::QdotT),
            "qmma_t" => return Some(Intrinsic::QmmaT),
            "gather" => return Some(Intrinsic::Gather),
            "argsel" => return Some(Intrinsic::ArgSel),
            "rms_norm_q_t" => return Some(Intrinsic::RmsNormQ),
            "warp_partial" => return Some(Intrinsic::WarpPartial),
            _ => {}
        }
        let (fmt, suffix) = callee.split_once('_')?;
        let fmt = RawFmt::from_name(fmt)?;
        Some(match suffix {
            "qdot_t" => Intrinsic::RawQdot(fmt),
            "qdot_i8_t" => Intrinsic::RawQdotI8(fmt),
            "qmma_t" => Intrinsic::RawQmma(fmt),
            "qmma_staged_t" => Intrinsic::RawQmmaStaged(fmt),
            "qgemm_t" => Intrinsic::RawQgemm(fmt),
            "qdecode_t" => Intrinsic::RawQdecode(fmt),
            _ => return None,
        })
    }
}

/// What an op does. Operands, results and blocks live beside the kind in
/// the arena; the kind holds only attributes.
///
/// The vocabulary is coarse: a builtin is one op carrying its format, a
/// per-element expression is one op carrying its tree, and a pass records
/// what it decided as an attribute on the op it decided about. The
/// declarations below this enum, the name, whether the kind ends a block,
/// which results alias which operands and which operands it writes, are
/// what the generic passes read; the per-kind typing rules live in
/// `verify.rs`.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    // Scalars.
    Const(Literal),
    Unary(UnOp),
    /// Two scalars of one type; the build inserts the widening casts.
    Binary(BinOp),
    /// Two `index` operands, an `index` or `bool` result.
    IndexOp(IndexOp),
    /// A numeric conversion, whatever the two types.
    Cast(Scalar),
    /// `arith.index_cast`: between `index` and a sized integer, one op.
    IndexCast(Scalar),
    ProgramId(u8),
    /// The runtime extent of a tensor's dimension.
    Dim(usize),
    /// `(mem, indices..) -> elem`; a private slot takes no indices.
    Load,
    /// `(value, mem, indices..)`.
    Store,
    /// `(tensor, slot, value) -> old`, over an `i32` tensor.
    AtomicAdd,

    // Control. Each owns blocks; the body's terminator carries the
    // loop-carried values back.
    /// `([lo, hi, step,] inits.., hoisted..) -> carried..`, one block
    /// `(iv, carried..)`. See [`ForInfo`].
    For(ForInfo),
    /// `(inits..) -> carried..`, a `before` block ending in `Condition`
    /// and an `after` block ending in `Yield`, both taking the carried values.
    While,
    /// `(cond) -> results..`, a `then` block and an optional `else`.
    If,
    Yield,
    /// `(cond, carried..)`: a `While`'s test, and what it hands on.
    Condition,
    /// A CTA barrier.
    Barrier,
    /// `(bar) -> index`: every block of the grid waits for every other. The
    /// result is the zero the source sees, since a call is an expression.
    GridBarrier,

    // Tiles.
    /// `(tensor) -> tensor`: the same parameter, promised 16-byte aligned,
    /// which is what its slices vectorize on. The parameter's dims are
    /// still read off the raw argument.
    AssumeAlign,
    /// A fresh buffer; the result type says where and how big.
    Alloc,
    Slice(Slice),
    /// A rank-2 tile viewed as one row, no copy.
    Flat,
    /// `(view) -> tile`: a copy into a fresh shared buffer.
    Stage(Stage),
    /// `(masked view) -> tile`: the in-bounds elements copied, the rest zero.
    Materialize,
    /// `(view) -> tile`: a loop-invariant dot operand staged to f16 in the
    /// loop's preheader, for the dots inside to read instead of staging.
    HoistStage,
    /// `(src, dst)`.
    Copy { sync: bool },
    /// `(src, dst)`: an element-type conversion into `dst`.
    Convert,
    /// `(scalar, dst)`.
    Fill,
    Map(Map),
    /// `(tiles.., dst)`.
    MapInto(Map),
    /// `(leaves.., dst)`: the tree in one sweep of `dst`.
    Fused(Tree),
    /// `(src, dst)`: a chain of per-element unary steps, outermost last.
    Chain(Vec<ElemStep>),
    /// `(s1, t1, s2, t2, dst)`: `dst = s1 * t1 + s2 * t2`.
    ScaledAdd,
    Reduce(Reduce),
    /// `(a, b) -> a * b`, or `a * b^T`.
    Dot { transpose: bool },
    /// `(a, b, dst)`. `aliased` when `dst` is one of the operands, which
    /// routes the product through a temporary.
    DotInto {
        transpose: bool,
        accumulate: bool,
        aliased: bool,
    },
    /// `(src, dst)`: `dst += src`, element-wise.
    Accumulate,
    Intrinsic(Intrinsic),
    /// `(operands.., dst)`.
    IntrinsicInto(Intrinsic),

    // Fragment accumulators: values, not buffers, so each update is a new
    // value and a loop carries them as iter args.
    /// `() -> frags`, every element `init`.
    FragInit(f64),
    /// `(acc, col) -> frags`: `acc = acc op col` for a `[m, 1]` column.
    FragScale(BinOp),
    /// `(acc, a, b) -> frags`: `acc += dot(a, b)`.
    FragDot,
    /// `(acc, dst)`: the fragments scattered straight to a tensor slice.
    FragStore,

    // The register matmul: an accumulator seeded before its k-loop, carried
    // through it as the loop's one iter arg, drained after it.
    /// `(init) -> gemm`: every element the scalar `init`.
    GemmInit,
    /// `(acc, a, b) -> gemm`: `acc += dot(a, b)` for an `[m, k]` a and a
    /// `[k, n]` b, both unmasked tensor slices.
    GemmDot,
    /// `(acc, dst[, alpha][, beta])`: `dst = alpha * acc + beta * dst`, with
    /// each coefficient absent, one or given, and beta reading `dst` before
    /// it is written.
    GemmStore { alpha: Coeff, beta: Coeff },
}

/// Which result of an op is a view of which operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alias {
    pub result: usize,
    pub operand: usize,
}

impl OpKind {
    /// The mnemonic the printer shows and the tests grep.
    pub fn name(&self) -> String {
        match self {
            OpKind::Const(_) => "const".into(),
            OpKind::Unary(UnOp::Neg) => "neg".into(),
            OpKind::Unary(UnOp::Not) => "not".into(),
            OpKind::Binary(op) => binop_name(*op).into(),
            OpKind::IndexOp(op) => op.name().into(),
            OpKind::Cast(s) => format!("cast {s}"),
            OpKind::IndexCast(s) => format!("index_cast {s}"),
            OpKind::ProgramId(d) => format!("program_id {d}"),
            OpKind::Dim(d) => format!("dim {d}"),
            OpKind::Load => "load".into(),
            OpKind::Store => "store".into(),
            OpKind::AtomicAdd => "atomic_add".into(),
            OpKind::For(info) => {
                let mut out = match info.bounds {
                    Bounds::Affine { lo, hi, step } => format!("for {lo}..{hi} step {step}"),
                    Bounds::Dynamic => "for".to_string(),
                };
                if info.ragged {
                    out.push_str(" ragged");
                }
                if info.hoisted > 0 {
                    out.push_str(&format!(" hoisted {}", info.hoisted));
                }
                if let Some(p) = info.pipeline {
                    out.push_str(&format!(" pipeline {}", p.staged));
                }
                out
            }
            OpKind::While => "while".into(),
            OpKind::If => "if".into(),
            OpKind::Yield => "yield".into(),
            OpKind::Condition => "condition".into(),
            OpKind::Barrier => "barrier".into(),
            OpKind::GridBarrier => "grid_barrier".into(),
            OpKind::AssumeAlign => "assume_align".into(),
            OpKind::Alloc => "alloc".into(),
            OpKind::Slice(s) => {
                let sizes: Vec<String> = s.sizes.iter().map(|e| e.to_string()).collect();
                let mut out = format!("slice [{}]", sizes.join(", "));
                if s.masked.iter().any(|&m| m) {
                    let masked: Vec<&str> = s
                        .masked
                        .iter()
                        .map(|&m| if m { "m" } else { "-" })
                        .collect();
                    out.push_str(&format!(" mask [{}]", masked.join(", ")));
                }
                out
            }
            OpKind::Flat => "flat".into(),
            OpKind::Stage(s) => {
                let mut out = "stage".to_string();
                if s.pad {
                    out.push_str(" pad");
                }
                if !s.sync {
                    out.push_str(" nosync");
                }
                out
            }
            OpKind::Materialize => "materialize".into(),
            OpKind::HoistStage => "hoist_stage".into(),
            OpKind::Copy { sync } => if *sync { "copy" } else { "copy nosync" }.into(),
            OpKind::Convert => "convert".into(),
            OpKind::Fill => "fill".into(),
            OpKind::Map(m) => format!("map {}", map_name(*m)),
            OpKind::MapInto(m) => format!("map_into {}", map_name(*m)),
            OpKind::Fused(tree) => format!("fused {tree}"),
            OpKind::Chain(steps) => {
                let steps: Vec<String> = steps.iter().map(|s| s.to_string()).collect();
                format!("chain {}", steps.join(" "))
            }
            OpKind::ScaledAdd => "scaled_add".into(),
            OpKind::Reduce(r) => r.name().into(),
            OpKind::Dot { transpose } => if *transpose { "dot_t" } else { "dot" }.into(),
            OpKind::DotInto {
                transpose,
                accumulate,
                aliased,
            } => {
                let mut out = if *transpose { "dot_t_into" } else { "dot_into" }.to_string();
                if *accumulate {
                    out.push_str(" acc");
                }
                if *aliased {
                    out.push_str(" aliased");
                }
                out
            }
            OpKind::Accumulate => "accumulate".into(),
            OpKind::Intrinsic(i) => i.name(),
            OpKind::IntrinsicInto(i) => format!("{}_into", i.name()),
            OpKind::FragInit(init) => format!("frag_init {init:?}"),
            OpKind::FragScale(op) => format!("frag_scale {}", binop_name(*op)),
            OpKind::FragDot => "frag_dot".into(),
            OpKind::FragStore => "frag_store".into(),
            OpKind::GemmInit => "gemm_init".into(),
            OpKind::GemmDot => "gemm_dot".into(),
            OpKind::GemmStore { alpha, beta } => {
                let coeff = |c: &Coeff| match c {
                    Coeff::Absent => "",
                    Coeff::One => "1",
                    Coeff::Given => "s",
                };
                format!("gemm_store alpha={} beta={}", coeff(alpha), coeff(beta))
            }
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(self, OpKind::Yield | OpKind::Condition)
    }

    /// Whether the op owns blocks, and so is a region of the structured
    /// control flow.
    pub fn has_blocks(&self) -> bool {
        matches!(self, OpKind::For(_) | OpKind::While | OpKind::If)
    }

    /// Whether the result is a fresh buffer of its own, as opposed to a view
    /// of an operand's bytes, a scalar, or nothing.
    pub fn makes_buffer(&self) -> bool {
        matches!(
            self,
            OpKind::Alloc
                | OpKind::Stage(_)
                | OpKind::Materialize
                | OpKind::HoistStage
                | OpKind::Map(_)
                | OpKind::Reduce(_)
                | OpKind::Dot { .. }
                | OpKind::Intrinsic(_)
        )
    }

    /// Results that are views of an operand: reading or writing through
    /// them touches the operand's bytes.
    pub fn aliases(&self) -> Vec<Alias> {
        match self {
            OpKind::Slice(_) | OpKind::Flat | OpKind::AssumeAlign => vec![Alias {
                result: 0,
                operand: 0,
            }],
            _ => Vec::new(),
        }
    }

    /// Operand indices the op writes through, given how many operands it
    /// has. Everything else it only reads.
    pub fn writes(&self, operand_count: usize) -> Vec<usize> {
        let last = operand_count.saturating_sub(1);
        match self {
            OpKind::Store => vec![1],
            OpKind::AtomicAdd => vec![0],
            OpKind::GridBarrier => vec![0],
            OpKind::Copy { .. }
            | OpKind::Convert
            | OpKind::Fill
            | OpKind::MapInto(_)
            | OpKind::Fused(_)
            | OpKind::Chain(_)
            | OpKind::ScaledAdd
            | OpKind::DotInto { .. }
            | OpKind::Accumulate
            | OpKind::IntrinsicInto(_)
            | OpKind::FragStore => vec![last],
            OpKind::GemmStore { .. } => vec![1],
            // Both of these may rewrite their operand in place.
            OpKind::Reduce(Reduce::Tril) => vec![0],
            OpKind::Map(Map::Unary(ElemStep::Exp)) => vec![0],
            // rms_norm_q_t writes every tile after the gain and eps.
            OpKind::Intrinsic(Intrinsic::RmsNormQ) => (3..operand_count).collect(),
            _ => Vec::new(),
        }
    }
}

fn map_name(m: Map) -> String {
    match m {
        Map::Binary(op) => binop_name(op).into(),
        Map::Scalar { op, scalar_left } => {
            if scalar_left {
                format!("s{}", binop_name(op))
            } else {
                format!("{}s", binop_name(op))
            }
        }
        Map::Unary(step) => step.to_string(),
        Map::Max => "max".into(),
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::Const(Literal::Int(n)) => write!(f, "const {n}"),
            OpKind::Const(Literal::Float(v)) => write!(f, "const {v:?}"),
            OpKind::Const(Literal::Bool(b)) => write!(f, "const {b}"),
            other => f.write_str(&other.name()),
        }
    }
}
