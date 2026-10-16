// Which blocks of a `@persistent` grid touch which elements of a buffer in
// device memory.
//
// A unit loop hands block `p` the units `u = (p + k) % blocks + i * blocks`
// (`k` zero unrotated), under `if u < UN`, or as a grid-stride loop
// `for u in range((p + k) % blocks, UN, blocks)`. An access whose element is
// `a * u + b` of its buffer, flattened row-major, is made by the block owning
// `u`, so two accesses with the same `k`, `a` and `b`, neither spanning more
// than `|a|` elements, are made by the same block for every unit. An access at block-same indices under block-same
// control is made alike by every block. Anything else may be made by any
// block.

use std::collections::BTreeMap;

use crate::ast::BinOp;
use crate::ir::uniform::{Kind, Uniformity};
use crate::ir::{BlockId, Bounds, Def, ForInfo, Ir, Literal, OpId, OpKind, ThreadIndex, Type, ValueId};

/// Which blocks make an access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Owner {
    /// Every block, alike.
    Every,
    /// The block owning unit `u` of a loop rotated by `k`, at element
    /// `a * u + b` and the `span` elements from it, when known.
    Unit { k: i64, a: i64, b: i64, span: Option<i64> },
    /// Some block, which the analysis cannot name.
    Some,
}

/// One access to a buffer in device memory.
#[derive(Clone, Debug)]
pub(super) struct Access {
    pub op: OpId,
    /// The buffer: a kernel parameter or a scratch tensor's marker.
    pub root: ValueId,
    pub write: bool,
    pub owner: Owner,
    /// The flattened elements it may touch, `[lo, hi)`; None for any.
    pub range: Option<(i64, i64)>,
}

impl Access {
    /// Whether another block's `other` may touch an element this does,
    /// with at least one of them writing it.
    pub fn conflicts(&self, other: &Access) -> bool {
        if self.root != other.root || !(self.write || other.write) {
            return false;
        }
        if let (Some((l1, h1)), Some((l2, h2))) = (self.range, other.range)
            && (h1 <= l2 || h2 <= l1)
        {
            return false;
        }
        match (self.owner, other.owner) {
            (Owner::Every, Owner::Every) => false,
            // The same unit's block, and neither reaches into the next unit.
            (
                Owner::Unit { k, a, b, span: Some(s) },
                Owner::Unit { k: k2, a: a2, b: b2, span: Some(s2) },
            ) => !((k, a, b) == (k2, a2, b2) && s.max(s2) <= a.abs()),
            _ => true,
        }
    }
}

/// An index as `a * u + b` for the unit `u` of a loop, or as the constant
/// `b` when `unit` is None.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Affine {
    unit: Option<ValueId>,
    a: i64,
    b: i64,
}

impl Affine {
    fn constant(b: i64) -> Affine {
        Affine { unit: None, a: 0, b }
    }

    fn add(self, other: Affine, sign: i64) -> Option<Affine> {
        let unit = match (self.unit, other.unit) {
            (Some(x), Some(y)) if x != y => return None,
            (x, y) => x.or(y),
        };
        Some(Affine { unit, a: self.a + sign * other.a, b: self.b + sign * other.b })
    }

    fn scale(self, c: i64) -> Affine {
        Affine { unit: self.unit, a: self.a * c, b: self.b * c }
    }
}

/// A unit variable: its loop's rotation, and for a grid-stride loop's
/// variable, the units the loop runs to.
#[derive(Clone, Copy, Debug)]
struct Unit {
    k: i64,
    to: Option<i64>,
}

pub(super) struct Owners<'a> {
    ir: &'a Ir,
    blocks: i64,
    grid: &'a Uniformity,
    units: BTreeMap<ValueId, Unit>,
}

impl<'a> Owners<'a> {
    /// The unit variables of `ir`, whose grid has `blocks` blocks.
    pub fn new(ir: &'a Ir, blocks: i64, grid: &'a Uniformity) -> Owners<'a> {
        let mut o = Owners { ir, blocks, grid, units: BTreeMap::new() };
        for op in ir.all_ops() {
            if let Some(&r) = ir.results(op).first()
                && let Some(k) = o.unit_of(r)
            {
                o.units.insert(r, Unit { k, to: None });
            }
            if let OpKind::For(ForInfo { bounds: Bounds::Dynamic, ragged: false, .. }) = ir.kind(op)
                && o.constant(ir.operand(op, 2)) == Some(blocks)
                && let Some(k) = o.rotation(ir.operand(op, 0))
                && let Some(to) = o.constant(ir.operand(op, 1))
            {
                let iv = ir.args(ir.blocks_of(op)[0])[0];
                o.units.insert(iv, Unit { k, to: Some(to) });
            }
        }
        o
    }

    /// `base + i * blocks` with `i` a loop's variable and `base` the block or
    /// `(block + c) % blocks`: the rotation `k = blocks - c`, mod `blocks`.
    fn unit_of(&self, v: ValueId) -> Option<i64> {
        let ir = self.ir;
        let op = ir.def_op(v)?;
        let OpKind::Binary(BinOp::Add) = ir.kind(op) else { return None };
        let (base, step) = (ir.operand(op, 0), ir.operand(op, 1));
        let step_op = ir.def_op(step)?;
        let OpKind::Binary(BinOp::Mul) = ir.kind(step_op) else { return None };
        let (iv, b) = (ir.operand(step_op, 0), ir.operand(step_op, 1));
        if !matches!(ir.def(iv), Def::Arg { .. }) || self.constant(b) != Some(self.blocks) {
            return None;
        }
        self.rotation(base)
    }

    /// The rotation `k` of a loop's first unit, the block or
    /// `(block + c) % blocks`.
    fn rotation(&self, base: ValueId) -> Option<i64> {
        let ir = self.ir;
        if self.is_block(base) {
            return Some(0);
        }
        let rem = ir.def_op(base)?;
        let OpKind::Binary(BinOp::Rem) = ir.kind(rem) else { return None };
        if self.constant(ir.operand(rem, 1)) != Some(self.blocks) {
            return None;
        }
        let sum = ir.def_op(ir.operand(rem, 0))?;
        let OpKind::Binary(BinOp::Add) = ir.kind(sum) else { return None };
        let (p, c) = (ir.operand(sum, 0), ir.operand(sum, 1));
        (self.is_block(p)).then_some(())?;
        Some((self.blocks - self.constant(c)?).rem_euclid(self.blocks))
    }

    /// Whether `iv` is a unit, or a unit variable steps by the loop
    /// variable `iv`.
    pub fn hands_out_units(&self, iv: ValueId) -> bool {
        self.units.contains_key(&iv)
            || self.units.keys().any(|&u| {
                let step = self.ir.def_op(u).map(|op| self.ir.operand(op, 1));
                step.and_then(|s| self.ir.def_op(s)).is_some_and(|m| self.ir.operand(m, 0) == iv)
            })
    }

    fn is_block(&self, v: ValueId) -> bool {
        self.ir.def_op(v).is_some_and(|op| matches!(self.ir.kind(op), OpKind::ProgramId(0)))
    }

    fn constant(&self, v: ValueId) -> Option<i64> {
        let op = self.ir.def_op(v)?;
        match self.ir.kind(op) {
            OpKind::Const(Literal::Int(n)) => Some(*n),
            _ => None,
        }
    }

    fn affine(&self, v: ValueId) -> Option<Affine> {
        if self.units.contains_key(&v) {
            return Some(Affine { unit: Some(v), a: 1, b: 0 });
        }
        if let Some(n) = self.constant(v) {
            return Some(Affine::constant(n));
        }
        let ir = self.ir;
        let op = ir.def_op(v)?;
        let [x, y] = ir.operands(op)[..] else { return None };
        match ir.kind(op) {
            OpKind::Binary(BinOp::Add) => self.affine(x)?.add(self.affine(y)?, 1),
            OpKind::Binary(BinOp::Sub) => self.affine(x)?.add(self.affine(y)?, -1),
            OpKind::Binary(BinOp::Mul) => match (self.constant(x), self.constant(y)) {
                (Some(c), _) => Some(self.affine(y)?.scale(c)),
                (_, Some(c)) => Some(self.affine(x)?.scale(c)),
                _ => None,
            },
            _ => None,
        }
    }

    /// The values `v` may take at `op`, `[lo, hi]`, from its constants, the
    /// thread's indices, the units the guards around `op` allow, and index
    /// arithmetic by constants.
    fn interval(&self, op: OpId, v: ValueId) -> Option<(i64, i64)> {
        let ir = self.ir;
        if let Some(n) = self.constant(v) {
            return Some((n, n));
        }
        if self.units.contains_key(&v) {
            return Some((0, self.guard(op, v)? - 1));
        }
        let def = ir.def_op(v)?;
        let threads = ir.kernel.cta_threads;
        let operands = ir.operands(def);
        let pair = |i: usize| self.interval(op, operands[i]);
        match ir.kind(def) {
            OpKind::Thread(ThreadIndex::Thread) => Some((0, threads - 1)),
            OpKind::Thread(ThreadIndex::Warp) => Some((0, threads / 32 - 1)),
            OpKind::Thread(ThreadIndex::Lane) => Some((0, 31)),
            OpKind::ProgramId(0) => Some((0, self.blocks - 1)),
            OpKind::Binary(BinOp::Add) => {
                let ((a, b), (c, d)) = (pair(0)?, pair(1)?);
                Some((a + c, b + d))
            }
            OpKind::Binary(BinOp::Sub) => {
                let ((a, b), (c, d)) = (pair(0)?, pair(1)?);
                Some((a - d, b - c))
            }
            OpKind::Binary(BinOp::Mul) => {
                let ((a, b), (c, d)) = (pair(0)?, pair(1)?);
                let all = [a * c, a * d, b * c, b * d];
                Some((*all.iter().min()?, *all.iter().max()?))
            }
            OpKind::Binary(BinOp::Div) => {
                let ((a, b), c) = (pair(0)?, self.constant(operands[1])?);
                (c > 0 && a >= 0).then_some((a / c, b / c))
            }
            OpKind::Binary(BinOp::Rem) => {
                let ((a, _), c) = (pair(0)?, self.constant(operands[1])?);
                (c > 0 && a >= 0).then_some((0, c - 1))
            }
            _ => None,
        }
    }

    /// The units a block takes of `unit`, `[0, UN)`: from its grid-stride
    /// loop's bound, or the guard `unit < UN` it is read under.
    fn guard(&self, op: OpId, unit: ValueId) -> Option<i64> {
        if let Some(to) = self.units.get(&unit).and_then(|u| u.to) {
            return Some(to);
        }
        let ir = self.ir;
        for (anc, _) in ir.ancestors(ir.parent_block(op)) {
            if !matches!(ir.kind(anc), OpKind::If) {
                continue;
            }
            let cond = ir.def_op(ir.operand(anc, 0))?;
            if matches!(ir.kind(cond), OpKind::Binary(BinOp::Lt)) && ir.operand(cond, 0) == unit {
                return self.constant(ir.operand(cond, 1));
            }
        }
        None
    }

    /// Every access `op` makes to a buffer in device memory.
    pub fn accesses(&self, op: OpId) -> Vec<Access> {
        let ir = self.ir;
        let kind = ir.kind(op);
        // A view or a shape query reads no element.
        if !kind.aliases().is_empty() || kind.has_blocks() || matches!(kind, OpKind::Dim(_)) {
            return Vec::new();
        }
        let n = ir.operands(op).len();
        let writes = kind.writes(n);
        let mut out = Vec::new();
        for (i, &v) in ir.operands(op).iter().enumerate() {
            let Some(view) = self.view(v) else { continue };
            // An op that indexes the buffer itself names the element in its
            // operands after it.
            let rank = view.dims.len();
            let (idx, len, dim): (&[ValueId], i64, Option<usize>) = match kind {
                OpKind::Load if i == 0 => (&ir.operands(op)[1..], 1, None),
                OpKind::Store if i == 1 => (&ir.operands(op)[2..], 1, None),
                OpKind::LaneLoad { dim, len, .. } | OpKind::LaneStore { dim, len, .. } if i == 0 => {
                    (&ir.operands(op)[1..1 + rank], *len, Some(*dim))
                }
                _ => (&[], 0, None),
            };
            let (offset, span) = match idx.is_empty() {
                true => (view.offset, view.span()),
                false => self.at(&view, idx, len, dim),
            };
            let block_same = self.grid.control(ir.parent_block(op)) == Kind::Uniform
                && view.indices.iter().chain(idx).all(|&x| self.grid.of(x) == Kind::Uniform);
            let (owner, range) = match (block_same, offset) {
                (true, Some(f)) if f.unit.is_none() => (Owner::Every, span.map(|s| (f.b, f.b + s))),
                (true, _) => (Owner::Every, None),
                (false, Some(Affine { unit: Some(u), a, b })) => {
                    let k = self.units[&u].k;
                    let range = self.guard(op, u).zip(span).map(|(un, s)| {
                        let (lo, hi) = if a >= 0 { (b, b + a * (un - 1)) } else { (b + a * (un - 1), b) };
                        (lo, hi + s)
                    });
                    (Owner::Unit { k, a, b, span }, range)
                }
                _ => {
                    // The elements it may touch, from the bounds of every index.
                    let terms = view.indices.iter().zip(&view.strides).chain(idx.iter().zip(view.dims.iter().map(|(_, st)| st)));
                    let range = terms
                        .map(|(&x, &st)| st.zip(self.interval(op, x)).map(|(st, (lo, hi))| (lo * st, hi * st)))
                        .try_fold((0, 0), |acc, t| t.map(|(lo, hi)| (acc.0 + lo, acc.1 + hi)))
                        .zip(span)
                        .map(|((lo, hi), s)| (lo, hi + s));
                    (Owner::Some, range)
                }
            };
            out.push(Access { op, root: view.root, write: writes.contains(&i), owner, range });
        }
        out
    }

    /// The flattened element `view[idx]`, and the elements a run of `len`
    /// along `dim` from it spans.
    fn at(&self, view: &View, idx: &[ValueId], len: i64, dim: Option<usize>) -> (Option<Affine>, Option<i64>) {
        let mut offset = view.offset;
        for (d, &x) in idx.iter().enumerate() {
            let term = view.dims[d].1.and_then(|st| self.affine(x).map(|f| f.scale(st)));
            offset = match (offset, term) {
                (Some(o), Some(t)) => o.add(t, 1),
                _ => None,
            };
        }
        let span = dim.map_or(Some(1), |d| view.dims[d].1.map(|st| (len - 1) * st + 1));
        (offset, span)
    }

    /// `v` as a view of a kernel parameter or a scratch tensor, if it is one.
    fn view(&self, v: ValueId) -> Option<View> {
        let ir = self.ir;
        match ir.def(v) {
            Def::Arg { block, .. } => {
                if block != ir.entry() {
                    return None;
                }
                let Type::Tensor(t) = ir.ty(v) else { return None };
                let dims = row_major(&t.shape.iter().map(|e| e.fixed()).collect::<Vec<_>>());
                Some(View { root: v, offset: Some(Affine::constant(0)), dims, indices: Vec::new(), strides: Vec::new() })
            }
            Def::Result { op, .. } => match ir.kind(op) {
                OpKind::Scratch => {
                    let Type::Tensor(t) = ir.ty(v) else { return None };
                    let dims = row_major(&t.shape.iter().map(|e| e.fixed()).collect::<Vec<_>>());
                    Some(View { root: v, offset: Some(Affine::constant(0)), dims, indices: Vec::new(), strides: Vec::new() })
                }
                OpKind::AssumeAlign => self.view(ir.operand(op, 0)),
                OpKind::Flat => {
                    let src = self.view(ir.operand(op, 0))?;
                    let total: Option<i64> = src.dims.iter().map(|&(s, _)| s).product();
                    Some(View { dims: vec![(Some(1), total), (total, Some(1))], ..src })
                }
                OpKind::Slice(s) => {
                    let src = self.view(ir.operand(op, 0))?;
                    let mut offset = src.offset;
                    let mut indices = src.indices.clone();
                    let mut strides = src.strides.clone();
                    let mut dims = Vec::with_capacity(s.rank());
                    for d in 0..s.rank() {
                        let at = ir.operand(op, s.offset_operand(d));
                        indices.push(at);
                        let stride = src.dims[d].1;
                        strides.push(stride);
                        let term = stride.and_then(|st| self.affine(at).map(|f| f.scale(st)));
                        offset = match (offset, term) {
                            (Some(o), Some(t)) => o.add(t, 1),
                            _ => None,
                        };
                        dims.push((s.sizes[d].fixed().or(src.dims[d].0), stride));
                    }
                    Some(View { root: src.root, offset, dims, indices, strides })
                }
                _ => None,
            },
        }
    }
}

/// A view's place in its buffer: its first element, flattened, and each
/// dimension's size and stride in elements, None where the shape is dynamic.
struct View {
    root: ValueId,
    offset: Option<Affine>,
    dims: Vec<(Option<i64>, Option<i64>)>,
    /// Every index the view was cut at, with its stride in the buffer.
    indices: Vec<ValueId>,
    strides: Vec<Option<i64>>,
}

impl View {
    /// The elements from the first to the last it holds.
    fn span(&self) -> Option<i64> {
        self.dims.iter().map(|&(s, st)| Some((s? - 1).max(0) * st?)).sum::<Option<i64>>().map(|s| s + 1)
    }
}

/// Row-major strides, known while every later dimension's size is.
fn row_major(shape: &[Option<i64>]) -> Vec<(Option<i64>, Option<i64>)> {
    let mut out = vec![(None, None); shape.len()];
    let mut stride = Some(1);
    for d in (0..shape.len()).rev() {
        out[d] = (shape[d], stride);
        stride = stride.zip(shape[d]).map(|(st, s)| st * s);
    }
    out
}

/// Whether `block` sits under control every block takes alike.
pub(super) fn grid_uniform(grid: &Uniformity, block: BlockId) -> bool {
    grid.control(block) == Kind::Uniform
}
