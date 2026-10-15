// Fragment accumulators. A shared f32 tile that is filled with a float, or
// written whole by a dot, and from then on only read and updated in ways the
// tensor cores' accumulator registers allow, becomes a register value: each
// update yields a new one, and each loop it is updated in carries it.
// `carry.rs` rewrites.
//
// The forms, as the build leaves them:
//
// - `fused (add acc (mul (mul (f32 p) s1) s2))` with `p = dot_t a, w` on int8
//   operands and f32 row and column scales: the scaled int8 dot.
// - `fused (add acc (mul r c))` and `fused (sub acc (mul r c))`: the int8
//   accumulator's outer product.
// - `fused (mul acc col)` and `fused (div acc col)`: row scaling.
// - `dot_into a, b, acc`, accumulating or not, transposed or not: the f16
//   dot. One that does not accumulate seeds the tile, as its first use.
// - any other `fused` into the tile reading it at the element, other tiles
//   broadcast to it, and scalars: an f16 sweep.
// - `rowmax` and `rowsum` of it: a reduction in the lanes, when every warp
//   owns whole rows.
// - a read as another dot's operand: the fragments written to a shared tile
//   for it, the one conversion.
// - `copy acc, slice` or `convert acc, slice`: the store.
//
// The int8 forms and the f16 ones never mix in one accumulator. The f16
// fragments live at once fit a lane's register budget.

use std::collections::BTreeSet;

use crate::ast::BinOp;
use crate::ir::build::Target;
use crate::ir::{
    BlockId, DistType, ElemStep, Ir, Literal, MmaAcc, OpId, OpKind, Reduce, RegLayout, Scalar, Space, Tree, Type,
    ValueId,
};
use crate::shape;

/// One use of the accumulator tile, and what replaces it.
pub(super) enum Update {
    /// `fused`, with the dot it reads and any staging the build put in front
    /// of the dot's operands, which the fragments read in place.
    Scaled { dot: OpId, unstaged: Vec<OpId>, operands: [ValueId; 4] },
    /// `[col, row]`, the column first.
    Outer { negate: bool, operands: [ValueId; 2] },
    Scale { op: BinOp, col: ValueId },
    /// `fresh` for the dot that seeds the tile rather than adding to it.
    Dot { transpose: bool, fresh: bool, a: ValueId, b: ValueId },
    /// `tree` over `leaves`, the `fused`'s operands but its destination; a
    /// leaf that is the tile reads the fragments.
    Map { tree: Tree, leaves: Vec<ValueId> },
    Reduce { how: Reduce },
    /// The tile is operand `index` of a dot.
    Operand { index: usize },
    Store { dst: ValueId },
}

/// An accumulator the pass takes: the tile, its seed, and every use.
pub(super) struct Accumulator {
    pub alloc: OpId,
    /// The `fill` seeding the tile; none when a dot does.
    pub fill: Option<OpId>,
    pub init: f64,
    pub frags: MmaAcc,
    /// In program order, each with the op it replaces.
    pub updates: Vec<(OpId, Update)>,
    /// The loops between the tile and its uses, which carry it.
    pub loops: BTreeSet<OpId>,
}

impl Accumulator {
    pub fn ty(&self) -> Type {
        Type::Dist(DistType {
            elem: Scalar::F32,
            shape: vec![self.frags.m, self.frags.n],
            layout: RegLayout::MmaAcc {
                wm: self.frags.wm,
                wn: self.frags.wn,
                imma: self.frags.imma,
            },
        })
    }
}

/// Every accumulator in `ir` the pass takes, in program order.
pub(super) fn find(ir: &Ir, target: &Target) -> Vec<Accumulator> {
    ir.all_ops().into_iter().filter_map(|op| candidate(ir, target, op)).collect()
}

fn candidate(ir: &Ir, target: &Target, alloc: OpId) -> Option<Accumulator> {
    if !matches!(ir.kind(alloc), OpKind::Alloc) {
        return None;
    }
    let tile = ir.result(alloc);
    let Type::Tile(t) = ir.ty(tile) else { return None };
    if t.space != Space::Shared || t.elem != Scalar::F32 || t.layout.row_stride.is_some() {
        return None;
    }
    let [m, n] = t.static_shape()?[..] else { return None };

    // The seed: `const <float>` then `fill`, right after the alloc, or else
    // a dot writing the whole tile as its first use.
    let block = ir.parent_block(alloc);
    let at = ir.position(alloc);
    let filled = match ir.ops(block).get(at + 1..at + 3) {
        Some(&[seed, fill]) => match *ir.kind(seed) {
            OpKind::Const(Literal::Float(init))
                if matches!(ir.kind(fill), OpKind::Fill) && ir.operands(fill) == [ir.result(seed), tile] =>
            {
                Some((fill, init))
            }
            _ => None,
        },
        _ => None,
    };
    let fill = filled.map(|(fill, _)| fill);
    let init = filled.map_or(0.0, |(_, init)| init);

    let mut scan = Scan::default();
    let mut users: Vec<OpId> = ir.uses(tile).iter().map(|u| u.op).filter(|&u| Some(u) != fill).collect();
    users.sort_by_key(|&u| order(ir, u));
    users.dedup();
    let mut updates = Vec::new();
    let mut loops = BTreeSet::new();
    for user in users {
        // The masked replay of a ragged loop goes when the loop is rebuilt,
        // its uses with it.
        if in_replay(ir, user, block) {
            continue;
        }
        enclosing_loops(ir, user, block, &mut loops)?;
        let u = update(ir, tile, user, m, n, &mut scan)?;
        // Only the first use may seed the tile, and only where nothing did.
        let seeds = matches!(u, Update::Dot { fresh: true, .. });
        if seeds != (fill.is_none() && updates.is_empty()) || (seeds && ir.parent_block(user) != block) {
            return None;
        }
        updates.push((user, u));
    }
    let frags = if scan.imma? {
        if !target.has_int8_mma {
            return None;
        }
        let (wm, wn) = shape::imma_plan(m, n, ir.kernel.cta_threads)?;
        MmaAcc { m, n, wm, wn, imma: true }
    } else {
        if !(ir.kernel.mma_sync && target.has_mma_sync) {
            return None;
        }
        let plan = if scan.rows { shape::wmma_rows_plan } else { shape::wmma_plan };
        let (wm, wn) = plan(m, n, scan.kk?, ir.kernel.cta_threads)?;
        MmaAcc { m, n, wm, wn, imma: false }
    };
    if !frags.imma && !fits(ir, alloc, &frags) {
        return None;
    }
    Some(Accumulator { alloc, fill, init, frags, updates, loops })
}

/// What the uses so far have settled.
#[derive(Default)]
struct Scan {
    /// The int8 forms rather than the f16 ones.
    imma: Option<bool>,
    /// The f16 dot's contraction step.
    kk: Option<i64>,
    /// Whether a row reduction reads the tile.
    rows: bool,
}

impl Scan {
    fn kind(&mut self, imma: bool) -> Option<()> {
        (*self.imma.get_or_insert(imma) == imma).then_some(())
    }
}

/// Whether f16 fragments `frags` fit beside those the loops around `alloc`
/// already carry.
fn fits(ir: &Ir, alloc: OpId, frags: &MmaAcc) -> bool {
    let mut lanes = shape::frag_lane_f32(frags.m, frags.n, frags.wm, frags.wn, false);
    for (op, _) in ir.ancestors(ir.parent_block(alloc)) {
        let OpKind::For(info) = ir.kind(op) else { continue };
        let first = info.bound_operands();
        for &v in &ir.operands(op)[first..first + info.carried] {
            if let Type::Dist(d) = ir.ty(v)
                && let Some(a) = d.mma_acc()
            {
                lanes += shape::frag_lane_f32(a.m, a.n, a.wm, a.wn, a.imma);
            }
        }
    }
    lanes <= shape::FRAG_LANE_BUDGET
}

/// A key that sorts ops into program order, nested ones included.
fn order(ir: &Ir, op: OpId) -> Vec<usize> {
    let mut key = vec![ir.position(op)];
    let mut block = ir.parent_block(op);
    while let Some(parent) = ir.parent_op(block) {
        key.push(ir.position(parent));
        block = ir.parent_block(parent);
    }
    key.reverse();
    key
}

/// Whether `op` sits in the masked replay of a ragged loop below `home`.
fn in_replay(ir: &Ir, op: OpId, home: BlockId) -> bool {
    let mut block = ir.parent_block(op);
    while block != home {
        let Some(parent) = ir.parent_op(block) else { return false };
        let ragged = matches!(ir.kind(parent), OpKind::For(info) if info.ragged) && ir.blocks_of(parent)[1] == block;
        if ragged || pipelined_replay(ir, parent).is_some() {
            return true;
        }
        block = ir.parent_block(parent);
    }
    false
}

/// For an `if` that replays a split pipelined loop's last iteration, masked,
/// the loop: a pipelined `for` over `lo .. full`, `full < hi`, then the `if`
/// on it.
pub(super) fn pipelined_replay(ir: &Ir, op: OpId) -> Option<OpId> {
    if !matches!(ir.kind(op), OpKind::If) {
        return None;
    }
    let ops = ir.ops(ir.parent_block(op));
    let at = ir.position(op);
    let cond = *ops.get(at.checked_sub(1)?)?;
    let prev = *ops.get(at.checked_sub(2)?)?;
    let OpKind::For(info) = ir.kind(prev) else { return None };
    let split = ir.def_op(ir.operand(op, 0)) == Some(cond)
        && info.pipeline.is_some()
        && matches!(info.bounds, crate::ir::Bounds::Dynamic)
        && matches!(ir.kind(cond), OpKind::Binary(BinOp::Lt))
        && ir.operand(cond, 0) == ir.operand(prev, 1);
    split.then_some(prev)
}

/// Records the loops between `op` and `home`, failing on any other region:
/// an update under a branch could not hand its value on.
fn enclosing_loops(ir: &Ir, op: OpId, home: BlockId, loops: &mut BTreeSet<OpId>) -> Option<()> {
    let mut block = ir.parent_block(op);
    while block != home {
        let parent = ir.parent_op(block)?;
        let OpKind::For(_) = ir.kind(parent) else { return None };
        loops.insert(parent);
        block = ir.parent_block(parent);
    }
    Some(())
}

/// What `user` does to the tile, if it is a form the fragments hold.
fn update(ir: &Ir, tile: ValueId, user: OpId, m: i64, n: i64, scan: &mut Scan) -> Option<Update> {
    let operands = ir.operands(user);
    match ir.kind(user) {
        OpKind::Fused(tree) => {
            let (&dst, leaves) = operands.split_last()?;
            if dst != tile {
                return None;
            }
            let leaf = |t: &Tree| match t {
                Tree::Leaf(i) => leaves.get(*i).copied(),
                _ => None,
            };
            let is_acc = |t: &Tree| leaf(t) == Some(tile);
            let others = |t: &[&Tree]| -> Option<Vec<ValueId>> {
                let vs = t.iter().map(|&x| leaf(x)).collect::<Option<Vec<_>>>()?;
                (!vs.contains(&tile)).then_some(vs)
            };
            match tree {
                Tree::Binary(BinOp::Add, acc, rest) if is_acc(acc) => {
                    if let Tree::Binary(BinOp::Mul, lhs, s2) = &**rest
                        && let Tree::Binary(BinOp::Mul, cast, s1) = &**lhs
                        && let Tree::Unary(ElemStep::Cast(Scalar::F32), p) = &**cast
                    {
                        let [p, s1, s2] = others(&[p, s1, s2])?[..] else { return None };
                        return scaled(ir, p, [s1, s2], m, n, scan);
                    }
                    let Tree::Binary(BinOp::Mul, r, c) = &**rest else { return None };
                    let [r, c] = others(&[r, c])?[..] else { return None };
                    outer(ir, false, [r, c], m, n, scan)
                }
                Tree::Binary(BinOp::Sub, acc, rest) if is_acc(acc) => {
                    let Tree::Binary(BinOp::Mul, r, c) = &**rest else { return None };
                    let [r, c] = others(&[r, c])?[..] else { return None };
                    outer(ir, true, [r, c], m, n, scan)
                }
                Tree::Binary(op @ (BinOp::Mul | BinOp::Div), acc, col)
                    if is_acc(acc) && others(&[col]).is_some_and(|c| column(ir, c[0], m)) =>
                {
                    scan.kind(false)?;
                    Some(Update::Scale { op: *op, col: others(&[col])?[0] })
                }
                _ => map(ir, tile, tree, leaves, m, n, scan),
            }
        }
        OpKind::Reduce(how @ (Reduce::RowMax | Reduce::RowSum)) if operands == [tile] => {
            scan.kind(false)?;
            scan.rows = true;
            Some(Update::Reduce { how: *how })
        }
        // The factors: `a, b` of `dot_into a, b, dst`, and of `frag_dot acc, a, b`.
        OpKind::DotInto { aliased: false, .. } | OpKind::FragDot { .. } if factor(ir.kind(user), operands, tile) => {
            let index = operands.iter().position(|&v| v == tile)?;
            scan.kind(false)?;
            Some(Update::Operand { index })
        }
        &OpKind::DotInto {
            transpose,
            accumulate,
            aliased: false,
        } => {
            let &[a, b, dst] = operands else { return None };
            if dst != tile || a == tile || b == tile {
                return None;
            }
            // Staged as any dot's operands are, from named tiles.
            let wide = |v: ValueId| {
                let t = ir.ty(v).as_tile()?;
                let named = ir.name(unstage(ir, v).0).is_some();
                (named && matches!(t.elem, Scalar::F16 | Scalar::F32)).then(|| t.static_shape())?
            };
            let (&[am, ak], &[b0, b1]) = (&wide(a)?[..], &wide(b)?[..]) else { return None };
            let (bk, bn) = if transpose { (b1, b0) } else { (b0, b1) };
            scan.kind(false)?;
            if am != m || bn != n || ak != bk || ak % 16 != 0 {
                return None;
            }
            scan.kk.get_or_insert(ak);
            Some(Update::Dot { transpose, fresh: !accumulate, a, b })
        }
        OpKind::Copy { sync: true } | OpKind::Convert => {
            let &[src, dst] = operands else { return None };
            (src == tile && tensor_store(ir, dst, m, n)).then_some(Update::Store { dst })
        }
        _ => None,
    }
}

/// Whether `tile` is exactly one factor of the dot `kind` and nothing else.
fn factor(kind: &OpKind, operands: &[ValueId], tile: ValueId) -> bool {
    let factors = match kind {
        OpKind::FragDot { .. } => 1..3,
        _ => 0..2,
    };
    let count = |r: std::ops::Range<usize>| operands.get(r).map_or(0, |o| o.iter().filter(|&&v| v == tile).count());
    count(factors) == 1 && operands.iter().filter(|&&v| v == tile).count() == 1
}

/// Whether `v` is a named f32 `[m, 1]` tile, the column row scaling reads.
fn column(ir: &Ir, v: ValueId, m: i64) -> bool {
    matches!(ir.ty(v), Type::Tile(_)) && read_in_place(ir, v, Scalar::F32) == Some(vec![m, 1])
}

/// Any other `fused` into the tile: an f16 sweep whose leaves are the tile,
/// unmasked tiles broadcast to it, and scalars.
fn map(ir: &Ir, tile: ValueId, tree: &Tree, leaves: &[ValueId], m: i64, n: i64, scan: &mut Scan) -> Option<Update> {
    let mut read = Vec::new();
    tree.operands(&mut read);
    for i in read {
        let v = *leaves.get(i)?;
        let ok = v == tile
            || match ir.ty(v) {
                Type::Scalar(_) => true,
                Type::Tile(t) => {
                    !masked(ir, v)
                        && t.static_shape().is_some_and(|s| {
                            s.len() == 2 && (s[0] == m || s[0] == 1) && (s[1] == n || s[1] == 1)
                        })
                }
                _ => false,
            };
        if !ok {
            return None;
        }
    }
    scan.kind(false)?;
    Some(Update::Map { tree: tree.clone(), leaves: leaves.to_vec() })
}

/// `acc += f32(dot_t(a, w)) * s1 * s2` for the dot whose product is `p`.
fn scaled(ir: &Ir, p: ValueId, scales: [ValueId; 2], m: i64, n: i64, scan: &mut Scan) -> Option<Update> {
    let dot = ir.def_op(p)?;
    if !matches!(ir.kind(dot), OpKind::Dot { transpose: true }) || ir.uses(p).len() != 1 {
        return None;
    }
    let mut unstaged = Vec::new();
    let [a, w] = [0, 1].map(|i| {
        let (v, stage) = unstage(ir, ir.operand(dot, i));
        unstaged.extend(stage);
        v
    });
    let (Some(ash), Some(wsh)) = (read_in_place(ir, a, Scalar::I8), read_in_place(ir, w, Scalar::I8)) else {
        return None;
    };
    let (&[am, ak], &[wn, wk]) = (&ash[..], &wsh[..]) else { return None };
    if am != m || wn != n || ak != wk || ak <= 0 || ak % 16 != 0 {
        return None;
    }
    broadcasts(ir, scales, m, n)?;
    scan.kind(true)?;
    let [s1, s2] = scales;
    Some(Update::Scaled { dot, unstaged, operands: [a, w, s1, s2] })
}

/// `acc += r * c` or `acc -= r * c`.
fn outer(ir: &Ir, negate: bool, parts: [ValueId; 2], m: i64, n: i64, scan: &mut Scan) -> Option<Update> {
    let column = broadcasts(ir, parts, m, n)?;
    scan.kind(true)?;
    let [r, c] = parts;
    let operands = if column == 0 { [r, c] } else { [c, r] };
    Some(Update::Outer { negate, operands })
}

/// Whether `vs` are an f32 `[m, 1]` column and a `[1, n]` row, in either
/// order: the index of the column.
fn broadcasts(ir: &Ir, vs: [ValueId; 2], m: i64, n: i64) -> Option<usize> {
    let [a, b] = vs.map(|v| read_in_place(ir, v, Scalar::F32));
    let (col, row) = (Some(vec![m, 1]), Some(vec![1, n]));
    if a == col && b == row {
        Some(0)
    } else if a == row && b == col {
        Some(1)
    } else {
        None
    }
}

/// The static shape of a value the fragments read with no bounds guard: a
/// named tile of element `elem`, unmasked, or a generated one.
fn read_in_place(ir: &Ir, v: ValueId, elem: Scalar) -> Option<Vec<i64>> {
    match ir.ty(v) {
        Type::Dist(d) if d.layout == RegLayout::Generated && d.elem == elem => Some(d.shape.clone()),
        Type::Tile(t) if t.elem == elem && ir.name(v).is_some() && !masked(ir, v) => t.static_shape(),
        _ => None,
    }
}

fn masked(ir: &Ir, v: ValueId) -> bool {
    ir.def_op(v)
        .is_some_and(|op| matches!(ir.kind(op), OpKind::Slice(s) if s.masked.iter().any(|&m| m)))
}

/// A dot operand without the copy into shared memory the build makes when
/// no tensor core reads it: the fragments read the source in place.
fn unstage(ir: &Ir, v: ValueId) -> (ValueId, Option<OpId>) {
    match ir.def_op(v) {
        Some(op) if matches!(ir.kind(op), OpKind::Stage(_)) && ir.name(v).is_none() && ir.uses(v).len() == 1 => {
            (ir.operand(op, 0), Some(op))
        }
        _ => (v, None),
    }
}

/// Whether `dst` is an unmasked `[m, n]` f16 or f32 slice of a tensor.
fn tensor_store(ir: &Ir, dst: ValueId, m: i64, n: i64) -> bool {
    let Some(slice) = ir.def_op(dst) else { return false };
    let OpKind::Slice(_) = ir.kind(slice) else { return false };
    let Type::Tile(t) = ir.ty(dst) else { return false };
    matches!(ir.ty(ir.operand(slice, 0)), Type::Tensor(_))
        && matches!(t.elem, Scalar::F16 | Scalar::F32)
        && t.static_shape().as_deref() == Some(&[m, n])
        && !masked(ir, dst)
}
