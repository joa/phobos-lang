// A dot followed by the elementwise update of its product becomes one
// `DotFused`, which applies the update to each element as the product is
// written out, instead of storing the product and sweeping it again.

use crate::ir::{At, Ir, OpId, OpKind, Space, Tree, Type, ValueId};

/// Fuses, in every block:
///
/// - `dot_into a, b, d` then an update of `d`: every read of `d` in the
///   update is the product.
/// - `p = dot a, b` then an update of some `t` that reads `p` and nothing
///   else does: the update reads `t` at the element it writes.
///
/// Only scalar ops may sit between the two. The destination is a shared tile
/// of its own, since the write-out has no bounds guard, and neither factor
/// reads it, since a factor is read across the whole contraction.
///
/// Returns each fused op with the dot it replaced.
pub(super) fn fuse_dot_epilogues(ir: &mut Ir) -> Vec<(OpId, OpId)> {
    let mut fused = Vec::new();
    let dots: Vec<OpId> = ir
        .all_ops()
        .into_iter()
        .filter(|&op| {
            matches!(
                ir.kind(op),
                OpKind::Dot { .. }
                    | OpKind::DotInto {
                        accumulate: false,
                        aliased: false,
                        ..
                    }
            )
        })
        .collect();
    for dot in dots {
        let Some(update) = next_tile_op(ir, dot) else { continue };
        let Some((kind, operands)) = plan(ir, dot, update) else { continue };
        let op = ir.create_op(At::Before(update), kind, &operands, Vec::new(), Vec::new());
        ir.erase_op(update);
        ir.erase_op(dot);
        fused.push((op, dot));
    }
    fused
}

/// The op after `dot`, past any scalar ops.
fn next_tile_op(ir: &Ir, dot: OpId) -> Option<OpId> {
    let ops = ir.ops(ir.parent_block(dot));
    ops[ir.position(dot) + 1..].iter().copied().find(|&o| !is_scalar_op(ir, o))
}

/// An op that only computes scalars from scalars.
fn is_scalar_op(ir: &Ir, op: OpId) -> bool {
    !ir.results(op).is_empty()
        && !ir.kind(op).has_blocks()
        && ir
            .operands(op)
            .iter()
            .chain(ir.results(op))
            .all(|&v| matches!(ir.ty(v), Type::Scalar(_)))
}

/// The fused op and its operands, `(a, b, leaves.., dst)`.
fn plan(ir: &Ir, dot: OpId, update: OpId) -> Option<(OpKind, Vec<ValueId>)> {
    let OpKind::Fused(tree) = ir.kind(update) else {
        return None;
    };
    let (&dst, leaves) = ir.operands(update).split_last()?;
    let (a, b) = (ir.operand(dot, 0), ir.operand(dot, 1));
    let (transpose, product, in_place) = match *ir.kind(dot) {
        OpKind::DotInto { transpose, .. } => (transpose, ir.operand(dot, 2), true),
        OpKind::Dot { transpose } => (transpose, ir.result(dot), false),
        _ => return None,
    };
    if in_place && product != dst {
        return None;
    }
    if !in_place {
        // The product accumulates in the destination, so it must be of the
        // same type and shape, and the update its only reader.
        let same = matches!((ir.ty(product), ir.ty(dst)), (Type::Tile(p), Type::Tile(d))
            if p.elem == d.elem && p.static_shape().is_some() && p.static_shape() == d.static_shape());
        if product == dst || !same || ir.uses(product).iter().any(|u| u.op != update) {
            return None;
        }
    }
    let owned = ir.def_op(dst).is_some_and(|d| matches!(ir.kind(d), OpKind::Alloc))
        && matches!(ir.ty(dst), Type::Tile(t) if t.space == Space::Shared);
    if !owned || root(ir, a) == dst || root(ir, b) == dst {
        return None;
    }

    let mut operands = vec![a, b];
    let tree = remap(tree, leaves, product, &mut operands)?;
    let mut read = Vec::new();
    tree.operands(&mut read);
    if !read.contains(&0) {
        return None;
    }
    operands.push(dst);
    Some((
        OpKind::DotFused {
            transpose,
            tree,
            in_place,
        },
        operands,
    ))
}

/// `tree` over `operands`: a read of `product` becomes `Leaf(0)`, and every
/// other leaf is appended to `operands` once.
fn remap(tree: &Tree, leaves: &[ValueId], product: ValueId, operands: &mut Vec<ValueId>) -> Option<Tree> {
    let index = |v: ValueId, operands: &mut Vec<ValueId>| match operands[2..].iter().position(|&o| o == v) {
        Some(k) => 2 + k,
        None => {
            operands.push(v);
            operands.len() - 1
        }
    };
    let go = |t: &Tree, operands: &mut Vec<ValueId>| remap(t, leaves, product, operands).map(Box::new);
    Some(match tree {
        Tree::Leaf(i) if leaves[*i] == product => Tree::Leaf(0),
        Tree::Leaf(i) => Tree::Leaf(index(leaves[*i], operands)),
        Tree::Scalar(i) if leaves[*i] == product => return None,
        Tree::Scalar(i) => Tree::Scalar(index(leaves[*i], operands)),
        Tree::Iota(d) => Tree::Iota(*d),
        Tree::Unary(step, x) => Tree::Unary(*step, go(x, operands)?),
        Tree::Binary(op, x, y) => Tree::Binary(*op, go(x, operands)?, go(y, operands)?),
        Tree::Max(x, y) => Tree::Max(go(x, operands)?, go(y, operands)?),
        Tree::Select(c, x, y) => Tree::Select(go(c, operands)?, go(x, operands)?, go(y, operands)?),
    })
}

/// The buffer `v` is a view of, through slices and the like.
fn root(ir: &Ir, mut v: ValueId) -> ValueId {
    while let Some(op) = ir.def_op(v)
        && let Some(alias) = ir.kind(op).aliases().first()
    {
        v = ir.operand(op, alias.operand);
    }
    v
}
