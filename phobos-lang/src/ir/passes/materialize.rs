// A generated tile the fragments do not read. The build leaves a `generate`
// before each update an int8 accumulator takes; when the accumulator stays
// in shared memory, its updates read tiles, so each such `generate` is
// written out to one first: every thread evaluates the block for units
// `ctx.thread`, `ctx.thread + threads`, .., stores them, and a barrier
// closes the copy.

use std::collections::BTreeMap;

use crate::diag::Span;
use crate::ir::build::{alloc_align_div, extent};
use crate::ir::{
    At, BinOp, Bounds, Builder, ForInfo, Ir, Layout, OpId, OpKind, Scalar, Space, ThreadIndex, TileType, Type,
};

/// Materializes every `generate` read by anything but a fragment update.
pub(super) fn leftovers(ir: &mut Ir, spans: &mut BTreeMap<OpId, Span>) {
    for op in ir.all_ops() {
        if !ir.is_alive(op) || !matches!(ir.kind(op), OpKind::Generate) {
            continue;
        }
        let g = ir.result(op);
        let readers: Vec<_> = ir
            .uses(g)
            .iter()
            .copied()
            .filter(|u| !matches!(ir.kind(u.op), OpKind::FragDotScaled | OpKind::FragOuter { .. }))
            .collect();
        if readers.is_empty() {
            continue;
        }
        let first = ir.op_count();
        let tile = materialize(ir, op);
        for u in readers {
            ir.set_operand(u.op, u.index, tile);
        }
        if let Some(&span) = spans.get(&op) {
            for made in (first..ir.op_count()).map(OpId::from_index) {
                spans.insert(made, span);
            }
        }
        if ir.uses(g).is_empty() {
            ir.erase_op(op);
        }
    }
}

/// The tile `generate` op `op` stands for, filled before it.
fn materialize(ir: &mut Ir, op: OpId) -> crate::ir::ValueId {
    let Type::Dist(d) = ir.ty(ir.result(op)).clone() else {
        unreachable!("a generate yields a generated tile")
    };
    let [rows, cols] = d.shape[..] else { unreachable!("a generated tile has rank 2") };
    let bytes = d.elem.bytes().expect("a generated element has a width");
    // A unit is a word of an 8-bit tile, else one element.
    let unit = if bytes == 1 { 4 } else { 1 };
    let units = cols / unit;
    let threads = ir.kernel.cta_threads;
    let outer = ir.set_span(ir.span(op));
    let mut b = Builder::new(ir, At::Before(op));
    let tile = b.value(
        OpKind::Alloc,
        &[],
        Type::Tile(TileType {
            elem: d.elem,
            shape: vec![extent(rows), extent(cols)],
            layout: Layout {
                row_stride: None,
                swizzle: None,
                align_div: alloc_align_div(&[rows, cols]),
            },
            space: Space::Shared,
        }),
    );
    let index = |b: &mut Builder<'_>, n: i64| b.value(OpKind::Const(crate::ir::Literal::Int(n)), &[], Type::INDEX);
    let lo = b.value(OpKind::Thread(ThreadIndex::Thread), &[], Type::INDEX);
    let hi = index(&mut b, rows * units);
    let st = index(&mut b, threads);
    let body = b.block_with(&[Type::INDEX]);
    let u = b.ir.args(body)[0];
    let region = b.ir.blocks_of(op)[0];
    let [row_arg, unit_arg] = b.ir.args(region)[..] else { unreachable!("a generate block takes a row and a unit") };
    b.in_block(body, |b| {
        let n = index(b, units);
        let row = b.value(OpKind::Binary(BinOp::Div), &[u, n], Type::INDEX);
        let at = b.value(OpKind::Binary(BinOp::Rem), &[u, n], Type::INDEX);
        let mut map = BTreeMap::from([(row_arg, row), (unit_arg, at)]);
        let [value] = b.ir.clone_body(region, At::End(body), &mut map)[..] else {
            unreachable!("a generate block yields one value")
        };
        let width = index(b, unit);
        let col = b.value(OpKind::Binary(BinOp::Mul), &[at, width], Type::INDEX);
        let words = bytes == 1;
        let value = if words { b.value(OpKind::IndexCast(Scalar::I32), &[value], Type::Scalar(Scalar::I32)) } else { value };
        let store = OpKind::LaneStore { dim: 1, len: unit, align: unit * bytes, words };
        b.stmt(store, &[tile, row, col, value]);
        b.stmt(OpKind::Yield, &[]);
    });
    let info = ForInfo {
        bounds: Bounds::Dynamic,
        ragged: false,
        carried: 0,
        hoisted: 0,
        pipeline: None,
    };
    b.op(OpKind::For(info), &[lo, hi, st], Vec::new(), vec![body]);
    b.stmt(OpKind::Barrier, &[]);
    ir.set_span(outer);
    tile
}
