// Rewrites an accumulator `frag.rs` took: the tile's seed becomes a
// `frag_init`, each use the fragment op that does the same to a value, and
// each loop between them a loop carrying that value.
//
// The build shaped those loops for a tile in shared memory, which the
// fragments do not want, so the rebuilt loop drops the shaping:
//
// - A constant range was folded to an affine loop; its bounds come back as
//   constants.
// - A ragged loop split off a masked last iteration; the fragments run the
//   whole range in the trimmed body, which reads its slices unmasked. That
//   is what the build's own check promised before this pass existed: a slice
//   riding the loop variable counted as in bounds.
// - A loop the build pipelined on its own initiative runs plain, its
//   staged prefix a staging run. One the kernel asked `@pipeline` of stays
//   pipelined, over the whole range where the build had split it and
//   replayed its last iteration in an `if` after it.

use std::collections::{BTreeMap, BTreeSet};

use super::frag::{Accumulator, Update};
use crate::ir::build::Report;
use crate::diag::Span;
use crate::ir::{At, Bounds, ForInfo, Ir, Literal, OpId, OpKind, Stage, Type, ValueId};

/// What a rewrite reads and records as it goes.
struct Carry<'a> {
    ty: Type,
    updates: BTreeMap<OpId, Update>,
    loops: BTreeSet<OpId>,
    spans: &'a mut BTreeMap<OpId, Span>,
    bounds: &'a BTreeMap<OpId, [Span; 3]>,
    /// The shared tile the fragments replace, which a read as a dot's
    /// operand still wants.
    tile: ValueId,
}

pub(super) fn rewrite(ir: &mut Ir, acc: Accumulator, report: &mut Report) {
    let tile = ir.result(acc.alloc);
    let mut c = Carry {
        ty: acc.ty(),
        updates: acc.updates.into_iter().collect(),
        loops: acc.loops,
        spans: &mut report.spans,
        bounds: &report.bounds,
        tile,
    };
    let init = c.make(ir, acc.alloc, At::Before(acc.alloc), OpKind::FragInit(acc.init), &[]);
    if let Some(name) = ir.name(tile).map(str::to_owned) {
        ir.set_name(init, name);
    }
    c.thread(ir, ir.parent_block(acc.alloc), init);
    if let Some(fill) = acc.fill {
        let seed = ir.def_op(ir.operand(fill, 0)).expect("the seed is a const");
        ir.erase_op(fill);
        ir.erase_op(seed);
    }
    ir.erase_op(acc.alloc);
}

impl Carry<'_> {
    /// A fragment op standing for `like`, at `at`, yielding the next value.
    fn make(&mut self, ir: &mut Ir, like: OpId, at: At, kind: OpKind, operands: &[ValueId]) -> ValueId {
        let op = self.op(ir, like, at, kind, operands, vec![self.ty.clone()], Vec::new());
        ir.result(op)
    }

    #[allow(clippy::too_many_arguments)]
    fn op(
        &mut self,
        ir: &mut Ir,
        like: OpId,
        at: At,
        kind: OpKind,
        operands: &[ValueId],
        types: Vec<Type>,
        blocks: Vec<crate::ir::BlockId>,
    ) -> OpId {
        let outer = ir.set_span(ir.span(like));
        let op = ir.create_op(at, kind, operands, types, blocks);
        ir.set_span(outer);
        if let Some(&span) = self.spans.get(&like) {
            self.spans.insert(op, span);
        }
        op
    }

    /// Threads `acc` through `block`'s uses and loops, returning the value
    /// it has at the end.
    fn thread(&mut self, ir: &mut Ir, block: crate::ir::BlockId, mut acc: ValueId) -> ValueId {
        for op in ir.ops(block).to_vec() {
            if !ir.is_alive(op) {
                continue;
            }
            if let Some(update) = self.updates.remove(&op) {
                acc = self.replace(ir, op, update, acc);
            } else if self.loops.contains(&op) {
                acc = self.carry(ir, op, acc);
            }
        }
        acc
    }

    fn replace(&mut self, ir: &mut Ir, op: OpId, update: Update, acc: ValueId) -> ValueId {
        let at = At::Before(op);
        let next = match &update {
            Update::Scaled { operands, .. } => {
                self.make(ir, op, at, OpKind::FragDotScaled, &[&[acc][..], operands].concat())
            }
            Update::Outer { negate, operands } => {
                self.make(ir, op, at, OpKind::FragOuter { negate: *negate }, &[&[acc][..], operands].concat())
            }
            Update::Scale { op: bop, col } => self.make(ir, op, at, OpKind::FragScale(*bop), &[acc, *col]),
            Update::Dot { transpose, a, b, .. } => {
                self.make(ir, op, at, OpKind::FragDot { transpose: *transpose }, &[acc, *a, *b])
            }
            Update::Map { tree, leaves } => {
                let leaves: Vec<ValueId> = leaves.iter().map(|&v| if v == self.tile { acc } else { v }).collect();
                self.make(ir, op, at, OpKind::FragMap(tree.clone()), &leaves)
            }
            // In place: its result keeps its id, which later uses name.
            Update::Reduce { how } => {
                *ir.kind_mut(op) = OpKind::FragReduce(*how);
                ir.set_operand(op, 0, acc);
                return acc;
            }
            Update::Operand { index } => {
                // The one conversion: the fragments written to a shared
                // tile of the same name for the dot to read.
                let ty = ir.ty(self.tile).clone();
                let made = self.op(ir, op, at, OpKind::Alloc, &[], vec![ty], Vec::new());
                let tile = ir.result(made);
                if let Some(name) = ir.name(self.tile).map(str::to_owned) {
                    ir.set_name(tile, name);
                }
                self.op(ir, op, at, OpKind::FragStore, &[acc, tile], Vec::new(), Vec::new());
                ir.set_operand(op, *index, tile);
                return acc;
            }
            Update::Store { dst } => {
                self.op(ir, op, at, OpKind::FragStore, &[acc, *dst], Vec::new(), Vec::new());
                acc
            }
        };
        ir.erase_op(op);
        if let Update::Scaled { dot, unstaged, .. } = update {
            ir.erase_op(dot);
            for stage in unstaged {
                ir.erase_op(stage);
            }
        }
        next
    }

    /// Rebuilds loop `old` to carry `acc` after what it carries already, and
    /// returns its result.
    fn carry(&mut self, ir: &mut Ir, old: OpId, acc: ValueId) -> ValueId {
        let OpKind::For(info) = *ir.kind(old) else {
            unreachable!("the pass carries values through loops only")
        };
        let operands = ir.operands(old).to_vec();
        let at = At::Before(old);
        let replay = ir.ops(ir.parent_block(old)).get(ir.position(old) + 2).copied().filter(|&next| {
            super::frag::pipelined_replay(ir, next) == Some(old)
        });
        let (bounds, full) = match info.bounds {
            Bounds::Affine { lo, hi, step } => {
                // Each constant where its bound is written, the loop where
                // a step is not.
                let spans = self.bounds.get(&old).copied().unwrap_or_default();
                let index = |c: &mut Self, ir: &mut Ir, n: i64, span: Span| {
                    let op = c.op(ir, old, at, OpKind::Const(Literal::Int(n)), &[], vec![Type::INDEX], Vec::new());
                    if span != Span::default() {
                        ir.set_op_span(op, span);
                    }
                    ir.result(op)
                };
                let [s0, s1, s2] = spans;
                ([index(self, ir, lo, s0), index(self, ir, hi, s1), index(self, ir, step, s2)], None)
            }
            Bounds::Dynamic if info.ragged => ([operands[0], operands[3], operands[2]], Some(operands[1])),
            Bounds::Dynamic => match replay {
                Some(next) => {
                    let cond = ir.def_op(ir.operand(next, 0)).expect("the replay's condition");
                    ([operands[0], ir.operand(cond, 1), operands[2]], Some(operands[1]))
                }
                None => ([operands[0], operands[1], operands[2]], None),
            },
        };
        let first = info.bound_operands();
        let carried = &operands[first..first + info.carried];
        let hoisted = &operands[first + info.carried..];

        let body = ir.blocks_of(old)[0];
        let mut arg_types: Vec<Type> = ir.args(body).iter().map(|&a| ir.ty(a).clone()).collect();
        arg_types.push(self.ty.clone());
        let block = ir.new_block(&arg_types);
        for (&from, &to) in ir.args(body).to_vec().iter().zip(&ir.args(block).to_vec()) {
            if let Some(name) = ir.name(from).map(str::to_owned) {
                ir.set_name(to, name);
            }
            ir.replace_all_uses(from, to);
        }
        let mut finals = Vec::new();
        for op in ir.ops(body).to_vec() {
            if ir.kind(op).is_terminator() {
                finals = ir.operands(op).to_vec();
                ir.erase_op(op);
            } else {
                ir.move_op(op, At::End(block));
            }
        }
        let pipeline = match info.pipeline {
            Some(p) if !ir.kernel.pipeline_assert => {
                as_staging_run(ir, block, p.staged);
                None
            }
            p => p,
        };
        let inner = *ir.args(block).last().expect("the carried value's argument");
        finals.push(self.thread(ir, block, inner));
        self.op(ir, old, At::End(block), OpKind::Yield, &finals, Vec::new(), Vec::new());

        let mut types: Vec<Type> = ir.results(old).iter().map(|&r| ir.ty(r).clone()).collect();
        types.push(self.ty.clone());
        let mut new_operands = bounds.to_vec();
        new_operands.extend_from_slice(carried);
        new_operands.push(acc);
        new_operands.extend_from_slice(hoisted);
        let kind = OpKind::For(ForInfo {
            bounds: Bounds::Dynamic,
            ragged: false,
            carried: info.carried + 1,
            hoisted: info.hoisted,
            pipeline,
        });
        let new = self.op(ir, old, at, kind, &new_operands, types, vec![block]);
        for (&from, &to) in ir.results(old).to_vec().iter().zip(&ir.results(new).to_vec()) {
            ir.replace_all_uses(from, to);
        }
        ir.erase_op(old);
        let cond = replay.map(|next| {
            let cond = ir.operand(next, 0);
            ir.erase_op(next);
            cond
        });
        // `full` first: while the replay's condition reads it, it stays, and
        // the condition takes it along.
        for v in full.into_iter().chain(cond) {
            erase_dead(ir, v);
        }
        *ir.results(new).last().expect("the carried value's result")
    }
}

/// The first `staged` stages of a pipelined loop's body as the staging run
/// the build makes of them in a plain one: every barrier but the last
/// deferred.
fn as_staging_run(ir: &mut Ir, block: crate::ir::BlockId, staged: usize) {
    let stages: Vec<OpId> = ir
        .ops(block)
        .iter()
        .copied()
        .filter(|&op| matches!(ir.kind(op), OpKind::Stage(_)))
        .take(staged)
        .collect();
    if stages.len() < 2 {
        return;
    }
    for &op in &stages[..stages.len() - 1] {
        if let OpKind::Stage(Stage { sync, .. }) = ir.kind_mut(op) {
            *sync = false;
        }
    }
}

/// Erases the index arithmetic that computed `v` once nothing reads it.
fn erase_dead(ir: &mut Ir, v: ValueId) {
    let Some(op) = ir.def_op(v) else { return };
    let pure = matches!(ir.kind(op), OpKind::Const(_) | OpKind::Binary(_) | OpKind::IndexOp(_));
    if !pure || ir.results(op).iter().any(|&r| !ir.uses(r).is_empty()) {
        return;
    }
    let operands = ir.operands(op).to_vec();
    ir.erase_op(op);
    for o in operands {
        erase_dead(ir, o);
    }
}
