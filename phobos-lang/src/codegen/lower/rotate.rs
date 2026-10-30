// Rotating stages. In a loop that carries register accumulators, on a chip
// with cp.async, a synchronous stage the pipeline does not double-buffer is
// issued for the next turn as soon as its tile's last reader in this one is
// done, into the same buffer, and waited for where the stage stands. Its
// load then overlaps whatever runs between its last reader and its next
// turn, as flash attention's keys overlap the softmax and the value mix and
// its values the next turn's scores, for no more shared memory.
//
// Each issue commits a group, an empty one past the loop's end, so the
// groups a wait may leave in flight are a count fixed by where the issues
// sit: those issued after this stage's in the turn before, and those issued
// before the stage in this one. The loop's first turn is issued ahead of the
// loop in the same order, so the count holds from the start.

use std::collections::{BTreeSet, HashMap};

use anyhow::Result;
use melior::dialect::arith;
use melior::ir::{Block, BlockLike, Value, operation::OperationLike};

use super::Lowered;
use super::defer::writes_global;
use crate::codegen::{Codegen, MemVal};
use crate::ir::{BlockId, Def, Ir, OpId, OpKind, Space, Type, ValueId};

/// One stage the loop rotates.
pub(super) struct Rotation {
    stage: OpId,
    /// The body op after which the next turn's copy is issued.
    after: OpId,
    /// The body ops the stage's source is computed by, in order.
    source: Vec<OpId>,
    /// The groups the wait at the stage leaves in flight.
    pending: u32,
}

/// A rotating stage's buffer and wait, as the stage op finds them.
#[derive(Clone)]
pub(in crate::codegen) struct Rotating<'c> {
    buf: MemVal<'c>,
    token: Value<'c, 'c>,
    pending: u32,
}

/// The copies a stage's last reader issues for the next turn, and the loop
/// they belong to.
#[derive(Clone)]
pub(in crate::codegen) struct Issues<'c> {
    iv: ValueId,
    step: Value<'c, 'c>,
    hi: Value<'c, 'c>,
    stages: Vec<(OpId, Vec<OpId>)>,
}

/// The stages of `body` that rotate. None when the body writes global
/// memory, which a load issued a turn early could read before.
pub(super) fn plan(ir: &Ir, body: BlockId) -> Vec<Rotation> {
    let ops = ir.ops(body);
    if ops.iter().any(|&o| writes_global(ir, o)) {
        return Vec::new();
    }
    let iv = ir.args(body)[0];
    let pos = |op: OpId| ops.iter().position(|&o| o == top(ir, body, op)).expect("an op of the body");
    let mut rots: Vec<(Rotation, usize, usize)> = Vec::new();
    for (at, &stage) in ops.iter().enumerate() {
        let OpKind::Stage(s) = ir.kind(stage) else { continue };
        if !s.sync {
            continue;
        }
        let mut source = BTreeSet::new();
        if !global_source(ir, ir.operand(stage, 0)) || !computed_from(ir, body, iv, ir.operand(stage, 0), &mut source) {
            continue;
        }
        let Some(readers) = readers(ir, ir.result(stage)) else { continue };
        let writes = readers.iter().any(|u| ir.kind(u.op).writes(ir.operands(u.op).len()).contains(&u.index));
        let Some(last) = readers.iter().map(|u| pos(u.op)).max() else { continue };
        if writes || readers.iter().any(|u| pos(u.op) <= at) {
            continue;
        }
        let mut source: Vec<OpId> = source.into_iter().collect();
        source.sort_by_key(|&o| pos(o));
        rots.push((Rotation { stage, after: ops[last], source, pending: 0 }, at, last));
    }
    let issued: Vec<usize> = rots.iter().map(|r| r.2).collect();
    for (r, at, last) in &mut rots {
        let later = issued.iter().filter(|&&p| p > *last).count();
        let earlier = issued.iter().filter(|&&p| p < *at).count();
        r.pending = (later + earlier) as u32;
    }
    rots.into_iter().map(|r| r.0).collect()
}

/// Every use of `tile`, and of each view of it a slice or a flattening
/// takes, which read the buffer as the tile does. None when a view leaves
/// the loop turn, through a yield or a loop's operands.
fn readers(ir: &Ir, tile: ValueId) -> Option<Vec<crate::ir::Use>> {
    let mut out = Vec::new();
    let mut views = vec![tile];
    while let Some(v) = views.pop() {
        for u in ir.uses(v) {
            match ir.kind(u.op) {
                OpKind::Slice(_) | OpKind::Flat if u.index == 0 => views.push(ir.result(u.op)),
                OpKind::Yield | OpKind::For(_) => return None,
                _ => {}
            }
            out.push(*u);
        }
    }
    Some(out)
}

/// Whether the last op emitted into `block` is a CTA barrier.
fn ends_in_barrier(block: &Block<'_>) -> bool {
    let mut last = None;
    let mut next = block.first_operation();
    while let Some(op) = next {
        next = op.next_in_block();
        last = Some(op);
    }
    last.is_some_and(|op| op.name().as_string_ref().as_str() == Ok("gpu.barrier"))
}

/// The op of `body` that holds `op`, itself or an ancestor.
fn top(ir: &Ir, body: BlockId, mut op: OpId) -> OpId {
    while ir.parent_block(op) != body {
        let parent = ir.ancestors(ir.parent_block(op))[0].0;
        op = parent;
    }
    op
}

/// Whether `v` is an unmasked slice of global memory.
fn global_source(ir: &Ir, v: ValueId) -> bool {
    let Some(op) = ir.def_op(v) else { return false };
    matches!(ir.kind(op), OpKind::Slice(s) if s.masked.iter().all(|&m| !m))
        && matches!(ir.ty(v), Type::Tile(t) if t.space == Space::Global)
}

/// Whether `v` is computed from the loop variable and values from outside
/// the loop by pure ops of `body`, which `ops` collects.
fn computed_from(ir: &Ir, body: BlockId, iv: ValueId, v: ValueId, ops: &mut BTreeSet<OpId>) -> bool {
    if v == iv {
        return true;
    }
    let def = ir.def_op(v);
    let Some(op) = def.filter(|&op| ir.parent_block(op) == body) else {
        // A value from outside, but never another of the body's arguments.
        return !matches!(ir.def(v), Def::Arg { block, .. } if block == body);
    };
    let pure = matches!(
        ir.kind(op),
        OpKind::Const(_) | OpKind::Binary(_) | OpKind::IndexOp(_) | OpKind::IndexCast(_) | OpKind::Cast(_) | OpKind::Slice(_)
    );
    if !pure {
        return false;
    }
    ops.insert(op);
    ir.operands(op).to_vec().into_iter().all(|o| computed_from(ir, body, iv, o, ops))
}

impl<'c> Codegen<'c> {
    /// Allocates each rotating stage's buffer and issues its copy for the
    /// first turn, `lo`, ahead of the loop, in the order the loop issues
    /// them. Returns the buffers to release after the loop.
    pub(super) fn start_rotation(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        iv: ValueId,
        (lo, hi, step): (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        rots: &[Rotation],
    ) -> Result<Vec<MemVal<'c>>> {
        let mut order: Vec<&Rotation> = rots.iter().collect();
        let body = ir.parent_block(rots[0].stage);
        let pos = |op: OpId| ir.ops(body).iter().position(|&o| o == op).expect("an op of the body");
        order.sort_by_key(|r| pos(r.after));
        let mut bufs = HashMap::new();
        let mut token = None;
        for r in order {
            let buf = self.rotation_buffer(block, ir, r.stage)?;
            self.issue_rotation(block, ir, iv, lo, r, &buf)?;
            token = Some(self.async_create_group(block)?);
            bufs.insert(r.stage, buf);
        }
        let token = token.expect("at least one rotating stage");
        let mut owned = Vec::new();
        for r in rots {
            let buf = bufs.remove(&r.stage).expect("a buffer each");
            self.rotating.insert(r.stage, Rotating { buf: MemVal { owned: false, ..buf.clone() }, token, pending: r.pending });
            self.rotate_after
                .entry(r.after)
                .or_insert_with(|| Issues { iv, step, hi, stages: Vec::new() })
                .stages
                .push((r.stage, r.source.clone()));
            owned.push(buf);
        }
        Ok(owned)
    }

    /// Waits out every copy the loop issued and returns its buffers.
    pub(super) fn end_rotation(&mut self, block: &Block<'c>, rots: &[Rotation], bufs: Vec<MemVal<'c>>) -> Result<()> {
        if let Some(r) = rots.first() {
            let token = self.rotating[&r.stage].token;
            self.async_wait_pending(block, token, 0)?;
        }
        for r in rots {
            self.rotating.remove(&r.stage);
            self.rotate_after.remove(&r.after);
        }
        for buf in &bufs {
            self.release(buf);
        }
        Ok(())
    }

    /// The stage `op` as a rotating one: its turn's copy, issued a turn
    /// earlier, waited for, and its buffer bound. False when it does not
    /// rotate.
    pub(super) fn wait_rotation(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<bool> {
        let Some(r) = self.rotating.get(&op).cloned() else { return Ok(false) };
        self.async_wait_pending(block, r.token, r.pending)?;
        self.barrier(block)?;
        self.set_ir(ir.result(op), Lowered::Tile(r.buf));
        Ok(true)
    }

    /// After `op`, the last reader of some rotating stages: a barrier, so
    /// every thread is done reading, then each stage's copy for the next
    /// turn when there is one, and a group committed either way.
    pub(super) fn issue_after(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let Some(Issues { iv, step, hi, stages }) = self.rotate_after.get(&op).cloned() else { return Ok(()) };
        // A reader that ends in a kept barrier, as a fragment dot does,
        // already orders its reads before the copy.
        if !ends_in_barrier(block) {
            self.barrier(block)?;
        }
        let now = self.scalar_ir(iv)?;
        let next = self.addi(block, now, step)?;
        let more = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, next, hi, self.loc))?;
        for (stage, source) in stages {
            let buf = self.rotating[&stage].buf.clone();
            let r = Rotation { stage, after: op, source, pending: 0 };
            self.if_then(block, more, |cg, then| cg.issue_rotation(then, ir, iv, next, &r, &buf))?;
            self.async_create_group(block)?;
        }
        Ok(())
    }

    /// The copy of stage `r` at loop value `at` into `buf`: its source
    /// recomputed with the loop variable at `at`, copied asynchronously.
    fn issue_rotation(&mut self, block: &Block<'c>, ir: &Ir, iv: ValueId, at: Value<'c, 'c>, r: &Rotation, buf: &MemVal<'c>) -> Result<()> {
        self.lowered.push(HashMap::new());
        self.set_ir(iv, Lowered::Scalar(at));
        let copied = (|| {
            for &op in &r.source {
                self.emit_op_inner(block, ir, op)?;
            }
            let src = self.tile_ir(ir, ir.operand(r.stage, 0))?;
            self.tile_copy(block, &src, buf, false, true)
        })();
        self.lowered.pop();
        copied
    }

    /// A buffer laid out as the stage `op`'s result is.
    fn rotation_buffer(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<MemVal<'c>> {
        let Type::Tile(t) = ir.ty(ir.result(op)) else { anyhow::bail!("a stage of a non-tile") };
        let elem = self.ir_scalar_type(t.elem);
        self.alloc_tile_as(block, elem, &super::dims(&t.shape), &t.layout)
    }
}
