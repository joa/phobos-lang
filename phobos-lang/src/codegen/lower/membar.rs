use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::plan::{Event, Plan, Trace};
use crate::ir::{BlockId, Ir, Map, OpId, OpKind, ValueId};

/// One shared-memory access by one op.
#[derive(Clone, Debug)]
struct Access {
    lo: i64,
    hi: i64,
    write: bool,
    /// The elementwise sweep the access was made under, or None for one
    /// any thread might make.
    mapping: Option<(Vec<i64>, i64)>,
}

impl Access {
    fn conflicts(&self, other: &Access) -> bool {
        if !(self.write || other.write) {
            return false;
        }
        if self.hi <= other.lo || other.hi <= self.lo {
            return false;
        }
        match (&self.mapping, &other.mapping) {
            (Some(a), Some(b)) => a != b,
            _ => true,
        }
    }
}

/// A pending access: what made it, where in emission order, in which block.
#[derive(Clone, Debug)]
struct Pending {
    seq: usize,
    block: BlockId,
    access: Access,
}

/// A trailing barrier not yet needed by anything.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    op: OpId,
    seq: usize,
    block: BlockId,
}

/// What the recording emission said about one op: the ranges its scratch
/// took, its sweeps, and how many barriers it emitted, the last of them
/// trailing or not.
#[derive(Clone, Debug, Default)]
struct Window {
    /// The `Alloc` events of scratch made inside the op, then their ranges.
    scratch_events: Vec<usize>,
    scratch: Vec<(i64, i64)>,
    sweeps: Vec<(Vec<i64>, i64)>,
    barriers: usize,
    ends_in_barrier: bool,
    instances: usize,
}

/// Which ops skip their trailing barrier, and how many barriers each of
/// those emits, so the replay knows which call to skip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Elision {
    pub(crate) skip: BTreeMap<OpId, usize>,
}

struct Membar<'a> {
    ir: &'a Ir,
    windows: &'a HashMap<OpId, Window>,
    ranges: &'a HashMap<ValueId, Vec<(i64, i64)>>,
    seq: usize,
    kept: BTreeSet<OpId>,
    /// Every candidate seen, whether or not later kept.
    candidates: Vec<Candidate>,
}

impl Membar<'_> {
    /// Whether the op's kind may lose its trailing barrier at all.
    fn candidate_kind(&self, op: OpId) -> bool {
        !matches!(
            self.ir.kind(op),
            OpKind::GridBarrier
                | OpKind::Barrier
                | OpKind::Intrinsic(crate::ir::Intrinsic::WarpPartial)
                | OpKind::Intrinsic(crate::ir::Intrinsic::RmsNormQ)
        ) && !self.ir.kind(op).has_blocks()
    }

    /// Whether the op reads its same-shaped tile operands elementwise, so a
    /// recorded sweep is the mapping of every access it makes to them.
    fn elementwise(&self, op: OpId) -> bool {
        matches!(
            self.ir.kind(op),
            OpKind::Map(Map::Binary(_) | Map::Scalar { .. } | Map::Unary(_) | Map::Max)
                | OpKind::MapInto(_)
                | OpKind::Fused(_)
                | OpKind::Chain(_)
                | OpKind::Fill
                | OpKind::Copy { .. }
                | OpKind::Convert
                | OpKind::Accumulate
                | OpKind::Stage(_)
                | OpKind::Materialize
                | OpKind::ScaledAdd
        )
    }

    fn walk_block(&mut self, block: BlockId, pending: &mut Vec<Pending>, elide: bool) {
        for &op in self.ir.ops(block) {
            self.walk_op(op, pending, elide);
        }
    }

    fn walk_op(&mut self, op: OpId, pending: &mut Vec<Pending>, elide: bool) {
        let ir = self.ir;
        match ir.kind(op) {
            OpKind::For(info) => {
                let body = ir.blocks_of(op)[0];
                if info.pipeline.is_some() {
                    // Instantiated twice with prefetches interleaved: keep
                    // every barrier inside, carry every access out.
                    let entry = pending.clone();
                    self.walk_block(body, pending, false);
                    pending.extend(entry);
                    return;
                }
                let entry = pending.clone();
                self.walk_block(body, pending, elide);
                // The body again, with the first pass's trailing accesses
                // flowing into its head, for what one iteration hands the next.
                self.walk_block(body, pending, elide);
                pending.extend(entry);
                if info.ragged {
                    self.walk_block(ir.blocks_of(op)[1], pending, elide);
                }
            }
            OpKind::While => {
                let entry = pending.clone();
                for _ in 0..2 {
                    for &b in ir.blocks_of(op) {
                        self.walk_block(b, pending, false);
                    }
                }
                pending.extend(entry);
            }
            OpKind::If => {
                let mut after = Vec::new();
                for &b in ir.blocks_of(op) {
                    let mut branch = pending.clone();
                    self.walk_block(b, &mut branch, false);
                    after.extend(branch);
                }
                if ir.blocks_of(op).len() == 1 {
                    after.extend(pending.iter().cloned());
                }
                *pending = after;
            }
            _ => self.leaf(op, pending, elide),
        }
    }

    fn leaf(&mut self, op: OpId, pending: &mut Vec<Pending>, elide: bool) {
        let ir = self.ir;
        self.seq += 1;
        let seq = self.seq;
        let block = ir.parent_block(op);
        let accesses = self.accesses(op);

        // A conflict with a pending access wants a barrier between them:
        // the latest candidate after the pending access and before this op
        // that sits in the pending access's block or one enclosing it, so
        // that it lies on every path from the one to the other.
        for a in &accesses {
            let clashes: Vec<Pending> = pending
                .iter()
                .filter(|p| p.access.conflicts(a))
                .cloned()
                .collect();
            for p in clashes {
                let above = self.enclosing(p.block);
                let chosen = self
                    .candidates
                    .iter()
                    .filter(|c| c.seq >= p.seq && c.seq < seq && !self.kept.contains(&c.op))
                    .filter(|c| above.contains(&c.block))
                    .max_by_key(|c| c.seq)
                    .copied();
                if let Some(c) = chosen {
                    self.keep(c, pending);
                }
            }
        }

        pending.extend(accesses.into_iter().map(|access| Pending {
            seq,
            block,
            access,
        }));

        let w = self.windows.get(&op);
        if w.is_some_and(|w| w.ends_in_barrier && w.instances == 1) && self.candidate_kind(op) {
            let c = Candidate { op, seq, block };
            if elide {
                self.candidates.push(c);
            } else {
                self.candidates.push(c);
                self.keep(c, pending);
            }
        }
    }

    /// Keeps a barrier: it orders every pending access before it on a path
    /// through it, which is every access in its block or below it.
    fn keep(&mut self, c: Candidate, pending: &mut Vec<Pending>) {
        self.kept.insert(c.op);
        pending.retain(|p| !(p.seq <= c.seq && self.enclosing(p.block).contains(&c.block)));
    }

    /// A block and every block enclosing it.
    fn enclosing(&self, block: BlockId) -> Vec<BlockId> {
        let mut out = vec![block];
        out.extend(self.ir.ancestors(block).into_iter().map(|(_, b)| b));
        out
    }

    /// Everything the op touches in shared memory.
    fn accesses(&self, op: OpId) -> Vec<Access> {
        let ir = self.ir;
        let w = self.windows.get(&op);
        let sweep = match w {
            Some(w) if self.elementwise(op) && w.sweeps.len() == 1 => Some(w.sweeps[0].clone()),
            _ => None,
        };
        let mut out = Vec::new();
        let n = ir.operands(op).len();
        let writes = ir.kind(op).writes(n);
        let mut values: Vec<(ValueId, bool)> = ir
            .operands(op)
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, writes.contains(&i)))
            .collect();
        // A fresh result is written by the op that makes it; an `Alloc` only
        // names bytes and touches none.
        if !matches!(ir.kind(op), OpKind::Alloc) {
            values.extend(ir.results(op).iter().map(|&r| (r, true)));
        }
        for (v, write) in values {
            let Some(ranges) = self.ranges_of(v) else { continue };
            let mapping = sweep
                .as_ref()
                .filter(|(shape, _)| unmasked_shape(ir, v).as_deref() == Some(shape.as_slice()))
                .cloned();
            for &(lo, hi) in ranges {
                out.push(Access {
                    lo,
                    hi,
                    write,
                    mapping: mapping.clone(),
                });
            }
        }
        if let Some(w) = w {
            for &(lo, hi) in &w.scratch {
                out.push(Access {
                    lo,
                    hi,
                    write: true,
                    mapping: None,
                });
            }
        }
        // A dot inside a loop reads the operand an enclosing loop staged in
        // its preheader in place of the operand it names, so those buffers
        // are among what it reads.
        if matches!(
            ir.kind(op),
            OpKind::Dot { .. } | OpKind::DotInto { .. } | OpKind::FragDot
        ) {
            for (ancestor, _) in ir.ancestors(ir.parent_block(op)) {
                let OpKind::For(info) = ir.kind(ancestor) else { continue };
                let first = info.bound_operands() + info.carried;
                for &buf in &ir.operands(ancestor)[first..] {
                    for &(lo, hi) in self.ranges_of(buf).map(Vec::as_slice).unwrap_or(&[]) {
                        out.push(Access {
                            lo,
                            hi,
                            write: false,
                            mapping: None,
                        });
                    }
                }
            }
        }
        out
    }

    /// The planned byte ranges behind a tile value, through its views.
    fn ranges_of(&self, v: ValueId) -> Option<&Vec<(i64, i64)>> {
        let mut cur = v;
        loop {
            if let Some(r) = self.ranges.get(&cur) {
                return Some(r);
            }
            let op = self.ir.def_op(cur)?;
            let alias = self.ir.kind(op).aliases().into_iter().find(|a| {
                self.ir.results(op).get(a.result) == Some(&cur)
            })?;
            cur = self.ir.operand(op, alias.operand);
        }
    }
}

/// The logical shape of an unmasked tile value; None for a masked slice or
/// a non-tile, which no sweep maps element for element.
fn unmasked_shape(ir: &Ir, v: ValueId) -> Option<Vec<i64>> {
    let crate::ir::Type::Tile(t) = ir.ty(v) else {
        return None;
    };
    if let Some(op) = ir.def_op(v)
        && let OpKind::Slice(s) = ir.kind(op)
        && s.masked.iter().any(|&m| m)
    {
        return None;
    }
    t.static_shape()
}

/// One op being recorded: what its window has seen so far.
struct Frame {
    op: OpId,
    last_was_barrier: bool,
    barriers: usize,
    sweeps: Vec<(Vec<i64>, i64)>,
    allocs: Vec<usize>,
}

/// Per op, over every instance the trace recorded.
fn windows(trace: &Trace) -> HashMap<OpId, Window> {
    let mut out: HashMap<OpId, Window> = HashMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    for (i, e) in trace.events.iter().enumerate() {
        match e {
            Event::OpBegin(op) => stack.push(Frame {
                op: *op,
                last_was_barrier: false,
                barriers: 0,
                sweeps: Vec::new(),
                allocs: Vec::new(),
            }),
            Event::OpEnd(op) => {
                let Some(frame) = stack.pop() else {
                    continue;
                };
                debug_assert_eq!(frame.op, *op);
                let w = out.entry(*op).or_default();
                w.instances += 1;
                w.ends_in_barrier =
                    frame.last_was_barrier && (w.instances == 1 || w.ends_in_barrier);
                w.barriers = frame.barriers;
                w.sweeps = frame.sweeps;
                w.scratch_events.extend(frame.allocs);
            }
            Event::Barrier => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = true;
                    top.barriers += 1;
                }
            }
            Event::Sweep { shape, width } => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = false;
                    top.sweeps.push((shape.clone(), *width));
                }
            }
            Event::Alloc { .. } => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = false;
                    if !trace.is_assigned(i) {
                        top.allocs.push(i);
                    }
                }
            }
            Event::Release { .. } => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = false;
                }
            }
        }
    }
    out
}

/// The planned byte ranges of every buffer that is a graph value.
fn buffer_ranges(trace: &Trace, plan: &Plan) -> HashMap<ValueId, Vec<(i64, i64)>> {
    let mut out: HashMap<ValueId, Vec<(i64, i64)>> = HashMap::new();
    for (value, allocs) in trace.assignments() {
        for &at in allocs {
            if let Some(range) = plan.range_of(trace, at) {
                out.entry(*value).or_default().push(range);
            }
        }
    }
    out
}

impl Elision {
    /// Decides for one kernel, from its graph, its recorded emission and
    /// the placement the plan gave every buffer.
    ///
    /// Every tile op closes with a CTA barrier, whether or not anything
    /// after it reads what it wrote. The walk runs in emission order
    /// carrying the shared-memory accesses made since the last barrier
    /// kept, and when an op's accesses conflict with a pending one it keeps
    /// the latest barrier site standing between them on every path. A
    /// trailing barrier no conflict ever needed is elided.
    ///
    /// An access is a byte range of the planned buffer, read or written,
    /// and the mapping it was made under. An elementwise sweep touches
    /// element `t, t + blockDim, ...` from thread `t`, so two sweeps over
    /// the same shape at the same vector width read and write the same
    /// elements from the same thread and need no barrier between them.
    /// Everything else, a reduction, a dot, a transpose, the quantized
    /// paths and the emitters' own scratch, is an access any thread might
    /// make.
    ///
    /// A barrier inside an `if`, a `while`, a pipelined loop or the
    /// register matmul's k-loop is never elided: those sites are not on
    /// every path, or the body is instantiated more than once and the
    /// emission order is not the graph's. Their accesses still flow into
    /// what follows.
    pub(crate) fn decide(ir: &Ir, trace: &Trace, plan: &Plan) -> Elision {
        let mut windows = windows(trace);
        for w in windows.values_mut() {
            w.scratch = w
                .scratch_events
                .iter()
                .filter_map(|&at| plan.range_of(trace, at))
                .collect();
        }
        let ranges = buffer_ranges(trace, plan);
        let mut m = Membar {
            ir,
            windows: &windows,
            ranges: &ranges,
            seq: 0,
            kept: BTreeSet::new(),
            candidates: Vec::new(),
        };
        let mut pending = Vec::new();
        m.walk_block(ir.entry(), &mut pending, true);
        let mut skip = BTreeMap::new();
        for c in &m.candidates {
            if !m.kept.contains(&c.op)
                && let Some(w) = windows.get(&c.op)
            {
                skip.insert(c.op, w.barriers);
            }
        }
        Elision { skip }
    }
}
