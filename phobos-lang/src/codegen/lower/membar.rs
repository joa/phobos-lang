use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::plan::{Event, Plan, Trace};
use crate::ir::uniform::{self, Kind, Uniformity};
use crate::ir::{BlockId, Ir, Map, OpId, OpKind, ValueId};

/// One shared-memory access by one op.
#[derive(Clone, Debug, PartialEq)]
struct Access {
    lo: i64,
    hi: i64,
    write: bool,
    /// The elementwise sweep the access was made under. None means any
    /// thread might make it.
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

/// An access not yet ordered by a kept barrier, with its emission order and
/// block.
#[derive(Clone, Debug, PartialEq)]
struct Pending {
    op: OpId,
    seq: usize,
    block: BlockId,
    access: Access,
}

/// An op's trailing barrier, which may be elided.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    op: OpId,
    seq: usize,
    block: BlockId,
}

/// What the recording emission saw of one op: its scratch ranges, its
/// sweeps, how many barriers it emitted, and whether it ends in one.
#[derive(Clone, Debug, Default)]
struct Window {
    /// The `Alloc` events of scratch made inside the op, and their ranges.
    scratch_events: Vec<usize>,
    scratch: Vec<(i64, i64)>,
    sweeps: Vec<(Vec<i64>, i64)>,
    barriers: usize,
    ends_in_barrier: bool,
    /// Ends in a barrier with nothing after it but scratch going back to
    /// the pool, in every instance.
    fenced: bool,
    instances: usize,
}

/// Which ops skip their trailing barrier, with how many barriers each emits
/// so the replay knows which call to skip, and which ops need a barrier ahead
/// of them that no op's trailing barrier provides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Elision {
    pub(crate) skip: BTreeMap<OpId, usize>,
    pub(crate) before: BTreeSet<OpId>,
}

struct Membar<'a> {
    ir: &'a Ir,
    windows: &'a HashMap<OpId, Window>,
    ranges: &'a HashMap<ValueId, Vec<(i64, i64)>>,
    seq: usize,
    kept: BTreeSet<OpId>,
    /// Every candidate seen, kept or not.
    candidates: Vec<Candidate>,
    /// Ops that end in a barrier that is never elided: an explicit
    /// `barrier()`, a CTA collective, an atomic, an op emitted more than once.
    fences: Vec<Candidate>,
    /// Barriers added ahead of an op, each placed as if it trailed the op
    /// before it.
    added: Vec<Candidate>,
    /// Where the latest walk of each op with blocks began.
    starts: HashMap<OpId, usize>,
    uniform: Uniformity,
}

impl Membar<'_> {
    /// Whether the op's kind may lose its trailing barrier at all.
    fn candidate_kind(&self, op: OpId) -> bool {
        !matches!(
            self.ir.kind(op),
            OpKind::GridBarrier
                | OpKind::Barrier
                | OpKind::Exchange
                | OpKind::AtomicAdd
                | OpKind::AtomicCas
                | OpKind::LoadAcquire
                | OpKind::StoreRelease
                | OpKind::RegsFused(_)
                | OpKind::RegsStore
                | OpKind::RegsReduce(_)
        ) && !self.ir.kind(op).has_blocks()
    }

    /// Whether the op accesses its same-shaped tile operands elementwise, so
    /// its recorded sweep describes every such access.
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
        if ir.kind(op).has_blocks() {
            self.starts.insert(op, self.seq + 1);
        }
        match ir.kind(op) {
            OpKind::For(info) => {
                let body = ir.blocks_of(op)[0];
                if info.pipeline.is_some() && self.single_turn(body) {
                    // One buffer a stage: a turn is the body once, and the
                    // stages are refilled between two barriers after it, so
                    // nothing is pending from one turn to the next. The
                    // stage ops' own accesses are those refills.
                    let entry = pending.clone();
                    self.walk_block(body, pending, elide);
                    pending.clear();
                    merge(pending, entry);
                    return;
                }
                if info.pipeline.is_some() {
                    // The body is instantiated twice with prefetches
                    // interleaved. Keep every barrier inside, and carry
                    // every access out.
                    let entry = pending.clone();
                    self.walk_block(body, pending, false);
                    merge(pending, entry);
                    return;
                }
                let entry = pending.clone();
                self.walk_block(body, pending, elide);
                // Walk the body again so one iteration's trailing accesses
                // meet the next iteration's head.
                self.walk_block(body, pending, elide);
                merge(pending, entry);
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
                merge(pending, entry);
            }
            OpKind::If => {
                let mut after = Vec::new();
                for &b in ir.blocks_of(op) {
                    let mut branch = pending.clone();
                    self.walk_block(b, &mut branch, false);
                    merge(&mut after, branch);
                }
                if ir.blocks_of(op).len() == 1 {
                    merge(&mut after, pending.clone());
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

        // A conflict with a pending access needs a barrier between them.
        // Keep the latest candidate between the two that sits in the pending
        // access's block or an enclosing one, so it lies on every path.
        for a in &accesses {
            let clashes: Vec<Pending> = pending
                .iter()
                .filter(|p| p.access.conflicts(a))
                .cloned()
                .collect();
            for p in clashes {
                let above = self.enclosing(p.block);
                let between = |c: &&Candidate| c.seq >= p.seq && c.seq < seq && above.contains(&c.block);
                // A barrier kept for an earlier clash, or one that is never
                // elided, may already order this one.
                if self.candidates.iter().filter(between).any(|c| self.kept.contains(&c.op))
                    || self.added.iter().any(|c| between(&c))
                    || self.fences.iter().any(|c| between(&c))
                {
                    continue;
                }
                let chosen = self.candidates.iter().filter(between).max_by_key(|c| c.seq).copied();
                if let Some(c) = chosen {
                    self.keep(c, pending);
                } else if self.needs_added(op, a, &p) && !self.fences.iter().any(|c| between(&c)) {
                    // Nothing between the two accesses ends in a barrier, as
                    // when a loop's last per-thread read of a tile meets the
                    // next turn's copy into it.
                    self.add_before(op, seq, &above);
                }
            }
        }

        pending.extend(accesses.into_iter().map(|access| Pending {
            op,
            seq,
            block,
            access,
        }));

        let w = self.windows.get(&op);
        let candidate = w.is_some_and(|w| w.ends_in_barrier && w.instances == 1) && self.candidate_kind(op);
        if !candidate && w.is_some_and(|w| w.fenced) && !ir.kind(op).has_blocks() {
            self.fences.push(Candidate { op, seq, block });
        }
        if candidate {
            let c = Candidate { op, seq, block };
            if elide {
                self.candidates.push(c);
            } else {
                self.candidates.push(c);
                self.keep(c, pending);
            }
        }
    }

    /// Whether a conflict between `op`'s `access` and an `earlier` one that
    /// no trailing barrier orders needs one added. A view touches no bytes
    /// itself, and two per-thread accesses are the kernel's to order with
    /// `barrier()`; what is left involves a CTA-wide op, which the compiler
    /// orders, unless both sides reach each element from the same thread.
    fn needs_added(&self, op: OpId, access: &Access, earlier: &Pending) -> bool {
        let (a, b) = (self.ir.kind(op), self.ir.kind(earlier.op));
        a.aliases().is_empty()
            && b.aliases().is_empty()
            && !(uniform::per_thread(a) && uniform::per_thread(b))
            && !self.same_thread(op, &earlier.access)
            && !self.same_thread(earlier.op, access)
    }

    /// Whether `regs` makes a register tile whose elements each thread holds
    /// as `other` reaches them. A register tile keeps element `t + e * threads`
    /// in thread `t`, which is what a one-wide sweep of its shape gives `t`.
    fn same_thread(&self, regs: OpId, other: &Access) -> bool {
        let Some(&result) = self.ir.results(regs).first() else {
            return false;
        };
        let crate::ir::Type::Regs(tile) = self.ir.ty(result) else {
            return false;
        };
        other
            .mapping
            .as_ref()
            .is_some_and(|(shape, width)| *width == 1 && *shape == tile.shape)
    }

    /// Adds a barrier ahead of `op` or, when the access it orders against
    /// lies outside the block `op` sits in, ahead of the op holding `op` that
    /// shares a block with that access, so it lies on every path between
    /// the two. A barrier needs every thread, so under varying control none
    /// is added, and ordering is the kernel's own `barrier()`.
    fn add_before(&mut self, op: OpId, seq: usize, above: &[BlockId]) {
        let ir = self.ir;
        let mut at = op;
        while !above.contains(&ir.parent_block(at)) {
            let Some(holder) = ir.parent_op(ir.parent_block(at)) else {
                return;
            };
            at = holder;
        }
        let block = ir.parent_block(at);
        if self.uniform.control(block) != Kind::Uniform {
            return;
        }
        let start = if at == op { seq } else { self.starts[&at] };
        self.added.push(Candidate {
            op: at,
            seq: start - 1,
            block,
        });
    }

    /// Keeps a barrier. It orders every earlier pending access in its block
    /// or a nested one, so those are dropped.
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

    /// Whether a pipelined loop's body was emitted once, one buffer a stage
    /// refilled from registers, rather than as two alternating halves.
    /// The prefix up to the last stage is emitted for the prologue and for
    /// every prefetch, so only the compute after it tells.
    fn single_turn(&self, body: BlockId) -> bool {
        let ops = self.ir.ops(body);
        let after = ops
            .iter()
            .rposition(|&o| matches!(self.ir.kind(o), OpKind::Stage(_)))
            .map_or(0, |i| i + 1);
        ops[after..]
            .iter()
            .all(|o| self.windows.get(o).is_none_or(|w| w.instances <= 1))
    }

    /// Everything the op touches in shared memory.
    fn accesses(&self, op: OpId) -> Vec<Access> {
        let ir = self.ir;
        // A view names bytes and touches none. Whatever reads or writes
        // through it records the access.
        if !ir.kind(op).aliases().is_empty() {
            return Vec::new();
        }
        // A single-turn pipelined loop's stage is filled outside its body's
        // ops, between the lowering's own barriers.
        let body = ir.parent_block(op);
        if matches!(ir.kind(op), OpKind::Stage(_))
            && let Some(owner) = ir.parent_op(body)
            && matches!(ir.kind(owner), OpKind::For(info) if info.pipeline.is_some())
            && self.single_turn(body)
        {
            return Vec::new();
        }
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
        // An op writes its fresh results. An `Alloc` only names bytes and
        // touches none.
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
        // A dot inside a loop may read operands an enclosing loop staged in
        // its preheader, instead of the ones it names. Count those as reads.
        if matches!(
            ir.kind(op),
            OpKind::Dot { .. } | OpKind::DotInto { .. } | OpKind::DotFused { .. } | OpKind::FragDot
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

/// Adds the accesses of `more` that `pending` does not already hold. A loop
/// re-adds what was pending at its entry, for the case it runs no turn;
/// appending blindly doubled the list at every loop.
fn merge(pending: &mut Vec<Pending>, more: Vec<Pending>) {
    for p in more {
        if !pending.contains(&p) {
            pending.push(p);
        }
    }
}

/// The logical shape of an unmasked tile value. None for a masked slice or a
/// non-tile, which no sweep maps element for element.
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

/// One op while scanning the trace, with what its window has seen so far.
struct Frame {
    op: OpId,
    last_was_barrier: bool,
    fenced: bool,
    barriers: usize,
    sweeps: Vec<(Vec<i64>, i64)>,
    allocs: Vec<usize>,
}

/// Each op's window, over every instance the trace recorded.
fn windows(trace: &Trace) -> HashMap<OpId, Window> {
    let mut out: HashMap<OpId, Window> = HashMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    for (i, e) in trace.events.iter().enumerate() {
        match e {
            Event::OpBegin(op) => stack.push(Frame {
                op: *op,
                last_was_barrier: false,
                fenced: false,
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
                w.fenced = frame.fenced && (w.instances == 1 || w.fenced);
                w.barriers = frame.barriers;
                w.sweeps = frame.sweeps;
                w.scratch_events.extend(frame.allocs);
            }
            Event::Barrier => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = true;
                    top.fenced = true;
                    top.barriers += 1;
                }
            }
            Event::Sweep { shape, width } => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = false;
                    top.fenced = false;
                    top.sweeps.push((shape.clone(), *width));
                }
            }
            Event::Alloc { .. } => {
                if let Some(top) = stack.last_mut() {
                    top.last_was_barrier = false;
                    top.fenced = false;
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
            Event::Copy { .. } | Event::Contract { .. } => {}
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
    /// Decides which trailing barriers one kernel can drop, from its graph,
    /// its recorded emission and the plan's buffer placement.
    ///
    /// Every tile op ends with a CTA barrier, needed or not. The walk goes in
    /// emission order and tracks the shared-memory accesses since the last
    /// kept barrier. When a new access conflicts with a pending one, it keeps
    /// the latest barrier between them that lies on every path. Trailing
    /// barriers that no conflict needs are elided.
    ///
    /// An access is a byte range of a planned buffer, read or written, plus
    /// the mapping it was made under. An elementwise sweep has thread `t`
    /// touch elements `t, t + blockDim, ...`. Two sweeps over the same shape
    /// at the same vector width touch the same elements from the same thread,
    /// so they need no barrier between them. Every other access, such as a
    /// reduction, a dot, a transpose or scratch, may come from any thread.
    ///
    /// Two accesses with no trailing barrier between them get one ahead of
    /// the later op, where control lets every thread reach it.
    ///
    /// Barriers inside an `if`, a `while`, a pipelined loop or the register
    /// matmul's k-loop are never elided. They are not on every path, or the
    /// body is emitted more than once in a different order. Their accesses
    /// still flow into what follows.
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
            fences: Vec::new(),
            added: Vec::new(),
            starts: HashMap::new(),
            uniform: uniform::analyze(ir),
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
        let before = m.added.iter().map(|c| c.op).collect();
        Elision { skip, before }
    }
}
