use std::collections::HashMap;

use crate::ir::{Ir, OpId, OpKind, ValueId};

/// One moment of the recorded emission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    OpBegin(OpId),
    OpEnd(OpId),
    /// A buffer handed out: its running index, the pool's name for it, and
    /// its size, 16-byte aligned.
    Alloc { name: String, bytes: i64 },
    /// The pool's name returned; the allocation it ends is the latest with
    /// that name, since the pool reuses names last-in first-out.
    Release { name: String },
    /// A CTA barrier.
    Barrier,
    /// An elementwise sweep of a tile of `shape`, `width` elements to the
    /// thread: what `distribute` runs, and the mapping its accesses share.
    Sweep { shape: Vec<i64>, width: i64 },
}

/// What the first emission recorded.
#[derive(Debug, Default)]
pub(crate) struct Trace {
    pub(crate) events: Vec<Event>,
    /// Which allocations each buffer-valued graph value ended up in, by the
    /// index of their `Alloc` events: one per instantiation of its body.
    assigned: HashMap<ValueId, Vec<usize>>,
    /// The latest `Alloc` event index per pool name.
    latest: HashMap<String, usize>,
    /// Each `Alloc` event's index among the allocations, in order.
    alloc_order: HashMap<usize, usize>,
    allocs: usize,
}

impl Trace {
    pub(crate) fn alloc(&mut self, name: &str, bytes: i64) {
        self.events.push(Event::Alloc {
            name: name.to_string(),
            bytes,
        });
        let at = self.events.len() - 1;
        self.latest.insert(name.to_string(), at);
        self.alloc_order.insert(at, self.allocs);
        self.allocs += 1;
    }

    pub(crate) fn barrier(&mut self) {
        self.events.push(Event::Barrier);
    }

    pub(crate) fn sweep(&mut self, shape: &[i64], width: i64) {
        self.events.push(Event::Sweep {
            shape: shape.to_vec(),
            width,
        });
    }

    /// Whether the allocation at event `at` became a graph value.
    pub(crate) fn is_assigned(&self, at: usize) -> bool {
        self.assigned.values().any(|allocs| allocs.contains(&at))
    }

    pub(crate) fn assignments(&self) -> &HashMap<ValueId, Vec<usize>> {
        &self.assigned
    }

    pub(crate) fn release(&mut self, name: &str) {
        self.events.push(Event::Release {
            name: name.to_string(),
        });
    }

    /// Records that `value` is the buffer the pool most recently named
    /// `name`: what the emitter returned for the op that defines it.
    pub(crate) fn assign(&mut self, value: ValueId, name: &str) {
        if let Some(&at) = self.latest.get(name) {
            let allocs = self.assigned.entry(value).or_default();
            if !allocs.contains(&at) {
                allocs.push(at);
            }
        }
    }
}

/// Where each allocation of the emission goes, in the order the emission
/// makes them.
///
/// The emitters allocate scratch the graph never sees: dot staging, the
/// quantized paths' tiles, the register matmul's staging pairs and slab.
/// A first emission into a throwaway module records every allocation and
/// release tagged with the op being emitted, and the graph adds what the
/// trace cannot say, that a buffer which is a value lives until the last
/// use of anything derived from it and through every loop that reads it
/// without defining it. The second emission hands out the offsets this
/// computes, in the order the first recorded them.
///
/// Reuse rests on every tile op ending in a CTA barrier, so a buffer whose
/// last reader is a tile op can be overwritten by the next. A per-thread
/// load or store on a tile does not barrier, and this planner does not
/// account for it; [`super::membar`] is where that is handled.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) offsets: Vec<i64>,
    pub(crate) peak: i64,
}

/// One buffer's life on the event timeline and its size.
#[derive(Clone, Debug)]
struct Interval {
    alloc: usize,
    start: usize,
    end: usize,
    bytes: i64,
}

impl Plan {
    /// The bytes the allocation at event `at` was given.
    pub(crate) fn range_of(&self, trace: &Trace, at: usize) -> Option<(i64, i64)> {
        let k = *trace.alloc_order.get(&at)?;
        let Event::Alloc { bytes, .. } = trace.events.get(at)? else {
            return None;
        };
        let offset = *self.offsets.get(k)?;
        Some((offset, offset + bytes))
    }

    /// Places every allocation the trace recorded.
    ///
    /// Each one becomes an interval, alive from its event until its release or
    /// the end of the trace. A body a loop instantiates more than once gets an
    /// interval per instance, not one spanning the loop.
    pub(crate) fn compute(ir: &Ir, trace: &Trace) -> Plan {
        let events = &trace.events;
        let end_of = events.len();

        // Every moment each op finished, ascending. A body a loop instantiates
        // more than once emits its ops more than once, and each instance's
        // buffers live within that instance.
        let mut op_ends: HashMap<OpId, Vec<usize>> = HashMap::new();
        for (i, e) in events.iter().enumerate() {
            if let Event::OpEnd(op) = e {
                op_ends.entry(*op).or_default().push(i);
            }
        }
        let first_end_after = |op: OpId, at: usize| -> Option<usize> {
            let ends = op_ends.get(&op)?;
            let k = ends.partition_point(|&i| i <= at);
            ends.get(k).copied()
        };

        // Every allocation, first as scratch: alive until its release, or
        // to the end when never released.
        let mut intervals: Vec<Interval> = Vec::new();
        let mut open: HashMap<String, usize> = HashMap::new();
        for (i, e) in events.iter().enumerate() {
            match e {
                Event::Alloc { name, bytes } => {
                    open.insert(name.clone(), intervals.len());
                    intervals.push(Interval {
                        alloc: i,
                        start: i,
                        end: end_of,
                        bytes: *bytes,
                    });
                }
                Event::Release { name } => {
                    if let Some(k) = open.remove(name) {
                        intervals[k].end = i;
                    }
                }
                _ => {}
            }
        }

        // A buffer that is a value lives by the graph instead: to the last
        // op that reads it or a view of it, through every loop that reads
        // it without defining it.
        let by_alloc: HashMap<usize, usize> = intervals
            .iter()
            .enumerate()
            .map(|(k, iv)| (iv.alloc, k))
            .collect();
        let mut value_end: HashMap<usize, usize> = HashMap::new();
        for (&value, allocs) in &trace.assigned {
            for &alloc in allocs {
                let Some(&k) = by_alloc.get(&alloc) else { continue };
                let def = ir.def_op(value);
                let mut end = intervals[k].start;
                for user in users_through_views(ir, value) {
                    let Some(at) = first_end_after(user, alloc) else { continue };
                    end = end.max(at);
                    for (ancestor, _) in ir.ancestors(ir.parent_block(user)) {
                        let loops = matches!(ir.kind(ancestor), OpKind::For(_) | OpKind::While);
                        let contains_def =
                            def.is_some_and(|d| d == ancestor || ir.contains(ancestor, d));
                        if loops
                            && !contains_def
                            && let Some(close) = first_end_after(ancestor, at)
                        {
                            end = end.max(close);
                        }
                    }
                }
                let slot = value_end.entry(k).or_insert(end);
                *slot = (*slot).max(end);
            }
        }
        for (k, end) in value_end {
            intervals[k].end = end;
        }

        // First fit by size, largest first; ties by start so the order is
        // total and the plan deterministic.
        let mut order: Vec<usize> = (0..intervals.len()).collect();
        order.sort_by_key(|&k| (std::cmp::Reverse(intervals[k].bytes), intervals[k].start, k));
        let mut placed: Vec<(usize, i64, i64)> = Vec::new(); // (interval, offset, end offset)
        let mut offsets = vec![0i64; intervals.len()];
        let mut peak = 0i64;
        for k in order {
            let iv = &intervals[k];
            let mut offset = 0i64;
            loop {
                let clash = placed.iter().find(|&&(j, lo, hi)| {
                    let other = &intervals[j];
                    let overlap_time = iv.start < other.end && other.start < iv.end;
                    let overlap_bytes = offset < hi && lo < offset + iv.bytes;
                    overlap_time && overlap_bytes
                });
                match clash {
                    Some(&(_, _, hi)) => offset = hi,
                    None => break,
                }
            }
            offsets[k] = offset;
            peak = peak.max(offset + iv.bytes);
            placed.push((k, offset, offset + iv.bytes));
        }
        Plan { offsets, peak }
    }
}

/// The ops reading `value` or any view of it.
fn users_through_views(ir: &Ir, value: ValueId) -> Vec<OpId> {
    let mut out = Vec::new();
    let mut stack = vec![value];
    let mut seen = vec![value];
    while let Some(v) = stack.pop() {
        for u in ir.uses(v) {
            out.push(u.op);
            for alias in ir.kind(u.op).aliases() {
                if alias.operand == u.index {
                    let r = ir.results(u.op)[alias.result];
                    if !seen.contains(&r) {
                        seen.push(r);
                        stack.push(r);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Builder, Bounds, ForInfo, KernelInfo, Literal, Scalar, Type};

    fn kernel() -> Ir {
        Ir::new(
            KernelInfo {
                name: "k".into(),
                cta_threads: 256,
                ..Default::default()
            },
            &[],
        )
    }

    fn tile() -> Type {
        Type::shared_tile(Scalar::F32, &[4, 4])
    }

    /// Two buffers whose lives do not overlap share an offset.
    #[test]
    fn disjoint_lives_share_bytes() {
        let mut ir = kernel();
        let entry = ir.entry();
        let mut b = Builder::at_end(&mut ir, entry);
        let a = b.value(OpKind::Alloc, &[], tile());
        let r = b.value(OpKind::Reduce(crate::ir::Reduce::Transpose), &[a], tile());
        let c = b.value(OpKind::Alloc, &[], tile());
        b.stmt(OpKind::Copy { sync: true }, &[r, c]);
        let ops = ir.ops(entry).to_vec();

        let mut t = Trace::default();
        t.events.push(Event::OpBegin(ops[0]));
        t.alloc("t0", 64);
        t.events.push(Event::OpEnd(ops[0]));
        t.assign(a, "t0");
        t.events.push(Event::OpBegin(ops[1]));
        t.alloc("t1", 64);
        t.events.push(Event::OpEnd(ops[1]));
        t.assign(r, "t1");
        t.events.push(Event::OpBegin(ops[2]));
        t.alloc("t2", 64);
        t.events.push(Event::OpEnd(ops[2]));
        t.assign(c, "t2");
        t.events.push(Event::OpBegin(ops[3]));
        t.events.push(Event::OpEnd(ops[3]));

        let plan = Plan::compute(&ir, &t);
        // a dies at the transpose; c takes its bytes; r lives to the copy.
        assert_eq!(plan.offsets[0], plan.offsets[2]);
        assert_ne!(plan.offsets[0], plan.offsets[1]);
        assert_eq!(plan.peak, 128);
    }

    /// A buffer read inside a loop and made outside it lives through the
    /// loop, so a scratch made inside cannot take its bytes.
    #[test]
    fn a_use_inside_a_loop_extends_to_the_loop_end() {
        let mut ir = kernel();
        let entry = ir.entry();
        let mut b = Builder::at_end(&mut ir, entry);
        let a = b.value(OpKind::Alloc, &[], tile());
        let lo = b.value(OpKind::Const(Literal::Int(0)), &[], Type::INDEX);
        let body = b.block_with(&[Type::INDEX]);
        let inner = b.in_block(body, |i| {
            let r = i.value(OpKind::Reduce(crate::ir::Reduce::Transpose), &[a], tile());
            i.stmt(OpKind::Yield, &[]);
            r
        });
        let for_op = b.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: 0,
                hoisted: 0,
                pipeline: None,
            }),
            &[lo, lo, lo],
            Vec::new(),
            vec![body],
        );
        let alloc_op = ir.def_op(a).unwrap();
        let inner_op = ir.def_op(inner).unwrap();

        let mut t = Trace::default();
        t.events.push(Event::OpBegin(alloc_op));
        t.alloc("t0", 64);
        t.events.push(Event::OpEnd(alloc_op));
        t.assign(a, "t0");
        t.events.push(Event::OpBegin(for_op));
        t.events.push(Event::OpBegin(inner_op));
        t.alloc("t1", 64);
        t.events.push(Event::OpEnd(inner_op));
        t.assign(inner, "t1");
        // Scratch after the last read of a, still inside the loop.
        t.alloc("s", 64);
        t.release("s");
        t.events.push(Event::OpEnd(for_op));

        let plan = Plan::compute(&ir, &t);
        assert_ne!(plan.offsets[0], plan.offsets[2], "a is live through the loop");
        // The transpose's result dies inside the loop, so the scratch may
        // take its bytes.
        assert_eq!(plan.offsets[1], plan.offsets[2]);
    }
}
