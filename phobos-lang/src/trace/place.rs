// What the recording emission and the plan say about one kernel: its shared
// buffers with their offsets and lives, its barriers, and how each op spread
// its accesses over the CTA.

use std::collections::HashMap;

use super::model::{self, ThreadMap};
use crate::codegen::Planned;
use crate::codegen::lower::plan::Event;
use crate::ir::{Ir, OpId, OpKind, Type, ValueId};

/// The events an op made itself, nested ops' excluded, from its first
/// instance in the recording.
#[derive(Default)]
struct Window {
    events: Vec<usize>,
}

pub(super) struct Placement {
    pub(super) buffers: Vec<model::Buffer>,
    pub(super) barriers: Vec<model::Barrier>,
    pub(super) peak_bytes: i64,
    /// Each buffer-valued graph value's buffer ids, one per instance.
    pub(super) of_value: HashMap<ValueId, Vec<usize>>,
    maps: HashMap<OpId, ThreadMap>,
    /// Whether an op's last event is a barrier, and whether it was kept.
    trailing: HashMap<OpId, bool>,
    /// Barriers each op emitted in its first instance.
    counts: HashMap<OpId, usize>,
    /// The instruction family each contraction lowered to.
    instrs: HashMap<OpId, &'static str>,
}

impl Placement {
    pub(super) fn new(p: &Planned<'_>, cta_threads: i64) -> Placement {
        let events = &p.trace.events;
        let windows = windows(events);
        let op_at = op_at(p.ir, events);

        let mut buffers = Vec::new();
        let mut alloc_ids = HashMap::new();
        for (at, e) in events.iter().enumerate() {
            let Event::Alloc { name, bytes } = e else {
                continue;
            };
            let k = buffers.len();
            alloc_ids.insert(at, k);
            let (start, end) = p.plan.lives.get(k).copied().unwrap_or((at, events.len()));
            buffers.push(model::Buffer {
                id: k,
                name: name.rsplit('_').next().unwrap_or(name).to_string(),
                bytes: *bytes,
                offset_bytes: p.plan.offsets.get(k).copied(),
                first_op: op_at[start.min(op_at.len() - 1)],
                last_op: op_at[end.min(op_at.len() - 1)],
                scratch: !p.trace.is_assigned(at),
                elem: None,
                shape: None,
            });
        }

        let mut of_value: HashMap<ValueId, Vec<usize>> = HashMap::new();
        for (&value, allocs) in p.trace.assignments() {
            let mut ids: Vec<usize> = allocs
                .iter()
                .filter_map(|at| alloc_ids.get(at).copied())
                .collect();
            ids.sort_unstable();
            for &k in &ids {
                let b = &mut buffers[k];
                if let Some(name) = p.ir.name(value) {
                    b.name = name.to_string();
                }
                if let Type::Tile(t) = p.ir.ty(value) {
                    b.elem = Some(t.elem.mlir_name().to_string());
                    b.shape = t.static_shape();
                }
            }
            of_value.insert(value, ids);
        }

        let mut maps = HashMap::new();
        let mut trailing = HashMap::new();
        let mut counts = HashMap::new();
        let mut barriers = Vec::new();
        let mut instrs = HashMap::new();
        for op in p.ir.all_ops() {
            let Some(w) = windows.get(&op) else { continue };
            if let Some(instr) = w.events.iter().find_map(|&i| match events[i] {
                Event::Contract { instr } => Some(instr),
                _ => None,
            }) {
                instrs.insert(op, instr);
            }
            let own = thread_maps(events, w, cta_threads, elem_bytes(p.ir, op));
            if let Some(map) = own.first() {
                maps.insert(op, map.clone());
            }
            // A loop that lowers its body itself (the register matmul, a
            // pipelined copy) makes its copies in its own window: the k-th
            // goes to the k-th body op that moves data. When some are
            // asynchronous, those are the steady state, after a synchronous
            // prologue.
            let own = if own.iter().any(|m| m.async_copy.is_some()) {
                own.into_iter().filter(|m| m.async_copy.is_some()).collect()
            } else {
                own
            };
            if matches!(p.ir.kind(op), OpKind::For(_)) && !own.is_empty() {
                let movers =
                    p.ir.blocks_of(op)
                        .iter()
                        .flat_map(|&b| p.ir.ops(b).iter().copied())
                        .filter(|o| {
                            !windows.contains_key(o)
                                && matches!(
                                    p.ir.kind(*o),
                                    OpKind::Stage(_) | OpKind::Materialize | OpKind::GemmDot
                                )
                        });
                for (k, mover) in movers.enumerate() {
                    maps.entry(mover)
                        .or_insert_with(|| own[k % own.len()].clone());
                }
            }
            let n = w
                .events
                .iter()
                .filter(|&&i| events[i] == Event::Barrier)
                .count();
            if n == 0 {
                continue;
            }
            let kept = !p.elision.skip.contains_key(&op);
            counts.insert(op, n);
            if w.events
                .last()
                .is_some_and(|&i| events[i] == Event::Barrier)
            {
                trailing.insert(op, kept);
            }
            barriers.push(model::Barrier {
                after_op: op.index(),
                kept,
            });
        }

        // The register matmul decides once, at its seed, for every op on
        // the same accumulator.
        let gemm_of = |op: OpId| {
            p.ir.results(op)
                .iter()
                .chain(p.ir.operands(op))
                .find_map(|&v| match p.ir.ty(v) {
                    Type::Gemm(g) => Some(*g),
                    _ => None,
                })
        };
        let seeds: Vec<_> = instrs
            .iter()
            .filter_map(|(&op, &instr)| Some((gemm_of(op)?, instr)))
            .collect();
        for op in p.ir.all_ops() {
            if matches!(p.ir.kind(op), OpKind::GemmDot | OpKind::GemmStore { .. })
                && let Some(g) = gemm_of(op)
                && let Some(&(_, instr)) = seeds.iter().find(|(s, _)| *s == g)
            {
                instrs.entry(op).or_insert(instr);
            }
        }

        Placement {
            buffers,
            barriers,
            peak_bytes: p.plan.peak,
            of_value,
            maps,
            trailing,
            counts,
            instrs,
        }
    }

    pub(super) fn map(&self, op: OpId) -> Option<ThreadMap> {
        self.maps.get(&op).cloned()
    }

    /// Whether the op ends in a barrier, and if so whether it was kept.
    pub(super) fn trailing_barrier(&self, op: OpId) -> Option<bool> {
        self.trailing.get(&op).copied()
    }

    pub(super) fn instr(&self, op: OpId) -> Option<&'static str> {
        self.instrs.get(&op).copied()
    }

    pub(super) fn barrier_count(&self, op: OpId) -> usize {
        self.counts.get(&op).copied().unwrap_or(0)
    }
}

fn windows(events: &[Event]) -> HashMap<OpId, Window> {
    let mut out: HashMap<OpId, Window> = HashMap::new();
    let mut stack: Vec<(OpId, Vec<usize>)> = Vec::new();
    for (i, e) in events.iter().enumerate() {
        match e {
            Event::OpBegin(op) => stack.push((*op, Vec::new())),
            Event::OpEnd(_) => {
                if let Some((op, own)) = stack.pop() {
                    out.entry(op).or_insert(Window { events: own });
                }
            }
            _ => {
                if let Some((_, own)) = stack.last_mut() {
                    own.push(i);
                }
            }
        }
    }
    out
}

/// The op each event happened under, the innermost one; one more entry for
/// the end of the kernel.
fn op_at(ir: &Ir, events: &[Event]) -> Vec<usize> {
    let last = ir.all_ops().last().map_or(0, |op| op.index());
    let mut out = Vec::with_capacity(events.len() + 1);
    let mut stack: Vec<OpId> = Vec::new();
    let mut recent = 0;
    for e in events {
        match e {
            Event::OpBegin(op) => stack.push(*op),
            Event::OpEnd(op) => {
                stack.pop();
                recent = op.index();
            }
            _ => {}
        }
        out.push(stack.last().map_or(recent, |op| op.index()));
    }
    out.push(last);
    out
}

/// The element width of what an op writes or yields, for a sweep's access.
fn elem_bytes(ir: &Ir, op: OpId) -> i64 {
    let tile = |v: ValueId| match ir.ty(v) {
        Type::Tile(t) => t.elem.bytes(),
        _ => None,
    };
    ir.results(op)
        .iter()
        .chain(ir.operands(op).iter().rev())
        .find_map(|&v| tile(v))
        .unwrap_or(4)
}

/// The op's sweeps in order, each with the copy that announced it if any.
fn thread_maps(events: &[Event], w: &Window, threads: i64, elem_bytes: i64) -> Vec<ThreadMap> {
    let mut out = Vec::new();
    let mut copy = None;
    for &i in &w.events {
        match &events[i] {
            Event::Copy {
                vector_bytes,
                async_copy,
            } => copy = Some((*vector_bytes, *async_copy)),
            Event::Sweep { shape, width } => {
                let (vector_bytes, async_copy) = copy.take().unwrap_or((width * elem_bytes, false));
                out.push(ThreadMap {
                    shape: shape.clone(),
                    width: *width,
                    threads,
                    vector_bytes,
                    async_copy: async_copy.then_some(true),
                });
            }
            _ => {}
        }
    }
    out
}
