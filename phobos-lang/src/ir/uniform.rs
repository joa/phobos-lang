// Which values may differ between the threads of a CTA, and the rules that
// keep per-thread code safe.
//
// Everything a kernel computes is uniform until `ctx.thread`, `ctx.warp` or
// `ctx.lane` flows into it. A value is as varying as the most varying of its
// operands; a `var` is as varying as anything stored to it, and as the
// control it was stored under; a loop's variable is as varying as its
// bounds. Loops can raise a `var` after it was read, so the analysis runs to
// a fixed point.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use super::{BlockId, Def, Ir, OpId, OpKind, Space, ValueId};
use crate::ast::lanes::Group;

/// How far a value may differ between threads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// The same in every thread of the CTA.
    #[default]
    Uniform,
    /// The same in every lane of a warp.
    Warp,
    /// May differ per lane.
    Varying,
}

impl Kind {
    fn describe(self) -> &'static str {
        match self {
            Kind::Uniform => "uniform",
            Kind::Warp => "differs between warps",
            Kind::Varying => "differs between lanes",
        }
    }
}

/// The kind of every value, and the control every block runs under.
#[derive(Debug, Default)]
pub struct Uniformity {
    values: HashMap<ValueId, Kind>,
    control: HashMap<BlockId, Kind>,
}

impl Uniformity {
    pub fn of(&self, value: ValueId) -> Kind {
        self.values.get(&value).copied().unwrap_or_default()
    }

    /// The kind of the conditions and bounds a block runs under.
    pub fn control(&self, block: BlockId) -> Kind {
        self.control.get(&block).copied().unwrap_or_default()
    }

    fn raise(&mut self, value: ValueId, kind: Kind) -> bool {
        let slot = self.values.entry(value).or_default();
        let raised = kind > *slot;
        *slot = (*slot).max(kind);
        raised
    }
}

pub fn analyze(ir: &Ir) -> Uniformity {
    let mut u = Uniformity::default();
    loop {
        let walk = Walk { ir, u: &mut u, changed: false };
        if !walk.run() {
            return u;
        }
    }
}

struct Walk<'a> {
    ir: &'a Ir,
    u: &'a mut Uniformity,
    changed: bool,
}

impl Walk<'_> {
    /// One pass over the kernel; whether anything was raised.
    fn run(mut self) -> bool {
        self.block(self.ir.entry(), Kind::Uniform);
        self.changed
    }

    fn set(&mut self, value: ValueId, kind: Kind) {
        self.changed |= self.u.raise(value, kind);
    }

    fn max_of(&self, values: &[ValueId]) -> Kind {
        values.iter().map(|&v| self.u.of(v)).max().unwrap_or_default()
    }

    fn block(&mut self, block: BlockId, control: Kind) {
        let entry = self.u.control.entry(block).or_default();
        if control > *entry {
            *entry = control;
            self.changed = true;
        }
        let control = self.u.control(block);
        for &op in self.ir.ops(block) {
            self.op(op, control);
        }
    }

    /// The operands of the terminator ending `block`.
    fn yielded(&self, block: BlockId) -> Vec<ValueId> {
        let ops = self.ir.ops(block);
        ops.last()
            .filter(|&&t| self.ir.kind(t).is_terminator())
            .map(|&t| self.ir.operands(t).to_vec())
            .unwrap_or_default()
    }

    fn op(&mut self, op: OpId, control: Kind) {
        let ir = self.ir;
        let operands = ir.operands(op);
        match ir.kind(op) {
            OpKind::Thread(which) => self.set(ir.result(op), which.kind()),
            // A full-warp butterfly leaves every lane with the same value.
            OpKind::Converged(Group::Lanes) => self.set(ir.result(op), self.u.of(operands[0]).min(Kind::Warp)),
            OpKind::Converged(Group::Cta) | OpKind::Exchange => {
                for &res in ir.results(op) {
                    self.set(res, Kind::Uniform);
                }
            }
            OpKind::Load if is_private(ir, operands[0]) => {
                self.set(ir.result(op), self.u.of(operands[0]));
            }
            OpKind::Store if is_private(ir, operands[1]) => {
                let kind = self.u.of(operands[0]).max(control);
                self.set(operands[1], kind);
            }
            OpKind::If => {
                let cond = self.u.of(operands[0]);
                let inner = control.max(cond);
                let mut results = vec![cond; ir.results(op).len()];
                for &b in ir.blocks_of(op) {
                    self.block(b, inner);
                    for (r, v) in results.iter_mut().zip(self.yielded(b)) {
                        *r = (*r).max(self.u.of(v));
                    }
                }
                for (&res, kind) in ir.results(op).iter().zip(results) {
                    self.set(res, kind);
                }
            }
            OpKind::For(info) => {
                let (bounds, rest) = operands.split_at(info.bound_operands());
                let bound = self.max_of(bounds);
                let inits = &rest[..info.carried];
                // Each block takes the induction variable, then the carried
                // values; a ragged loop's remainder block replays the body.
                let mut carried = vec![Kind::Uniform; info.carried];
                for &body in ir.blocks_of(op) {
                    let args = ir.args(body).to_vec();
                    let (iv, args) = args.split_first().map_or((None, &[][..]), |(iv, a)| (Some(*iv), a));
                    if let Some(iv) = iv {
                        self.set(iv, bound);
                    }
                    for (&arg, &init) in args.iter().zip(inits) {
                        self.set(arg, self.u.of(init));
                    }
                    self.block(body, control.max(bound));
                    for ((&arg, v), c) in args.iter().zip(self.yielded(body)).zip(&mut carried) {
                        self.set(arg, self.u.of(v));
                        *c = (*c).max(self.u.of(arg)).max(self.u.of(v));
                    }
                }
                for (&res, kind) in ir.results(op).iter().zip(carried) {
                    self.set(res, kind.max(bound));
                }
            }
            OpKind::While => {
                let [before, after] = ir.blocks_of(op)[..] else {
                    unreachable!("a while has a before and an after block");
                };
                // The condition decides how long each lane loops, so both
                // blocks run under it.
                let cond = self.yielded(before).first().map_or(Kind::Uniform, |&c| self.u.of(c));
                let inner = control.max(cond);
                for (&arg, &init) in ir.args(before).to_vec().iter().zip(operands) {
                    self.set(arg, self.u.of(init));
                }
                self.block(before, inner);
                let handed = self.yielded(before);
                for (&arg, &v) in ir.args(after).to_vec().iter().zip(handed.iter().skip(1)) {
                    self.set(arg, self.u.of(v));
                }
                self.block(after, inner);
                for (&arg, v) in ir.args(before).to_vec().iter().zip(self.yielded(after)) {
                    self.set(arg, self.u.of(v));
                }
                for (&res, &v) in ir.results(op).iter().zip(handed.iter().skip(1)) {
                    self.set(res, self.u.of(v).max(cond));
                }
            }
            kind => {
                let k = self.max_of(operands);
                for &res in ir.results(op) {
                    self.set(res, k);
                }
                for &b in ir.blocks_of(op) {
                    debug_assert!(!kind.has_blocks(), "every op with blocks is handled above");
                    self.block(b, control);
                }
            }
        }
    }
}

fn is_private(ir: &Ir, value: ValueId) -> bool {
    ir.ty(value).as_tile().is_some_and(|t| t.space == Space::Private)
}

fn is_shared(ir: &Ir, value: ValueId) -> bool {
    ir.ty(value).as_tile().is_some_and(|t| t.space == Space::Shared)
}

/// Whether an op runs in each thread on its own, so it may run under
/// control that differs between threads.
pub(crate) fn per_thread(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Const(_)
            | OpKind::Unary(_)
            | OpKind::Binary(_)
            | OpKind::IndexOp(_)
            | OpKind::Cast(_)
            | OpKind::IndexCast(_)
            | OpKind::ProgramId(_)
            | OpKind::GridDim(_)
            | OpKind::Thread(_)
            | OpKind::Converged(_)
            | OpKind::Math(_)
            | OpKind::Select
            | OpKind::Dim(_)
            | OpKind::Load
            | OpKind::LaneLoad { .. }
            | OpKind::LaneStore { .. }
            | OpKind::Dot4 { .. }
            | OpKind::F16Bits
            | OpKind::Word
            | OpKind::Dp4a
            | OpKind::BytePerm
            | OpKind::PopCount
            | OpKind::Store
            | OpKind::Trap
            | OpKind::For(_)
            | OpKind::While
            | OpKind::If
            | OpKind::Yield
            | OpKind::Condition
    )
}

/// The buffer a view reads: a tensor parameter or a shared tile's
/// allocation, if any.
fn root(ir: &Ir, mut value: ValueId) -> Option<ValueId> {
    loop {
        match ir.def(value) {
            Def::Arg { block, .. } => return (block == ir.entry()).then_some(value),
            Def::Result { op, .. } => match ir.kind(op) {
                OpKind::AssumeAlign | OpKind::Slice(_) | OpKind::Flat => value = ir.operand(op, 0),
                OpKind::Alloc if is_shared(ir, value) => return Some(value),
                _ => return None,
            },
        }
    }
}

fn describe(ir: &Ir, value: ValueId) -> String {
    ir.name(value).map_or_else(|| "a value".into(), |n| format!("`{n}`"))
}

/// The rules that keep code safe once values differ between threads.
pub fn check(ir: &Ir, u: &Uniformity) -> Result<()> {
    for op in ir.all_ops() {
        let control = u.control(ir.parent_block(op));
        let kind = ir.kind(op);
        let operands = ir.operands(op);
        match kind {
            // A `var`'s slot is each thread's own.
            OpKind::Alloc if is_private(ir, ir.result(op)) => continue,
            OpKind::Shuffle { .. } if control > Kind::Warp => bail!(
                "a shuffle needs every lane of the warp, but the control it sits under {}",
                control.describe()
            ),
            OpKind::Shuffle { .. } => continue,
            OpKind::Exchange if control != Kind::Uniform => bail!(
                "a collective over `ctx.thread` needs every thread of the CTA, but the control \
                 it sits under {}",
                control.describe()
            ),
            OpKind::Exchange => continue,
            _ => {}
        }
        if per_thread(kind) {
            continue;
        }
        if control != Kind::Uniform {
            bail!(
                "`{}` runs once per CTA, so every thread must reach it, but the control \
                 it sits under {}",
                kind.name(),
                control.describe()
            );
        }
        if let Some(&v) = operands.iter().find(|&&v| u.of(v) != Kind::Uniform) {
            bail!(
                "`{}` takes {}, which {}; a CTA-wide op needs the same value in every thread",
                kind.name(),
                describe(ir, v),
                u.of(v).describe()
            );
        }
    }
    let mut dirty = HashSet::new();
    Order { ir, u }.block(ir.entry(), &mut dirty)
}

/// The buffer a per-thread store writes: a varying value, index or control
/// stored into a tensor or a shared tile.
fn per_thread_store(ir: &Ir, u: &Uniformity, op: OpId) -> Option<ValueId> {
    let operands = ir.operands(op);
    // A thread's run is its own; any write of one is per thread.
    if matches!(ir.kind(op), OpKind::LaneStore { .. }) {
        return root(ir, operands[0]);
    }
    if !matches!(ir.kind(op), OpKind::Store) || is_private(ir, operands[1]) {
        return None;
    }
    let control = u.control(ir.parent_block(op));
    let indices = operands[2..].iter().map(|&i| u.of(i)).max().unwrap_or_default();
    let kind = control.max(u.of(operands[0])).max(indices);
    (kind != Kind::Uniform).then(|| root(ir, operands[1])).flatten()
}

/// Buffers stored per thread since the last `barrier()`, walked in program
/// order. A CTA-wide access to one before a barrier may miss another
/// thread's store.
struct Order<'a> {
    ir: &'a Ir,
    u: &'a Uniformity,
}

impl Order<'_> {
    fn block(&self, block: BlockId, dirty: &mut HashSet<ValueId>) -> Result<()> {
        for &op in self.ir.ops(block) {
            self.op(op, dirty)?;
        }
        Ok(())
    }

    fn op(&self, op: OpId, dirty: &mut HashSet<ValueId>) -> Result<()> {
        let ir = self.ir;
        match ir.kind(op) {
            OpKind::Barrier | OpKind::GridBarrier => {
                dirty.clear();
                return Ok(());
            }
            // Each branch from the state before it, then their union; with
            // no `else`, the path that takes neither.
            OpKind::If => {
                let blocks = ir.blocks_of(op);
                let mut out: HashSet<ValueId> = if blocks.len() == 1 { dirty.clone() } else { HashSet::new() };
                for &b in blocks {
                    let mut d = dirty.clone();
                    self.block(b, &mut d)?;
                    out.extend(d);
                }
                *dirty = out;
                return Ok(());
            }
            // Twice, so one iteration's stores meet the next's accesses.
            OpKind::For(_) | OpKind::While => {
                for _ in 0..2 {
                    for &b in ir.blocks_of(op) {
                        self.block(b, dirty)?;
                    }
                }
                return Ok(());
            }
            OpKind::AssumeAlign | OpKind::Slice(_) | OpKind::Flat => return Ok(()),
            _ => {}
        }
        let stored = per_thread_store(ir, self.u, op);
        // A thread's own point or run may be the one it stored, which only
        // its indices could tell; those accesses are the source's to fence.
        if matches!(
            ir.kind(op),
            OpKind::Load | OpKind::Store | OpKind::LaneLoad { .. } | OpKind::LaneStore { .. }
        ) {
            dirty.extend(stored);
            return Ok(());
        }
        let clash = ir.operands(op).iter().filter_map(|&v| root(ir, v)).find(|r| dirty.contains(r) && Some(*r) != stored);
        if let Some(buf) = clash {
            bail!(
                "`{}` reads {} after a per-thread store to it with no `barrier()` between: \
                 another thread's store may not have landed",
                ir.kind(op).name(),
                describe(ir, buf)
            );
        }
        dirty.extend(stored);
        Ok(())
    }
}
