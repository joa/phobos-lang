// A small interpreter over the graph's index arithmetic: one CTA at a time,
// the lanes of its warp 0 side by side, each with its own values and an
// active mask, so per-lane loops and conditions run as the hardware runs
// them. It computes addresses, partners and placement, never values: a load
// yields nothing, and an address that needs one fails the trace.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};

use super::model::{Access, Cta, LaneEvent, LaneKind, Mode, Run, Step, StepBarrier};
use super::place::Placement;
use super::view::{Root, View};
use crate::ast::lanes::{Group, Mode as AstMode};
use crate::ir::{
    BinOp, BlockId, Bounds, Coeff, IndexOp, Ir, Literal, OpId, OpKind, Scalar, Space, ThreadIndex,
    Type, UnOp, ValueId,
};

/// Op executions one CTA may take before the trace gives up.
const MAX_OPS_PER_CTA: usize = 2_000_000;

pub(super) const DATA_DEPENDENT: &str = "data-dependent addressing, not traceable yet";

/// One value in every simulated lane, `None` where it is not an address:
/// a float, or anything loaded.
type Lanes = Vec<Option<i64>>;

/// What stays the same across CTAs.
pub(super) struct Kernel<'a> {
    pub(super) ir: &'a Ir,
    pub(super) place: &'a Placement,
    /// The runtime extent of each tensor parameter's dimensions, by entry
    /// argument.
    pub(super) tensors: HashMap<ValueId, (String, Vec<i64>)>,
    pub(super) grid: Cta,
    /// Lanes of warp 0 simulated, and lanes in a warp on the target.
    pub(super) lanes: usize,
    pub(super) warp_lanes: i64,
    /// The shuffles each lane collective folds, in program order.
    pub(super) collectives: HashMap<OpId, Vec<OpId>>,
}

struct Cursor<'a, 'k> {
    k: &'k Kernel<'a>,
    cta: Cta,
    vals: HashMap<ValueId, Lanes>,
    views: HashMap<ValueId, View>,
    slots: HashMap<ValueId, Lanes>,
    iter: Vec<(String, i64)>,
    /// The iteration count of each enclosing loop, innermost last.
    trips: Vec<usize>,
    /// Iterations each loop ran the first time it was entered.
    loops: HashMap<OpId, usize>,
    steps: Vec<Step>,
    executed: usize,
}

/// What one CTA did: its steps, and how many times each loop ran.
pub(super) struct Ran {
    pub(super) steps: Vec<Step>,
    pub(super) loops: HashMap<OpId, usize>,
}

impl Kernel<'_> {
    /// Every step one CTA takes, in program order.
    pub(super) fn run(&self, cta: Cta) -> Result<Ran> {
        let mut r = Cursor {
            k: self,
            cta,
            vals: HashMap::new(),
            views: HashMap::new(),
            slots: HashMap::new(),
            iter: Vec::new(),
            trips: Vec::new(),
            loops: HashMap::new(),
            steps: Vec::new(),
            executed: 0,
        };
        for (&arg, (_, dims)) in &self.tensors {
            r.views.insert(arg, View::tensor(arg, dims));
        }
        let mask = vec![true; self.lanes];
        r.block(self.ir.entry(), &mask)?;
        Ok(Ran {
            steps: r.steps,
            loops: r.loops,
        })
    }
}

fn any(mask: &[bool]) -> bool {
    mask.iter().any(|&m| m)
}

fn integral(ty: &Type) -> bool {
    matches!(ty, Type::Scalar(s) if !matches!(s, Scalar::F16 | Scalar::BF16 | Scalar::F32 | Scalar::F64))
}

impl<'a> Cursor<'a, '_> {
    fn ir(&self) -> &'a Ir {
        self.k.ir
    }

    fn splat(&self, v: i64) -> Lanes {
        vec![Some(v); self.k.lanes]
    }

    fn get(&self, v: ValueId) -> Lanes {
        self.vals
            .get(&v)
            .cloned()
            .unwrap_or_else(|| vec![None; self.k.lanes])
    }

    /// The value in every active lane, or the trace fails.
    fn known(&self, v: ValueId, mask: &[bool], op: OpId) -> Result<Vec<i64>> {
        let vals = self.get(v);
        vals.iter()
            .zip(mask)
            .map(|(x, &m)| match (x, m) {
                (Some(x), _) => Ok(*x),
                (None, false) => Ok(0),
                (None, true) => bail!("{DATA_DEPENDENT}: {op} `{}`", self.ir().kind(op)),
            })
            .collect()
    }

    /// The value the first active lane holds, for an operand every lane
    /// shares.
    fn lead(&self, v: ValueId, mask: &[bool], op: OpId) -> Result<i64> {
        let at = mask.iter().position(|&m| m).unwrap_or(0);
        Ok(self.known(v, mask, op)?[at])
    }

    fn iter_map(&self) -> BTreeMap<String, i64> {
        self.iter.iter().cloned().collect()
    }

    fn block(&mut self, block: BlockId, mask: &[bool]) -> Result<()> {
        for &op in self.ir().ops(block) {
            self.op(op, mask)?;
        }
        Ok(())
    }

    /// The operands of the terminator ending `block`.
    fn yielded(&self, block: BlockId) -> Vec<ValueId> {
        let ir = self.ir();
        ir.ops(block)
            .last()
            .filter(|&&t| ir.kind(t).is_terminator())
            .map(|&t| ir.operands(t).to_vec())
            .unwrap_or_default()
    }

    fn bind(&mut self, to: ValueId, from: ValueId) {
        let vals = self.get(from);
        self.vals.insert(to, vals);
        if let Some(view) = self.views.get(&from).cloned() {
            self.views.insert(to, view);
        }
    }

    /// Takes `from`'s lanes into `to` where `mask` is set.
    fn merge(&mut self, to: ValueId, from: ValueId, mask: &[bool]) {
        let src = self.get(from);
        let mut dst = self.get(to);
        for (l, &m) in mask.iter().enumerate() {
            if m {
                dst[l] = src[l];
            }
        }
        self.vals.insert(to, dst);
        if let Some(view) = self.views.get(&from).cloned() {
            self.views.insert(to, view);
        }
    }

    fn op(&mut self, op: OpId, mask: &[bool]) -> Result<()> {
        self.executed += 1;
        if self.executed > MAX_OPS_PER_CTA {
            bail!(
                "the kernel runs more than {MAX_OPS_PER_CTA} ops per CTA; trace it with smaller dims"
            );
        }
        let kind = self.ir().kind(op).clone();
        match kind {
            OpKind::For(info) => return self.for_op(op, info, mask),
            OpKind::While => return self.while_op(op, mask),
            OpKind::If => return self.if_op(op, mask),
            OpKind::Yield | OpKind::Condition => return Ok(()),
            OpKind::Trap => bail!(
                "the launch traps: the dims break a promise the kernel checks, such as @aligned"
            ),
            _ => {}
        }
        self.scalar(op, &kind, mask)?;
        self.rebind_iter(op, mask);
        self.step(op, &kind, mask)
    }

    /// A value named for the loop variable in the loop's own body binds it
    /// anew, as each copy of an unrolled body does with `iv + k * step`; the
    /// steps after it carry that value.
    fn rebind_iter(&mut self, op: OpId, mask: &[bool]) {
        let ir = self.k.ir;
        let [result] = ir.results(op) else {
            return;
        };
        let in_loop_body = ir
            .parent_op(ir.parent_block(op))
            .is_some_and(|p| matches!(ir.kind(p), OpKind::For(_)));
        let Some(name) = ir.name(*result).filter(|_| in_loop_body) else {
            return;
        };
        let at = mask.iter().position(|&m| m).unwrap_or(0);
        let value = self.vals.get(result).and_then(|v| v.get(at).copied().flatten());
        if let (Some(last), Some(value)) = (self.iter.last_mut(), value)
            && last.0 == name
        {
            last.1 = value;
        }
    }

    fn for_op(&mut self, op: OpId, info: crate::ir::ForInfo, mask: &[bool]) -> Result<()> {
        let ir = self.k.ir;
        let operands = ir.operands(op).to_vec();
        let (lo, hi, step, tail_hi) = match info.bounds {
            Bounds::Affine { lo, hi, step } => (
                vec![lo; mask.len()],
                vec![hi; mask.len()],
                vec![step; mask.len()],
                None,
            ),
            Bounds::Dynamic => (
                self.known(operands[0], mask, op)?,
                self.known(operands[1], mask, op)?,
                self.known(operands[2], mask, op)?,
                if info.ragged {
                    Some(self.known(operands[3], mask, op)?)
                } else {
                    None
                },
            ),
        };
        if step.iter().zip(mask).any(|(&s, &m)| m && s <= 0) {
            bail!("{op}: a loop step that is not positive");
        }
        let nb = info.bound_operands();
        let inits = &operands[nb..nb + info.carried];
        let blocks = ir.blocks_of(op).to_vec();
        let body = blocks[0];
        let args = ir.args(body).to_vec();
        let name = ir.name(args[0]).map(str::to_string);
        let results = ir.results(op).to_vec();
        for (&r, &init) in results.iter().zip(inits) {
            self.bind(r, init);
        }

        let mut iv = lo.clone();
        let mut trips = 0;
        loop {
            let active: Vec<bool> = (0..mask.len()).map(|l| mask[l] && iv[l] < hi[l]).collect();
            if !any(&active) {
                break;
            }
            self.vals
                .insert(args[0], iv.iter().map(|&x| Some(x)).collect());
            for (&a, &r) in args[1..].iter().zip(&results) {
                self.bind(a, r);
            }
            self.enter(&name, &iv, &active);
            self.trips.push(trips);
            self.block(body, &active)?;
            self.trips.pop();
            trips += 1;
            if name.is_some() {
                self.iter.pop();
            }
            for (&r, y) in results.iter().zip(self.yielded(body)) {
                self.merge(r, y, &active);
            }
            for l in 0..mask.len() {
                if active[l] {
                    iv[l] += step[l];
                }
            }
        }

        if let (Some(tail_hi), Some(&tail)) = (tail_hi, blocks.get(1)) {
            let active: Vec<bool> = (0..mask.len())
                .map(|l| mask[l] && hi[l] < tail_hi[l])
                .collect();
            if any(&active) {
                for (&a, &r) in ir.args(tail).iter().zip(&results) {
                    self.bind(a, r);
                }
                self.enter(&name, &hi, &active);
                self.trips.push(trips);
                self.block(tail, &active)?;
                self.trips.pop();
                trips += 1;
                if name.is_some() {
                    self.iter.pop();
                }
                for (&r, y) in results.iter().zip(self.yielded(tail)) {
                    self.merge(r, y, &active);
                }
            }
        }
        self.loops.entry(op).or_insert(trips);
        Ok(())
    }

    fn enter(&mut self, name: &Option<String>, iv: &[i64], active: &[bool]) {
        if let Some(name) = name {
            let at = active.iter().position(|&m| m).unwrap_or(0);
            self.iter.push((name.clone(), iv[at]));
        }
    }

    fn while_op(&mut self, op: OpId, mask: &[bool]) -> Result<()> {
        let ir = self.k.ir;
        let blocks = ir.blocks_of(op).to_vec();
        let (before, after) = (blocks[0], blocks[1]);
        let results = ir.results(op).to_vec();
        let mut carried: Vec<ValueId> = ir.operands(op).to_vec();
        let mut active = mask.to_vec();
        let mut trips = 0;
        loop {
            for (&a, &c) in ir.args(before).iter().zip(&carried) {
                self.bind(a, c);
            }
            self.block(before, &active)?;
            let cond_op = *ir
                .ops(before)
                .last()
                .expect("a while's test ends its before block");
            let handed = ir.operands(cond_op).to_vec();
            let cond = self.known(handed[0], &active, cond_op)?;
            let leaving: Vec<bool> = (0..mask.len()).map(|l| active[l] && cond[l] == 0).collect();
            for (&r, &v) in results.iter().zip(&handed[1..]) {
                self.merge(r, v, &leaving);
            }
            active = (0..mask.len()).map(|l| active[l] && cond[l] != 0).collect();
            if !any(&active) {
                self.loops.entry(op).or_insert(trips);
                return Ok(());
            }
            for (&a, &v) in ir.args(after).iter().zip(&handed[1..]) {
                self.bind(a, v);
            }
            self.block(after, &active)?;
            carried = self.yielded(after);
            trips += 1;
        }
    }

    fn if_op(&mut self, op: OpId, mask: &[bool]) -> Result<()> {
        let ir = self.k.ir;
        let cond = self.known(ir.operand(op, 0), mask, op)?;
        let blocks = ir.blocks_of(op).to_vec();
        let results = ir.results(op).to_vec();
        for (i, &block) in blocks.iter().enumerate() {
            let taken: Vec<bool> = (0..mask.len())
                .map(|l| mask[l] && (cond[l] != 0) == (i == 0))
                .collect();
            if !any(&taken) {
                continue;
            }
            self.block(block, &taken)?;
            for (&r, y) in results.iter().zip(self.yielded(block)) {
                self.merge(r, y, &taken);
            }
        }
        Ok(())
    }

    /// Computes the op's results where they are addresses.
    fn scalar(&mut self, op: OpId, kind: &OpKind, mask: &[bool]) -> Result<()> {
        let ir = self.k.ir;
        let operands = ir.operands(op).to_vec();
        let results = ir.results(op).to_vec();
        let lanes = self.k.lanes;
        let map2 = |r: &Self, f: &dyn Fn(i64, i64) -> Option<i64>| -> Lanes {
            let (a, b) = (r.get(operands[0]), r.get(operands[1]));
            (0..lanes).map(|l| f(a[l]?, b[l]?)).collect()
        };
        let out: Option<Lanes> = match kind {
            OpKind::Const(Literal::Int(n)) => Some(self.splat(*n)),
            OpKind::Const(Literal::Bool(b)) => Some(self.splat(i64::from(*b))),
            OpKind::ProgramId(d) => Some(self.splat(self.cta[usize::from(*d)])),
            OpKind::GridDim(d) => Some(self.splat(self.k.grid[usize::from(*d)])),
            OpKind::Thread(which) => Some(
                (0..lanes as i64)
                    .map(|l| {
                        Some(match which {
                            ThreadIndex::Thread | ThreadIndex::Lane => l,
                            ThreadIndex::Warp => 0,
                        })
                    })
                    .collect(),
            ),
            OpKind::Dim(d) => {
                let arg = self.tensor_arg(operands[0]);
                let extent = arg
                    .and_then(|a| self.k.tensors.get(&a))
                    .and_then(|(_, dims)| dims.get(*d));
                extent.map(|&e| self.splat(e))
            }
            OpKind::Binary(bop) if integral(ir.ty(results[0])) => {
                Some(map2(self, &|a, b| binary(*bop, a, b)))
            }
            OpKind::IndexOp(iop) => Some(map2(self, &|a, b| index_op(*iop, a, b))),
            OpKind::Unary(uop) if integral(ir.ty(results[0])) => Some(
                self.get(operands[0])
                    .into_iter()
                    .map(|a| {
                        a.map(|a| match uop {
                            UnOp::Neg => a.wrapping_neg(),
                            UnOp::Not => i64::from(a == 0),
                            UnOp::BitNot => !a,
                        })
                    })
                    .collect(),
            ),
            OpKind::Cast(_) | OpKind::IndexCast(_) | OpKind::Converged(_)
                if integral(ir.ty(results[0])) && integral(ir.ty(operands[0])) =>
            {
                Some(self.get(operands[0]))
            }
            OpKind::Select if integral(ir.ty(results[0])) => {
                let (c, a, b) = (
                    self.get(operands[0]),
                    self.get(operands[1]),
                    self.get(operands[2]),
                );
                Some(
                    (0..lanes)
                        .map(|l| if c[l]? != 0 { a[l] } else { b[l] })
                        .collect(),
                )
            }
            OpKind::Shuffle { mode, width } if integral(ir.ty(results[0])) => {
                let v = self.get(operands[0]);
                let src = self.known(operands[1], mask, op)?;
                Some(
                    (0..lanes)
                        .map(|l| {
                            v.get(partner(*mode, *width, l as i64, src[l]) as usize)
                                .copied()
                                .flatten()
                        })
                        .collect(),
                )
            }
            OpKind::Alloc => {
                if let Type::Tile(t) = ir.ty(results[0]) {
                    if t.space == Space::Private {
                        self.slots.insert(results[0], vec![None; lanes]);
                    } else {
                        self.views.insert(results[0], View::buffer(results[0], t));
                    }
                }
                None
            }
            OpKind::Load if self.slots.contains_key(&operands[0]) => {
                Some(self.slots[&operands[0]].clone())
            }
            OpKind::Store if self.slots.contains_key(&operands[1]) => {
                let v = self.get(operands[0]);
                let slot = self.slots.get_mut(&operands[1]).expect("checked above");
                for (l, &m) in mask.iter().enumerate() {
                    if m {
                        slot[l] = v[l];
                    }
                }
                None
            }
            OpKind::AssumeAlign | OpKind::Flat => {
                if let Some(view) = self.views.get(&operands[0]).cloned() {
                    self.views.insert(results[0], view);
                }
                None
            }
            OpKind::Slice(s) => {
                let base = self.views.get(&operands[0]).cloned();
                if let Some(base) = base {
                    let mut offsets = Vec::new();
                    for d in 0..s.rank() {
                        offsets.push(self.lead(operands[s.offset_operand(d)], mask, op)?);
                    }
                    let mut dyn_at = 1 + s.rank();
                    let mut sizes = Vec::new();
                    for e in &s.sizes {
                        sizes.push(match e.fixed() {
                            Some(n) => n,
                            None => {
                                dyn_at += 1;
                                self.lead(operands[dyn_at - 1], mask, op)?
                            }
                        });
                    }
                    let mut limits = Vec::new();
                    for d in 0..s.rank() {
                        limits.push(match s.mask_operand(d) {
                            Some(at) => Some(self.lead(operands[at], mask, op)?),
                            None => None,
                        });
                    }
                    self.views
                        .insert(results[0], base.slice(&offsets, &sizes, &limits));
                }
                None
            }
            _ => None,
        };
        if let Some(out) = out {
            self.vals.insert(results[0], out);
        }
        if kind.makes_buffer() {
            for &r in &results {
                if let Type::Tile(t) = ir.ty(r)
                    && t.space != Space::Private
                {
                    self.views.insert(r, View::buffer(r, t));
                }
            }
        }
        Ok(())
    }

    /// The entry argument a tensor value is a view of.
    fn tensor_arg(&self, v: ValueId) -> Option<ValueId> {
        match self.views.get(&v)?.root {
            Root::Tensor(arg) => Some(arg),
            Root::Buffer(_) => None,
        }
    }

    fn access(&self, view: &View) -> Access {
        view.access(self.k.ir, self.k.place, &self.k.tensors)
    }

    /// The point or run each active lane touches through `mem` at `indices`,
    /// `len` elements along `dim`.
    fn lane_runs(
        &self,
        op: OpId,
        mem: ValueId,
        indices: &[ValueId],
        dim: usize,
        len: i64,
        mask: &[bool],
    ) -> Result<Vec<Run>> {
        let Some(view) = self.views.get(&mem) else {
            return Ok(Vec::new());
        };
        let idx: Vec<Vec<i64>> = indices
            .iter()
            .map(|&i| self.known(i, mask, op))
            .collect::<Result<_>>()?;
        let mut runs = Vec::new();
        for (lane, &m) in mask.iter().enumerate() {
            if !m {
                continue;
            }
            let at: Vec<i64> = idx.iter().map(|v| v[lane]).collect();
            let sizes: Vec<i64> = (0..at.len())
                .map(|d| if d == dim { len } else { 1 })
                .collect();
            runs.push(Run {
                lane,
                access: self.access(&view.slice(&at, &sizes, &vec![None; at.len()])),
            });
        }
        Ok(runs)
    }

    fn step(&mut self, op: OpId, kind: &OpKind, mask: &[bool]) -> Result<()> {
        // Views and extents name memory without touching it.
        if matches!(
            kind,
            OpKind::AssumeAlign | OpKind::Flat | OpKind::Slice(_) | OpKind::Dim(_) | OpKind::Alloc
        ) {
            return Ok(());
        }
        let ir = self.k.ir;
        let operands = ir.operands(op).to_vec();
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut lanes = None;
        let warp = self.k.warp_lanes;
        match kind {
            OpKind::Load | OpKind::Store => {
                let (mem, at) = if matches!(kind, OpKind::Load) {
                    (0, 1)
                } else {
                    (1, 2)
                };
                if self.slots.contains_key(&operands[mem]) {
                    return Ok(());
                }
                let runs = self.lane_runs(op, operands[mem], &operands[at..], 0, 1, mask)?;
                let side = if matches!(kind, OpKind::Load) {
                    &mut reads
                } else {
                    &mut writes
                };
                for run in runs {
                    if !side.contains(&run.access) {
                        side.push(run.access);
                    }
                }
            }
            OpKind::LaneLoad { dim, len, align } | OpKind::LaneStore { dim, len, align, .. } => {
                let load = matches!(kind, OpKind::LaneLoad { .. });
                let rank = match ir.ty(operands[0]) {
                    Type::Tensor(t) => t.shape.len(),
                    Type::Tile(t) => t.rank(),
                    _ => 0,
                };
                let runs =
                    self.lane_runs(op, operands[0], &operands[1..=rank], *dim, *len, mask)?;
                if let Some(bound) = super::view::union(runs.iter().map(|r| &r.access)) {
                    if load {
                        reads.push(bound)
                    } else {
                        writes.push(bound)
                    }
                }
                let elem = match ir.ty(operands[0]) {
                    Type::Tensor(t) => t.elem.bytes(),
                    Type::Tile(t) => t.elem.bytes(),
                    _ => None,
                }
                .unwrap_or(4);
                lanes = Some(LaneEvent {
                    kind: if load {
                        LaneKind::LaneLoad
                    } else {
                        LaneKind::LaneStore
                    },
                    width: warp,
                    vector_bytes: Some(if *align > 0 { len * elem } else { elem }),
                    align_bytes: (*align > 0).then_some(*align),
                    runs: Some(runs),
                    ..lane_event(LaneKind::LaneLoad, warp)
                });
            }
            OpKind::Shuffle { mode, width } => {
                let src = self.known(operands[1], mask, op)?;
                lanes = Some(LaneEvent {
                    mode: Some(mode_of(*mode)),
                    partners: Some(self.partners(*mode, *width, &src)),
                    ..lane_event(LaneKind::Shuffle, *width)
                });
            }
            OpKind::Converged(Group::Lanes) => {
                let Some(shuffles) = self.k.collectives.get(&op) else {
                    return Ok(());
                };
                let mut stages = Vec::new();
                let mut width = warp;
                for &s in shuffles {
                    let OpKind::Shuffle { mode, width: w } = ir.kind(s) else {
                        continue;
                    };
                    width = *w;
                    let src = self.known(ir.operand(s, 1), mask, s)?;
                    stages.push(self.partners(*mode, *w, &src));
                }
                lanes = Some(LaneEvent {
                    mode: Some(Mode::Xor),
                    stages: Some(stages),
                    ..lane_event(LaneKind::Collective, width)
                });
            }
            OpKind::Exchange => {
                lanes = Some(LaneEvent {
                    barriers: Some(self.k.place.barrier_count(op)),
                    ..lane_event(LaneKind::Exchange, warp)
                });
            }
            OpKind::Dp4a | OpKind::Dot4 { .. } => {
                lanes = Some(LaneEvent {
                    vector_bytes: Some(4),
                    ..lane_event(LaneKind::Dp4a, warp)
                });
            }
            OpKind::AtomicAdd | OpKind::AtomicCas | OpKind::LoadAcquire | OpKind::StoreRelease => {
                if let Some(view) = self.views.get(&operands[0]) {
                    let slot = self.lead(operands[1], mask, op)?;
                    let access = self.access(&view.flat_point(slot));
                    reads.push(access.clone());
                    if !matches!(kind, OpKind::LoadAcquire) {
                        writes.push(access);
                    }
                }
            }
            _ => {
                let written = kind.writes(operands.len());
                for (i, &v) in operands.iter().enumerate() {
                    let Some(view) = self.views.get(&v) else {
                        continue;
                    };
                    let access = self.access(view);
                    let beta_reads =
                        matches!(kind, OpKind::GemmStore { beta, .. } if *beta != Coeff::Absent);
                    if written.contains(&i) {
                        if beta_reads {
                            reads.push(access.clone());
                        }
                        writes.push(access);
                    } else {
                        reads.push(access);
                    }
                }
                if kind.makes_buffer() {
                    for &r in ir.results(op) {
                        if let Some(view) = self.views.get(&r) {
                            writes.push(self.access(view));
                        }
                    }
                }
            }
        }
        if let Some(sets) = self.k.place.staging(op) {
            let set = &sets[self.trips.last().copied().unwrap_or(0) % sets.len()];
            reads.extend(set.iter().cloned());
            writes.extend(set.iter().cloned());
        }
        let map = self.k.place.map(op);
        let barrier = self
            .k
            .place
            .trailing_barrier(op)
            .map(|kept| StepBarrier { kept });
        let barrier = barrier
            .or_else(|| matches!(kind, OpKind::Barrier).then_some(StepBarrier { kept: true }));
        if reads.is_empty()
            && writes.is_empty()
            && map.is_none()
            && lanes.is_none()
            && barrier.is_none()
        {
            return Ok(());
        }
        self.steps.push(Step {
            op: op.index(),
            cta: self.cta,
            iter: self.iter_map(),
            reads,
            writes,
            map,
            lanes,
            barrier,
        });
        Ok(())
    }

    /// The source lane of each simulated lane.
    fn partners(&self, mode: AstMode, width: i64, src: &[i64]) -> Vec<i64> {
        (0..self.k.lanes)
            .map(|l| partner(mode, width, l as i64, src[l]))
            .collect()
    }
}

fn lane_event(kind: LaneKind, width: i64) -> LaneEvent {
    LaneEvent {
        kind,
        mode: None,
        width,
        partners: None,
        stages: None,
        runs: None,
        vector_bytes: None,
        align_bytes: None,
        barriers: None,
    }
}

fn mode_of(mode: AstMode) -> Mode {
    match mode {
        AstMode::Xor => Mode::Xor,
        AstMode::Down => Mode::Down,
        AstMode::Up => Mode::Up,
        AstMode::From => Mode::From,
    }
}

/// The lane `lane` reads from, within its `width`-lane piece; a source past
/// the piece reads the lane's own value.
fn partner(mode: AstMode, width: i64, lane: i64, src: i64) -> i64 {
    let at = lane % width;
    match mode {
        AstMode::Xor => lane ^ src,
        AstMode::Down if at + src < width => lane + src,
        AstMode::Up if at >= src => lane - src,
        AstMode::Down | AstMode::Up => lane,
        AstMode::From => lane - at + src.rem_euclid(width),
    }
}

fn binary(op: BinOp, a: i64, b: i64) -> Option<i64> {
    Some(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => a.checked_div(b)?,
        BinOp::Rem => a.checked_rem(b)?,
        BinOp::Eq => i64::from(a == b),
        BinOp::Ne => i64::from(a != b),
        BinOp::Lt => i64::from(a < b),
        BinOp::Le => i64::from(a <= b),
        BinOp::Gt => i64::from(a > b),
        BinOp::Ge => i64::from(a >= b),
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.checked_shl(u32::try_from(b).ok()?)?,
        BinOp::Shr => a.checked_shr(u32::try_from(b).ok()?)?,
        BinOp::And => i64::from(a != 0 && b != 0),
        BinOp::Or => i64::from(a != 0 || b != 0),
        BinOp::Min => a.min(b),
        BinOp::Max => a.max(b),
    })
}

fn index_op(op: IndexOp, a: i64, b: i64) -> Option<i64> {
    let (a, b) = (a as u64, b as u64);
    Some(match op {
        IndexOp::DivU => a.checked_div(b)? as i64,
        IndexOp::RemU => a.checked_rem(b)? as i64,
        IndexOp::CmpUlt => i64::from(a < b),
    })
}

/// The shuffles each full-warp lane collective folds: those its operand is
/// built from in the same block, through arithmetic only.
pub(super) fn collectives(ir: &Ir) -> HashMap<OpId, Vec<OpId>> {
    let mut out = HashMap::new();
    for op in ir.all_ops() {
        if !matches!(ir.kind(op), OpKind::Converged(Group::Lanes)) {
            continue;
        }
        let block = ir.parent_block(op);
        let mut shuffles = Vec::new();
        let mut stack = vec![ir.operand(op, 0)];
        let mut seen = Vec::new();
        while let Some(v) = stack.pop() {
            let Some(def) = ir.def_op(v) else { continue };
            if seen.contains(&def) || ir.parent_block(def) != block {
                continue;
            }
            seen.push(def);
            match ir.kind(def) {
                OpKind::Shuffle { .. } => {
                    shuffles.push(def);
                    stack.push(ir.operand(def, 0));
                }
                OpKind::Binary(_) | OpKind::Unary(_) | OpKind::Select | OpKind::Cast(_) => {
                    stack.extend(ir.operands(def).iter().copied());
                }
                _ => {}
            }
        }
        shuffles.sort_by_key(|&s| ir.position(s));
        out.insert(op, shuffles);
    }
    out
}
