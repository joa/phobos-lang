use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};

use super::{Buf, FusedProject, L2_EPS, Q8_BLOCK, QBuf};

/// Rows of the folded activation a redundant normalization sweeps at a time.
///
/// The whole row in one tile is 16 KB at the model dimension, which alone spends
/// the budget that reaches four blocks per SM; halving it costs one more turn of
/// a two-turn loop. Occupancy is load-bearing here, because the kernel *is* the
/// projections.
pub(crate) const NORM_ROWS: usize = 16;

/// Outputs one block takes of a contraction accumulating into its target.
/// Mirrors `Q8_QDOT_TN`, which the device backend asserts against.
pub(crate) const OUT_TILE: usize = 8;

/// Threads per block. A grid barrier ties the block count to the compiled code,
/// so the thread count has to be fixed here too.
const CTA: usize = 256;

/// Guards a Q8_0 scale against an all-zero run.
const QUANT_EPS: &str = "0.00000001";

/// A tensor a stage reads or writes, as an index into a [`Chain`]'s value table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Val(usize);

/// What a value is, as far as the pass is concerned.
///
/// Deliberately free of buffer handles: this is half the module cache's key, and
/// two layers with different weights want the same compiled kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    /// A caller's row of `len` f32, read or written in place.
    Given { len: usize },
    /// A Q8_0 weight of `rows` by `k`, one scale per [`Q8_BLOCK`] of `k`.
    Weight { rows: usize, k: usize },
    /// A Q8_0 activation the pass gives storage to, `len` elements before any
    /// per-block replication.
    Quant { len: usize },
    /// A value that only ever lives in registers, `len` elements wide. Crossing
    /// a nest with one is what the pass declines on.
    Temp { len: usize },
}

/// The storage behind a value. The cache key leaves this out; the backend
/// resolves it at launch.
#[derive(Clone, Copy, Debug)]
enum Bind {
    Given(Buf),
    Weight(QBuf),
    /// Storage the pass allocates, or none at all.
    Internal,
}

/// One recorded op of a decode step.
///
/// Each variant is a stage of the pass's IR, not a kernel: whether it becomes a
/// loop nest of its own or a few registers inside someone else's is what
/// [`ChainKey::plan`] decides.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Stage {
    /// `out = quantize(x * rsqrt(mean(x * x) + eps) * gain)` over the whole row.
    NormQ {
        x: Val,
        gain: Val,
        out: Val,
        width: usize,
        /// The epsilon's bit pattern, so a chain can key a hash map. Build this
        /// with [`Stage::norm_q`] rather than spelling the bits.
        eps_bits: u32,
    },
    /// `out = w[row_off + unit * Q8_BLOCK ..] . a`, contracting the whole of
    /// `a`. One Q8_0 run of outputs per unit, `units` of them, so a unit's run
    /// is exactly what one output scale covers.
    ProjQ {
        a: Val,
        w: Val,
        out: Val,
        units: usize,
        row_off: usize,
    },
    /// `out[out_off + unit * Q8_BLOCK ..] = w[row_off + unit * Q8_BLOCK ..] . a`,
    /// contracting the whole of `a` and publishing f32 where the caller asked
    /// for it, which is what removes the copy a consumer reading a window of it
    /// would otherwise need.
    ProjF {
        a: Val,
        w: Val,
        out: Val,
        out_off: usize,
        units: usize,
        row_off: usize,
    },
    /// `out = (g * sigmoid(g)) * u` on a unit's run.
    Swiglu { g: Val, u: Val, out: Val },
    /// `out = quantize(h)`, one scale per unit.
    QuantQ { h: Val, out: Val, units: usize },
    /// `y += w[unit * OUT_TILE ..] . a`, contracting the whole of `a`.
    ProjAdd {
        a: Val,
        w: Val,
        y: Val,
        width: usize,
    },
    /// The delta net's causal depthwise convolution over one position, one
    /// (plane, head) pair per unit, writing the packed planes a delta rule reads.
    ///
    /// A unit is a head's row rather than a run of channels because the L2
    /// normalization couples the whole of it, which is also why this cannot share
    /// the projection's partition and so why it costs a barrier. The plane rides
    /// the unit index alongside the head: folding the three into one unit and
    /// unrolling them costs a third of the parallelism and measured worse than
    /// the launch it replaced.
    ///
    /// The unit is also the packed output's row, which holds only at one
    /// position, where a plane is exactly `heads` rows wide.
    Conv {
        history: Val,
        taps: Val,
        out: Val,
        /// Planes, which is three: the query, the key and the value.
        planes: usize,
        heads: usize,
        head_dim: usize,
        kernel: usize,
        /// Elements in one position of the convolution's stream.
        channels: usize,
        /// Element offset of head zero of the first plane within one position,
        /// and the distance to the next plane. The three are evenly spaced in
        /// both layouts a file uses, so this is a stride and not a table.
        plane_base: usize,
        plane_stride: usize,
        /// Distance between consecutive heads within a plane.
        head_stride: usize,
        /// L2-normalize the query and key planes, the value never.
        normalize: bool,
        /// The query's scale, as bits so a chain can key a hash map.
        scale_bits: u32,
    },
    /// `decay = exp(rate * softplus(a + bias))` and `beta = sigmoid(b)`, one
    /// head per unit, appended to the same buffer the planes went in.
    ///
    /// This rides the convolution's nest rather than taking one of its own, which
    /// is what makes it free: it reads across blocks and so needs a barrier, and
    /// sharing the convolution's partition means sharing the barrier the
    /// convolution already forced. Hence `units`, which is wider than the work.
    Gates {
        /// The raw decay and write-strength projections, and where in the value
        /// each starts. Both are windows of the same projection in every layout
        /// seen so far, hence one value.
        raw: Val,
        decay_at: usize,
        beta_at: usize,
        rate: Val,
        bias: Val,
        out: Val,
        heads: usize,
        /// Units of the nest this shares, of which it covers the first `heads`.
        units: usize,
        /// Elements in one packed plane; the gates follow three of them.
        span: usize,
    },
}

/// How a stage's work divides across the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    /// Every block does all of it, on a row narrow enough that repeating the
    /// read beats publishing the result. Carries the row's width in elements.
    Whole(usize),
    /// One unit per block, `count` in all, walked with a grid stride.
    Units(usize),
    /// Elementwise, so whatever partition the nest already has.
    Inherit,
}

/// How much of a value a stage reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Read {
    /// Only the reader's own unit, so nothing crosses a block.
    Local,
    /// All of it, which is what a contraction does, and what costs a barrier
    /// when the writer split the value across the grid.
    All,
}

impl Stage {
    pub(crate) fn norm_q(x: Val, gain: Val, out: Val, width: usize, eps: f32) -> Stage {
        Stage::NormQ {
            x,
            gain,
            out,
            width,
            eps_bits: eps.to_bits(),
        }
    }

    fn part(&self) -> Part {
        match *self {
            Stage::NormQ { width, .. } => Part::Whole(width),
            Stage::ProjQ { units, .. }
            | Stage::ProjF { units, .. }
            | Stage::QuantQ { units, .. } => Part::Units(units),
            Stage::Swiglu { .. } => Part::Inherit,
            Stage::ProjAdd { width, .. } => Part::Units(width / OUT_TILE),
            Stage::Conv { planes, heads, .. } => Part::Units(planes * heads),
            Stage::Gates { units, .. } => Part::Units(units),
        }
    }

    fn reads(&self) -> Vec<(Val, Read)> {
        match *self {
            Stage::NormQ { x, gain, .. } => vec![(x, Read::All), (gain, Read::All)],
            Stage::ProjQ { a, w, .. } | Stage::ProjF { a, w, .. } => {
                vec![(a, Read::All), (w, Read::All)]
            }
            Stage::Swiglu { g, u, .. } => vec![(g, Read::Local), (u, Read::Local)],
            Stage::QuantQ { h, .. } => vec![(h, Read::Local)],
            // `y` is read only where it is written, so the accumulation crosses
            // nothing of its own.
            Stage::ProjAdd { a, w, y, .. } => {
                vec![(a, Read::All), (w, Read::All), (y, Read::Local)]
            }
            // A head's row spans four of the projection's units, so the stream
            // is read across blocks however the convolution is partitioned.
            Stage::Conv { history, taps, .. } => vec![(history, Read::All), (taps, Read::All)],
            Stage::Gates {
                raw, rate, bias, ..
            } => vec![(raw, Read::All), (rate, Read::All), (bias, Read::All)],
        }
    }

    fn writes(&self) -> Val {
        match *self {
            Stage::NormQ { out, .. }
            | Stage::ProjQ { out, .. }
            | Stage::ProjF { out, .. }
            | Stage::Swiglu { out, .. }
            | Stage::QuantQ { out, .. }
            | Stage::Conv { out, .. }
            | Stage::Gates { out, .. } => out,
            Stage::ProjAdd { y, .. } => y,
        }
    }
}

/// The stages of a decode step as a frontend records them, with the storage
/// behind each value kept to one side.
#[derive(Default)]
pub(crate) struct Chain {
    key: ChainKey,
    binds: Vec<Bind>,
}

/// Everything about a chain that decides the emitted source, and so the module
/// cache's key. Buffer handles are excluded on purpose, which is what lets one
/// compiled kernel serve every layer of a model.
#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ChainKey {
    vals: Vec<Kind>,
    stages: Vec<Stage>,
    /// Blocks the kernel is compiled for. A grid barrier makes this part of what
    /// the kernel means, since a block that is not resident never arrives.
    blocks: u32,
}

impl Chain {
    /// A caller buffer of `len` f32, which the chain may read, write or both.
    pub(crate) fn given(&mut self, buf: Buf, len: usize) -> Val {
        self.add(Kind::Given { len }, Bind::Given(buf))
    }

    pub(crate) fn weight(&mut self, w: QBuf, rows: usize, k: usize) -> Val {
        self.add(Kind::Weight { rows, k }, Bind::Weight(w))
    }

    /// A quantized activation passed between stages, which the pass gives
    /// storage to.
    pub(crate) fn quant(&mut self, len: usize) -> Val {
        self.add(Kind::Quant { len }, Bind::Internal)
    }

    /// A value that has to stay in registers, so it must not outlive its nest.
    pub(crate) fn temp(&mut self, len: usize) -> Val {
        self.add(Kind::Temp { len }, Bind::Internal)
    }

    pub(crate) fn push(&mut self, stage: Stage) {
        self.key.stages.push(stage);
    }

    fn add(&mut self, kind: Kind, bind: Bind) -> Val {
        self.key.vals.push(kind);
        self.binds.push(bind);
        Val(self.binds.len() - 1)
    }

    pub(crate) fn buf(&self, val: Val) -> Result<Buf> {
        match self.binds[val.0] {
            Bind::Given(buf) => Ok(buf),
            _ => bail!("value {} is not a caller buffer", val.0),
        }
    }

    pub(crate) fn weight_of(&self, val: Val) -> Result<QBuf> {
        match self.binds[val.0] {
            Bind::Weight(w) => Ok(w),
            _ => bail!("value {} is not a quantized weight", val.0),
        }
    }

    /// The chain's shape at this grid, for looking a compiled plan up. Cloning
    /// the shape and leaving the bindings behind is what lets one compiled
    /// kernel serve every layer.
    pub(crate) fn key(&self, blocks: u32) -> ChainKey {
        ChainKey {
            blocks,
            ..self.key.clone()
        }
    }
}

/// Where a slot's bytes come from. The pass names the operand, the backend
/// resolves it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bound {
    /// A caller buffer.
    Given(Val),
    /// A weight's signed bytes, then its per-row scales.
    WeightQs(Val),
    WeightScales(Val),
    /// Scratch, by index into [`Plan::scratch`].
    ScratchQs(usize),
    ScratchScales(usize),
    /// The arrival counter and release generation.
    Barrier,
}

/// One tensor parameter of the emitted kernel, in declaration order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slot {
    pub(crate) bound: Bound,
    pub(crate) dims: [i64; 2],
}

/// Storage a published value wants, in elements.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scratch {
    pub(crate) bytes: usize,
    pub(crate) scales: usize,
}

/// What the pass produces: source to compile, operands to bind, scratch to
/// allocate, and the grid all three were decided for.
#[derive(Debug)]
pub(crate) struct Plan {
    pub(crate) source: String,
    pub(crate) slots: Vec<Slot>,
    pub(crate) scratch: Vec<Scratch>,
    pub(crate) blocks: u32,
    /// Grid barriers the chain turned out to need.
    pub(crate) barriers: usize,
    /// Values the chain keeps in shared memory rather than publishing. These
    /// cost nothing on the device, which is the point of counting them.
    pub(crate) held: usize,
}

/// A run of stages sharing a partition, and whether a barrier has to precede it.
struct Nest {
    part: Part,
    stages: Vec<usize>,
    barrier_before: bool,
}

impl ChainKey {
    /// Stages the chain records, which is what one fused launch replaces.
    pub(crate) fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Groups the stages into nests, decides where a barrier is genuinely
    /// required, and emits the kernel.
    ///
    /// `Ok(None)` means the pass has no fused form for this chain and the caller
    /// should run the stages as separate launches. An `Err` is a malformed
    /// chain, which is a bug in whoever recorded it.
    pub(crate) fn plan(&self) -> Result<Option<Plan>> {
        let mut nests = self.nests()?;
        let mut nest_of = vec![usize::MAX; self.stages.len()];
        for (n, nest) in nests.iter().enumerate() {
            for &s in &nest.stages {
                nest_of[s] = n;
            }
        }

        // The whole of the pass's judgement is here. A value read outside the
        // nest that wrote it has to be in memory, and it needs a barrier unless
        // the reader can prove it is reading its own block's work.
        //
        // The walk is in chain order and the write is recorded after the reads,
        // so a value the chain both reads early and writes late, which is what
        // accumulating into the residual is, does not look like a dependency of
        // the stage that read it first.
        let mut redundant: HashSet<Val> = HashSet::new();
        let mut writer: HashMap<Val, usize> = HashMap::new();
        for (s, stage) in self.stages.iter().enumerate() {
            let n = nest_of[s];
            for (val, read) in stage.reads() {
                let Some(&w) = writer.get(&val) else { continue };
                if nest_of[w] == n {
                    continue;
                }
                if matches!(self.vals[val.0], Kind::Temp { .. }) {
                    // No f32 scratch path, so a register value crossing a nest
                    // is a chain this pass cannot fuse.
                    return Ok(None);
                }
                // A barrier costs nothing to skip only when the block that reads
                // is the block that wrote: a redundant writer means every block
                // wrote its own copy, and two nests striding the same unit count
                // over the same grid hand unit `i` to the same block in both.
                let part = nests[nest_of[w]].part;
                if matches!(part, Part::Whole(_)) {
                    redundant.insert(val);
                } else if read == Read::All || part != nests[n].part {
                    nests[n].barrier_before = true;
                }
            }
            writer.insert(stage.writes(), s);
        }

        let mut emit = Emit::new(self.blocks, redundant);
        for (n, nest) in nests.iter().enumerate() {
            if nest.barrier_before {
                emit.barrier();
            }
            if !emit.nest(self, nest, n)? {
                return Ok(None);
            }
        }
        Ok(Some(emit.finish()))
    }

    fn nests(&self) -> Result<Vec<Nest>> {
        let mut nests: Vec<Nest> = Vec::new();
        for (s, stage) in self.stages.iter().enumerate() {
            let part = stage.part();
            // A whole-row stage emits a sweep of its own, so it never shares a
            // nest. An elementwise one always joins the nest it is found in.
            let joins = match part {
                Part::Whole(_) => false,
                Part::Inherit => true,
                Part::Units(n) => nests.last().is_some_and(|l| l.part == Part::Units(n)),
            };
            match (joins, nests.last_mut()) {
                (true, Some(last)) => last.stages.push(s),
                (true, None) => bail!("a chain cannot start with an elementwise stage"),
                (false, _) => nests.push(Nest {
                    part,
                    stages: vec![s],
                    barrier_before: false,
                }),
            }
        }
        Ok(nests)
    }

    fn len_of(&self, val: Val) -> usize {
        match self.vals[val.0] {
            Kind::Given { len } | Kind::Quant { len } | Kind::Temp { len } => len,
            Kind::Weight { rows, k } => rows * k,
        }
    }
}

/// The source being built, and the operand list it implies.
struct Emit {
    blocks: u32,
    tune: Vec<(String, usize)>,
    params: Vec<String>,
    slots: Vec<Slot>,
    scratch: Vec<Scratch>,
    /// Tile declarations and their flat views, which have to precede the loops
    /// that fill them however late the stage wanting one is emitted.
    decls: String,
    body: String,
    barriers: usize,
    /// Parameters already declared, by value and the shape it is seen under, so
    /// a second use under the same shape reuses the operand. A quantized value
    /// and a weight name two of them, the bytes and the scales.
    views: HashMap<(Val, View), String>,
    pairs: HashMap<(Val, View), (String, String)>,
    /// Scratch already claimed, by value.
    stored: HashMap<Val, usize>,
    /// Values every block wrote its own copy of, so a reader is reading its own
    /// work and the value belongs in shared memory rather than in scratch.
    redundant: HashSet<Val>,
    /// Those of them a stage has reached, so their tiles are declared exactly
    /// once in [`Emit::decls`].
    shared: HashSet<Val>,
    /// Tile variables holding a value that never leaves its nest.
    regs: HashMap<Val, String>,
    /// The barrier state's parameter, once some nest has needed one.
    bar: Option<String>,
}

/// The shape a value is seen under.
///
/// A normalization writes rows of [`Q8_BLOCK`] and a contraction reads one flat
/// row, and both are the same bytes: the CTA barrier that trails every tile
/// store is what orders the one against the other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum View {
    Folded,
    Flat,
    /// `[len / width, width]`, which is what the convolution walks: one position
    /// of the stream a row, so a tap is a row subscript rather than arithmetic
    /// on a flat offset.
    Grid(usize),
}

impl Emit {
    fn new(blocks: u32, redundant: HashSet<Val>) -> Emit {
        Emit {
            blocks,
            tune: vec![("BLOCKS".into(), blocks as usize)],
            params: Vec::new(),
            slots: Vec::new(),
            scratch: Vec::new(),
            decls: "  let p = program_id(0)\n".into(),
            body: String::new(),
            barriers: 0,
            views: HashMap::new(),
            pairs: HashMap::new(),
            stored: HashMap::new(),
            redundant,
            shared: HashSet::new(),
            regs: HashMap::new(),
            bar: None,
        }
    }

    /// Emit one nest, or decline the chain by returning `false`.
    fn nest(&mut self, key: &ChainKey, nest: &Nest, n: usize) -> Result<bool> {
        match nest.part {
            Part::Whole(width) => {
                let [only] = nest.stages[..] else {
                    bail!("a whole-row nest holds exactly one stage");
                };
                self.whole(key, only, width)
            }
            Part::Units(units) => self.units(key, nest, n, units),
            Part::Inherit => bail!("a nest cannot itself be elementwise"),
        }
    }

    /// The redundant sweep: every block normalizes and quantizes the whole row
    /// into its own copy, so nothing has to be published before the first
    /// contraction.
    fn whole(&mut self, key: &ChainKey, s: usize, width: usize) -> Result<bool> {
        let Stage::NormQ {
            x,
            gain,
            out,
            eps_bits,
            ..
        } = key.stages[s]
        else {
            bail!("only a normalization partitions as a whole row");
        };
        // The row is swept NORM_ROWS rows at a time and a row is Q8_BLOCK wide,
        // so a width that does not divide takes the unfused path.
        if !width.is_multiple_of(Q8_BLOCK * NORM_ROWS) {
            return Ok(false);
        }

        let rows = width / Q8_BLOCK;
        let nb = self.tune_const(format!("NB{s}"), rows);
        let sb = self.tune_const("SB".into(), NORM_ROWS);
        let xf = self.given(x, key.len_of(x), View::Folded);
        let gf = self.given(gain, key.len_of(gain), View::Folded);
        let (qs, sc) = self.quant_rows(out, key.len_of(out));
        let eps = f32::from_bits(eps_bits);
        let q8 = Q8_BLOCK;

        let _ = write!(
            self.body,
            "
  var acc{s}: tile<f32>[{sb}, 1] = 0.0
  for b{s} in range(0, {nb}, {sb}) {{
    let xb{s} = {xf}[b{s} :+ {sb}, 0 :+ {q8}]
    acc{s} = acc{s} + rowsum(xb{s} * xb{s})
  }}
  var tot{s}: tile<f32>[1, 1] = rowsum(transpose(acc{s}))
  var inv{s}: tile<f32>[1, 1] = 1.0 / sqrt(tot{s} / {width}.0 + {eps:.12})
  for b{s} in range(0, {nb}, {sb}) {{
    var y{s}: tile<f32>[{sb}, {q8}] = {xf}[b{s} :+ {sb}, 0 :+ {q8}] * inv{s} \
* {gf}[b{s} :+ {sb}, 0 :+ {q8}]
    var mx{s}: tile<f32>[{sb}, 1] = rowmax(tmax(y{s}, -y{s}))
    var q{s} = y{s} * (127.0 / (mx{s} + {QUANT_EPS}))
    {qs}[b{s} :+ {sb}, 0 :+ {q8}] = i8(i32(round(q{s})))
    {sc}[b{s} :+ {sb}, 0 :+ 1] = mx{s} / 127.0
  }}
"
        );
        Ok(true)
    }

    /// A grid-strided nest, one unit of work per block per turn.
    ///
    /// The iteration count is compiled in and the tail guarded, because a
    /// grid-stride loop over a dynamic extent splits and the masked remainder
    /// then wants a static shape.
    fn units(&mut self, key: &ChainKey, nest: &Nest, n: usize, units: usize) -> Result<bool> {
        let iters = units.div_ceil(self.blocks as usize);
        let it = self.tune_const(format!("IT{n}"), iters);
        let un = self.tune_const(format!("UN{n}"), units);
        let unit = format!("u{n}");
        let _ = write!(
            self.body,
            "
  for i{n} in range(0, {it}) {{
    let {unit} = p + i{n} * BLOCKS
    if {unit} < {un} {{
"
        );
        for &s in &nest.stages {
            if !self.unit_stage(key, s, &unit)? {
                return Ok(false);
            }
        }
        let _ = write!(self.body, "    }}\n  }}\n");
        Ok(true)
    }

    fn unit_stage(&mut self, key: &ChainKey, s: usize, unit: &str) -> Result<bool> {
        let q8 = Q8_BLOCK;
        match key.stages[s] {
            Stage::ProjQ {
                a, w, out, row_off, ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let at = self.strided(format!("OF{s}"), row_off, unit, q8);
                let name = self.reg(out, s);
                let head = format!("      var {name}: tile<f32>[1, {q8}] = qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let at{s} = {at}
{head}{aq}, {asc},
{pad}{wq}[at{s} :+ {q8}, :], {wsc}[at{s} :+ {q8}, :])
"
                );
            }
            Stage::ProjF {
                a,
                w,
                out,
                out_off,
                row_off,
                ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let dst = self.given(out, key.len_of(out), View::Flat);
                let at = self.strided(format!("OF{s}"), row_off, unit, q8);
                let to = self.strided(format!("DO{s}"), out_off, unit, q8);
                let head = format!("      {dst}[0 :+ 1, to{s} :+ {q8}] = qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let at{s} = {at}
      let to{s} = {to}
{head}{aq}, {asc},
{pad}{wq}[at{s} :+ {q8}, :], {wsc}[at{s} :+ {q8}, :])
"
                );
            }
            Stage::Swiglu { g, u, out } => {
                let (gn, un) = (self.reg_of(g)?, self.reg_of(u)?);
                let name = self.reg(out, s);
                let _ = writeln!(
                    self.body,
                    "      var {name}: tile<f32>[1, {q8}] = ({gn} / (1.0 + exp(-{gn}))) * {un}"
                );
            }
            Stage::QuantQ { h, out, .. } => {
                let hn = self.reg_of(h)?;
                let (qs, sc) = self.quant_rows(out, key.len_of(out));
                // The scale has to be bound before the store so the elementwise
                // chain fuses into one sweep: the fusion fires on a named tile
                // and not on an expression.
                let _ = write!(
                    self.body,
                    "      var m{s}: tile<f32>[1, 1] = rowmax(tmax({hn}, -{hn}))
      var q{s} = {hn} * (127.0 / (m{s} + {QUANT_EPS}))
      {qs}[{unit} :+ 1, 0 :+ {q8}] = i8(i32(round(q{s})))
      {sc}[{unit} :+ 1, 0 :+ 1] = m{s} / 127.0
"
                );
            }
            Stage::ProjAdd { a, w, y, width } => {
                if !width.is_multiple_of(OUT_TILE) {
                    return Ok(false);
                }
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let yn = self.given(y, key.len_of(y), View::Flat);
                let tn = self.tune_const("TN".into(), OUT_TILE);
                let head = format!("      {yn}[0 :+ 1, t{s} :+ {tn}] += qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let t{s} = {unit} * {tn}
{head}{aq}, {asc},
{pad}{wq}[t{s} :+ {tn}, :], {wsc}[t{s} :+ {tn}, :])
"
                );
            }
            Stage::Conv { .. } => self.conv(key, s, unit)?,
            Stage::Gates {
                raw,
                decay_at,
                beta_at,
                rate,
                bias,
                out,
                heads,
                span,
                ..
            } => {
                let src = self.given(raw, key.len_of(raw), View::Flat);
                let rt = self.given(rate, key.len_of(rate), View::Flat);
                let bs = self.given(bias, key.len_of(bias), View::Flat);
                let dst = self.given(out, key.len_of(out), View::Flat);
                let (da, ba) = (decay_at, beta_at);
                let (dec, bet) = (3 * span, 3 * span + heads);
                let nh = self.tune_const(format!("GH{s}"), heads);
                // The nest is wider than the gates, which cover a head each, so
                // the tail of it sits this out. The softplus is
                // max(x, 0) + log(1 + exp(-|x|)) rather than the direct
                // log(1 + exp(x)), which overflows well inside the range the
                // decay projection reaches.
                let _ = write!(
                    self.body,
                    "      if {unit} < {nh} {{
        var a{s}: tile<f32>[1, 1] = {src}[0 :+ 1, {da} + {unit} :+ 1] \
+ {bs}[0 :+ 1, {unit} :+ 1]
        var z{s}: tile<f32>[1, 1] = 0.0
        var sp{s}: tile<f32>[1, 1] = tmax(a{s}, z{s}) + log(1.0 + exp(-tmax(a{s}, -a{s})))
        {dst}[0 :+ 1, {dec} + {unit} :+ 1] = exp({rt}[0 :+ 1, {unit} :+ 1] * sp{s})
        var b{s}: tile<f32>[1, 1] = {src}[0 :+ 1, {ba} + {unit} :+ 1]
        {dst}[0 :+ 1, {bet} + {unit} :+ 1] = 1.0 / (1.0 + exp(-b{s}))
      }}
"
                );
            }
            Stage::NormQ { .. } => bail!("a normalization does not partition into units"),
        }
        Ok(true)
    }

    /// One (plane, head) pair of the causal depthwise convolution.
    ///
    /// The plane rides the unit index rather than being unrolled inside it: a
    /// block does one plane of one head, which is the parallelism the launched
    /// kernel had on its third grid axis. Unrolling instead measured 13.5
    /// microseconds a layer against the 9.5 the two launches it replaced cost.
    ///
    /// The epilogue tests the plane, which is derived from the block index and so
    /// is uniform across the CTA. That matters: the gain reduces the row, and a
    /// CTA-wide reduction inside a divergent branch would hang.
    fn conv(&mut self, key: &ChainKey, s: usize, unit: &str) -> Result<()> {
        let Stage::Conv {
            history,
            taps,
            out,
            heads,
            head_dim,
            kernel,
            channels,
            plane_base,
            plane_stride,
            head_stride,
            normalize,
            scale_bits,
            ..
        } = key.stages[s]
        else {
            bail!("only a convolution emits a convolution");
        };
        let hist = self.given(history, key.len_of(history), View::Grid(channels));
        let taps = self.given(taps, key.len_of(taps), View::Grid(channels));
        let dst = self.given(out, key.len_of(out), View::Flat);
        let ks = self.tune_const(format!("KS{s}"), kernel);
        let d = self.tune_const(format!("HD{s}"), head_dim);
        let nh = self.tune_const(format!("NH{s}"), heads);
        let ps = self.tune_const(format!("PS{s}"), plane_stride);
        let st = self.tune_const(format!("ST{s}"), head_stride);
        let base = match plane_base {
            0 => String::new(),
            at => format!("{} + ", self.tune_const(format!("PB{s}"), at)),
        };
        // Only the query carries the readout scale, and only the query and key
        // are normalized: the value is written into the recurrent state rather
        // than matched against it, so it leaves the convolution as it is.
        let scale = f32::from_bits(scale_bits);
        let norm = format!("sqrt(rowsum(y{s} * y{s}) + {L2_EPS})");
        let query = match normalize {
            true => format!("{scale:.9} / {norm}"),
            false => format!("{scale:.9}"),
        };
        let gains: String = [Some(query), normalize.then(|| format!("1.0 / {norm}"))]
            .into_iter()
            .enumerate()
            .filter_map(|(plane, gain)| Some((plane, gain?)))
            .map(|(plane, g)| {
                format!("      if pl{s} == {plane} {{\n        g{s} = {g}\n      }}\n")
            })
            .collect();

        let _ = write!(
            self.body,
            "      let pl{s} = {unit} / {nh}
      let hd{s} = {unit} % {nh}
      let cb{s} = {base}pl{s} * {ps} + hd{s} * {st}
      var acc{s}: tile<f32>[1, {d}] = 0.0
      for k{s} in range(0, {ks}) {{
        acc{s} = acc{s} + {hist}[k{s} :+ 1, cb{s} :+ {d}] * {taps}[k{s} :+ 1, cb{s} :+ {d}]
      }}
      var y{s}: tile<f32>[1, {d}] = acc{s} / (1.0 + exp(-acc{s}))
      var g{s}: tile<f32>[1, 1] = 1.0
{gains}      {dst}[0 :+ 1, {unit} * {head_dim} :+ {d}] = y{s} * g{s}
"
        );
        Ok(())
    }

    /// Where a unit's run of `stride` elements starts, as source text. A zero
    /// offset is left out rather than compiled in as a constant nobody reads,
    /// which keeps the emitted source of a projection starting at row zero the
    /// shape it had before offsets existed.
    fn strided(&mut self, name: String, offset: usize, unit: &str, stride: usize) -> String {
        if offset == 0 {
            return format!("{unit} * {stride}");
        }
        let off = self.tune_const(name, offset);
        format!("{off} + {unit} * {stride}")
    }

    fn reg(&mut self, val: Val, s: usize) -> String {
        let name = format!("v{s}");
        self.regs.insert(val, name.clone());
        name
    }

    fn reg_of(&self, val: Val) -> Result<String> {
        self.regs
            .get(&val)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("value {} is read before its nest writes it", val.0))
    }

    /// One counter and generation pair serves every barrier of the kernel, since
    /// the expansion leaves both as it found them.
    fn barrier(&mut self) {
        let bar = match &self.bar {
            Some(name) => name.clone(),
            None => {
                let name = self.slot("BAR".into(), "i32", [2, 1], Bound::Barrier);
                self.bar = Some(name.clone());
                name
            }
        };
        let _ = writeln!(self.body, "\n  grid_barrier({bar})");
        self.barriers += 1;
    }

    /// Declare a parameter and record what the backend must bind to it.
    fn slot(&mut self, name: String, ty: &str, dims: [i64; 2], bound: Bound) -> String {
        self.params
            .push(format!("{name}: tensor<{ty}>[{}, {}]", dims[0], dims[1]));
        self.slots.push(Slot { bound, dims });
        name
    }

    fn tune_const(&mut self, name: String, value: usize) -> String {
        if !self.tune.iter().any(|(n, _)| *n == name) {
            self.tune.push((name.clone(), value));
        }
        name
    }

    /// The parameter a caller buffer is seen through, folded into rows of
    /// [`Q8_BLOCK`] or flat.
    fn given(&mut self, val: Val, len: usize, view: View) -> String {
        if let Some(name) = self.views.get(&(val, view)) {
            return name.clone();
        }
        let (dims, suffix) = match view {
            View::Folded => ([(len / Q8_BLOCK) as i64, Q8_BLOCK as i64], "F"),
            View::Flat => ([1, len as i64], ""),
            View::Grid(width) => ([(len / width) as i64, width as i64], "G"),
        };
        let name = self.slot(
            format!("X{}{suffix}", val.0),
            "f32",
            dims,
            Bound::Given(val),
        );
        self.views.insert((val, view), name.clone());
        name
    }

    fn weight(&mut self, key: &ChainKey, val: Val) -> Result<(String, String)> {
        if let Some(pair) = self.pairs.get(&(val, View::Flat)) {
            return Ok(pair.clone());
        }
        let Kind::Weight { rows, k } = key.vals[val.0] else {
            bail!(
                "value {} is used as a weight but was not recorded as one",
                val.0
            );
        };
        let qs = self.slot(
            format!("Wq{}", val.0),
            "i8",
            [rows as i64, k as i64],
            Bound::WeightQs(val),
        );
        let scales = self.slot(
            format!("Ws{}", val.0),
            "f32",
            [rows as i64, (k / Q8_BLOCK) as i64],
            Bound::WeightScales(val),
        );
        let pair = (qs, scales);
        self.pairs.insert((val, View::Flat), pair.clone());
        Ok(pair)
    }

    /// The bytes and scales of a quantized activation as rows of [`Q8_BLOCK`],
    /// which is the shape a per-block reduction produces and so what a stage
    /// stores through. The caller adds the row subscript.
    fn quant_rows(&mut self, val: Val, len: usize) -> (String, String) {
        self.quant(val, len, true)
    }

    /// The same value as one row, ready to pass to a contraction.
    ///
    /// Unlike [`Self::quant_rows`] this is a whole operand rather than a name to
    /// subscript, because the two storage classes reach the contraction
    /// differently: a shared value's flat view already *is* a one-row tile, where
    /// a global one is a tensor parameter that a slice has to turn into one.
    fn quant_row(&mut self, val: Val, len: usize) -> (String, String) {
        let (qs, scales) = self.quant(val, len, false);
        match self.shared.contains(&val) {
            true => (qs, scales),
            false => (format!("{qs}[0 :+ 1, :]"), format!("{scales}[0 :+ 1, :]")),
        }
    }

    /// Claims a quantized activation's storage on first sight under either shape.
    ///
    /// Where that storage is, is the one decision here. A value every block wrote
    /// its own copy of is read back only by the block that wrote it, so it goes
    /// in shared and the grid's worth of copies a redundant stage would otherwise
    /// publish never happens. A value one block wrote and another reads has to be
    /// global, and that is exactly the case a barrier already precedes.
    ///
    /// `folded` is a flag rather than a [`View`] because those two shapes are the
    /// only ones a quantized row has any meaning under.
    fn quant(&mut self, val: Val, len: usize, folded: bool) -> (String, String) {
        let view = if folded { View::Folded } else { View::Flat };
        if let Some(pair) = self.pairs.get(&(val, view)) {
            return pair.clone();
        }
        if self.redundant.contains(&val) {
            return self.quant_shared(val, len, folded);
        }
        let at = match self.stored.get(&val) {
            Some(&at) => at,
            None => {
                self.scratch.push(Scratch {
                    bytes: len,
                    scales: len / Q8_BLOCK,
                });
                let at = self.scratch.len() - 1;
                self.stored.insert(val, at);
                at
            }
        };
        let rows = (len / Q8_BLOCK) as i64;
        let (dims_q, dims_s, suffix) = match folded {
            true => ([rows, Q8_BLOCK as i64], [rows, 1], "F"),
            false => ([1, len as i64], [1, rows], ""),
        };
        let qs = self.slot(
            format!("Vq{}{suffix}", val.0),
            "i8",
            dims_q,
            Bound::ScratchQs(at),
        );
        let scales = self.slot(
            format!("Vs{}{suffix}", val.0),
            "f32",
            dims_s,
            Bound::ScratchScales(at),
        );
        let pair = (qs, scales);
        self.pairs.insert((val, view), pair.clone());
        pair
    }

    /// A redundantly written activation, in shared memory.
    ///
    /// The tile is declared without an initializer, since the sweep that follows
    /// writes every element of it, and the flat view is bound next to the
    /// declaration so both are in scope before any loop. The two views are the
    /// same bytes: the folded one is what the per-block reduction produces, the
    /// flat one what the contraction reads.
    fn quant_shared(&mut self, val: Val, len: usize, folded: bool) -> (String, String) {
        let rows = len / Q8_BLOCK;
        let (qs, scales) = (format!("Aq{}", val.0), format!("As{}", val.0));
        if self.shared.insert(val) {
            let q8 = Q8_BLOCK;
            let _ = write!(
                self.decls,
                "  var {qs}: tile<i8>[{rows}, {q8}]
  var {scales}: tile<f32>[{rows}, 1]
  let {qs}L = flat({qs})
  let {scales}L = flat({scales})
"
            );
        }
        let (pair, view) = match folded {
            true => ((qs, scales), View::Folded),
            false => ((format!("{qs}L"), format!("{scales}L")), View::Flat),
        };
        self.pairs.insert((val, view), pair.clone());
        pair
    }

    fn finish(self) -> Plan {
        let tune = self
            .tune
            .iter()
            .map(|(n, v)| format!("{n} in [{v}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let params = self.params.join(",\n             ");
        // The shared declarations go first whatever order the stages wanted them
        // in, since a tile has to be in scope before the loop that fills it.
        let source = format!(
            "@launch({CTA})\n@persistent\n@autotune({tune})\nkernel fused({params}) {{\n{}{}}}\n",
            self.decls, self.body
        );
        Plan {
            source,
            held: self.shared.len(),
            slots: self.slots,
            scratch: self.scratch,
            blocks: self.blocks,
            barriers: self.barriers,
        }
    }
}

/// The decode MLP as a chain: the normalization, the two halves of the gate/up
/// projection, the SwiGLU between them, and the down projection accumulating
/// back into the residual.
///
/// `x` is both the first stage's input and the last stage's target, which is
/// what the aliasing in the emitted kernel means, and the barrier is what makes
/// it safe: every block reads the residual in its own normalization before it
/// arrives, and the down projection adds into it only after.
pub(crate) fn mlp_chain(
    x: Buf,
    gain: Buf,
    gate_up: QBuf,
    down: QBuf,
    d_model: usize,
    d_ff: usize,
    eps: f32,
) -> Chain {
    let mut chain = Chain::default();
    let xv = chain.given(x, d_model);
    let gv = chain.given(gain, d_model);
    let act = chain.quant(d_model);
    chain.push(Stage::norm_q(xv, gv, act, d_model, eps));

    let wgu = chain.weight(gate_up, 2 * d_ff, d_model);
    let units = d_ff / Q8_BLOCK;
    let gate = chain.temp(Q8_BLOCK);
    let up = chain.temp(Q8_BLOCK);
    chain.push(Stage::ProjQ {
        a: act,
        w: wgu,
        out: gate,
        units,
        row_off: 0,
    });
    chain.push(Stage::ProjQ {
        a: act,
        w: wgu,
        out: up,
        units,
        row_off: d_ff,
    });
    let hidden = chain.temp(Q8_BLOCK);
    chain.push(Stage::Swiglu {
        g: gate,
        u: up,
        out: hidden,
    });
    let hq = chain.quant(d_ff);
    chain.push(Stage::QuantQ {
        h: hidden,
        out: hq,
        units,
    });

    let wdn = chain.weight(down, d_model, d_ff);
    chain.push(Stage::ProjAdd {
        a: hq,
        w: wdn,
        y: xv,
        width: d_model,
    });
    chain
}

/// A mixer's input normalization and the projection reading it as a chain, with
/// each run of the projection's outputs written where its consumer wants it, and
/// optionally the delta net's convolution and gates behind them.
///
/// The projection alone costs no barrier: the normalization is redundant, so its
/// quantized row crosses into the projection for free, and the runs are
/// independent nests writing disjoint windows of the caller's buffers. The
/// convolution costs exactly one, for the reason [`super::FusedMix`] gives, and
/// the gates ride in its nest.
///
/// `None` means a shape the pass has no stage for, so the caller keeps its own
/// launches: a run not dividing into whole Q8_0 output blocks, more than one
/// position, or gates split across two projections.
pub(crate) fn project_chain(project: &FusedProject) -> Option<Chain> {
    // One value per buffer, at the widest extent any stage touches. Two values
    // naming one buffer would hide a dependency: the convolution reads the
    // stream position the projection wrote, and the pass sees that only if both
    // stages name the same value.
    let mut extents: Vec<(Buf, usize)> = Vec::new();
    let mut want = |buf: Buf, len: usize| match extents.iter_mut().find(|(b, _)| *b == buf) {
        Some((_, at)) => *at = (*at).max(len),
        None => extents.push((buf, len)),
    };
    for run in project.runs {
        want(run.dst, run.dst_off + run.width);
    }
    if let Some(m) = &project.mix {
        // Two projections feeding the gates would need a value each, and no
        // layout splits them; the stage carries one on purpose.
        if m.decay.0 != m.beta.0 {
            return None;
        }
        let spec = &m.spec;
        want(m.history, spec.history_len());
        want(m.taps, spec.kernel * spec.channels());
        want(m.packed, spec.packed_len());
        want(m.decay.0, m.decay.1 + spec.gates());
        want(m.beta.0, m.beta.1 + spec.gates());
        want(m.rate, spec.heads);
        want(m.dt_bias, spec.heads);
    }

    let mut chain = Chain::default();
    let xv = chain.given(project.x, project.d_model);
    let gv = chain.given(project.gain, project.d_model);
    let vals: Vec<Val> = extents
        .iter()
        .map(|&(buf, len)| chain.given(buf, len))
        .collect();
    let val_of = |buf: Buf| {
        let at = extents.iter().position(|(b, _)| *b == buf)?;
        Some(vals[at])
    };

    let act = chain.quant(project.d_model);
    chain.push(Stage::norm_q(xv, gv, act, project.d_model, project.eps));

    let w = chain.weight(project.w, project.out_dim, project.d_model);
    for run in project.runs {
        if !run.width.is_multiple_of(Q8_BLOCK) || !run.row_off.is_multiple_of(Q8_BLOCK) {
            return None;
        }
        chain.push(Stage::ProjF {
            a: act,
            w,
            out: val_of(run.dst)?,
            out_off: run.dst_off,
            units: run.width / Q8_BLOCK,
            row_off: run.row_off,
        });
    }

    if let Some(m) = &project.mix {
        let spec = &m.spec;
        // One head a unit, and a position at a time, which is the decode shape
        // the whole fused path is for. A prompt pass keeps its own launches.
        if spec.rows != 1 {
            return None;
        }
        // The plane rides the unit index as a stride, so unevenly spaced planes
        // have no stage. No layout uses them, and the device backend's own
        // convolution asserts the same thing.
        let [first, second, third] = spec.planes;
        if third - second != second - first {
            return None;
        }
        chain.push(Stage::Conv {
            history: val_of(m.history)?,
            taps: val_of(m.taps)?,
            out: val_of(m.packed)?,
            planes: spec.planes.len(),
            heads: spec.heads,
            head_dim: spec.head_dim,
            kernel: spec.kernel,
            channels: spec.channels(),
            plane_base: first,
            plane_stride: second - first,
            head_stride: spec.head_stride,
            normalize: spec.normalize,
            scale_bits: spec.query_scale.to_bits(),
        });
        chain.push(Stage::Gates {
            raw: val_of(m.decay.0)?,
            decay_at: m.decay.1,
            beta_at: m.beta.1,
            rate: val_of(m.rate)?,
            bias: val_of(m.dt_bias)?,
            out: val_of(m.packed)?,
            heads: spec.heads,
            units: spec.planes.len() * spec.heads,
            span: spec.span(),
        });
    }
    Some(chain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeltaMix, FusedMix, ProjRun};

    /// Qwen3.5-0.8B's MLP on the grid the card settles at, which is the chain
    /// the hand-written kernel of step 1 covered. The pass has to find one
    /// barrier and put it in one place: after the SwiGLU, because the down
    /// projection contracts over the whole hidden row and a block only wrote a
    /// run of 32 of it.
    fn qwen_plan() -> Plan {
        let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 1024, 3584, 1e-6);
        chain
            .key(192)
            .plan()
            .expect("a well-formed chain")
            .expect("a shape the pass fuses")
    }

    /// Qwen3.5-0.8B's delta-net input: the block normalization and the stacked
    /// projection of the query/key/value stream, the output gate, the decay and
    /// the write strength, with the widest run going straight into the
    /// convolution's padded stream. `mix` continues into the convolution and the
    /// gates reading that projection.
    fn qwen_project_plan(mix: bool) -> Plan {
        let (channels, carried, gates) = (6144, 3 * 6144, 2048 + 2 * 16);
        let runs = [
            ProjRun {
                row_off: 0,
                width: channels,
                dst: Buf(2),
                dst_off: carried,
            },
            ProjRun {
                row_off: channels,
                width: gates,
                dst: Buf(3),
                dst_off: channels,
            },
        ];
        let spec = DeltaMix {
            rows: 1,
            heads: 16,
            head_dim: 128,
            kernel: 4,
            planes: [0, 2048, 4096],
            head_stride: 128,
            normalize: true,
            query_scale: (128.0f32).sqrt().recip(),
        };
        let project = FusedProject {
            x: Buf(0),
            d_model: 1024,
            gain: Buf(1),
            eps: 1e-6,
            w: QBuf(0),
            out_dim: 8448,
            runs: &runs,
            mix: mix.then_some(FusedMix {
                spec,
                history: Buf(2),
                taps: Buf(4),
                // The decay and write strength sit at the tail of the second
                // run, the output gate taking the 2048 ahead of them.
                decay: (Buf(3), channels + 2048),
                beta: (Buf(3), channels + 2048 + 16),
                rate: Buf(5),
                dt_bias: Buf(6),
                packed: Buf(7),
            }),
        };
        project_chain(&project)
            .expect("a shape the pass records")
            .key(192)
            .plan()
            .expect("a well-formed chain")
            .expect("a shape the pass fuses")
    }

    /// Prints the emitted kernel, so it can be fed to
    /// `cargo run -p phobos-lang --example emit` when the generated source is
    /// what is under suspicion.
    ///
    ///     cargo test -p phobos-gguf fused_source -- --nocapture --ignored
    #[test]
    #[ignore = "prints the emitted source rather than checking anything"]
    fn fused_source() {
        println!("{}", qwen_plan().source);
        println!("{}", qwen_project_plan(false).source);
        println!("{}", qwen_project_plan(true).source);
    }

    #[test]
    fn the_mlp_chain_needs_exactly_one_barrier() {
        let plan = qwen_plan();
        assert_eq!(plan.barriers, 1);
        assert_eq!(plan.source.matches("grid_barrier").count(), 1);
        // And in one place: after the SwiGLU's quantization, because the down
        // projection contracts over the whole hidden row and a block wrote a run
        // of 32 of it. The normalization is redundant, so its own output crosses
        // a nest without one.
        let (before, after) = plan.source.split_once("grid_barrier").expect("a barrier");
        assert!(before.contains("rowsum"), "the normalization comes first");
        assert!(before.contains("exp(-v1)"), "the SwiGLU comes first");
        assert!(after.contains("+= qdot_t"), "the accumulation comes after");
    }

    /// What reaches global memory is exactly what crosses a barrier, and nothing
    /// else. Everything between the projections and the quantization stays in
    /// registers; the normalized row is redundant, so the block that reads it is
    /// the block that wrote it and it stays in shared.
    #[test]
    fn only_a_value_crossing_a_barrier_reaches_global_memory() {
        let plan = qwen_plan();
        // The SwiGLU's quantized row, which the down projection contracts over
        // the whole of, so a block reads what other blocks wrote.
        assert_eq!(plan.scratch.len(), 1);
        assert_eq!(plan.scratch[0].bytes, 3584);
        assert_eq!(plan.scratch[0].scales, 112);
        // And the normalized row, held per block instead of published 192 times.
        assert_eq!(plan.held, 1);
        assert_eq!(plan.source.matches("flat(").count(), 2);
        assert!(
            plan.source.contains("var Aq2: tile<i8>[32, 32]"),
            "the activation should be a tile, not an operand"
        );
    }

    /// The grid's block count decides how many copies a published value needs, so
    /// a redundant value moving to shared memory is what takes the block count out
    /// of the emitted source. Two grids, one kernel.
    #[test]
    fn the_held_row_does_not_scale_with_the_grid() {
        let at = |blocks: u32| {
            let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 1024, 3584, 1e-6);
            let plan = chain
                .key(blocks)
                .plan()
                .expect("well-formed")
                .expect("fusable");
            let held = plan
                .source
                .lines()
                .filter(|l| l.contains("tile<i8>") || l.contains("flat("))
                .map(str::trim)
                .map(str::to_string)
                .collect::<Vec<_>>();
            (plan.scratch, held)
        };
        let (small, small_held) = at(48);
        let (large, large_held) = at(192);
        assert_eq!(small_held, large_held);
        assert_eq!(small[0].bytes, large[0].bytes);
    }

    /// The residual is read folded and accumulated into flat, so it is bound
    /// twice under two shapes. Losing that is how the fusion would silently stop
    /// writing where the caller reads.
    #[test]
    fn the_residual_is_bound_under_both_views() {
        let plan = qwen_plan();
        let x: Vec<[i64; 2]> = plan
            .slots
            .iter()
            .filter(|s| matches!(s.bound, Bound::Given(v) if v == Val(0)))
            .map(|s| s.dims)
            .collect();
        assert_eq!(x, vec![[32, 32], [1, 1024]]);
    }

    /// A mixer's projection costs no barrier at all: the normalization ahead of
    /// it is redundant, and the runs are independent nests writing disjoint
    /// windows of buffers nothing in the chain reads back.
    #[test]
    fn the_mixer_projection_needs_no_barrier() {
        let plan = qwen_project_plan(false);
        assert_eq!(plan.barriers, 0);
        assert!(!plan.source.contains("grid_barrier"));
        // And with no barrier, nothing at all has to be published: the one value
        // the chain passes between its nests is the normalized row, which every
        // block wrote for itself and reads from shared.
        assert!(plan.scratch.is_empty());
        assert_eq!(plan.held, 1);
    }

    /// Each run is declared only as far as it writes, and lands at the offset
    /// its consumer reads from. Getting either wrong is how the projection would
    /// quietly write outside the window the convolution walks.
    #[test]
    fn a_run_is_bound_only_as_far_as_it_writes() {
        let plan = qwen_project_plan(false);
        let given: Vec<[i64; 2]> = plan
            .slots
            .iter()
            .filter(|s| matches!(s.bound, Bound::Given(v) if v.0 > 1))
            .map(|s| s.dims)
            .collect();
        // The convolution's stream, up to the end of the fresh position, and the
        // stacked projection up to the end of the write strength. Neither says
        // how large the caller's buffer is.
        assert_eq!(given, vec![[1, 4 * 6144], [1, 6144 + 2080]]);
        assert!(plan.source.contains("DO1 in [18432]"));
        assert!(plan.source.contains("OF2 in [6144], DO2 in [6144]"));
    }

    /// The convolution costs one barrier and the gates behind it cost none: they
    /// take the same unit count, so they share its nest and the barrier it
    /// already forced covers their read too.
    #[test]
    fn the_convolution_costs_one_barrier_and_the_gates_none() {
        let plan = qwen_project_plan(true);
        assert_eq!(plan.barriers, 1);
        let (before, after) = plan.source.split_once("grid_barrier").expect("a barrier");
        assert!(before.contains("qdot_t"), "the projection comes first");
        assert!(!before.contains("rowsum(y"), "the convolution comes after");
        // Both stages after it, and inside the same grid-strided loop rather
        // than one each.
        assert!(after.contains("rowsum(y3"), "the convolution");
        assert!(after.contains("exp(X6["), "the gates");
        // One nest, which is the whole reason the gates are free: a nest of their
        // own would read across blocks and want a barrier of its own.
        assert_eq!(after.matches("in range(0, IT").count(), 1);
        assert!(after.contains("UN3 in [48]") || plan.source.contains("UN3 in [48]"));
    }

    /// The convolution reads the stream position the projection wrote, and the
    /// pass sees that only because both name one value.
    ///
    /// Worth its own test because splitting them is a *latent* miscompile: the
    /// gates read the other projection across blocks too, so the barrier
    /// survives on their account and this chain stays correct. The first chain
    /// with a convolution and no gates would lose it silently.
    #[test]
    fn the_stream_is_one_value_under_two_shapes() {
        let plan = qwen_project_plan(true);
        let stream: Vec<[i64; 2]> = plan
            .slots
            .iter()
            .filter(|s| matches!(s.bound, Bound::Given(v) if v == Val(2)))
            .map(|s| s.dims)
            .collect();
        // Flat, which is where the projection writes its run, and by position,
        // which is how the convolution walks its taps.
        assert_eq!(stream, vec![[1, 24576], [4, 6144]]);
    }

    /// And the stream alone is enough: a convolution with no gates behind it
    /// still cannot be shown to read its own block's work, because a head's row
    /// spans four of the projection's units however the two are partitioned.
    #[test]
    fn the_convolution_alone_still_costs_the_barrier() {
        let (channels, carried) = (6144, 3 * 6144);
        let mut chain = Chain::default();
        let x = chain.given(Buf(0), 1024);
        let gain = chain.given(Buf(1), 1024);
        let stream = chain.given(Buf(2), carried + channels);
        let taps = chain.given(Buf(3), 4 * channels);
        let packed = chain.given(Buf(4), 6176);
        let act = chain.quant(1024);
        chain.push(Stage::norm_q(x, gain, act, 1024, 1e-6));
        let w = chain.weight(QBuf(0), channels, 1024);
        chain.push(Stage::ProjF {
            a: act,
            w,
            out: stream,
            out_off: carried,
            units: channels / Q8_BLOCK,
            row_off: 0,
        });
        chain.push(Stage::Conv {
            history: stream,
            taps,
            out: packed,
            planes: 3,
            heads: 16,
            head_dim: 128,
            kernel: 4,
            channels,
            plane_base: 0,
            plane_stride: 2048,
            head_stride: 128,
            normalize: true,
            scale_bits: 1.0f32.to_bits(),
        });
        let plan = chain
            .key(192)
            .plan()
            .expect("well-formed")
            .expect("fusable");
        assert_eq!(plan.barriers, 1);
    }

    /// The gates go where the delta rule reads them: after the three planes, the
    /// decay then the write strength. An offset wrong here is a plausible-looking
    /// model that decays by a stale number.
    #[test]
    fn the_gates_land_behind_the_planes() {
        let plan = qwen_project_plan(true);
        let (span, heads) = (2048, 16);
        assert!(
            plan.source
                .contains(&format!("X5[0 :+ 1, {} + u3 :+ 1] = exp(", 3 * span))
        );
        assert!(plan.source.contains(&format!(
            "X5[0 :+ 1, {} + u3 :+ 1] = 1.0 /",
            3 * span + heads
        )));
        // And the planes ahead of them. At one position a plane is exactly
        // `heads` rows, so the unit index *is* the packed row and the three
        // planes need no offset of their own.
        assert!(plan.source.contains("X5[0 :+ 1, u3 * 128 :+ HD3]"));
        assert_eq!(span, heads * 128);
    }

    /// Only the query carries the readout scale and only the query and key are
    /// normalized. The value goes into the recurrent state rather than being
    /// matched against it, so a gain on it is silently wrong arithmetic. The
    /// gain is chosen by testing the plane, which comes from the block index and
    /// so is uniform across the CTA: the normalization reduces the row, and a
    /// CTA-wide reduction under a divergent branch would hang.
    #[test]
    fn only_the_query_and_key_are_normalized() {
        let plan = qwen_project_plan(true);
        let gains: Vec<&str> = plan
            .source
            .lines()
            .skip_while(|l| !l.contains("var g3"))
            .filter(|l| l.contains("g3 = "))
            .map(str::trim)
            .collect();
        assert_eq!(gains.len(), 2, "the value plane keeps the default gain");
        assert_eq!(
            gains[0],
            "g3 = 0.088388346 / sqrt(rowsum(y3 * y3) + 0.000000000001)"
        );
        assert_eq!(
            gains[1],
            "g3 = 1.0 / sqrt(rowsum(y3 * y3) + 0.000000000001)"
        );
        // Guarded on the plane, not on the head, and defaulted to 1.0 so the
        // value plane falls through.
        assert!(plan.source.contains("if pl3 == 0 {"));
        assert!(plan.source.contains("if pl3 == 1 {"));
        assert!(plan.source.contains("var g3: tile<f32>[1, 1] = 1.0"));
    }

    /// More than one position is a different convolution: the taps then walk a
    /// window per position rather than the whole stream, and the packed planes
    /// are strided. The pass declines rather than emitting the decode shape.
    #[test]
    fn a_prompt_pass_is_not_recorded() {
        let (channels, carried) = (6144, 3 * 6144);
        let runs = [ProjRun {
            row_off: 0,
            width: channels,
            dst: Buf(2),
            dst_off: carried,
        }];
        let spec = DeltaMix {
            rows: 4,
            heads: 16,
            head_dim: 128,
            kernel: 4,
            planes: [0, 2048, 4096],
            head_stride: 128,
            normalize: true,
            query_scale: 1.0,
        };
        let project = FusedProject {
            x: Buf(0),
            d_model: 1024,
            gain: Buf(1),
            eps: 1e-6,
            w: QBuf(0),
            out_dim: 6144,
            runs: &runs,
            mix: Some(FusedMix {
                spec,
                history: Buf(2),
                taps: Buf(4),
                decay: (Buf(3), 0),
                beta: (Buf(3), 16),
                rate: Buf(5),
                dt_bias: Buf(6),
                packed: Buf(7),
            }),
        };
        assert!(project_chain(&project).is_none());
    }

    /// A run that does not divide into whole Q8_0 output blocks has no unit
    /// count, so the chain cannot even be recorded and the caller keeps its own
    /// projection and copy.
    #[test]
    fn a_ragged_run_is_not_recorded() {
        let runs = [ProjRun {
            row_off: 0,
            width: 48,
            dst: Buf(2),
            dst_off: 0,
        }];
        let project = FusedProject {
            x: Buf(0),
            d_model: 1024,
            gain: Buf(1),
            eps: 1e-6,
            w: QBuf(0),
            out_dim: 48,
            runs: &runs,
            mix: None,
        };
        assert!(project_chain(&project).is_none());
    }

    /// A shape the sweep cannot fold is declined rather than mis-emitted, which
    /// leaves the caller running the four stages itself.
    #[test]
    fn an_unfoldable_width_is_declined() {
        let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 480, 3584, 1e-6);
        assert!(chain.key(192).plan().expect("well-formed").is_none());
    }

    /// A register value read outside the nest that wrote it has nowhere to live,
    /// so the pass declines instead of emitting a kernel that reads a stale
    /// tile.
    #[test]
    fn a_register_crossing_a_nest_is_declined() {
        let mut chain = Chain::default();
        let x = chain.given(Buf(0), 1024);
        let gain = chain.given(Buf(1), 1024);
        let act = chain.quant(1024);
        chain.push(Stage::norm_q(x, gain, act, 1024, 1e-6));
        let w = chain.weight(QBuf(0), 7168, 1024);
        let gate = chain.temp(Q8_BLOCK);
        chain.push(Stage::ProjQ {
            a: act,
            w,
            out: gate,
            units: 112,
            row_off: 0,
        });
        // A second nest, since the unit count differs, reading the first's
        // register output.
        let up = chain.temp(Q8_BLOCK);
        chain.push(Stage::ProjQ {
            a: act,
            w,
            out: up,
            units: 64,
            row_off: 3584,
        });
        let out = chain.temp(Q8_BLOCK);
        chain.push(Stage::Swiglu {
            g: gate,
            u: up,
            out,
        });
        assert!(chain.key(192).plan().expect("well-formed").is_none());
    }
}
