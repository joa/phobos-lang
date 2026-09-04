// Lowering a chain of decode stages into one persistent kernel.
//
// This file is the vocabulary: what a stage is, what it reads and writes,
// and how a chain keys into a compiled plan. `emit.rs` turns that into
// Phobos source and `chains.rs` builds the two chains the model uses.

mod chains;
mod emit;

#[cfg(test)]
mod tests;

pub(crate) use chains::{attn_out_chain, mlp_chain, mlp_chain_raw, project_chain};

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};

use super::{Buf, FusedProject, L2_EPS, ProjWeight, Q8_BLOCK, QBuf, RawBuf};
use crate::quant::Quant;

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

/// Outputs one block takes of a raw-format contraction, both kinds: the
/// `<fmt>_qdot_i8_t` decode gives a warp eight columns and a CTA of eight
/// warps wants 64 to keep every warp busy. Two Q8_0 blocks, so a
/// quantization of a unit's run writes two rows.
pub(crate) const RAW_UNIT: usize = 64;

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
    /// A raw-format weight of `rows` by `k`, its block bytes and `d` plane
    /// as `constant_raw` uploaded them, decoded by `quant`'s own intrinsic.
    Raw { rows: usize, k: usize, quant: Quant },
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
    Raw(RawBuf),
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
    /// `out = w[row_off + unit * RAW_UNIT ..] . a` for a raw-format weight,
    /// contracting the whole of `a`: one [`RAW_UNIT`] run of outputs per unit.
    ProjRaw {
        a: Val,
        w: Val,
        out: Val,
        units: usize,
        row_off: usize,
    },
    /// `out[out_off + unit * RAW_UNIT ..] = w[row_off + unit * RAW_UNIT ..] . a`
    /// for a raw-format weight, `width` outputs a unit: [`RAW_UNIT`], or the
    /// remainder of a run for the one unit that finishes it.
    ProjRawF {
        a: Val,
        w: Val,
        out: Val,
        out_off: usize,
        units: usize,
        width: usize,
        row_off: usize,
    },
    /// `out = (g * sigmoid(g)) * u` on a unit's run.
    Swiglu { g: Val, u: Val, out: Val },
    /// `out = quantize(h)`, `blocks` scales per unit: one where a unit is a
    /// Q8_0 block, two where it is a raw format's run of [`RAW_UNIT`].
    QuantQ { h: Val, out: Val, units: usize, blocks: usize },
    /// `y += w[unit * RAW_UNIT ..] . a` for a raw-format weight.
    ProjAddRaw {
        a: Val,
        w: Val,
        y: Val,
        width: usize,
    },
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
            | Stage::ProjRaw { units, .. }
            | Stage::ProjRawF { units, .. }
            | Stage::QuantQ { units, .. } => Part::Units(units),
            Stage::Swiglu { .. } => Part::Inherit,
            Stage::ProjAdd { width, .. } => Part::Units(width / OUT_TILE),
            Stage::ProjAddRaw { width, .. } => Part::Units(width / RAW_UNIT),
            Stage::Conv { planes, heads, .. } => Part::Units(planes * heads),
            Stage::Gates { units, .. } => Part::Units(units),
        }
    }

    fn reads(&self) -> Vec<(Val, Read)> {
        match *self {
            Stage::NormQ { x, gain, .. } => vec![(x, Read::All), (gain, Read::All)],
            Stage::ProjQ { a, w, .. }
            | Stage::ProjF { a, w, .. }
            | Stage::ProjRaw { a, w, .. }
            | Stage::ProjRawF { a, w, .. } => vec![(a, Read::All), (w, Read::All)],
            Stage::Swiglu { g, u, .. } => vec![(g, Read::Local), (u, Read::Local)],
            Stage::QuantQ { h, .. } => vec![(h, Read::Local)],
            // `y` is read only where it is written, so the accumulation crosses
            // nothing of its own.
            Stage::ProjAdd { a, w, y, .. } | Stage::ProjAddRaw { a, w, y, .. } => {
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
            | Stage::ProjRaw { out, .. }
            | Stage::ProjRawF { out, .. }
            | Stage::Swiglu { out, .. }
            | Stage::QuantQ { out, .. }
            | Stage::Conv { out, .. }
            | Stage::Gates { out, .. } => out,
            Stage::ProjAdd { y, .. } | Stage::ProjAddRaw { y, .. } => y,
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

    /// A raw-format weight, decoded inside the kernel by its own intrinsic.
    pub(crate) fn raw_weight(&mut self, w: RawBuf, rows: usize, k: usize, quant: Quant) -> Val {
        self.add(Kind::Raw { rows, k, quant }, Bind::Raw(w))
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

    pub(crate) fn raw_of(&self, val: Val) -> Result<RawBuf> {
        match self.binds[val.0] {
            Bind::Raw(w) => Ok(w),
            _ => bail!("value {} is not a raw weight", val.0),
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
    /// A raw weight's block bytes, then its `d` plane.
    RawBytes(Val),
    RawD(Val),
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
