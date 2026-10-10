// Lowering a chain of decode stages into one persistent kernel.
//
// This file defines the vocabulary: stages, what they read and write, and how
// a chain keys a compiled plan. `emit.rs` turns a chain into Phobos source.
// `chains.rs` builds the chains the model uses.

mod chains;
mod emit;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) use chains::attn_out_chain;
pub(crate) use chains::{mlp_chain, mlp_chain_raw, project_chain};

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};

use super::{Buf, FusedProject, L2_EPS, ProjWeight, Q8_BLOCK, QBuf, RawBuf};
use crate::quant::Quant;

/// Outputs per block of an accumulating contraction. Must equal
/// `Q8_QDOT_TN`; the device backend asserts it.
pub(crate) const OUT_TILE: usize = 8;

/// Outputs per block of a raw-format contraction. Eight warps of eight
/// columns each. This is two Q8_0 blocks, so quantizing a unit's run writes
/// two scales.
pub(crate) const RAW_UNIT: usize = 32;

/// Threads per block. Fixed, since the grid barrier ties the launch shape to
/// the compiled code.
const CTA: usize = 256;

/// Guards a Q8_0 scale against an all-zero run.
const QUANT_EPS: &str = "0.00000001";

/// A tensor a stage reads or writes, as an index into a [`Chain`]'s value table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Val(usize);

/// The shape of a value.
///
/// Holds no buffer handles, since it is part of the module cache key and
/// layers with different weights share one compiled kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    /// A caller's row of `len` f32, read or written in place.
    Given { len: usize },
    /// A Q8_0 weight of `rows` by `k`, one scale per [`Q8_BLOCK`] of `k`.
    Weight { rows: usize, k: usize },
    /// A raw-format weight of `rows` by `k`, as `constant_raw` uploaded it,
    /// decoded by `quant`'s own intrinsic.
    Raw { rows: usize, k: usize, quant: Quant },
    /// A Q8_0 activation the pass allocates, `len` elements before any
    /// per-block replication.
    Quant { len: usize },
    /// A value that lives only in registers, `len` elements wide. The pass
    /// declines a chain where one crosses a nest.
    Temp { len: usize },
}

/// The storage behind a value. Not part of the cache key; the backend
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
/// A stage of the pass's IR, not a kernel. [`ChainKey::plan`] decides whether
/// it gets a loop nest of its own or joins another's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Stage {
    /// `out = quantize(x * rsqrt(mean(x * x) + eps) * gain)` over the whole row.
    NormQ {
        x: Val,
        gain: Val,
        out: Val,
        width: usize,
        /// The epsilon's bits, so the stage can be hashed. Build with
        /// [`Stage::norm_q`].
        eps_bits: u32,
    },
    /// `out = w[row_off + unit * Q8_BLOCK ..] . a`, contracting all of `a`.
    /// Each of the `units` produces one Q8_0 block of outputs, covered by one
    /// output scale.
    ProjQ {
        a: Val,
        w: Val,
        out: Val,
        units: usize,
        row_off: usize,
    },
    /// `out[out_off + unit * Q8_BLOCK ..] = w[row_off + unit * Q8_BLOCK ..] . a`,
    /// contracting all of `a` and writing f32 at the caller's offset.
    ProjF {
        a: Val,
        w: Val,
        out: Val,
        out_off: usize,
        units: usize,
        row_off: usize,
    },
    /// `out = w[row_off + unit * RAW_UNIT ..] . a` for a raw-format weight,
    /// contracting all of `a`. One [`RAW_UNIT`] run of outputs per unit.
    ProjRaw {
        a: Val,
        w: Val,
        out: Val,
        units: usize,
        row_off: usize,
    },
    /// `out[out_off + unit * RAW_UNIT ..] = w[row_off + unit * RAW_UNIT ..] . a`
    /// for a raw-format weight. Each unit writes `width` outputs, which is
    /// [`RAW_UNIT`] or, for the last unit, the remainder.
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
    /// `out = quantize(h)` with `blocks` scales per unit: one for a Q8_0
    /// block, two for a raw format's [`RAW_UNIT`] run.
    QuantQ { h: Val, out: Val, units: usize, blocks: usize },
    /// `y += w[unit * RAW_UNIT ..] . a` for a raw-format weight.
    ProjAddRaw {
        a: Val,
        w: Val,
        y: Val,
        width: usize,
    },
    /// `y += w[unit * OUT_TILE ..] . a`, contracting all of `a`.
    ProjAdd {
        a: Val,
        w: Val,
        y: Val,
        width: usize,
    },
    /// The delta net's causal depthwise convolution over one position. One
    /// (plane, head) pair per unit, writing the packed planes the delta rule
    /// reads.
    ///
    /// A unit is a whole head's row because the L2 normalization spans it.
    /// So this cannot share the projection's partition and needs its own
    /// barrier.
    ///
    /// A unit is also a row of the packed output, which holds only for a
    /// single position.
    Conv {
        history: Val,
        taps: Val,
        out: Val,
        /// Three: the query, the key and the value.
        planes: usize,
        heads: usize,
        head_dim: usize,
        kernel: usize,
        /// Elements in one position of the convolution's stream.
        channels: usize,
        /// Element offset of the first plane's head zero within a position,
        /// and the distance to the next plane. The planes are evenly spaced
        /// in every supported layout.
        plane_base: usize,
        plane_stride: usize,
        /// Distance between consecutive heads within a plane.
        head_stride: usize,
        /// L2-normalize the query and key planes. The value is never
        /// normalized.
        normalize: bool,
        /// The query's scale, as bits so the stage can be hashed.
        scale_bits: u32,
    },
    /// `decay = exp(rate * softplus(a + bias))` and `beta = sigmoid(b)`, one
    /// head per unit, appended after the planes in the same buffer.
    ///
    /// It shares the convolution's nest, so it reuses the barrier the
    /// convolution already needs. Hence `units` is wider than the work.
    Gates {
        /// The raw decay and write-strength projections, and where each
        /// starts. Both are windows of one projection.
        raw: Val,
        decay_at: usize,
        beta_at: usize,
        rate: Val,
        bias: Val,
        out: Val,
        heads: usize,
        /// Units of the shared nest. Only the first `heads` do work.
        units: usize,
        /// Elements in one packed plane. The gates follow three planes.
        span: usize,
    },
}

/// How a stage's work divides across the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    /// Every block does all of it, redundantly. Carries the row's width in
    /// elements.
    Whole(usize),
    /// One unit per block, `count` in all, walked with a grid stride.
    Units(usize),
    /// Elementwise, so it takes the nest's partition.
    Inherit,
}

/// How much of a value a stage reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Read {
    /// Only the reader's own unit, so nothing crosses a block.
    Local,
    /// All of it, as a contraction does. Needs a barrier when the writer
    /// split the value across the grid.
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
            // `y` is read only where it is written, so it crosses no block.
            Stage::ProjAdd { a, w, y, .. } | Stage::ProjAddRaw { a, w, y, .. } => {
                vec![(a, Read::All), (w, Read::All), (y, Read::Local)]
            }
            // A head's row spans several projection units, so the stream is
            // read across blocks.
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

/// The recorded stages of a decode step, with each value's storage kept
/// separately from the key.
#[derive(Default)]
pub(crate) struct Chain {
    key: ChainKey,
    binds: Vec<Bind>,
}

/// Everything that decides the emitted source, used as the module cache key.
/// It excludes buffer handles, so one compiled kernel serves every layer.
#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ChainKey {
    vals: Vec<Kind>,
    stages: Vec<Stage>,
    /// Blocks the kernel is compiled for. Every block must be resident, or
    /// the grid barrier never completes.
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

    /// A quantized activation passed between stages, allocated by the pass.
    pub(crate) fn quant(&mut self, len: usize) -> Val {
        self.add(Kind::Quant { len }, Bind::Internal)
    }

    /// A register-only value. It must not outlive its nest.
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

    /// Every Q8_0-family weight the chain binds.
    pub(crate) fn weights(&self) -> impl Iterator<Item = QBuf> + '_ {
        self.binds.iter().filter_map(|bind| match bind {
            Bind::Weight(w) => Some(*w),
            _ => None,
        })
    }

    pub(crate) fn raw_of(&self, val: Val) -> Result<RawBuf> {
        match self.binds[val.0] {
            Bind::Raw(w) => Ok(w),
            _ => bail!("value {} is not a raw weight", val.0),
        }
    }

    /// The chain's cache key at this grid size. Bindings are left out, so
    /// one compiled kernel serves every layer.
    pub(crate) fn key(&self, blocks: u32) -> ChainKey {
        ChainKey {
            blocks,
            ..self.key.clone()
        }
    }
}

/// Where a slot's bytes come from. The pass names the operand and the
/// backend resolves it.
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

/// The pass's output: source to compile, operands to bind, scratch to
/// allocate, and the grid size they assume.
#[derive(Debug)]
pub(crate) struct Plan {
    pub(crate) source: String,
    pub(crate) slots: Vec<Slot>,
    pub(crate) scratch: Vec<Scratch>,
    pub(crate) blocks: u32,
    /// Grid barriers the chain needs.
    pub(crate) barriers: usize,
    /// Values kept in shared memory rather than published to device memory.
    pub(crate) held: usize,
}
