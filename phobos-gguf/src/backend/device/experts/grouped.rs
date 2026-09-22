// The prompt pass's routed feed-forward as grouped GEMMs: the rows sorted
// by expert on the host, a permuted activation whose segments are padded to
// the GEMM's tile, the experts brought in a group at a time (as many as the
// block has slots), each group's segments contracted through a schedule
// table, and each row's eight results gathered back with its weights. Each
// expert crosses the bus once a block, where the row path's cache thrashes
// as soon as a pass's rows choose more experts than the block holds.
// `PHOBOS_MOE_GROUPED=0` opts out; see `ENV.md`.

use anyhow::Result;
use cust::memory::{DeviceBuffer, LockedBuffer};
use cust::stream::Stream;

use super::super::kernels::{MOE_COMBINE_TN, MOE_USED, QGEMM_TM, QGEMM_TN};
use super::super::{DeviceBackend, Plane};
use super::op::kernel_for;
use super::{Experts, Kind, MAX_ROWS, copy_async};
use crate::backend::{Backend, Buf, Moe};
use crate::experts::ExpertSet;

/// Whether the grouped path can run a set: gate and up in one format, and
/// widths in whole blocks and tiles.
pub(super) fn fits(set: &ExpertSet, d: usize, d_ff: usize) -> bool {
    set.gate.quant() == set.up.quant() && [d, d_ff].iter().all(|w| w.is_multiple_of(256) && w.is_multiple_of(QGEMM_TN))
}

/// One expert's rows of the permuted activation: its first padded row and
/// the real rows there.
struct Segment {
    expert: usize,
    start: usize,
    len: usize,
}

/// Columns a permutation program copies.
const PERMUTE_TK: usize = 256;

/// Integers the host writes for the device to read: a pinned mirror and
/// its device copy, filled by asynchronous copies in stream order.
struct Staged {
    host: LockedBuffer<i32>,
    dev: DeviceBuffer<i32>,
}

impl Staged {
    fn new(len: usize) -> Result<Staged> {
        Ok(Staged { host: LockedBuffer::new(&0, len)?, dev: DeviceBuffer::from_slice(&vec![0; len])? })
    }

    /// Copies `values` in at `at` and returns their device address.
    fn push(&mut self, at: usize, values: &[i32], stream: &Stream) -> Result<u64> {
        self.host.as_mut_slice()[at..at + values.len()].copy_from_slice(values);
        let dst = self.dev.as_device_ptr().as_raw() + (at * size_of::<i32>()) as u64;
        // SAFETY: the region is not written again before the block's end
        // synchronizes.
        unsafe { copy_async(dst, self.host.as_slice()[at..].as_ptr().cast(), size_of_val(values), stream)? };
        Ok(dst)
    }
}

/// A pass's permutation, positions and schedule tables, reused block after
/// block: a block's end synchronizes before the next one writes them.
pub(super) struct GroupedScratch {
    perm: Staged,
    pos: Staged,
    sched: Staged,
    /// Integers a group's schedule table may take.
    sched_stride: usize,
}

impl GroupedScratch {
    fn new(n_expert: usize, per_block: usize) -> Result<GroupedScratch> {
        let uses = MAX_ROWS * MOE_USED;
        // A segment is at least one tile and the rows past the first tile
        // of each add at most `uses / TM` more over the whole block.
        let sched_stride = 2 * (per_block + uses / QGEMM_TM);
        Ok(GroupedScratch {
            perm: Staged::new(uses + n_expert * (QGEMM_TM - 1))?,
            pos: Staged::new(uses)?,
            sched: Staged::new(n_expert.div_ceil(per_block) * sched_stride)?,
            sched_stride,
        })
    }
}

pub(crate) fn moe_permute_src() -> String {
    format!(
        "@launch(256)
@autotune(TK in [{PERMUTE_TK}])
@aligned(K = TK)
kernel moe_permute(PERM: tensor<i32>[1, M], A: tensor<f32>[R, K], OUT: tensor<f32>[M, K]) {{
  let p = program_id(0)
  let pk = program_id(1)
  let r = PERM[0, p]
  OUT[p :+ 1, pk * TK :+ TK] = A[r :+ 1, pk * TK :+ TK]
}}
"
    )
}

/// The K-quant prompt GEMM over a schedule table: program `p` contracts
/// activation tile `SCHED[1, p]` against slot `SCHED[0, p]`'s rows.
pub(crate) fn moe_qgemm_src(name: &str, ne: usize) -> String {
    format!(
        "@launch(256, 2)
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}], NE in [{ne}])
@aligned(M = TM, N = TN, NS = NE, K = 256)
kernel {name}_moe_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB], SCHED: tensor<i32>[2, T],
                       QB: tensor<i8>[NS, RB], D: tensor<f16>[NS, NB], C: tensor<f32>[M, N]) {{
  let p = program_id(0)
  let pn = program_id(1)
  let s = SCHED[0, p]
  let t = SCHED[1, p]
  let r = s * NE + pn * TN
  C[t * TM :+ TM, pn * TN :+ TN] = {name}_qgemm_t(A[t * TM :+ TM, :], AS[t * TM :+ TM, :], QB[r :+ TN, :], D[r :+ TN, :])
}}
"
    )
}

/// Each row's eight down rows gathered by position and summed with the
/// router's weights, plus the gated shared expert, into the residual.
pub(crate) fn moe_gather_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}], U in [{MOE_USED}])
@aligned(N = TN)
kernel moe_gather(POS: tensor<i32>[R, U], W: tensor<f32>[R, U], C: tensor<f32>[M, N], G: tensor<f32>[R, 1],
                  S: tensor<f32>[R, N], X: tensor<f32>[R, N]) {{
  let r = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[1, TN] = X[r :+ 1, pn * TN :+ TN]
  for j in range(0, U, 1) {{
    let p = POS[r, j]
    let w = W[r, j]
    acc = acc + C[p :+ 1, pn * TN :+ TN] * w
  }}
  var g: tile<f32>[1, 1] = G[r :+ 1, 0 :+ 1]
  g = 1.0 / (1.0 + exp(0.0 - g))
  acc = acc + S[r :+ 1, pn * TN :+ TN] * g
  X[r :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

impl DeviceBackend {
    /// The rows' feed-forward as grouped GEMMs, after the sync point has
    /// filled the block's `topk_host`. Eager: the caller flushed.
    pub(super) fn run_grouped(&self, experts: &mut Experts, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let mut scratch = match experts.grouped.take() {
            Some(scratch) => scratch,
            None => GroupedScratch::new(req.n_expert, experts.per_block)?,
        };
        let result = self.grouped_blocks(experts, &mut scratch, req, shared);
        experts.grouped = Some(scratch);
        result
    }

    fn grouped_blocks(&self, experts: &mut Experts, scratch: &mut GroupedScratch, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let (d, d_ff, rows, block) = (req.d_model, req.d_ff, req.rows, req.experts.0);

        // Rows by expert, each expert's segment padded to the tile; padding
        // rows read row zero and nothing gathers their outputs.
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); req.n_expert];
        for r in 0..rows {
            for (j, e) in experts.blocks[block].chosen(r)?.into_iter().enumerate() {
                by_expert[e].push((r, j));
            }
        }
        let mut segments = Vec::new();
        let mut perm: Vec<i32> = Vec::new();
        let mut pos = vec![0i32; rows * MOE_USED];
        for (expert, users) in by_expert.iter().enumerate().filter(|(_, u)| !u.is_empty()) {
            let start = perm.len();
            for &(r, j) in users {
                pos[r * MOE_USED + j] = perm.len() as i32;
                perm.push(r as i32);
            }
            perm.resize(perm.len().next_multiple_of(QGEMM_TM), 0);
            segments.push(Segment { expert, start, len: users.len() });
        }
        let m_pad = perm.len();

        // Groups of at most the block's slots, each a contiguous run of
        // padded rows, so a group's activation, GEMMs and SwiGLU are its
        // own rows only: quantized into the transient ring rather than a
        // pass-long slot apiece, which at these row counts was gigabytes.
        let groups: Vec<&[Segment]> = segments.chunks(experts.per_block).collect();
        let span = |g: &[Segment]| (g[0].start, g.last().map_or(0, |s| s.start + s.len.next_multiple_of(QGEMM_TM)) - g[0].start);
        // Scratch sized by bounds on the rows rather than by this block's
        // routing: the pool hands a buffer out again only for its exact
        // length, and a size that moved with the block left every block's
        // scratch dead on the free list until the card paged.
        let uses = rows * MOE_USED;
        let padded = |segments: usize| (uses + segments * (QGEMM_TM - 1)).next_multiple_of(QGEMM_TM);
        let m_max = padded(req.n_expert.min(uses));
        let widest = padded(experts.per_block);
        let perm_ptr = scratch.perm.push(0, &perm, &self.stream)?;
        let pos_ptr = scratch.pos.push(0, &pos, &self.stream)?;
        let xp = self.alloc(widest * d)?;
        let bufs = [self.alloc(widest * d_ff)?, self.alloc(widest * d_ff)?, self.alloc(widest * d_ff)?];
        let [gate_out, up_out, h] = bufs;
        let down_out = self.alloc(m_max * d)?;

        for (gi, group) in groups.into_iter().enumerate() {
            let (g_start, g_rows) = span(group);
            self.with_kernel(&self.moe_permute, (), "moe_permute", moe_permute_src, |module| {
                self.launch(
                    module,
                    "moe_permute",
                    &[
                        (perm_ptr + (g_start * size_of::<i32>()) as u64, [1, g_rows as i64]),
                        (self.ptr(req.x, 0)?, [rows as i64, d as i64]),
                        (self.ptr(xp, 0)?, [g_rows as i64, d as i64]),
                    ],
                    (g_rows as u32, (d / PERMUTE_TK) as u32, 1),
                )
            })?;
            let act = self.quantize_act_into(self.act_slot_transient(g_rows, d)?, xp, g_rows, d)?;
            let (qa, das) = self.act_ptrs(act)?;

            let ids: Vec<usize> = group.iter().map(|s| s.expert).collect();
            let tick = experts.step(block, &ids);
            let mut sched = [Vec::new(), Vec::new()];
            for seg in group {
                let slot = experts
                    .place(block, seg.expert, tick, &self.stream)?
                    .ok_or_else(|| anyhow::anyhow!("a group of {} experts does not fit the block's slots", group.len()))?;
                for t in 0..seg.len.div_ceil(QGEMM_TM) {
                    sched[0].push(slot as i32);
                    sched[1].push(((seg.start - g_start) / QGEMM_TM + t) as i32);
                }
            }
            let tiles = sched[0].len();
            let table_ptr = scratch.sched.push(gi * scratch.sched_stride, &sched.concat(), &self.stream)?;
            self.grouped_gemm(experts, block, Kind::Gate, qa, das, g_rows, d, table_ptr, tiles, d_ff, self.ptr(gate_out, 0)?)?;
            self.grouped_gemm(experts, block, Kind::Up, qa, das, g_rows, d, table_ptr, tiles, d_ff, self.ptr(up_out, 0)?)?;
            let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
            self.swiglu_planes(plane(gate_out), plane(up_out), h, g_rows, d_ff)?;
            let hact = self.quantize_act_into(self.act_slot_transient(g_rows, d_ff)?, h, g_rows, d_ff)?;
            let (hqa, hdas) = self.act_ptrs(hact)?;
            // The down rows land at their global padded rows, where the
            // gather finds them.
            self.grouped_gemm(experts, block, Kind::Down, hqa, hdas, g_rows, d_ff, table_ptr, tiles, d, self.ptr(down_out, g_start * d)?)?;
        }

        let w_ptr = experts.blocks[block].weights.as_device_ptr().as_raw();
        let u = MOE_USED as i64;
        self.with_kernel(&self.moe_gather, (), "moe_gather", moe_gather_src, |module| {
            self.launch(
                module,
                "moe_gather",
                &[
                    (pos_ptr, [rows as i64, u]),
                    (w_ptr, [rows as i64, u]),
                    (self.ptr(down_out, 0)?, [m_pad as i64, d as i64]),
                    (self.ptr(shared.1, 0)?, [rows as i64, 1]),
                    (self.ptr(shared.0, 0)?, [rows as i64, d as i64]),
                    (self.ptr(req.dest, 0)?, [rows as i64, d as i64]),
                ],
                (rows as u32, (d / MOE_COMBINE_TN) as u32, 1),
            )
        })?;
        // The next block rewrites the staging the launches just issued read.
        self.stream.synchronize()?;
        for buf in [xp, down_out].into_iter().chain(bufs) {
            self.release(buf);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_gemm(&self, experts: &Experts, block: usize, kind: Kind, qa: u64, das: u64, m_pad: usize, k: usize, table: u64, tiles: usize, n: usize, out: u64) -> Result<()> {
        let slab = experts.slab(block, kind);
        let b = &experts.blocks[block];
        let (name, [_, _, function], _) = kernel_for(kind.stack(&b.set).quant())?;
        let ns = (experts.per_block * slab.n) as i64;
        let (bytes_at, d_at) = slab.at(b.base[kind as usize]);
        self.with_kernel(&self.moe_qgemm, (name, n), "moe_qgemm", || moe_qgemm_src(name, n), |module| {
            self.launch(
                module,
                function,
                &[
                    (qa, [m_pad as i64, k as i64]),
                    (das, [m_pad as i64, (k / 32) as i64]),
                    (table, [2, tiles as i64]),
                    (bytes_at, [ns, slab.rb as i64]),
                    (d_at, [ns, slab.nb as i64]),
                    (out, [m_pad as i64, n as i64]),
                ],
                (tiles as u32, (n / QGEMM_TN) as u32, 1),
            )
        })
    }
}
