// The prompt pass's routed feed-forward as grouped GEMMs.
//
// The host sorts the rows by expert and builds the permutation for a
// quantized activation, with each expert's segment padded to the GEMM tile. Experts
// are brought in one group at a time, as many as the block has slots. A
// schedule table drives each group's GEMMs, and the group's rows are added
// into the residual before the next group overwrites them.
//
// Each expert crosses the bus once per block. The per-row path would
// thrash the cache whenever the rows choose more experts than a block
// holds. `PHOBOS_MOE_GROUPED=0` opts out; see `ENV.md`.
//
// The experts with the fewest rows run on the host instead, which saves
// their copies. The host's share is tuned from the last block's timings.
// The host runs alongside the device and its rows are added at the end.
// `PHOBOS_MOE_HOST=0` opts out.

use std::time::Instant;

use anyhow::Result;
use cust::event::{Event, EventFlags};
use cust::memory::LockedBuffer;

use super::super::kernels::{MOE_COMBINE_TN, MOE_USED, QGEMM_TM, QGEMM_TN};
use super::super::{DeviceBackend, Plane};
use super::op::{U, kernels};
use super::{Experts, MAX_ROWS, Staged, copy_async};
use crate::backend::{Backend, Buf, Moe};
use crate::experts::{ExpertSet, Stack};
use crate::simd::{self, Job};

/// The host's initial share of a block's experts, before any timings.
pub(super) const HOST_SHARE_START: f32 = 0.4;

/// Upper bound on the padded row count for `segments` experts, each padded
/// to the tile.
fn padded(rows: usize, segments: usize) -> usize {
    (rows * MOE_USED + segments * (QGEMM_TM - 1)).next_multiple_of(QGEMM_TM)
}

/// Device bytes a pass of `rows` needs on this path, beyond the cache.
///
/// Covers the widest group's permuted activation, its gate, up, SwiGLU and
/// down outputs, the quantized SwiGLU copies, and the host's rows.
pub(super) fn scratch_bytes(rows: usize, d: usize, d_ff: usize, per_block: usize) -> usize {
    let widest = padded(rows, per_block);
    let f32s = widest * (d + 3 * d_ff) + rows * d;
    let quantized = |k: usize| widest * k + widest * k / 32 * size_of::<f32>();
    f32s * size_of::<f32>() + quantized(d) + 4 * quantized(d_ff)
}

/// Whether the grouped path can run a set. Gate and up must share a format,
/// and both widths must be whole blocks and tiles.
pub(super) fn can_group(set: &ExpertSet, d: usize, d_ff: usize) -> bool {
    set.gate.quant() == set.up.quant() && [d, d_ff].iter().all(|w| w.is_multiple_of(256) && w.is_multiple_of(QGEMM_TN))
}

/// One expert's rows in the permuted activation: the first padded row and
/// the count of real rows.
struct Segment {
    expert: usize,
    start: usize,
    len: usize,
}

/// The widest tile that divides a row of `k` columns, up to 256.
fn permute_tile(k: usize) -> usize {
    (1..=k.min(256)).rev().find(|t| k.is_multiple_of(*t)).unwrap_or(1)
}

/// A pass's permutation, each group's schedule, positions and weights, and
/// the pinned rows the host's experts produce.
///
/// Reused across blocks. Each block synchronizes at its end, before the
/// next one writes these.
pub(super) struct GroupedScratch {
    perm: Staged<i32>,
    sched: Staged<i32>,
    pos: Staged<i32>,
    w: Staged<f32>,
    /// Maximum entries in one group's schedule table.
    sched_stride: usize,
    /// `[MAX_ROWS, d_model]` the host's share sums into.
    y_host: LockedBuffer<f32>,
    /// The permutation kernels' names and tiles, for the int8 rows and for
    /// their scales. A launch needs a `'static` name, so these are leaked.
    permute: [(&'static str, usize); 2],
}

impl GroupedScratch {
    fn new(n_expert: usize, per_block: usize, d: usize) -> Result<GroupedScratch> {
        let uses = MAX_ROWS * MOE_USED;
        // Each segment is at least one tile, and the rows past each first
        // tile add at most `uses / TM` tiles over the whole block.
        let sched_stride = 2 * (per_block + uses / QGEMM_TM);
        let groups = n_expert.div_ceil(per_block);
        let scale_tile = permute_tile(d / 32);
        let name = |elem: &str, tk: usize| -> &'static str { Box::leak(format!("moe_permute_{elem}_{tk}").into_boxed_str()) };
        Ok(GroupedScratch {
            perm: Staged::new(uses + n_expert * (QGEMM_TM - 1))?,
            sched: Staged::new(groups * sched_stride)?,
            pos: Staged::new(groups * uses)?,
            w: Staged::new(groups * uses)?,
            sched_stride,
            y_host: LockedBuffer::new(&0.0, MAX_ROWS * d)?,
            permute: [(name("i8", 256), 256), (name("f32", scale_tile), scale_tile)],
        })
    }
}

/// Gathers rows of `elem` through a permutation, `tk` columns per program.
fn moe_permute_src(elem: &str, tk: usize) -> String {
    format!(
        "@launch(256)
@autotune(TK in [{tk}])
@aligned(K = TK)
kernel moe_permute_{elem}_{tk}(PERM: tensor<i32>[1, M], A: tensor<{elem}>[R, K], OUT: tensor<{elem}>[M, K]) {{
  let p = program_id(0)
  let pk = program_id(1)
  let r = PERM[0, p]
  OUT[p :+ 1, pk * TK :+ TK] = A[r :+ 1, pk * TK :+ TK]
}}
"
    )
}

/// The K-quant prompt GEMM over a schedule table. Program `p` contracts
/// activation tile `SCHED[1, p]` against the rows of slot `SCHED[0, p]`.
fn moe_qgemm_src(name: &str, ne: usize) -> String {
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

/// Gathers each row's down outputs for one group by position and adds
/// them, weighted by the router, into the residual. Entries outside the
/// group carry weight zero.
fn moe_gather_add_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}], U in [{MOE_USED}])
@aligned(N = TN)
kernel moe_gather_add(POS: tensor<i32>[R, U], W: tensor<f32>[R, U], C: tensor<f32>[M, N], X: tensor<f32>[R, N]) {{
  let r = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[1, TN] = X[r :+ 1, pn * TN :+ TN]
  for j in range(0, U, 1) {{
    let p = POS[r, j]
    let w = W[r, j]
    acc = acc + C[p :+ 1, pn * TN :+ TN] * w
  }}
  X[r :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

/// The gated shared expert added into the residual.
fn moe_shared_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}])
@aligned(N = TN)
kernel moe_shared(G: tensor<f32>[R, 1], S: tensor<f32>[R, N], X: tensor<f32>[R, N]) {{
  let r = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[1, TN] = X[r :+ 1, pn * TN :+ TN]
  var g: tile<f32>[1, 1] = G[r :+ 1, 0 :+ 1]
  g = 1.0 / (1.0 + exp(0.0 - g))
  acc = acc + S[r :+ 1, pn * TN :+ TN] * g
  X[r :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

impl DeviceBackend {
    /// Runs the rows' feed-forward as grouped GEMMs. Call after the sync
    /// point has filled the block's top-k, with the stream flushed.
    pub(super) fn run_grouped(&self, experts: &mut Experts, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let mut scratch = match experts.grouped.take() {
            Some(scratch) => scratch,
            None => GroupedScratch::new(req.n_expert, experts.per_block, req.d_model)?,
        };
        let result = self.grouped_blocks(experts, &mut scratch, req, shared);
        experts.grouped = Some(scratch);
        result
    }

    fn grouped_blocks(&self, experts: &mut Experts, scratch: &mut GroupedScratch, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let (d, d_ff, rows, block) = (req.d_model, req.d_ff, req.rows, req.experts.0);
        let entries = rows * MOE_USED;

        // Rows by expert, and the router's weights.
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); req.n_expert];
        for r in 0..rows {
            for (j, e) in experts.blocks[block].chosen(r)?.into_iter().enumerate() {
                by_expert[e].push((r, j));
            }
        }
        let weights = experts.blocks[block].weights.host()[..entries].to_vec();
        let host_jobs = if self.moe_host && simd::supports(&*experts.blocks[block].set) {
            host_jobs(experts, block, &by_expert, &weights)
        } else {
            Vec::new()
        };
        let mut on_host = vec![false; req.n_expert];
        for job in &host_jobs {
            on_host[job.expert] = true;
        }
        // Read the activation before queuing device work, since a read
        // drains the stream.
        let mut x_host = Vec::new();
        if !host_jobs.is_empty() {
            x_host.resize(rows * d, 0.0);
            self.read(req.x, &mut x_host)?;
        }

        // Times the device's experts, to tune the host share.
        let device_span = [Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?];
        device_span[0].record(&self.stream)?;

        // Pad each device expert's segment to the tile. Padding rows read
        // row zero and their outputs are never gathered. Record each
        // entry's padded row and segment for the gathers.
        let mut segments = Vec::new();
        let mut perm: Vec<i32> = Vec::new();
        let mut pos = vec![0usize; entries];
        let mut segment_of = vec![usize::MAX; entries];
        for (expert, users) in by_expert.iter().enumerate().filter(|&(e, u)| !u.is_empty() && !on_host[e]) {
            let start = perm.len();
            for &(r, j) in users {
                pos[r * MOE_USED + j] = perm.len();
                segment_of[r * MOE_USED + j] = segments.len();
                perm.push(r as i32);
            }
            perm.resize(perm.len().next_multiple_of(QGEMM_TM), 0);
            segments.push(Segment { expert, start, len: users.len() });
        }

        // Groups of at most the block's slot count, each a contiguous run
        // of padded rows. Scratch is sized from a row bound, not from this
        // block's routing, because the pool reuses buffers only at their
        // exact length.
        let groups: Vec<&[Segment]> = segments.chunks(experts.per_block).collect();
        let span = |g: &[Segment]| (g[0].start, g.last().map_or(0, |s| s.start + s.len.next_multiple_of(QGEMM_TM)) - g[0].start);
        let widest = padded(rows, experts.per_block);
        let perm_ptr = scratch.perm.push(0, &perm, &self.stream)?;
        // The last two are the permuted int8 activation and its scales,
        // sized in the pool's f32 units.
        let bufs = [
            self.alloc(widest * d_ff)?,
            self.alloc(widest * d_ff)?,
            self.alloc(widest * d_ff)?,
            self.alloc(widest * d)?,
            self.alloc(widest * d / size_of::<f32>())?,
            self.alloc(widest * d / 32)?,
        ];
        let [gate_out, up_out, h, down_out, xq_g, xs_g] = bufs;
        let (qa, das) = (self.ptr(xq_g, 0)?, self.ptr(xs_g, 0)?);
        let act = req.act.map_or_else(|| self.quantize_act(req.x, rows, d), Ok)?;
        let (xq, xs) = self.act_ptrs(act)?;
        let [rows_kernel, scales_kernel] = scratch.permute;

        for (gi, group) in groups.into_iter().enumerate() {
            let (g_start, g_rows) = span(group);
            // Permute the group's rows of the quantized activation and its
            // scales.
            let perm_at = perm_ptr + (g_start * size_of::<i32>()) as u64;
            for (elem, (name, tk), src, dst, cols) in [("i8", rows_kernel, xq, qa, d), ("f32", scales_kernel, xs, das, d / 32)] {
                self.with_kernel(&self.moe_permute, (elem, tk), "moe_permute", || moe_permute_src(elem, tk), |module| {
                    self.launch(
                        module,
                        name,
                        &[(perm_at, [1, g_rows as i64]), (src, [rows as i64, cols as i64]), (dst, [g_rows as i64, cols as i64])],
                        (g_rows as u32, (cols / tk) as u32, 1),
                    )
                })?;
            }

            let ids: Vec<usize> = group.iter().map(|s| s.expert).collect();
            let tick = experts.step(block, &ids);
            let mut sched = [Vec::new(), Vec::new()];
            for seg in group {
                let slot = experts
                    .place(block, seg.expert, tick, &self.stream, true)?
                    .ok_or_else(|| anyhow::anyhow!("a group of {} experts does not fit the block's slots", group.len()))?;
                for t in 0..seg.len.div_ceil(QGEMM_TM) {
                    sched[0].push(slot as i32);
                    sched[1].push(((seg.start - g_start) / QGEMM_TM + t) as i32);
                }
            }
            let tiles = sched[0].len();
            let table_ptr = scratch.sched.push(gi * scratch.sched_stride, &sched.concat(), &self.stream)?;
            self.grouped_gemm(experts, block, Stack::Gate, qa, das, g_rows, d, table_ptr, tiles, d_ff, self.ptr(gate_out, 0)?)?;
            self.grouped_gemm(experts, block, Stack::Up, qa, das, g_rows, d, table_ptr, tiles, d_ff, self.ptr(up_out, 0)?)?;
            let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
            self.swiglu_planes(plane(gate_out), plane(up_out), h, g_rows, d_ff)?;
            let hact = self.quantize_act_into(self.act_slot_transient(g_rows, d_ff)?, h, g_rows, d_ff)?;
            let (hqa, hdas) = self.act_ptrs(hact)?;
            self.grouped_gemm(experts, block, Stack::Down, hqa, hdas, g_rows, d_ff, table_ptr, tiles, d, self.ptr(down_out, 0)?)?;

            // Map this group's entries to their rows in the down output.
            // Other entries point at row zero with weight zero.
            let (first, last) = (gi * experts.per_block, gi * experts.per_block + group.len());
            let in_group = |i: usize| (first..last).contains(&segment_of[i]);
            let pos_g: Vec<i32> = (0..entries).map(|i| if in_group(i) { (pos[i] - g_start) as i32 } else { 0 }).collect();
            let w_g: Vec<f32> = (0..entries).map(|i| if in_group(i) { weights[i] } else { 0.0 }).collect();
            let pos_ptr = scratch.pos.push(gi * MAX_ROWS * MOE_USED, &pos_g, &self.stream)?;
            let w_ptr = scratch.w.push(gi * MAX_ROWS * MOE_USED, &w_g, &self.stream)?;
            self.with_kernel(&self.moe_gather_add, (), "moe_gather_add", moe_gather_add_src, |module| {
                self.launch(
                    module,
                    "moe_gather_add",
                    &[
                        (pos_ptr, [rows as i64, U]),
                        (w_ptr, [rows as i64, U]),
                        (self.ptr(down_out, 0)?, [g_rows as i64, d as i64]),
                        (self.ptr(req.dest, 0)?, [rows as i64, d as i64]),
                    ],
                    (rows as u32, (d / MOE_COMBINE_TN) as u32, 1),
                )
            })?;
        }
        self.with_kernel(&self.moe_shared, (), "moe_shared", moe_shared_src, |module| {
            self.launch(
                module,
                "moe_shared",
                &[
                    (self.ptr(shared.1, 0)?, [rows as i64, 1]),
                    (self.ptr(shared.0, 0)?, [rows as i64, d as i64]),
                    (self.ptr(req.dest, 0)?, [rows as i64, d as i64]),
                ],
                (rows as u32, (d / MOE_COMBINE_TN) as u32, 1),
            )
        })?;
        device_span[1].record(&self.stream)?;

        // Run the host's share while the device runs its groups.
        let mut host_micros = 0.0;
        if !host_jobs.is_empty() {
            let started = Instant::now();
            let y_host = &mut scratch.y_host.as_mut_slice()[..rows * d];
            y_host.fill(0.0);
            let b = &experts.blocks[block];
            simd::experts_ffn(&b.mirror.source(&b.set), &host_jobs, &x_host, rows, y_host)?;
            host_micros = started.elapsed().as_secs_f64() * 1e6;
            experts.stats.cpu_misses += host_jobs.len() as u64;
            experts.stats.cpu_nanos += (host_micros * 1e3) as u64;
            self.add_host_rows(req.dest, &scratch.y_host, rows * d)?;
        }
        // The next block rewrites the staging these launches read.
        self.stream.synchronize()?;
        // Each side's time is roughly proportional to its expert count, so
        // the two rates give the balanced split. Move the share halfway
        // there, clamped.
        if !host_jobs.is_empty() && host_micros > 0.0 {
            let device_micros = f64::from(device_span[1].elapsed_time_f32(&device_span[0])?) * 1e3;
            let share = experts.host_share;
            let (host_rate, device_rate) = (host_micros / f64::from(share), device_micros / f64::from(1.0 - share));
            let balanced = (device_rate / (host_rate + device_rate)) as f32;
            experts.host_share = (0.5 * share + 0.5 * balanced).clamp(0.05, 0.95);
        }
        for buf in bufs {
            self.release(buf);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_gemm(&self, experts: &Experts, block: usize, stack: Stack, qa: u64, das: u64, m_pad: usize, k: usize, table: u64, tiles: usize, n: usize, out: u64) -> Result<()> {
        let kernels = kernels(experts.blocks[block].set.stack(stack).quant())?;
        let [bytes, d] = experts.slab_operands(block, stack);
        self.with_kernel(&self.moe_qgemm, (kernels.prefix, n), "moe_qgemm", || moe_qgemm_src(kernels.prefix, n), |module| {
            self.launch(
                module,
                kernels.qgemm,
                &[(qa, [m_pad as i64, k as i64]), (das, [m_pad as i64, (k / 32) as i64]), (table, [2, tiles as i64]), bytes, d, (out, [m_pad as i64, n as i64])],
                (tiles as u32, (n / QGEMM_TN) as u32, 1),
            )
        })
    }

    /// Adds `len` floats of pinned rows into `dest`, in stream order.
    ///
    /// The rows must not be rewritten before the next sync point. Meant for
    /// a prompt pass, where no refill competes for the copy queue.
    fn add_host_rows(&self, dest: Buf, rows: &LockedBuffer<f32>, len: usize) -> Result<()> {
        let buf = self.alloc(len)?;
        // SAFETY: the pinned rows live with the grouped scratch.
        unsafe { copy_async(self.ptr(buf, 0)?, rows.as_slice().as_ptr().cast(), len * size_of::<f32>(), &self.stream)? };
        self.add_into(dest, buf)?;
        self.release(buf);
        Ok(())
    }

}

/// The experts the host takes: the host share of the non-resident experts,
/// fewest rows first. Their entries are in no group, so no gather adds
/// them.
fn host_jobs(experts: &Experts, block: usize, by_expert: &[Vec<(usize, usize)>], weights: &[f32]) -> Vec<Job> {
    let mut order: Vec<usize> = (0..by_expert.len()).filter(|&e| !by_expert[e].is_empty() && !experts.resident(block, e)).collect();
    order.sort_by_key(|&e| by_expert[e].len());
    let take = (order.len() as f32 * experts.host_share).round() as usize;
    order
        .into_iter()
        .take(take)
        .map(|e| {
            let users = &by_expert[e];
            Job {
                expert: e,
                rows: users.iter().map(|&(r, _)| r).collect(),
                weights: users.iter().map(|&(r, j)| weights[r * MOE_USED + j]).collect(),
            }
        })
        .collect()
}
