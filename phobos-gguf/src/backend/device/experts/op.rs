// The device side of `Backend::moe`: top-k over every row, one sync point
// a block where the rows' choices come back, then either the row path
// here, per row the misses into their slots and the slot matvecs and
// combine that read them, or the grouped path of `grouped.rs`.
//
// With `PHOBOS_MOE_HOST_DECODE=1` a decode step's misses go to the host:
// the row's kernels run over the hits with a zero slot standing in for
// each miss, replayed at once so the device works while the host computes
// the misses' share, which is added afterwards; each miss is also copied
// into its slot on the second stream, so the next token finds it.

use anyhow::{Result, bail, ensure};
use cust::memory::CopyDestination;
use cust::stream::StreamWaitEventFlags;
use phobos_kernels::cuda_ok;

use super::super::kernels::{MOE_COMBINE_TN, MOE_QDOT_TN, MOE_USED, moe_combine_src, moe_gateup_src, moe_qdot_src, moe_topk_src};
use super::super::{DeviceBackend, Plane};
use super::{Experts, MAX_ROWS};
use crate::backend::{Backend, Buf, Moe};
use crate::experts::Stack;
use crate::quant::Quant;
use crate::simd;

pub(super) const U: i64 = MOE_USED as i64;

/// Bytes of one row's eight table entries.
const ROW_BYTES: usize = MOE_USED * size_of::<i32>();

/// A format's slot kernels: the source prefix, the kernels' names and the
/// register budget's CTA count.
pub(super) struct Kernels {
    pub(super) prefix: &'static str,
    pub(super) qdot: &'static str,
    pub(super) gateup: &'static str,
    pub(super) qgemm: &'static str,
    pub(super) resident: usize,
}

pub(super) fn kernels(quant: Quant) -> Result<Kernels> {
    let (prefix, qdot, gateup, qgemm, resident) = match quant {
        Quant::Q4_K => ("q4k", "q4k_moe_qdot", "q4k_moe_gateup", "q4k_moe_qgemm", 4),
        Quant::Q5_K => ("q5k", "q5k_moe_qdot", "q5k_moe_gateup", "q5k_moe_qgemm", 3),
        Quant::Q6_K => ("q6k", "q6k_moe_qdot", "q6k_moe_gateup", "q6k_moe_qgemm", 3),
        other => bail!("no expert kernels for {} experts", other.name()),
    };
    Ok(Kernels { prefix, qdot, gateup, qgemm, resident })
}

/// A row's expert the host computes: its rank among the row's eight and
/// its id.
struct Miss {
    rank: usize,
    expert: usize,
}

/// What a decode row owes the host: its misses and the row itself.
struct Owed {
    misses: Vec<Miss>,
    x: Vec<f32>,
}

impl DeviceBackend {
    pub(in super::super) fn run_moe(&self, req: Moe) -> Result<()> {
        ensure!(req.n_used == MOE_USED, "the kernels route {MOE_USED} experts a token, the model {}", req.n_used);
        ensure!(req.rows <= MAX_ROWS, "a mixture-of-experts pass takes at most {MAX_ROWS} rows, got {}", req.rows);
        let (d, d_ff, rows, block) = (req.d_model, req.d_ff, req.rows, req.experts.0);
        let mut experts = self.experts.borrow_mut();
        self.ensure_slabs(&mut experts)?;
        let set = experts.blocks.get(block).ok_or_else(|| anyhow::anyhow!("use of an unknown expert set handle"))?.set.clone();
        ensure!(
            set.gate.n() == d_ff && set.gate.k() == d && set.down.n() == d && set.down.k() == d_ff,
            "the expert set does not match the request's widths"
        );
        let Some(shared) = req.shared else {
            bail!("the device mixture-of-experts path expects the shared expert")
        };
        ensure!(d.is_multiple_of(MOE_COMBINE_TN), "d_model {d} is not a multiple of the combine tile {MOE_COMBINE_TN}");

        // A pass of several rows evicts within a block, so a later row's
        // copies can land on a slot an earlier row's kernels read: fine in
        // stream order, not in a segment replayed at the block's end.
        if rows > 1 {
            self.flush_pending()?;
        }
        self.route(&mut experts, &req)?;

        // Grouped once the rows choose more experts than the block holds,
        // where the row path's cache would thrash.
        let grouped = self.moe_grouped && rows * MOE_USED > experts.per_block && super::grouped::can_group(&set, d, d_ff);
        if grouped {
            self.run_grouped(&mut experts, &req, shared)?;
        } else {
            self.run_rows(&mut experts, &req, shared)?;
        }
        if let Some(routes) = req.routes {
            // Past the sync point nothing recorded reads this buffer.
            let ids: Vec<f32> = experts.blocks[block].topk.host()[..rows * MOE_USED].iter().map(|&id| id as f32).collect();
            self.write_now(routes, &ids)?;
        }
        Ok(())
    }

    /// The router's top-k over the rows, the sync point where its choices
    /// come back, and the lookahead's prefetch of the next block.
    fn route(&self, experts: &mut Experts, req: &Moe) -> Result<()> {
        let (rows, block) = (req.rows, req.experts.0);
        let b = &experts.blocks[block];
        let iota = experts.iota[&req.n_expert].as_device_ptr().as_raw();
        self.launch_topk(self.ptr(req.logits, 0)?, rows, req.n_expert, iota, b.topk.dev(), b.weights.dev())?;
        let next = self.lookahead(experts, req, iota)?;
        self.sync_point()?;
        let b = &mut experts.blocks[block];
        if let Some(event) = b.fetched.take() {
            self.stream.wait_event(event, StreamWaitEventFlags::DEFAULT)?;
        }
        b.topk.pull(rows * MOE_USED)?;
        if let Some(next) = next {
            self.prefetch(experts, next)?;
        }
        Ok(())
    }

    /// The rows one at a time: each row's experts into slots, then the
    /// slot matvecs, the SwiGLU, the down matvec and the combine over its
    /// eight.
    fn run_rows(&self, experts: &mut Experts, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let (d, d_ff, rows, block) = (req.d_model, req.d_ff, req.rows, req.experts.0);
        let (shared_out, gate_logit) = shared;
        let act = req.act.map_or_else(|| self.quantize_act(req.x, rows, d), Ok)?;
        let (qa, das) = self.act_ptrs(act)?;
        let bufs = [self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d)?];
        let [gate_out, up_out, h, down_out] = bufs;
        let b = &experts.blocks[block];
        let joint = b.set.gate.quant() == b.set.up.quant();
        let host_misses = self.moe_host_decode && rows == 1 && simd::supports(&*b.set);
        let (slot_ptr, w_ptr) = (b.slots.dev(), b.weights.dev());
        let mut owed = Vec::new();
        for r in 0..rows {
            let misses = self.place_row(experts, block, r, host_misses, rows > 1)?;
            if host_misses {
                // A row with no misses still owes a zero row, so every step
                // records the same launches and its graphs stay cached.
                let x = self.read_row(experts, block, req.x, d, !misses.is_empty())?;
                owed.push(Owed { misses, x });
            }
            let row_slots = slot_ptr + (r * ROW_BYTES) as u64;
            let (row_qa, row_das) = (qa + (r * d) as u64, das + (r * d / 32 * 4) as u64);
            if joint {
                self.slot_gateup(experts, block, row_qa, row_das, d, row_slots, d_ff, h)?;
            } else {
                self.slot_matvec(experts, block, Stack::Gate, row_qa, row_das, 1, d, row_slots, d_ff, gate_out)?;
                self.slot_matvec(experts, block, Stack::Up, row_qa, row_das, 1, d, row_slots, d_ff, up_out)?;
                let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
                self.swiglu_planes(plane(gate_out), plane(up_out), h, MOE_USED, d_ff)?;
            }
            let (hqa, hdas) = self.act_ptrs(self.quantize_act(h, MOE_USED, d_ff)?)?;
            self.slot_matvec(experts, block, Stack::Down, hqa, hdas, MOE_USED, d_ff, row_slots, d, down_out)?;
            self.with_kernel(&self.moe_combine, (), "moe_combine", moe_combine_src, |module| {
                self.launch(
                    module,
                    "moe_combine",
                    &[
                        (w_ptr + (r * ROW_BYTES) as u64, [1, U]),
                        (self.ptr(down_out, 0)?, [U, d as i64]),
                        (self.ptr(gate_logit, r)?, [1, 1]),
                        (self.ptr(shared_out, r * d)?, [1, d as i64]),
                        (self.ptr(req.dest, r * d)?, [1, d as i64]),
                    ],
                    ((d / MOE_COMBINE_TN) as u32, 1, 1),
                )
            })?;
        }
        for buf in bufs {
            self.release(buf);
        }
        // The hits' kernels go to the device now rather than at the next
        // sync point, so it works while the host does.
        if host_misses {
            self.replay_segment()?;
        }
        for Owed { misses, x } in owed {
            let y = self.host_misses(experts, block, &x, &misses)?;
            let b = &mut experts.blocks[block];
            b.y_row.as_mut_slice()[..d].copy_from_slice(&y);
            self.add_row(req.dest, &b.y_row, d)?;
        }
        Ok(())
    }

    /// The router's top-k over `rows` rows of `logits`.
    fn launch_topk(&self, logits: u64, rows: usize, n_expert: usize, iota: u64, topk: u64, w: u64) -> Result<()> {
        self.with_kernel(&self.moe_topk, n_expert, "moe_topk", || moe_topk_src(n_expert), |module| {
            self.launch(
                module,
                "moe_topk",
                &[(logits, [rows as i64, n_expert as i64]), (iota, [1, n_expert as i64]), (topk, [rows as i64, U]), (w, [rows as i64, U])],
                (rows as u32, 1, 1),
            )
        })
    }

    /// The next block's router on the residual as it stands, into that
    /// block's lookahead table, to be read at this block's sync point.
    /// Returns the block it predicted for.
    fn lookahead(&self, experts: &Experts, req: &Moe, iota: u64) -> Result<Option<usize>> {
        let look = match req.lookahead {
            Some(look) if self.moe_lookahead && req.rows == 1 => look,
            _ => return Ok(None),
        };
        let d = req.d_model;
        let scratch = self.alloc(d)?;
        let logits = self.alloc(req.n_expert)?;
        self.rms_norm(req.dest, 1, d, look.gain, look.eps, scratch)?;
        self.matmul(scratch, 1, d, look.router, req.n_expert, logits)?;
        let n = &experts.blocks[look.experts.0];
        self.launch_topk(self.ptr(logits, 0)?, 1, req.n_expert, iota, n.look.dev(), n.look_w.as_device_ptr().as_raw())?;
        self.release(logits);
        self.release(scratch);
        Ok(Some(look.experts.0))
    }

    /// Block `next`'s predicted experts into its slots on the copy stream.
    /// A wrong prediction costs a copy and a slot.
    fn prefetch(&self, experts: &mut Experts, next: usize) -> Result<()> {
        let b = &mut experts.blocks[next];
        let n = b.set.count() as i32;
        let mut predicted = [0usize; MOE_USED];
        for (p, &id) in predicted.iter_mut().zip(b.look.pull(MOE_USED)?) {
            ensure!((0..n).contains(&id), "the lookahead chose expert {id} of {n}");
            *p = id as usize;
        }
        let tick = experts.step(next, &predicted);
        experts.fetch(next, predicted, tick, &self.copy_stream)
    }

    /// Row `r`'s experts into slots and its table row published, all on
    /// the stream ahead of its kernels. With `host_misses`, a miss is left
    /// for the host with the zero slot standing in, and copied into a slot
    /// on the second stream for the next token. `prompt` counts the lookups
    /// as a prompt pass's.
    fn place_row(&self, experts: &mut Experts, block: usize, r: usize, host_misses: bool, prompt: bool) -> Result<Vec<Miss>> {
        let chosen = experts.blocks[block].chosen(r)?;
        let tick = experts.step(block, &chosen);
        let mut slots = [0i32; MOE_USED];
        let mut misses = Vec::new();
        for (rank, &expert) in chosen.iter().enumerate() {
            let slot = if host_misses && !experts.resident(block, expert) {
                experts.stats.misses += 1;
                misses.push(Miss { rank, expert });
                experts.per_block
            } else {
                experts.place(block, expert, tick, &self.stream, prompt)?.ok_or_else(|| anyhow::anyhow!("every slot of the block is in use by this row"))?
            };
            slots[rank] = slot as i32;
        }
        experts.blocks[block].slots.push(r * MOE_USED, &slots, &self.stream)?;
        if host_misses {
            experts.fetch(block, misses.iter().map(|m| m.expert), tick, &self.copy_stream)?;
        }
        Ok(misses)
    }

    /// A decode row `x` back to the host through the block's pinned row,
    /// while the stream is idle from the sync point; zeros when `wanted`
    /// is not.
    fn read_row(&self, experts: &mut Experts, block: usize, x: Buf, d: usize, wanted: bool) -> Result<Vec<f32>> {
        if !wanted {
            return Ok(vec![0.0; d]);
        }
        let row = &mut experts.blocks[block].x_row;
        // SAFETY: the row is `d` floats on the device and the pinned row
        // holds `d`.
        cuda_ok(
            unsafe { cust::sys::cuMemcpyDtoH_v2(row.as_mut_slice().as_mut_ptr().cast(), self.ptr(x, 0)?, d * size_of::<f32>()) },
            "reading the row",
        )?;
        Ok(row.as_slice()[..d].to_vec())
    }

    /// The misses' share of a decode row `x`, computed on the host from the
    /// mirror's bytes and weighted by the router, summed into a `[d]` row
    /// for the device to add.
    fn host_misses(&self, experts: &mut Experts, block: usize, x: &[f32], misses: &[Miss]) -> Result<Vec<f32>> {
        let mut y = vec![0.0f32; x.len()];
        if misses.is_empty() {
            return Ok(y);
        }
        let started = std::time::Instant::now();
        let b = &mut experts.blocks[block];
        let weights = b.weights.pull(MOE_USED)?;
        let jobs: Vec<(usize, f32)> = misses.iter().map(|m| (m.expert, weights[m.rank])).collect();
        simd::experts_row(&b.mirror.source(&b.set), &jobs, x, &mut y)?;
        experts.stats.cpu_misses += misses.len() as u64;
        experts.stats.cpu_nanos += started.elapsed().as_nanos() as u64;
        Ok(y)
    }

    /// One slot matvec over the eight slots `row_slots` names in `block`'s
    /// `stack` slab: `n` outputs of `k` inputs, the activation `rows` rows
    /// of which a program reads row zero (one row) or its own (eight).
    #[allow(clippy::too_many_arguments)]
    fn slot_matvec(&self, experts: &Experts, block: usize, stack: Stack, qa: u64, das: u64, rows: usize, k: usize, row_slots: u64, n: usize, out: Buf) -> Result<()> {
        let b = &experts.blocks[block];
        ensure!(experts.slab(block, stack).n == n && n.is_multiple_of(MOE_QDOT_TN), "a slot matvec of {n} outputs does not fit the slab's rows or the tile");
        let kernels = kernels(b.set.stack(stack).quant())?;
        let act_row = if rows == 1 { "0" } else { "j" };
        let [bytes, d] = experts.slab_operands(block, stack);
        self.with_kernel(&self.moe_qdot, (kernels.prefix, n, rows == 1), "moe_qdot", || moe_qdot_src(kernels.prefix, n, act_row, kernels.resident), |module| {
            self.launch(
                module,
                kernels.qdot,
                &[(qa, [rows as i64, k as i64]), (das, [rows as i64, (k / 32) as i64]), (row_slots, [1, U]), bytes, d, (self.ptr(out, 0)?, [U, n as i64])],
                ((n / MOE_QDOT_TN) as u32, MOE_USED as u32, 1),
            )
        })
    }

    /// Gate, up and the SwiGLU of the eight slots `row_slots` names, into
    /// `h` (`[8, n]`), for a block whose gate and up share a format.
    #[allow(clippy::too_many_arguments)]
    fn slot_gateup(&self, experts: &Experts, block: usize, qa: u64, das: u64, k: usize, row_slots: u64, n: usize, h: Buf) -> Result<()> {
        let b = &experts.blocks[block];
        ensure!(
            experts.slab(block, Stack::Gate).n == n && experts.slab(block, Stack::Up).n == n && n.is_multiple_of(MOE_QDOT_TN),
            "a joint gate and up of {n} outputs does not fit the slabs or the tile"
        );
        let kernels = kernels(b.set.gate.quant())?;
        let [gb, gd] = experts.slab_operands(block, Stack::Gate);
        let [ub, ud] = experts.slab_operands(block, Stack::Up);
        self.with_kernel(&self.moe_gateup, (kernels.prefix, n), "moe_gateup", || moe_gateup_src(kernels.prefix, n, kernels.resident), |module| {
            self.launch(
                module,
                kernels.gateup,
                &[(qa, [1, k as i64]), (das, [1, (k / 32) as i64]), (row_slots, [1, U]), gb, gd, ub, ud, (self.ptr(h, 0)?, [U, n as i64])],
                ((n / MOE_QDOT_TN) as u32, MOE_USED as u32, 1),
            )
        })
    }

    /// Writes `data` into a pool buffer now, outside the recording: only
    /// safe right after a sync point.
    fn write_now(&self, buf: Buf, data: &[f32]) -> Result<()> {
        let slots = self.slots.borrow();
        let Some(super::super::mem::Slot::Owned(buffer)) = slots.get(buf.0).and_then(Option::as_ref) else {
            bail!("writing routes into a released handle or a constant")
        };
        ensure!(buffer.len() >= data.len(), "the routes buffer is too small");
        buffer.index(0..data.len()).copy_from(data)?;
        Ok(())
    }
}
