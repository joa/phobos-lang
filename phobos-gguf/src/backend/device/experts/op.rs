// The device side of `Backend::moe`: top-k over every row, one sync point
// a block where the rows' choices come back, then per row the misses
// copied into their slots and the slot matvecs and combine that read them.

use anyhow::{Context, Result, bail, ensure};
use cust::memory::CopyDestination;
use phobos_kernels::cuda_ok;

use super::super::kernels::{
    MOE_COMBINE_TN, MOE_QDOT_TN, MOE_USED, moe_combine_src, moe_qdot_src, moe_topk_src,
};
use super::super::{DeviceBackend, Plane};
use super::{Experts, KINDS, Kind, MAX_ROWS, NONE};
use crate::backend::{Backend, Buf, Moe};
use crate::quant::Quant;

/// Bytes of one row's eight table entries.
const ROW_BYTES: usize = MOE_USED * size_of::<i32>();

impl DeviceBackend {
    pub(in super::super) fn run_moe(&self, req: Moe) -> Result<()> {
        ensure!(
            req.n_used == MOE_USED,
            "the kernels are written for {MOE_USED} experts a token, the model routes {}",
            req.n_used
        );
        ensure!(req.rows <= MAX_ROWS, "a mixture-of-experts pass takes at most {MAX_ROWS} rows, got {}", req.rows);
        let (d, d_ff, n_expert, rows) = (req.d_model, req.d_ff, req.n_expert, req.rows);
        let mut experts = self.experts.borrow_mut();
        self.ensure_slabs(&mut experts)?;
        let block = req.experts.0;
        ensure!(block < experts.blocks.len(), "use of an unknown expert set handle");
        {
            let set = &experts.blocks[block].set;
            ensure!(
                set.gate.n() == d_ff && set.gate.k() == d && set.down.n() == d && set.down.k() == d_ff,
                "the expert set does not match the request's widths"
            );
        }
        let Some((shared, gate_logit)) = req.shared else {
            bail!("the device mixture-of-experts path expects the shared expert")
        };
        ensure!(
            d.is_multiple_of(MOE_COMBINE_TN),
            "d_model {d} is not a multiple of the combine tile {MOE_COMBINE_TN}"
        );

        // A pass of several rows evicts within a block, so a later row's
        // copies can land on a slot an earlier row's kernels read. Issued
        // in stream order that is fine; recorded and replayed at the
        // segment's end it is not, so such a pass runs eagerly from here.
        // A prompt pass is not replayed anyway.
        if rows > 1 {
            self.flush_pending()?;
        }

        // Every row's choice in one launch, then the block's sync point.
        let (iota, topk_ptr, w_ptr, slot_ptr) = {
            let b = &experts.blocks[block];
            (
                experts.iota[&n_expert].as_device_ptr().as_raw(),
                b.topk.as_device_ptr().as_raw(),
                b.weights.as_device_ptr().as_raw(),
                b.slot_table.as_device_ptr().as_raw(),
            )
        };
        let u = MOE_USED as i64;
        self.with_kernel(&self.moe_topk, n_expert, "moe_topk", || moe_topk_src(n_expert), |module| {
            self.launch(
                module,
                "moe_topk",
                &[
                    (self.ptr(req.logits, 0)?, [rows as i64, n_expert as i64]),
                    (iota, [1, n_expert as i64]),
                    (topk_ptr, [rows as i64, u]),
                    (w_ptr, [rows as i64, u]),
                ],
                (rows as u32, 1, 1),
            )
        })?;
        self.sync_point()?;
        {
            let b = &mut experts.blocks[block];
            let wanted = rows * MOE_USED;
            b.topk.index(0..wanted).copy_to(&mut b.topk_host.as_mut_slice()[..wanted])?;
        }

        // The activation, quantized once for every row.
        let act = req.act.map_or_else(|| self.quantize_act(req.x, rows, d), Ok)?;
        let (qa, das) = self.act_ptrs(act)?;
        let gate_out = self.alloc(MOE_USED * d_ff)?;
        let up_out = self.alloc(MOE_USED * d_ff)?;
        let h = self.alloc(MOE_USED * d_ff)?;
        let down_out = self.alloc(MOE_USED * d)?;

        for r in 0..rows {
            // The row's misses into slots, its table row published, both
            // asynchronous on the stream ahead of its kernels.
            self.place_row(&mut experts, block, r, n_expert)?;
            let row_slots = slot_ptr + (r * ROW_BYTES) as u64;
            let (row_qa, row_das) = (qa + (r * d) as u64, das + (r * d / 32 * 4) as u64);
            self.slot_matvec(&experts, block, Kind::Gate, row_qa, row_das, 1, d, row_slots, d_ff, gate_out)?;
            self.slot_matvec(&experts, block, Kind::Up, row_qa, row_das, 1, d, row_slots, d_ff, up_out)?;
            let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
            self.swiglu_planes(plane(gate_out), plane(up_out), h, MOE_USED, d_ff)?;
            let hact = self.quantize_act(h, MOE_USED, d_ff)?;
            let (hqa, hdas) = self.act_ptrs(hact)?;
            self.slot_matvec(&experts, block, Kind::Down, hqa, hdas, MOE_USED, d_ff, row_slots, d, down_out)?;
            self.with_kernel(&self.moe_combine, (), "moe_combine", moe_combine_src, |module| {
                self.launch(
                    module,
                    "moe_combine",
                    &[
                        (w_ptr + (r * ROW_BYTES) as u64, [1, u]),
                        (self.ptr(down_out, 0)?, [u, d as i64]),
                        (self.ptr(gate_logit, r)?, [1, 1]),
                        (self.ptr(shared, r * d)?, [1, d as i64]),
                        (self.ptr(req.dest, r * d)?, [1, d as i64]),
                    ],
                    ((d / MOE_COMBINE_TN) as u32, 1, 1),
                )
            })?;
        }
        if let Some(routes) = req.routes {
            // After the sync point every earlier launch has run, and
            // nothing recorded since reads this buffer.
            let ids: Vec<f32> = experts.blocks[block].topk_host.as_slice()[..rows * MOE_USED]
                .iter()
                .map(|&id| id as f32)
                .collect();
            self.write_now(routes, &ids)?;
        }
        for buf in [gate_out, up_out, h, down_out] {
            self.release(buf);
        }
        Ok(())
    }

    /// Row `r`'s eight experts into slots: hits stamped, misses copied from
    /// the mirror into the least recently used slots the row does not
    /// itself read, the row's slot table published, all on the stream.
    fn place_row(&self, experts: &mut Experts, block: usize, r: usize, n_expert: usize) -> Result<()> {
        let Experts { blocks, slabs, per_block, tick, stats, .. } = &mut *experts;
        *tick += 1;
        let (tick, per_block) = (*tick, *per_block);
        let slabs = slabs.as_ref().expect("ensured by the caller");
        let b = &mut blocks[block];
        let mut chosen = [0usize; MOE_USED];
        for (j, &id) in b.topk_host.as_slice()[r * MOE_USED..(r + 1) * MOE_USED].iter().enumerate() {
            ensure!((0..n_expert as i32).contains(&id), "the router chose expert {id} of {n_expert}");
            chosen[j] = id as usize;
        }
        // Hits first, stamped, so no victim below is one of this row's
        // experts.
        for &e in &chosen {
            let s = b.slot_of[e];
            if s != NONE {
                b.held[s as usize].1 = tick;
            }
        }
        let mut slots = [0i32; MOE_USED];
        for (j, &e) in chosen.iter().enumerate() {
            let local = match b.slot_of[e] {
                s if s != NONE => {
                    stats.hits += 1;
                    s as usize
                }
                _ => {
                    stats.misses += 1;
                    let victim = (0..per_block)
                        .filter(|&s| b.held[s].1 != tick)
                        .min_by_key(|&s| b.held[s].1)
                        .context("every slot of the block is in use by this row")?;
                    let (old, _) = b.held[victim];
                    if old != NONE {
                        b.slot_of[old as usize] = NONE;
                    }
                    let src = b.mirror.expert(e);
                    for (k, kind) in KINDS.into_iter().enumerate() {
                        let slab = &slabs[&(kind, kind.stack(&b.set).quant())];
                        let (bytes_at, d_at) = slab.at(b.base[k] + victim);
                        ensure!(
                            src[k].1 == slab.slot_bytes() && src[3 + k].1 == slab.n * slab.nb * 2,
                            "mirror and slab disagree on an expert's size: {} against {}",
                            src[k].1,
                            slab.slot_bytes()
                        );
                        for (dst, (ptr, len)) in [(bytes_at, src[k]), (d_at, src[3 + k])] {
                            // SAFETY: the mirror is pinned and outlives the
                            // copy; the slot is inside the slab.
                            cuda_ok(
                                unsafe { cust::sys::cuMemcpyHtoDAsync_v2(dst, ptr, len, self.stream.as_inner()) },
                                "copying an expert into its slot",
                            )?;
                            stats.bytes += len as u64;
                        }
                    }
                    b.held[victim] = (e as u32, tick);
                    b.slot_of[e] = victim as u32;
                    victim
                }
            };
            slots[j] = local as i32;
        }
        // The table row, from its pinned mirror so the copy needs no wait:
        // the row's region is not written again before the next sync point.
        b.slot_host.as_mut_slice()[r * MOE_USED..(r + 1) * MOE_USED].copy_from_slice(&slots);
        let src = b.slot_host.as_slice()[r * MOE_USED..].as_ptr();
        let dst = b.slot_table.as_device_ptr().as_raw() + (r * ROW_BYTES) as u64;
        // SAFETY: both sides are live buffers of at least `ROW_BYTES` past
        // the offsets; the host side is page-locked.
        cuda_ok(
            unsafe { cust::sys::cuMemcpyHtoDAsync_v2(dst, src.cast(), ROW_BYTES, self.stream.as_inner()) },
            "publishing a row's slot table",
        )?;
        Ok(())
    }

    /// One slot matvec over the eight slots `row_slots` names in `block`'s
    /// `kind` slab: `n` outputs of `k` inputs each, the activation `rows`
    /// rows of which a program reads row zero (one row) or its own (eight).
    /// The slab operand starts at the block's base, so the table's entries
    /// are local to the block.
    #[allow(clippy::too_many_arguments)]
    fn slot_matvec(
        &self,
        experts: &Experts,
        block: usize,
        kind: Kind,
        qa: u64,
        das: u64,
        rows: usize,
        k: usize,
        row_slots: u64,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let slab = experts.slab(block, kind);
        let b = &experts.blocks[block];
        let quant = kind.stack(&b.set).quant();
        ensure!(
            slab.n == n && n.is_multiple_of(MOE_QDOT_TN),
            "a slot matvec of {n} outputs does not fit the slab's {} rows or the tile",
            slab.n
        );
        let (name, function, resident) = match quant {
            Quant::Q4_K => ("q4k", "q4k_moe_qdot", 4),
            Quant::Q5_K => ("q5k", "q5k_moe_qdot", 3),
            Quant::Q6_K => ("q6k", "q6k_moe_qdot", 3),
            other => bail!("no slot matvec for {} experts", other.name()),
        };
        let act_row = if rows == 1 { "0" } else { "j" };
        let key = (name, n, rows == 1);
        let u = MOE_USED as i64;
        let ns = (experts.per_block * slab.n) as i64;
        let (bytes_at, d_at) = slab.at(b.base[kind as usize]);
        self.with_kernel(&self.moe_qdot, key, "moe_qdot", || moe_qdot_src(name, n, act_row, resident), |module| {
            self.launch(
                module,
                function,
                &[
                    (qa, [rows as i64, k as i64]),
                    (das, [rows as i64, (k / 32) as i64]),
                    (row_slots, [1, u]),
                    (bytes_at, [ns, slab.rb as i64]),
                    (d_at, [ns, slab.nb as i64]),
                    (self.ptr(out, 0)?, [u, n as i64]),
                ],
                ((n / MOE_QDOT_TN) as u32, MOE_USED as u32, 1),
            )
        })
    }

    /// Writes `data` into a pool buffer now, outside the recording. Only
    /// safe right after a sync point, when nothing recorded earlier is
    /// still to run against it.
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
