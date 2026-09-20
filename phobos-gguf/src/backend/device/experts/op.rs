// The device side of `Backend::moe`: top-k over every row, one sync point
// a block where the rows' choices come back, then per row the misses
// copied into their slots and the slot matvecs and combine that read them.

use anyhow::{Context, Result, bail, ensure};
use cust::event::{Event, EventFlags};
use cust::memory::CopyDestination;
use cust::stream::StreamWaitEventFlags;
use phobos_kernels::cuda_ok;

use super::super::kernels::{
    MOE_COMBINE_TN, MOE_QDOT_TN, MOE_USED, moe_combine_src, moe_gateup_src, moe_qdot_src,
    moe_topk_src,
};
use super::super::{DeviceBackend, Plane};
use super::{Experts, KINDS, Kind, MAX_ROWS, NONE};
use crate::backend::{Backend, Buf, Moe};
use crate::quant::Quant;

/// Bytes of one row's eight table entries.
const ROW_BYTES: usize = MOE_USED * size_of::<i32>();

/// Expert `e` of `stack` applied to `x`: each of its `n` rows decoded into
/// a scratch and dotted, on the calling thread.
fn dequant_matvec(stack: &crate::experts::ExpertStack, e: usize, x: &[f32]) -> Result<Vec<f32>> {
    let (n, k) = (stack.n(), stack.k());
    let block = stack.quant().spec().block;
    let block_bytes = stack.quant().spec().block_bytes;
    let rb = k / block * block_bytes;
    let bytes = stack.expert(e);
    let mut decoded = vec![0.0f32; k];
    let mut out = vec![0.0f32; n];
    for (j, o) in out.iter_mut().enumerate() {
        (stack.quant().spec().dequantize)(&bytes[j * rb..(j + 1) * rb], &mut decoded);
        *o = decoded.iter().zip(x).map(|(&w, &v)| w * v).sum();
    }
    Ok(out)
}

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
        // The next block's router on the residual as it stands, into that
        // block's lookahead table, read at the same sync point.
        let next = match (self.moe_lookahead && rows == 1, req.lookahead) {
            (true, Some(look)) => {
                let scratch = self.alloc(d)?;
                self.rms_norm(req.dest, 1, d, look.gain, look.eps, scratch)?;
                let (look_logits, look_topk, look_w) = {
                    let n = &experts.blocks[look.experts.0];
                    (
                        n.look_logits.as_device_ptr().as_raw(),
                        n.look_topk.as_device_ptr().as_raw(),
                        n.look_w.as_device_ptr().as_raw(),
                    )
                };
                // The matvec through the pool's handle so the logits land
                // in the block's own buffer: a temporary, then a copy.
                let logits = self.alloc(n_expert)?;
                self.matmul(scratch, 1, d, look.router, n_expert, logits)?;
                self.with_kernel(&self.moe_topk, n_expert, "moe_topk", || moe_topk_src(n_expert), |module| {
                    self.launch(
                        module,
                        "moe_topk",
                        &[
                            (self.ptr(logits, 0)?, [1, n_expert as i64]),
                            (iota, [1, n_expert as i64]),
                            (look_topk, [1, u]),
                            (look_w, [1, u]),
                        ],
                        (1, 1, 1),
                    )
                })?;
                let _ = look_logits;
                self.release(logits);
                self.release(scratch);
                Some(look.experts.0)
            }
            _ => None,
        };
        self.sync_point()?;
        // What the previous block predicted for this one is on the copy
        // stream; this block's copies and kernels queue behind it.
        if let Some(event) = experts.blocks[block].fetched.take() {
            self.stream.wait_event(event, StreamWaitEventFlags::DEFAULT)?;
        }
        if let Some(next) = next {
            self.prefetch(&mut experts, next, n_expert)?;
        }
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
        // Gate and up in one launch with the SwiGLU when they share a
        // format, which is what a file does unless it mixes them.
        let joint = {
            let set = &experts.blocks[block].set;
            set.gate.quant() == set.up.quant()
        };

        // The host's share of a decode step: the misses, computed there
        // from the mirror while the device sums the hits.
        let host_misses = self.moe_cpu_miss && rows == 1;
        for r in 0..rows {
            // The row's misses into slots, its table row published, both
            // asynchronous on the stream ahead of its kernels.
            let misses = self.place_row(&mut experts, block, r, n_expert, host_misses)?;
            if !misses.is_empty() {
                let y = self.host_misses(&mut experts, block, req.x, d, d_ff, &misses)?;
                let ybuf = self.upload(&y)?;
                self.add_into(req.dest, ybuf)?;
                self.release(ybuf);
            }
            let row_slots = slot_ptr + (r * ROW_BYTES) as u64;
            let (row_qa, row_das) = (qa + (r * d) as u64, das + (r * d / 32 * 4) as u64);
            if joint {
                self.slot_gateup(&experts, block, row_qa, row_das, d, row_slots, d_ff, h)?;
            } else {
                self.slot_matvec(&experts, block, Kind::Gate, row_qa, row_das, 1, d, row_slots, d_ff, gate_out)?;
                self.slot_matvec(&experts, block, Kind::Up, row_qa, row_das, 1, d, row_slots, d_ff, up_out)?;
                let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
                self.swiglu_planes(plane(gate_out), plane(up_out), h, MOE_USED, d_ff)?;
            }
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
    fn place_row(
        &self,
        experts: &mut Experts,
        block: usize,
        r: usize,
        n_expert: usize,
        host_misses: bool,
    ) -> Result<Vec<(usize, usize)>> {
        let mut for_host = Vec::new();
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
                    if std::mem::take(&mut b.prefetched[s as usize]) {
                        stats.prefetch_hits += 1;
                    }
                    s as usize
                }
                _ if host_misses => {
                    // The zero slot past the block's share; the host adds
                    // this expert's share of the row itself.
                    stats.misses += 1;
                    for_host.push((j, e));
                    per_block
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
        Ok(for_host)
    }

    /// The misses' share of a decode row, computed on the host: for each
    /// `(rank, expert)`, the expert decoded from the mirror's file bytes and
    /// applied to the row, weighted by the router's weight of that rank,
    /// summed into a `[d]` row for the device to add. One thread an expert
    /// and matrix.
    fn host_misses(
        &self,
        experts: &mut Experts,
        block: usize,
        x: Buf,
        d: usize,
        d_ff: usize,
        misses: &[(usize, usize)],
    ) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        let mut row = vec![0.0f32; d];
        self.read(x, &mut row)?;
        let mut weights = [0.0f32; MOE_USED];
        experts.blocks[block].weights.index(0..MOE_USED).copy_to(&mut weights[..])?;
        let set = std::sync::Arc::clone(&experts.blocks[block].set);
        let parts: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = misses
                .iter()
                .map(|&(j, e)| {
                    let (set, row) = (&set, &row);
                    let w = weights[j];
                    scope.spawn(move || -> Result<Vec<f32>> {
                        // Gate and up on two threads, the down after them.
                        let (g, u) = std::thread::scope(|inner| {
                            let gate = inner.spawn(|| dequant_matvec(&set.gate, e, row));
                            let up = inner.spawn(|| dequant_matvec(&set.up, e, row));
                            (gate.join().expect("gate thread"), up.join().expect("up thread"))
                        });
                        let (g, u) = (g?, u?);
                        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
                        let mut y = dequant_matvec(&set.down, e, &h)?;
                        for v in &mut y {
                            *v *= w;
                        }
                        Ok(y)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("expert thread")).collect::<Result<Vec<_>>>()
        })?;
        let mut y = vec![0.0f32; d];
        for part in parts {
            for (acc, v) in y.iter_mut().zip(part) {
                *acc += v;
            }
        }
        debug_assert_eq!(d_ff, set.gate.n());
        experts.stats.cpu_misses += misses.len() as u64;
        experts.stats.cpu_nanos += started.elapsed().as_nanos() as u64;
        Ok(y)
    }

    /// Block `next`'s predicted experts into its slots on the copy stream:
    /// the ones not resident, into its least recently used slots, with an
    /// event the block waits on before it runs. A wrong prediction costs a
    /// copy and a slot; the stamp it gets is the current tick, so a right
    /// one is not the next victim.
    fn prefetch(&self, experts: &mut Experts, next: usize, n_expert: usize) -> Result<()> {
        let Experts { blocks, slabs, per_block, tick, stats, .. } = &mut *experts;
        *tick += 1;
        let (tick, per_block) = (*tick, *per_block);
        let slabs = slabs.as_ref().expect("ensured by the caller");
        let b = &mut blocks[next];
        b.look_topk.copy_to(&mut b.look_host.as_mut_slice()[..MOE_USED])?;
        let mut predicted = [0usize; MOE_USED];
        for (j, &id) in b.look_host.as_slice()[..MOE_USED].iter().enumerate() {
            ensure!((0..n_expert as i32).contains(&id), "the lookahead chose expert {id} of {n_expert}");
            predicted[j] = id as usize;
        }
        for &e in &predicted {
            let s = b.slot_of[e];
            if s != NONE {
                b.held[s as usize].1 = tick;
            }
        }
        let mut copied = false;
        for &e in &predicted {
            if b.slot_of[e] != NONE {
                continue;
            }
            let Some(victim) = (0..per_block).filter(|&s| b.held[s].1 != tick).min_by_key(|&s| b.held[s].1) else {
                break;
            };
            let (old, _) = b.held[victim];
            if old != NONE {
                b.slot_of[old as usize] = NONE;
            }
            let src = b.mirror.expert(e);
            for (k, kind) in KINDS.into_iter().enumerate() {
                let slab = &slabs[&(kind, kind.stack(&b.set).quant())];
                let (bytes_at, d_at) = slab.at(b.base[k] + victim);
                for (dst, (ptr, len)) in [(bytes_at, src[k]), (d_at, src[3 + k])] {
                    // SAFETY: the mirror is pinned and outlives the copy;
                    // the slot is inside the slab; nothing reads the slot
                    // until the event below.
                    cuda_ok(
                        unsafe { cust::sys::cuMemcpyHtoDAsync_v2(dst, ptr, len, self.copy_stream.as_inner()) },
                        "prefetching an expert into its slot",
                    )?;
                    stats.bytes += len as u64;
                }
            }
            b.held[victim] = (e as u32, tick);
            b.slot_of[e] = victim as u32;
            b.prefetched[victim] = true;
            stats.prefetches += 1;
            copied = true;
        }
        if copied {
            let event = Event::new(EventFlags::DISABLE_TIMING)?;
            event.record(&self.copy_stream)?;
            b.fetched = Some(event);
        }
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

    /// Gate, up and the SwiGLU of the eight slots `row_slots` names, into
    /// `h` (`[8, n]`), for a block whose gate and up share a format.
    #[allow(clippy::too_many_arguments)]
    fn slot_gateup(
        &self,
        experts: &Experts,
        block: usize,
        qa: u64,
        das: u64,
        k: usize,
        row_slots: u64,
        n: usize,
        h: Buf,
    ) -> Result<()> {
        let (gate, up) = (experts.slab(block, Kind::Gate), experts.slab(block, Kind::Up));
        let b = &experts.blocks[block];
        let quant = b.set.gate.quant();
        ensure!(
            gate.n == n && up.n == n && n.is_multiple_of(MOE_QDOT_TN),
            "a joint gate and up of {n} outputs does not fit the slabs or the tile"
        );
        let (name, function, resident) = match quant {
            Quant::Q4_K => ("q4k", "q4k_moe_gateup", 4),
            Quant::Q5_K => ("q5k", "q5k_moe_gateup", 3),
            Quant::Q6_K => ("q6k", "q6k_moe_gateup", 3),
            other => bail!("no slot matvec for {} experts", other.name()),
        };
        let u = MOE_USED as i64;
        let ns = (experts.per_block * n) as i64;
        let (gb, gd) = gate.at(b.base[Kind::Gate as usize]);
        let (ub, ud) = up.at(b.base[Kind::Up as usize]);
        self.with_kernel(&self.moe_gateup, (name, n), "moe_gateup", || moe_gateup_src(name, n, resident), |module| {
            self.launch(
                module,
                function,
                &[
                    (qa, [1, k as i64]),
                    (das, [1, (k / 32) as i64]),
                    (row_slots, [1, u]),
                    (gb, [ns, gate.rb as i64]),
                    (gd, [ns, gate.nb as i64]),
                    (ub, [ns, up.rb as i64]),
                    (ud, [ns, up.nb as i64]),
                    (self.ptr(h, 0)?, [u, n as i64]),
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
