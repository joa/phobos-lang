// The device side of `Backend::moe`: top-k over every row, one sync point
// a block where the rows' choices come back, then per row the misses into
// their slots and the slot matvecs and combine that read them.

use anyhow::{Result, bail, ensure};
use cust::event::{Event, EventFlags};
use cust::memory::CopyDestination;
use cust::stream::StreamWaitEventFlags;

use super::super::kernels::{
    MOE_COMBINE_TN, MOE_QDOT_TN, MOE_USED, moe_combine_src, moe_gateup_src, moe_qdot_src,
    moe_topk_src,
};
use super::super::{DeviceBackend, Plane};
use super::{Experts, Kind, MAX_ROWS};
use crate::backend::{Backend, Buf, Moe};
use crate::experts::ExpertStack;
use crate::quant::Quant;

const U: i64 = MOE_USED as i64;

/// Bytes of one row's eight table entries.
const ROW_BYTES: usize = MOE_USED * size_of::<i32>();

/// A format's kernel prefix, its two slot kernels' names, and its register
/// budget's CTA count.
pub(super) fn kernel_for(quant: Quant) -> Result<(&'static str, [&'static str; 3], usize)> {
    Ok(match quant {
        Quant::Q4_K => ("q4k", ["q4k_moe_qdot", "q4k_moe_gateup", "q4k_moe_qgemm"], 4),
        Quant::Q5_K => ("q5k", ["q5k_moe_qdot", "q5k_moe_gateup", "q5k_moe_qgemm"], 3),
        Quant::Q6_K => ("q6k", ["q6k_moe_qdot", "q6k_moe_gateup", "q6k_moe_qgemm"], 3),
        other => bail!("no expert kernels for {} experts", other.name()),
    })
}

impl DeviceBackend {
    pub(in super::super) fn run_moe(&self, req: Moe) -> Result<()> {
        ensure!(req.n_used == MOE_USED, "the kernels route {MOE_USED} experts a token, the model {}", req.n_used);
        ensure!(req.rows <= MAX_ROWS, "a mixture-of-experts pass takes at most {MAX_ROWS} rows, got {}", req.rows);
        let (d, d_ff, n_expert, rows, block) = (req.d_model, req.d_ff, req.n_expert, req.rows, req.experts.0);
        let mut experts = self.experts.borrow_mut();
        self.ensure_slabs(&mut experts)?;
        let set = &experts.blocks.get(block).ok_or_else(|| anyhow::anyhow!("use of an unknown expert set handle"))?.set;
        ensure!(
            set.gate.n() == d_ff && set.gate.k() == d && set.down.n() == d && set.down.k() == d_ff,
            "the expert set does not match the request's widths"
        );
        let joint = set.gate.quant() == set.up.quant();
        let Some((shared, gate_logit)) = req.shared else {
            bail!("the device mixture-of-experts path expects the shared expert")
        };
        ensure!(d.is_multiple_of(MOE_COMBINE_TN), "d_model {d} is not a multiple of the combine tile {MOE_COMBINE_TN}");

        // A pass of several rows evicts within a block, so a later row's
        // copies can land on a slot an earlier row's kernels read: fine in
        // stream order, not in a segment replayed at the block's end.
        if rows > 1 {
            self.flush_pending()?;
        }

        let (iota, topk_ptr, w_ptr) = {
            let b = &experts.blocks[block];
            (experts.iota[&n_expert].as_device_ptr().as_raw(), b.topk.as_device_ptr().as_raw(), b.weights.as_device_ptr().as_raw())
        };
        self.launch_topk(self.ptr(req.logits, 0)?, rows, n_expert, iota, topk_ptr, w_ptr)?;
        let next = self.lookahead(&experts, &req, iota)?;
        self.sync_point()?;
        if let Some(event) = experts.blocks[block].fetched.take() {
            self.stream.wait_event(event, StreamWaitEventFlags::DEFAULT)?;
        }
        {
            let b = &mut experts.blocks[block];
            let wanted = rows * MOE_USED;
            b.topk.index(0..wanted).copy_to(&mut b.topk_host.as_mut_slice()[..wanted])?;
        }
        if let Some(next) = next {
            self.prefetch(&mut experts, next)?;
        }

        if self.moe_grouped && rows > 1 {
            self.run_grouped(&mut experts, &req, (shared, gate_logit))?;
        } else {
            let act = req.act.map_or_else(|| self.quantize_act(req.x, rows, d), Ok)?;
            let (qa, das) = self.act_ptrs(act)?;
            let bufs = [self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d_ff)?, self.alloc(MOE_USED * d)?];
            let [gate_out, up_out, h, down_out] = bufs;
            let host_misses = self.moe_cpu_miss && rows == 1;
            let slot_ptr = experts.blocks[block].slot_table.as_device_ptr().as_raw();
            for r in 0..rows {
                let misses = self.place_row(&mut experts, block, r, host_misses)?;
                if !misses.is_empty() {
                    let y = self.host_misses(&mut experts, block, req.x, &misses)?;
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
                let (hqa, hdas) = self.act_ptrs(self.quantize_act(h, MOE_USED, d_ff)?)?;
                self.slot_matvec(&experts, block, Kind::Down, hqa, hdas, MOE_USED, d_ff, row_slots, d, down_out)?;
                self.with_kernel(&self.moe_combine, (), "moe_combine", moe_combine_src, |module| {
                    self.launch(
                        module,
                        "moe_combine",
                        &[
                            (w_ptr + (r * ROW_BYTES) as u64, [1, U]),
                            (self.ptr(down_out, 0)?, [U, d as i64]),
                            (self.ptr(gate_logit, r)?, [1, 1]),
                            (self.ptr(shared, r * d)?, [1, d as i64]),
                            (self.ptr(req.dest, r * d)?, [1, d as i64]),
                        ],
                        ((d / MOE_COMBINE_TN) as u32, 1, 1),
                    )
                })?;
            }
            for buf in bufs {
                self.release(buf);
            }
        }
        if let Some(routes) = req.routes {
            // Past the sync point nothing recorded reads this buffer.
            let ids: Vec<f32> = experts.blocks[block].topk_host.as_slice()[..rows * MOE_USED].iter().map(|&id| id as f32).collect();
            self.write_now(routes, &ids)?;
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
        self.launch_topk(self.ptr(logits, 0)?, 1, req.n_expert, iota, n.look_topk.as_device_ptr().as_raw(), n.look_w.as_device_ptr().as_raw())?;
        self.release(logits);
        self.release(scratch);
        Ok(Some(look.experts.0))
    }

    /// Block `next`'s predicted experts into its slots on the copy stream,
    /// with an event the block waits on before it runs. A wrong prediction
    /// costs a copy and a slot.
    fn prefetch(&self, experts: &mut Experts, next: usize) -> Result<()> {
        let b = &mut experts.blocks[next];
        b.look_topk.copy_to(&mut b.look_host.as_mut_slice()[..MOE_USED])?;
        let n = b.set.count() as i32;
        let mut predicted = [0usize; MOE_USED];
        for (p, &id) in predicted.iter_mut().zip(b.look_host.as_slice()) {
            ensure!((0..n).contains(&id), "the lookahead chose expert {id} of {n}");
            *p = id as usize;
        }
        let tick = experts.step(next, &predicted);
        let mut copied = false;
        for e in predicted {
            if experts.resident(next, e) {
                continue;
            }
            let Some(victim) = experts.blocks[next].victim(tick) else {
                break;
            };
            experts.fill(next, e, victim, tick, &self.copy_stream)?;
            experts.mark_prefetched(next, victim);
            copied = true;
        }
        if copied {
            let event = Event::new(EventFlags::DISABLE_TIMING)?;
            event.record(&self.copy_stream)?;
            experts.blocks[next].fetched = Some(event);
        }
        Ok(())
    }

    /// Row `r`'s experts into slots and its table row published, all on
    /// the stream ahead of its kernels. With `host_misses`, a miss is left
    /// for the host and the zero slot stands in; the returned `(rank,
    /// expert)` pairs are what the host owes.
    fn place_row(&self, experts: &mut Experts, block: usize, r: usize, host_misses: bool) -> Result<Vec<(usize, usize)>> {
        let chosen = experts.blocks[block].chosen(r)?;
        let tick = experts.step(block, &chosen);
        let mut slots = [0i32; MOE_USED];
        let mut for_host = Vec::new();
        for (j, &e) in chosen.iter().enumerate() {
            let slot = if host_misses && !experts.resident(block, e) {
                experts.stats.misses += 1;
                for_host.push((j, e));
                experts.per_block
            } else {
                experts.place(block, e, tick, &self.stream)?.ok_or_else(|| anyhow::anyhow!("every slot of the block is in use by this row"))?
            };
            slots[j] = slot as i32;
        }
        experts.blocks[block].publish(r, &slots, &self.stream)?;
        Ok(for_host)
    }

    /// The misses' share of a decode row, computed on the host from the
    /// mirror's bytes and weighted by the router, summed into a `[d]` row
    /// for the device to add. One thread an expert, gate and up in parallel.
    fn host_misses(&self, experts: &mut Experts, block: usize, x: Buf, misses: &[(usize, usize)]) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        let d = experts.blocks[block].set.gate.k();
        let mut row = vec![0.0f32; d];
        self.read(x, &mut row)?;
        let mut weights = [0.0f32; MOE_USED];
        experts.blocks[block].weights.index(0..MOE_USED).copy_to(&mut weights[..])?;
        let set = &experts.blocks[block].set;
        let parts: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = misses
                .iter()
                .map(|&(j, e)| {
                    let (set, row, w) = (set, &row, weights[j]);
                    scope.spawn(move || -> Result<Vec<f32>> {
                        let (g, u) = std::thread::scope(|inner| {
                            let gate = inner.spawn(|| dequant_matvec(&set.gate, e, row));
                            let up = inner.spawn(|| dequant_matvec(&set.up, e, row));
                            (gate.join().expect("gate thread"), up.join().expect("up thread"))
                        });
                        let (g, u) = (g?, u?);
                        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
                        Ok(dequant_matvec(&set.down, e, &h)?.into_iter().map(|v| v * w).collect())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("expert thread")).collect::<Result<_>>()
        })?;
        let mut y = vec![0.0f32; d];
        for part in parts {
            for (acc, v) in y.iter_mut().zip(part) {
                *acc += v;
            }
        }
        experts.stats.cpu_misses += misses.len() as u64;
        experts.stats.cpu_nanos += started.elapsed().as_nanos() as u64;
        Ok(y)
    }

    /// One slot matvec over the eight slots `row_slots` names in `block`'s
    /// `kind` slab: `n` outputs of `k` inputs, the activation `rows` rows of
    /// which a program reads row zero (one row) or its own (eight). The
    /// slab operand starts at the block's base, so the table's entries are
    /// local to the block.
    #[allow(clippy::too_many_arguments)]
    fn slot_matvec(&self, experts: &Experts, block: usize, kind: Kind, qa: u64, das: u64, rows: usize, k: usize, row_slots: u64, n: usize, out: Buf) -> Result<()> {
        let slab = experts.slab(block, kind);
        let b = &experts.blocks[block];
        ensure!(slab.n == n && n.is_multiple_of(MOE_QDOT_TN), "a slot matvec of {n} outputs does not fit the slab's {} rows or the tile", slab.n);
        let (name, [function, ..], resident) = kernel_for(kind.stack(&b.set).quant())?;
        let act_row = if rows == 1 { "0" } else { "j" };
        // The block's share, not counting its spare slot past it: the
        // intrinsic's grouped addressing reads the extent, and declaring the
        // spare made a decode step wrong and five times slower.
        let ns = (experts.per_block * slab.n) as i64;
        let (bytes_at, d_at) = slab.at(b.base[kind as usize]);
        self.with_kernel(&self.moe_qdot, (name, n, rows == 1), "moe_qdot", || moe_qdot_src(name, n, act_row, resident), |module| {
            self.launch(
                module,
                function,
                &[
                    (qa, [rows as i64, k as i64]),
                    (das, [rows as i64, (k / 32) as i64]),
                    (row_slots, [1, U]),
                    (bytes_at, [ns, slab.rb as i64]),
                    (d_at, [ns, slab.nb as i64]),
                    (self.ptr(out, 0)?, [U, n as i64]),
                ],
                ((n / MOE_QDOT_TN) as u32, MOE_USED as u32, 1),
            )
        })
    }

    /// Gate, up and the SwiGLU of the eight slots `row_slots` names, into
    /// `h` (`[8, n]`), for a block whose gate and up share a format.
    #[allow(clippy::too_many_arguments)]
    fn slot_gateup(&self, experts: &Experts, block: usize, qa: u64, das: u64, k: usize, row_slots: u64, n: usize, h: Buf) -> Result<()> {
        let (gate, up) = (experts.slab(block, Kind::Gate), experts.slab(block, Kind::Up));
        let b = &experts.blocks[block];
        ensure!(gate.n == n && up.n == n && n.is_multiple_of(MOE_QDOT_TN), "a joint gate and up of {n} outputs does not fit the slabs or the tile");
        let (name, [_, function, _], resident) = kernel_for(b.set.gate.quant())?;
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
                    (row_slots, [1, U]),
                    (gb, [ns, gate.rb as i64]),
                    (gd, [ns, gate.nb as i64]),
                    (ub, [ns, up.rb as i64]),
                    (ud, [ns, up.nb as i64]),
                    (self.ptr(h, 0)?, [U, n as i64]),
                ],
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

/// Expert `e` of `stack` applied to `x`, each row decoded into a scratch
/// and dotted, on the calling thread.
fn dequant_matvec(stack: &ExpertStack, e: usize, x: &[f32]) -> Result<Vec<f32>> {
    let (n, k) = (stack.n(), stack.k());
    let spec = stack.quant().spec();
    let rb = k / spec.block * spec.block_bytes;
    let bytes = stack.expert(e);
    let mut decoded = vec![0.0f32; k];
    Ok((0..n)
        .map(|j| {
            (spec.dequantize)(&bytes[j * rb..(j + 1) * rb], &mut decoded);
            decoded.iter().zip(x).map(|(&w, &v)| w * v).sum()
        })
        .collect())
}
