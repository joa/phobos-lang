use std::collections::HashMap;

use anyhow::{Result, bail};
use melior::{
    dialect::{arith, scf},
    ir::{Block, BlockLike, Region, RegionLike, Type, Value, ValueLike, operation::OperationBuilder},
};

use super::Lowered;
use crate::codegen::{Codegen, MemVal, detach};
use crate::shape::DYN;
use crate::ir::{Ir, OpId, OpKind, Pipeline, ValueId};

impl<'c> Codegen<'c> {
    pub(super) fn emit_pipelined_for_ir(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        info: Pipeline,
        carried: usize,
    ) -> Result<()> {
        let lo = self.scalar_ir(ir.operand(op, 0))?;
        let hi = self.scalar_ir(ir.operand(op, 1))?;
        let st = self.scalar_ir(ir.operand(op, 2))?;
        let body = ir.blocks_of(op)[0];
        let iv = ir.args(body)[0];
        let ops = ir.ops(body);

        // The prefix ends at its `info.staged`-th stage op.
        let mut stages = Vec::with_capacity(info.staged);
        for &o in ops {
            if matches!(ir.kind(o), OpKind::Stage(_)) {
                stages.push(o);
                if stages.len() == info.staged {
                    break;
                }
            }
        }
        if stages.len() != info.staged {
            bail!("a pipelined loop with fewer stage ops than its prefix");
        }
        let end = ops.iter().position(|o| *o == stages[info.staged - 1]).expect("found above") + 1;
        let (prefix, rest) = ops.split_at(end);

        // Prologue: stage iteration 0 into each pair's first buffer.
        let mut bufs0 = Vec::with_capacity(info.staged);
        let mut pads = Vec::with_capacity(info.staged);
        let mut srcs = Vec::with_capacity(info.staged);
        self.lowered.push(HashMap::new());
        self.set_ir(iv, Lowered::Scalar(lo));
        let prologue = (|| {
            for &o in prefix {
                if let OpKind::Stage(stage) = ir.kind(o) {
                    let src = self.tile_ir(ir, ir.operand(o, 0))?;
                    let swizzled = matches!(ir.ty(ir.result(o)), crate::ir::Type::Tile(t) if t.layout.swizzle.is_some());
                    let b0 = self.alloc_stage_buf(block, &src, stage.pad, swizzled)?;
                    self.tile_copy(block, &src, &b0, true, false)?;
                    bufs0.push(b0);
                    pads.push((stage.pad, swizzled));
                    srcs.push(src);
                } else {
                    self.emit_op(block, ir, o)?;
                }
            }
            Ok::<(), anyhow::Error>(())
        })();
        self.lowered.pop();
        prologue?;

        // Without cp.async, a loop carrying register accumulators holds the
        // next stage in registers while the compute reads this one. A stage
        // every thread copies that way needs no second buffer: it is stored
        // over the one just read.
        let registers = !self.has_cp_async() && carried > 0;
        let single = registers
            && srcs.iter().zip(&bufs0).all(|(src, buf)| self.reg_copy_plan(src, buf).is_some());
        // How each stage is copied through registers, the same every turn.
        let copies: Vec<Option<RegCopy<'c>>> = if registers {
            srcs.iter().zip(&bufs0).map(|(src, buf)| self.reg_copy_plan(src, buf)).collect()
        } else {
            Vec::new()
        };
        let mut bufs1 = Vec::with_capacity(info.staged);
        if single {
            bufs1.clone_from(&bufs0);
        } else {
            for (src, &pad) in srcs.iter().zip(&pads) {
                bufs1.push(self.alloc_stage_buf(block, src, pad.0, pad.1)?);
            }
        }

        // Register accumulators the loop carries, flattened to their values.
        let carried_ops = &ir.operands(op)[3..3 + carried];
        let likes: Vec<Lowered<'c>> = carried_ops.iter().map(|&v| self.get_ir(v).cloned()).collect::<Result<_>>()?;
        let groups: Vec<Vec<Value<'c, 'c>>> = carried_ops.iter().map(|&v| self.carried_vals(v)).collect::<Result<_>>()?;
        let inits: Vec<Value<'c, 'c>> = groups.iter().flatten().copied().collect();
        let types: Vec<Type<'c>> = inits.iter().map(|v| v.r#type()).collect();
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();

        let mut args = vec![(self.index_t, self.loc)];
        args.extend(types.iter().map(|&t| (t, self.loc)));
        let body_block = Block::new(&args);
        let mlir_iv = detach(body_block.argument(0)?.into());
        let mut entry = Vec::with_capacity(inits.len());
        for i in 0..inits.len() {
            entry.push(detach(body_block.argument(1 + i)?.into()));
        }
        let next = self.addi(&body_block, mlir_iv, st)?;

        let half = Half {
            ir,
            iv,
            body,
            prefix,
            rest,
            stages: &stages,
            hi,
            ends_with_tile_op: info.ends_with_tile_op,
            likes: &likes,
            sizes: &sizes,
            registers,
            single,
            copies: &copies,
        };
        // Half A: compute iteration iv from bufs0, prefetch iv+st into bufs1.
        let after_a = self.emit_pipeline_stage_ir(&body_block, &half, mlir_iv, next, &bufs0, &bufs1, &entry)?;

        // With one buffer a stage, a turn is one iteration: there is no
        // pair to alternate between.
        let (out, step) = if single {
            (after_a, st)
        } else {
            let out = self.emit_pipeline_half_b(&body_block, &half, next, st, &bufs0, &bufs1, &after_a, &types)?;
            let two = self.const_index(block, 2)?;
            (out, self.muli(block, st, two)?)
        };
        body_block.append_operation(scf::r#yield(&out, self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let mut operands = vec![lo, hi, step];
        // prefetch iv+2*st into bufs0. Otherwise the carried values pass
        // through.

        operands.extend_from_slice(&inits);
        let loop_op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&types)
                .add_regions([region])
                .build()?,
        );
        let mut off = 0;
        for (k, (like, &n)) in likes.iter().zip(&sizes).enumerate() {
            let mut vals = Vec::with_capacity(n);
            for j in 0..n {
                vals.push(detach(loop_op.result(off + j)?.into()));
            }
            off += n;
            self.set_ir(ir.results(op)[k], self.with_vals(like, vals)?);
        }
        // Each half ends in a barrier, so what follows may reuse the buffers.
        let second = if single { &[][..] } else { &bufs1[..] };
        for buf in bufs0.iter().chain(second) {
            self.release(buf);
        }
        Ok(())
    }

    /// One unrolled half: a guarded prefetch of `prefetch_iv` into `dst`,
    /// then compute of `compute_iv` from `cur`, then a closing barrier. The
    /// barrier is skipped when the body's last tile op already ends in one.
    /// `carried` holds the carried values going in; the ones coming out are
    /// returned.
    #[allow(clippy::too_many_arguments)]
    fn emit_pipeline_stage_ir(
        &mut self,
        block: &Block<'c>,
        half: &Half<'c, '_>,
        compute_iv: Value<'c, 'c>,
        prefetch_iv: Value<'c, 'c>,
        cur: &[MemVal<'c>],
        dst: &[MemVal<'c>],
        carried: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let ir = half.ir;
        let use_async = self.has_cp_async();
        if half.registers {
            return self.emit_register_stage_ir(block, half, compute_iv, prefetch_iv, cur, dst, carried);
        }
        self.guarded_prefetch(block, prefetch_iv, half.hi, |cg, then| {
            cg.lowered.push(HashMap::new());
            cg.set_ir(half.iv, Lowered::Scalar(prefetch_iv));
            let mut d = dst.iter();
            let out = (|| {
                for &o in half.prefix {
                    if matches!(ir.kind(o), OpKind::Stage(_)) {
                        let src = cg.tile_ir(ir, ir.operand(o, 0))?;
                        let buf = d.next().expect("one buffer per stage");
                        cg.tile_copy(then, &src, buf, false, use_async)?;
                    } else {
                        cg.emit_op(then, ir, o)?;
                    }
                }
                Ok::<(), anyhow::Error>(())
            })();
            cg.lowered.pop();
            out
        })?;

        // Commit every cp.async this thread issued in the prefetch as one
        // group.
        let group = if use_async {
            Some(self.async_create_group(block)?)
        } else {
            None
        };

        let out = self.emit_pipeline_compute_ir(block, half, compute_iv, cur, carried)?;

        if let Some(group) = group {
            self.async_wait(block, group)?;
            self.barrier(block)?;
            return Ok(out);
        }
        if !half.ends_with_tile_op {
            self.barrier(block)?;
        }
        Ok(out)
    }
}

impl<'c> Codegen<'c> {
    /// Half B, when iteration `next` exists: compute it from `bufs1`,
    /// prefetch the one after into `bufs0`. Otherwise the carried values
    /// `after_a` pass through. Returns the carried values after the turn.
    #[allow(clippy::too_many_arguments)]
    fn emit_pipeline_half_b(
        &mut self,
        body: &Block<'c>,
        half: &Half<'c, '_>,
        next: Value<'c, 'c>,
        st: Value<'c, 'c>,
        bufs0: &[MemVal<'c>],
        bufs1: &[MemVal<'c>],
        after_a: &[Value<'c, 'c>],
        types: &[Type<'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let have_b = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, next, half.hi, self.loc),
        )?;
        let half_b = Block::new(&[]);
        let next2 = self.addi(&half_b, next, st)?;
        let after_b = self.emit_pipeline_stage_ir(&half_b, half, next, next2, bufs1, bufs0, after_a)?;
        half_b.append_operation(scf::r#yield(&after_b, self.loc));
        let half_b_region = Region::new();
        half_b_region.append_block(half_b);
        let skip_b = Block::new(&[]);
        skip_b.append_operation(scf::r#yield(after_a, self.loc));
        let skip_b_region = Region::new();
        skip_b_region.append_block(skip_b);
        let if_b = body.append_operation(scf::r#if(have_b, types, half_b_region, skip_b_region, self.loc));
        (0..types.len()).map(|i| Ok(detach(if_b.result(i)?.into()))).collect()
    }

    /// A half's compute: iteration `compute_iv` from the buffers `cur`, with
    /// the carried values `carried`, returning the ones it yields.
    fn emit_pipeline_compute_ir(
        &mut self,
        block: &Block<'c>,
        half: &Half<'c, '_>,
        compute_iv: Value<'c, 'c>,
        cur: &[MemVal<'c>],
        carried: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let ir = half.ir;
        self.lowered.push(HashMap::new());
        self.set_ir(half.iv, Lowered::Scalar(compute_iv));
        let mut off = 0;
        for (k, (like, &n)) in half.likes.iter().zip(half.sizes).enumerate() {
            let lowered = self.with_vals(like, carried[off..off + n].to_vec())?;
            self.set_ir(ir.args(half.body)[1 + k], lowered);
            off += n;
        }
        for (&stage, c) in half.stages.iter().zip(cur) {
            self.set_ir(ir.result(stage), Lowered::Tile(c.clone()));
        }
        let compute = self.emit_ops(block, ir, half.rest).and_then(|()| {
            let mut out = Vec::with_capacity(carried.len());
            for v in self.yielded(ir, half.body)? {
                out.extend(self.carried_vals(v)?);
            }
            Ok(out)
        });
        self.lowered.pop();
        compute
    }

    /// A half without `cp.async`: the next stage's share of each thread is
    /// loaded into registers before the compute and stored into `dst` after
    /// it, so the loads' latency hides behind the compute. A stage that is
    /// not a rank-2, unmasked copy of a static shape is copied before the
    /// compute instead.
    #[allow(clippy::too_many_arguments)]
    fn emit_register_stage_ir(
        &mut self,
        block: &Block<'c>,
        half: &Half<'c, '_>,
        compute_iv: Value<'c, 'c>,
        prefetch_iv: Value<'c, 'c>,
        cur: &[MemVal<'c>],
        dst: &[MemVal<'c>],
        carried: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let ir = half.ir;
        let more = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, prefetch_iv, half.hi, self.loc),
        )?;
        // Each chunk located once a turn, for both its load and its store.
        // Hoisted out of the loop they would hold registers across it.
        let copies: Vec<Option<RegCopy<'c>>> = half
            .copies
            .iter()
            .map(|c| c.clone().map(|plan| self.reg_copy_indexed(block, plan)).transpose())
            .collect::<Result<_>>()?;

        // The loads, in the guarded branch, yielded out of it.
        let then = Block::new(&[]);
        self.lowered.push(HashMap::new());
        self.set_ir(half.iv, Lowered::Scalar(prefetch_iv));
        let mut elems = Vec::new();
        let loaded = (|| {
            let mut d = dst.iter().zip(&copies);
            let mut vals = Vec::new();
            for &o in half.prefix {
                if !matches!(ir.kind(o), OpKind::Stage(_)) {
                    self.emit_op(&then, ir, o)?;
                    continue;
                }
                let src = self.tile_ir(ir, ir.operand(o, 0))?;
                let (buf, copy) = d.next().expect("one buffer per stage");
                match copy {
                    Some(copy) => {
                        vals.extend(self.reg_copy_load(&then, &src, copy)?);
                        elems.extend(std::iter::repeat_n(src.elem, copy.at.len()));
                    }
                    None => self.tile_copy(&then, &src, buf, false, false)?,
                }
            }
            Ok::<_, anyhow::Error>(vals)
        })();
        self.lowered.pop();
        let loaded = loaded?;
        then.append_operation(scf::r#yield(&loaded, self.loc));
        let types: Vec<Type<'c>> = loaded.iter().map(|v| v.r#type()).collect();
        let skip = Block::new(&[]);
        let mut zeros = Vec::with_capacity(types.len());
        for (&t, &elem) in types.iter().zip(&elems) {
            let zero = self.zero_scalar(&skip, elem)?;
            zeros.push(self.vec_broadcast(&skip, zero, t)?);
        }
        skip.append_operation(scf::r#yield(&zeros, self.loc));
        let (then_region, skip_region) = (Region::new(), Region::new());
        then_region.append_block(then);
        skip_region.append_block(skip);
        let fetch = block.append_operation(scf::r#if(more, &types, then_region, skip_region, self.loc));
        let mut held = Vec::with_capacity(types.len());
        for i in 0..types.len() {
            held.push(detach(fetch.result(i)?.into()));
        }

        // A synchronous copy above fills the buffer the next half reads;
        // the barrier closing this half orders it.
        let out = self.emit_pipeline_compute_ir(block, half, compute_iv, cur, carried)?;

        if !held.is_empty() {
            if half.single {
                // The stores overwrite the buffer the compute read.
                self.barrier(block)?;
            }
            let store = Block::new(&[]);
            let mut vals = held.iter();
            for (copy, buf) in copies.iter().zip(dst) {
                if let Some(copy) = copy {
                    let mine: Vec<Value<'c, 'c>> = vals.by_ref().take(copy.at.len()).copied().collect();
                    self.reg_copy_store(&store, buf, copy, &mine)?;
                }
            }
            store.append_operation(scf::r#yield(&[], self.loc));
            let region = Region::new();
            region.append_block(store);
            block.append_operation(scf::r#if(more, &[], region, Region::new(), self.loc));
        }
        // The stores must land before the next turn reads them, which a
        // barrier the body passes before that read already sees to.
        let staged = half.stages.iter().map(|&s| ir.result(s)).collect();
        if !(half.single && fenced_before_reads(ir, half.rest, staged)) {
            self.barrier(block)?;
        }
        Ok(out)
    }

    /// A buffer for a stage of `src`, padded when the stage asks.
    fn alloc_stage_buf(&mut self, block: &Block<'c>, src: &MemVal<'c>, pad: bool, swizzled: bool) -> Result<MemVal<'c>> {
        if pad {
            self.alloc_tile_padded(block, src.elem, &src.shape)
        } else if swizzled {
            self.alloc_tile_swizzled(block, src.elem, &src.shape)
        } else {
            self.alloc_tile_shaped(block, src.elem, &src.shape)
        }
    }

    /// How a thread copies its share of `src` into `dst` through registers,
    /// or None when the copy is not a rank-2, unmasked one of static shape.
    pub(super) fn reg_copy_plan(&self, src: &MemVal<'c>, dst: &MemVal<'c>) -> Option<RegCopy<'c>> {
        let &[rows, cols] = &src.shape[..] else { return None };
        if dst.shape.len() != 2 {
            return None;
        }
        if src.is_masked() || dst.is_masked() || rows == DYN || cols == DYN {
            return None;
        }
        let (width, align) = self.copy_width(src, dst);
        if cols % width != 0 {
            return None;
        }
        let across = cols / width;
        let total = rows * across;
        Some(RegCopy {
            width,
            align,
            across,
            total,
            per_thread: (total + self.cta_threads - 1) / self.cta_threads,
            vec_t: Type::vector(&[width as u64], src.elem),
            at: Vec::new(),
        })
    }

    /// `plan` with each of the thread's chunks located.
    pub(super) fn reg_copy_indexed(&self, block: &Block<'c>, mut plan: RegCopy<'c>) -> Result<RegCopy<'c>> {
        plan.at = (0..plan.per_thread).map(|i| self.reg_copy_at(block, &plan, i)).collect::<Result<_>>()?;
        Ok(plan)
    }

    /// The `[row, col]` of the thread's `i`-th chunk, clamped into the tile,
    /// and whether it is a real one.
    fn reg_copy_at(
        &self,
        block: &Block<'c>,
        plan: &RegCopy<'c>,
        i: i64,
    ) -> Result<([Value<'c, 'c>; 2], Value<'c, 'c>)> {
        let tid = self.thread_id(block)?;
        let off = self.const_index(block, i * self.cta_threads)?;
        let lin = self.addi(block, tid, off)?;
        let total = self.const_index(block, plan.total)?;
        let real = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, lin, total, self.loc))?;
        // Only a chunk some threads lack is clamped into the tile.
        let lin = if (i + 1) * self.cta_threads > plan.total {
            let last = self.const_index(block, plan.total - 1)?;
            self.minsi(block, lin, last)?
        } else {
            lin
        };
        let across = self.const_index(block, plan.across)?;
        let row = self.divui(block, lin, across)?;
        let chunk = self.remui(block, lin, across)?;
        let width = self.const_index(block, plan.width)?;
        let col = self.muli(block, chunk, width)?;
        Ok(([row, col], real))
    }

    pub(super) fn reg_copy_load(&self, block: &Block<'c>, src: &MemVal<'c>, plan: &RegCopy<'c>) -> Result<Vec<Value<'c, 'c>>> {
        plan.at
            .iter()
            .map(|(at, _)| self.vec_load_al(block, src.mem, at, plan.vec_t, plan.align))
            .collect()
    }

    pub(super) fn reg_copy_store(
        &self,
        block: &Block<'c>,
        dst: &MemVal<'c>,
        plan: &RegCopy<'c>,
        vals: &[Value<'c, 'c>],
    ) -> Result<()> {
        for (i, (&v, &(at, real))) in vals.iter().zip(&plan.at).enumerate() {
            // A chunk is at most 16 bytes at its own alignment, so it sits
            // inside one granule of a swizzled buffer.
            let at: [Value<'c, 'c>; 2] = self.swizzled_index(block, dst, &at)?.try_into().expect("a rank-2 index");
            if (i as i64 + 1) * self.cta_threads <= plan.total {
                self.vec_store_al(block, v, dst.mem, &at, plan.align)?;
                continue;
            }
            let guarded = Block::new(&[]);
            self.vec_store_al(&guarded, v, dst.mem, &at, plan.align)?;
            guarded.append_operation(scf::r#yield(&[], self.loc));
            let region = Region::new();
            region.append_block(guarded);
            block.append_operation(scf::r#if(real, &[], region, Region::new(), self.loc));
        }
        Ok(())
    }
}

/// A thread's share of a staged copy, held in registers: `per_thread`
/// vectors of `width` elements, chunk `i` the CTA's `tid + i * threads`-th
/// of `total`, `across` of them to a row. `at` holds each chunk's
/// `[row, col]` and whether the thread has it, once located.
#[derive(Clone)]
pub(super) struct RegCopy<'c> {
    width: i64,
    align: i64,
    across: i64,
    total: i64,
    per_thread: i64,
    vec_t: Type<'c>,
    at: Vec<([Value<'c, 'c>; 2], Value<'c, 'c>)>,
}

/// Whether `ops` pass an explicit `barrier()` before anything reads one of
/// `staged`. A view of one reads nothing itself; any use inside a nested
/// block counts as a read before the fence.
pub(super) fn fenced_before_reads(ir: &Ir, ops: &[OpId], mut staged: Vec<ValueId>) -> bool {
    fn nested_reads(ir: &Ir, op: OpId, staged: &[ValueId]) -> bool {
        ir.blocks_of(op).iter().any(|&b| {
            ir.ops(b).iter().any(|&o| {
                ir.operands(o).iter().any(|v| staged.contains(v)) || nested_reads(ir, o, staged)
            })
        })
    }
    for &op in ops {
        let reads = ir.operands(op).iter().any(|v| staged.contains(v)) || nested_reads(ir, op, &staged);
        match ir.kind(op) {
            OpKind::Barrier => return true,
            kind if reads && !kind.aliases().is_empty() => staged.push(ir.result(op)),
            _ if reads => return false,
            _ => {}
        }
    }
    false
}

/// What both halves of a pipelined loop share.
struct Half<'c, 'o> {
    ir: &'o Ir,
    iv: ValueId,
    body: crate::ir::BlockId,
    prefix: &'o [OpId],
    rest: &'o [OpId],
    stages: &'o [OpId],
    hi: Value<'c, 'c>,
    ends_with_tile_op: bool,
    /// Each carried accumulator's lowered form, and how many values it
    /// flattens to.
    likes: &'o [Lowered<'c>],
    sizes: &'o [usize],
    /// The next stage is carried in registers across the compute.
    registers: bool,
    /// Each stage has one buffer, refilled from registers after the compute.
    single: bool,
    /// With `registers`, how each stage is copied through them, or None for
    /// one copied synchronously.
    copies: &'o [Option<RegCopy<'c>>],
}
