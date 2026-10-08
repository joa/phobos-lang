// The `Backend` impl. Most method bodies live in the sibling modules.

use super::*;
use super::hadamard::HadamardExtra;

impl Backend for DeviceBackend {
    #[track_caller]
    fn alloc(&self, len: usize) -> Result<Buf> {
        self.note_alloc(len, 1);
        residency::note_big_alloc(len);
        Ok(self.store(self.pool.take(len)?))
    }

    fn device_memory(&self) -> Option<(usize, usize)> {
        cust::memory::mem_get_info().ok()
    }

    fn card_memory(&self) -> Option<(usize, usize)> {
        mem::card_memory().ok()
    }

    fn device_info(&self) -> Option<phobos_inference::DeviceInfo> {
        super::init::device_info()
    }

    fn cache_stats(&self) -> Option<phobos_inference::CacheStats> {
        let (buffers_reused, buffers_allocated) = self.pool.reuse_counts();
        let (buffer_live_bytes, buffer_idle_bytes) = self.pool.byte_counts();
        let experts = self.expert_stats();
        Some(phobos_inference::CacheStats {
            kernels_reused: self.kernels_reused.get(),
            kernels_compiled: self.kernels_compiled.get(),
            buffers_reused,
            buffers_allocated,
            buffer_live_bytes,
            buffer_idle_bytes,
            expert_hits: experts.hits,
            expert_misses: experts.misses,
            expert_prompt_hits: experts.prompt_hits,
            expert_prompt_misses: experts.prompt_misses,
            expert_bytes: experts.bytes,
            expert_prefetches: experts.prefetches,
            expert_prefetch_hits: experts.prefetch_hits,
            expert_cpu_misses: experts.cpu_misses,
            expert_cpu_nanos: experts.cpu_nanos,
        })
    }

    fn limit_expert_cache(&self, bytes: usize) -> Result<()> {
        self.experts.borrow_mut().limit = Some(bytes);
        Ok(())
    }

    fn budget_streamed(&self, resident_bytes: usize, sets: usize) -> Result<()> {
        self.set_streamed_budget(resident_bytes, sets)
    }

    fn constant_experts(&self, key: &str, set: &std::sync::Arc<crate::experts::ExpertSet>) -> Result<crate::backend::ExpertsBuf> {
        self.register_experts(key, set)
    }

    fn moe(&self, req: crate::backend::Moe) -> Result<()> {
        self.run_moe(req)
    }

    fn release(&self, buf: Buf) {
        let taken = self
            .slots
            .borrow_mut()
            .get_mut(buf.0)
            .and_then(Option::take);
        if let Some(slot) = taken {
            match slot {
                mem::Slot::Owned(buffer) => {
                    self.note_alloc(buffer.len(), -1);
                    self.pool.put(buffer);
                }
                mem::Slot::State { .. } => {
                    let live = self.state_live.get() - 1;
                    self.state_live.set(live);
                    if live == 0 {
                        self.state_arena.reset();
                    }
                }
                mem::Slot::Const { .. } => {}
            }
            self.free_slots.borrow_mut().push(buf.0);
        }
    }

    fn alloc_h(&self, len: usize) -> Result<HBuf> {
        Ok(HBuf(self.store(self.pool.take(len.div_ceil(2))?).0))
    }

    fn release_h(&self, buf: HBuf) {
        self.release(DeviceBackend::words(buf));
    }

    fn zeroed_h(&self, len: usize) -> Result<HBuf> {
        Ok(HBuf(self.zeroed(len.div_ceil(2))?.0))
    }

    fn read_h(&self, buf: HBuf, out: &mut [f32]) -> Result<()> {
        // Each f32 word holds two halves, low one first.
        let mut words = vec![0.0f32; out.len().div_ceil(2)];
        self.read(DeviceBackend::words(buf), &mut words)?;
        for (o, bits) in out.iter_mut().zip(
            words
                .iter()
                .flat_map(|w| [w.to_bits() as u16, (w.to_bits() >> 16) as u16]),
        ) {
            *o = f16_to_f32(bits);
        }
        Ok(())
    }

    fn copy_h(
        &self,
        src: HBuf,
        src_offset: usize,
        dst: HBuf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        ensure!(
            src_offset.is_multiple_of(2) && dst_offset.is_multiple_of(2) && len.is_multiple_of(2),
            "copy_h moves whole words, so it needs an even run at even offsets"
        );
        self.copy(
            DeviceBackend::words(src),
            src_offset / 2,
            DeviceBackend::words(dst),
            dst_offset / 2,
            len / 2,
        )
    }

    fn upload(&self, data: &[f32]) -> Result<Buf> {
        let buf = self.alloc_written_now(data.len())?;
        let slots = self.slots.borrow();
        let Some(mem::Slot::Owned(buffer)) = slots.get(buf.0).and_then(Option::as_ref) else {
            bail!("upload lost its buffer");
        };
        buffer.index(0..data.len()).copy_from(data)?;
        drop(slots);
        Ok(buf)
    }

    fn zeroed(&self, len: usize) -> Result<Buf> {
        // While recording, clear a pooled buffer with a recorded launch, so
        // it runs in order with the launches before it. An immediate clear
        // would need a fresh allocation, which is too slow per block.
        if self.recording.get() {
            let buf = self.alloc(len)?;
            self.pointwise("zero", &[(buf, 0)], len)?;
            return Ok(buf);
        }
        let buf = self.alloc_written_now(len)?;
        let ptr = self.ptr(buf, 0)?;
        // Async on the stream, so it needs no sync.
        cuda_ok(
            // SAFETY: the allocation is at least `len` floats and the handle
            // holds it alive for the duration.
            unsafe { cust::sys::cuMemsetD32Async(ptr, 0, len, self.stream.as_inner()) },
            "zeroing a device allocation",
        )?;
        Ok(buf)
    }

    fn begin_pass(&self, rows: usize) -> Result<()> {
        self.trim_after_prompt(rows)?;
        self.trim_after_dense()?;
        self.keep_headroom(rows)?;
        self.mark_pass_vram();
        // Restart the rings, so each step records the same slots and the
        // cached graph needs no patching.
        self.act_next.set(0);
        self.act_ring.set(0);
        self.act_shared.set(mem::ACT_RING);
        self.recorded_len.set(0);
        self.segment.set(0);
        self.flushed.set(false);
        self.recording.set(true);
        if self.report_pass.get() != 0 {
            self.report.borrow_mut().clear();
        }
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        // A flushed pass is only partly recorded. Issue the tail as launches
        // and keep the cached graph for the next whole pass to replace.
        if self.flushed.replace(false) {
            self.recording.set(false);
            return self.issue_recorded("issuing the tail of a flushed pass");
        }
        self.recording.set(false);
        self.replay()
    }

    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()> {
        // The only synchronization point in a block.
        self.stream.synchronize()?;
        // Any slot kind can be read, including recurrent state for a
        // checkpoint.
        let (at, len) = (self.ptr(buf, 0)?, self.len_of(buf)?);
        ensure!(
            len >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            len
        );
        // Copy through pinned staging. A copy straight into a Vec is bounced
        // by the driver a page at a time, which is slower.
        let mut staging = self.readback.borrow_mut();
        let too_small = staging.as_ref().is_none_or(|s| s.len() < out.len());
        if too_small {
            *staging = Some(LockedBuffer::new(&0.0f32, out.len())?);
        }
        let pinned = staging.as_mut().expect("filled above");
        let dst = pinned.as_mut_slice()[..out.len()].as_mut_ptr();
        // SAFETY: `at` holds at least `out.len()` elements, checked above,
        // and the pinned staging as many; the stream is drained.
        cuda_ok(
            unsafe { cust::sys::cuMemcpyDtoH_v2(dst.cast(), at, size_of_val(out)) },
            "reading a buffer back",
        )?;
        out.copy_from_slice(&pinned.as_slice()[..out.len()]);
        Ok(())
    }

    fn argmax(&self, buf: Buf, len: usize) -> Result<i64> {
        self.device_argmax(buf, len)
    }

    fn constant(&self, key: &str, data: &[f32]) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        residency::note_big_const(key, data.len());
        let buf = self.store_const(data)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        let data = fill()?;
        residency::note_big_const(key, data.len());
        let buf = self.store_const(&data)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.matmul_dense(a, m, k, w, n, out)
    }

    fn matmul_rows(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.matmul_rows_dense(a, m, k, w, n, out)
    }

    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf> {
        if let Some(&buf) = self.q_constants.borrow().get(key) {
            return Ok(buf);
        }
        let (k, n) = (packed.k(), packed.n());
        let mut planes = packed.planes()?;
        let q50 = packed.quant() == Quant::Q5_0;
        if q50 {
            planes.qs = packed.device_blocks();
        }
        let blocks = k / Q8_BLOCK;
        let mut row_scales = vec![0.0f32; planes.scales.len()];
        for (b, row) in planes.scales.chunks_exact(n).enumerate() {
            for (j, &s) in row.iter().enumerate() {
                row_scales[j * blocks + b] = s;
            }
        }
        // A large weight is bulk like any raw one and joins the arena, since
        // WDDM keeps a few large allocations resident where it fails many.
        // A small one stays apart: in a slab it drags the slab resident.
        let arena = match () {
            _ if self.arena_weights && planes.qs.len() >= BULK_QUANT_BYTES => Some(&self.arena),
            _ if self.arena_const => Some(&self.hot),
            _ => None,
        };
        let uploaded = if let Some(arena) = arena {
            DeviceQuant {
                qs: arena.upload(&planes.qs)?,
                scales: arena.upload(&planes.scales)?,
                row_scales: arena.upload(&row_scales)?,
                n,
                q50,
            }
        } else {
            let owned = (
                DeviceBuffer::from_slice(&planes.qs)?,
                DeviceBuffer::from_slice(&planes.scales)?,
                DeviceBuffer::from_slice(&row_scales)?,
            );
            let at = DeviceQuant {
                qs: owned.0.as_device_ptr().as_raw(),
                scales: owned.1.as_device_ptr().as_raw(),
                row_scales: owned.2.as_device_ptr().as_raw(),
                n,
                q50,
            };
            self.owned_quants.borrow_mut().push(owned);
            at
        };
        let mut quants = self.quants.borrow_mut();
        quants.push(uploaded);
        let buf = QBuf(quants.len() - 1);
        drop(quants);
        self.q_constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn constant_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {
        self.upload_raw(key, packed)
    }

    fn matmul_raw(&self, a: Buf, m: usize, k: usize, w: RawBuf, n: usize, out: Buf) -> Result<()> {
        // Formats with a dequant kernel, used for multi-row batches.
        const DEQUANT_FORMATS: [Quant; 9] = [
            Quant::IQ1_S,
            Quant::IQ2_XXS,
            Quant::IQ1_M,
            Quant::IQ2_S,
            Quant::IQ2_XS,
            Quant::IQ3_XXS,
            Quant::IQ3_S,
            Quant::IQ4_XS,
            Quant::Q2_K,
        ];
        // Prefer the fused projection, which decodes inside the contraction
        // and needs no scratch. See `raw_qmma_eligible` for the shapes it
        // takes.
        if m > 1 && self.raw_qmma.get() && self.raw_qmma_eligible(w, m, k, n) {
            return self.project_raw_qmma(None, a, m, k, w, n, out);
        }
        if m > 1 && DEQUANT_FORMATS.iter().any(|&q| self.raw_quant_is(w, q)) {
            return self.project_raw_dense(a, m, k, w, n, out);
        }
        self.project_raw(None, a, m, k, w, n, out)
    }

    fn matmul_raw_act(
        &self,
        act: QAct,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        if m > 1 && self.raw_qmma.get() && self.raw_qmma_eligible(w, m, k, n) {
            return self.project_raw_qmma(Some(act), a, m, k, w, n, out);
        }
        if m == 1 {
            return self.project_raw(Some(act), a, m, k, w, n, out);
        }
        self.matmul_raw(a, m, k, w, n, out)
    }

    fn zeroed_state(&self, len: usize) -> Result<Buf> {
        match self.state_arena_on {
            true => self.zeroed_state_buf(len),
            false => self.zeroed(len),
        }
    }

    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        self.quantize_act_into(self.act_slot(m, k)?, a, m, k)
    }

    fn matmul_quant_act(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        self.project_q8(act, m, k, w, n, out, false)
    }

    fn matmul_quant_add(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        // Only the single-row kernel accumulates. Other shapes project into
        // a temporary and add it separately.
        if m != 1 || !n.is_multiple_of(Q8_QDOT_TN) {
            let temp = self.alloc(m * n)?;
            self.matmul_quant_act(act, m, k, w, n, temp)?;
            self.add_into(out, temp)?;
            self.release(temp);
            return Ok(());
        }
        self.project_q8(act, m, k, w, n, out, true)
    }

    fn rms_norm(&self, x: Buf, rows: usize, width: usize, gain: Buf, eps: f32, out: Buf) -> Result<()> {
        self.rms_norm_rows(x, rows, width, gain, eps, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        gate: Buf,
        gate_at: usize,
        out: Buf,
    ) -> Result<QAct> {
        self.check_distinct("rms_norm_gated", out, &[x, gain, gate]);
        let shape = self.norm_shape(width)?;
        let (act, qa_ptr, das_ptr) = self.act_slot(rows, width)?;
        self.with_kernel(
            &self.gated_norms,
            (width, eps.to_bits()),
            "rms_norm_gated",
            || rms_norm_src(width, eps, NormForm::GatedQuantized),
            |module| {
                let rb = (rows as i64) * shape.0;
                self.launch(
                    module,
                    "rms_norm_gated",
                    &[
                        (self.ptr(x, 0)?, [rb, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [rb, shape.1]),
                        (self.ptr(gate, gate_at)?, [rb, shape.1]),
                        (qa_ptr, [rb, shape.1]),
                        (das_ptr, [rb, 1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn rms_norm_q(&self, x: Buf, rows: usize, width: usize, gain: Buf, eps: f32, out: Buf) -> Result<QAct> {
        let slot = self.act_slot(rows, width)?;
        self.rms_norm_q_into(slot, x, rows, width, gain, eps, out)
    }

    fn fused_mlp(&self, mlp: FusedMlp) -> Result<bool> {
        self.launch_fused_mlp(mlp)
    }

    fn fused_mlp_raw(&self, mlp: FusedMlpRaw) -> Result<bool> {
        self.launch_fused_mlp_raw(mlp)
    }

    fn fused_project(&self, project: FusedProject) -> Result<Fused> {
        self.launch_fused_project(project)
    }

    fn fused_attn_out(&self, out: FusedAttnOut) -> Result<bool> {
        self.launch_fused_attn_out(out)
    }

    fn store_2d_pair(
        &self,
        a: (Plane, HPlane),
        b: (Plane, HPlane),
        rows: usize,
        width: usize,
    ) -> Result<()> {
        if !self.fused_store2d {
            self.store_2d(a.0, a.1, rows, width)?;
            return self.store_2d(b.0, b.1, rows, width);
        }
        let from_a = (self.ptr(a.0.buf, a.0.offset)?, a.0.pitch);
        let to_a = (self.hptr(a.1.buf, a.1.offset)?, a.1.pitch);
        let from_b = (self.ptr(b.0.buf, b.0.offset)?, b.0.pitch);
        let to_b = (self.hptr(b.1.buf, b.1.offset)?, b.1.pitch);
        self.strided_pair(from_a, to_a, from_b, to_b, rows, width)
    }

    fn copy_2d(&self, src: Plane, dst: Plane, rows: usize, width: usize) -> Result<()> {
        self.check_distinct("copy_2d", dst.buf, &[src.buf]);
        let from = (self.ptr(src.buf, src.offset)?, src.pitch);
        let to = (self.ptr(dst.buf, dst.offset)?, dst.pitch);
        self.strided(Strided::Dense, from, to, rows, width)
    }

    fn store_2d(&self, src: Plane, dst: HPlane, rows: usize, width: usize) -> Result<()> {
        let from = (self.ptr(src.buf, src.offset)?, src.pitch);
        let to = (self.hptr(dst.buf, dst.offset)?, dst.pitch);
        self.strided(Strided::Store, from, to, rows, width)
    }

    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()> {
        let half = spec.rope_dim / 2;
        let (r, d) = ((rows * spec.heads) as i64, spec.head_dim as i64);
        self.with_kernel(
            &self.ropes,
            (spec.heads, half),
            "rope",
            || rope_src(spec.heads, half),
            |module| {
                self.launch(
                    module,
                    "rope",
                    &[
                        (self.ptr(x, 0)?, [r, d]),
                        (
                            // Offset to this call's first position, so the
                            // kernel indexes the table by row.
                            self.ptr(table, spec.start_pos * spec.rope_dim)?,
                            [rows as i64, spec.rope_dim as i64],
                        ),
                    ],
                    (r as u32, 1, 1),
                )
            },
        )
    }

    fn rope_gather(
        &self,
        src: Plane,
        rows: usize,
        table: Buf,
        spec: Rope,
        dest: Buf,
    ) -> Result<()> {
        self.rope_gather_impl(src, rows, table, spec, dest)
    }

    fn attention(&self, q: Buf, keys: HBuf, values: HBuf, spec: Attn, out: Buf) -> Result<()> {
        self.check_distinct("attention", out, &[q]);
        // Prefer the blocked kernel. Its causal mask needs the query blocks
        // aligned to the tile, so a misaligned start falls through.
        let block = attention_block_tile(spec.head_dim);
        if spec.rows > 1 && spec.start_pos.is_multiple_of(block) {
            return self.attention_blocked(q, keys, values, spec, block, out);
        }
        // Next, the two-matmul path if the shape tiles evenly.
        if attn_gemm_fits(spec) {
            return self.attention_gemm(q, keys, values, spec, out);
        }
        match spec.rows {
            1 => self.attention_decode(q, keys, values, spec, out),
            _ => self.attention_rows(q, keys, values, spec, out),
        }
    }

    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()> {
        let len = self.len_of(x)?.min(self.len_of(gate)?);
        self.pointwise("gate_into", &[(x, 0), (gate, 0)], len)
    }

    fn hadamard(&self, x: Buf, rows: usize, width: usize, signs: Buf, perm: Option<HeadPerm>, out: Buf) -> Result<()> {
        let plain = HadamardExtra { form: HadamardForm::Plain, norm: None };
        self.hadamard_rows(x, rows, width, signs, perm, out, plain)?;
        Ok(())
    }

    fn hadamard_q(&self, x: Buf, rows: usize, width: usize, signs: Buf, perm: Option<HeadPerm>, out: Buf) -> Result<QAct> {
        let quantized = HadamardExtra { form: HadamardForm::Quantized, norm: None };
        let act = self.hadamard_rows(x, rows, width, signs, perm, out, quantized)?;
        act.context("the quantized Hadamard transform left no quantized copy")
    }

    fn rms_norm_hadamard_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        signs: Buf,
        normed: Buf,
        out: Buf,
    ) -> Result<QAct> {
        // The one-kernel form has every program sum the whole row, so it
        // only suits a single narrow row.
        if rows > 1 || width > HADAMARD_NORM_MAX_WIDTH {
            self.rms_norm(x, rows, width, gain, eps, normed)?;
            return self.hadamard_q(normed, rows, width, signs, None, out);
        }
        let normed_form = HadamardExtra { form: HadamardForm::Normed(eps.to_bits()), norm: Some((gain, normed)) };
        let act = self.hadamard_rows(x, rows, width, signs, None, out, normed_form)?;
        act.context("the normalizing Hadamard transform left no quantized copy")
    }

    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()> {
        self.check_distinct("delta_conv", packed, &[history, taps]);
        let channels = mix.channels();
        let (pr, c) = ((mix.pad() + mix.rows) as i64, channels as i64);
        // The packed destination has one row per (position, head), so
        // `batch` positions of one head form an ordinary tile.
        let batch = delta_conv_batch(mix.rows);
        let (planes, width) = ((3 * mix.rows) as i64, (mix.heads * mix.head_dim) as i64);
        // The planes are evenly spaced, checked below.
        let plane_stride = mix.planes[1];
        ensure!(
            mix.planes == [0, plane_stride, 2 * plane_stride],
            "delta_conv expects evenly spaced query, key and value planes"
        );
        let key = (
            mix.heads,
            mix.kv_heads,
            mix.head_dim,
            mix.kernel,
            mix.head_stride,
            mix.normalize,
            mix.query_scale.to_bits(),
            batch,
        );
        self.with_kernel(
            &self.convs,
            key,
            "delta_conv",
            || {
                delta_conv_src(
                    mix.heads,
                    mix.kv_heads,
                    mix.head_dim,
                    mix.kernel,
                    mix.head_stride,
                    plane_stride,
                    batch,
                    mix.normalize,
                    mix.query_scale,
                )
            },
            |module| {
                self.launch(
                    module,
                    "delta_conv",
                    &[
                        (self.ptr(history, 0)?, [pr, c]),
                        (self.ptr(taps, 0)?, [mix.kernel as i64, c]),
                        (self.ptr(packed, 0)?, [planes, width]),
                    ],
                    ((mix.rows / batch) as u32, mix.heads as u32, 3),
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_gates(
        &self,
        decay_in: Buf,
        decay_at: usize,
        beta_in: Buf,
        beta_at: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()> {
        self.check_distinct("delta_gates", packed, &[decay_in, beta_in, rate, dt_bias]);
        let (heads, span, gates) = (mix.heads, mix.span(), mix.gates());
        let tile_rows = (ELEM_TILE / 4 / heads).max(1);
        let (r, h) = (mix.rows as i64, heads as i64);
        self.with_kernel(
            &self.gates,
            heads,
            "delta_gates",
            || delta_gates_src(heads, tile_rows),
            |module| {
                self.launch(
                    module,
                    "delta_gates",
                    &[
                        (self.ptr(decay_in, decay_at)?, [r, h]),
                        (self.ptr(beta_in, beta_at)?, [r, h]),
                        (self.ptr(rate, 0)?, [1, h]),
                        (self.ptr(dt_bias, 0)?, [1, h]),
                        (self.ptr(packed, 3 * span)?, [r, h]),
                        (self.ptr(packed, 3 * span + gates)?, [r, h]),
                    ],
                    (mix.rows.div_ceil(tile_rows) as u32, 1, 1),
                )
            },
        )
    }

    fn delta_rule(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        self.check_distinct("delta_rule", out, &[packed, state]);
        ensure!(
            head_dim.is_multiple_of(DELTA_TN),
            "delta_rule needs a head dimension ({head_dim}) that is a multiple of {DELTA_TN}"
        );
        // The five operands are windows of one allocation.
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (r, d) = ((rows * heads) as i64, head_dim as i64);

        // Use the chunked form when rows and head dim divide evenly.
        // Everything else, including decode, uses the sequential kernel.
        if rows.is_multiple_of(DELTA_CHUNK) && head_dim.is_multiple_of(DELTA_CHUNK_TN) {
            return self.delta_chunked(packed, rows, heads, head_dim, state, out);
        }

        self.with_kernel(
            &self.deltas,
            (heads, head_dim),
            "delta_rule",
            || {
                DELTA_SRC
                    .replace("{H}", &heads.to_string())
                    .replace("{D}", &head_dim.to_string())
                    .replace("{TN}", &DELTA_TN.to_string())
            },
            |module| {
                self.launch(
                    module,
                    "delta_rule",
                    &[
                        (self.ptr(packed, 0)?, [r, d]),
                        (self.ptr(packed, span)?, [r, d]),
                        (self.ptr(packed, 2 * span)?, [r, d]),
                        (self.ptr(packed, 3 * span)?, [r, 1]),
                        (self.ptr(packed, 3 * span + gates)?, [r, 1]),
                        (self.ptr(state, 0)?, [(heads * head_dim) as i64, d]),
                        (self.ptr(out, 0)?, [r, d]),
                    ],
                    (heads as u32, (head_dim / DELTA_TN) as u32, 1),
                )
            },
        )
    }

    fn add_into(&self, acc: Buf, add: Buf) -> Result<()> {
        let len = self.len_of(acc)?.min(self.len_of(add)?);
        self.pointwise("add_into", &[(acc, 0), (add, 0)], len)
    }

    fn rms_norm_add(
        &self,
        y: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        normed: Buf,
        x: Buf,
    ) -> Result<()> {
        // At one row the plain norm costs three times the quantizing one,
        // whose quantized copy then goes to a slot nothing reads.
        if rows == 1 && norm_q_cta(width).is_some() {
            self.rms_norm_q(y, rows, width, gain, eps, normed)?;
        } else {
            self.rms_norm(y, rows, width, gain, eps, normed)?;
        }
        self.add_into(x, normed)
    }

    fn add_rows(&self, x: Buf, rows: usize, width: usize, bias: Buf) -> Result<()> {
        self.check_distinct("add_rows", x, &[bias]);
        self.launch(
            &self.pointwise,
            "add_rows",
            &[
                (self.ptr(x, 0)?, [rows as i64, width as i64]),
                (self.ptr(bias, 0)?, [1, width as i64]),
            ],
            (width.div_ceil(ELEM_TILE) as u32, rows as u32, 1),
        )
    }

    fn swiglu_q(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<QAct> {
        self.check_distinct("swiglu_q", out, &[gate, up]);
        ensure!(
            len.is_multiple_of(ELEM_TILE),
            "a quantizing SwiGLU needs a length ({len}) that is a multiple of {ELEM_TILE}"
        );
        let (act, qa_ptr, das_ptr) = self.act_slot(1, len)?;
        let rb = (len / RMS_LANE) as i64;
        let lane = RMS_LANE as i64;
        self.with_kernel(
            &self.gated_swiglu,
            SWIGLU_Q_BLOCKS,
            "swiglu_q",
            || swiglu_q_src(SWIGLU_Q_BLOCKS),
            |module| {
                self.launch(
                    module,
                    "swiglu_q",
                    &[
                        (self.ptr(gate, gate_at)?, [rb, lane]),
                        (self.ptr(up, up_at)?, [rb, lane]),
                        (self.ptr(out, 0)?, [rb, lane]),
                        (qa_ptr, [rb, lane]),
                        (das_ptr, [rb, 1]),
                    ],
                    ((len / ELEM_TILE) as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn swiglu(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<()> {
        self.check_distinct("swiglu", out, &[gate, up]);
        self.pointwise(
            "swiglu",
            &[(gate, gate_at), (up, up_at), (out, 0)],
            len.min(self.len_of(out)?),
        )
    }

    fn swiglu_planes(
        &self,
        gate: Plane,
        up: Plane,
        out: Buf,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        self.check_distinct("swiglu_planes", out, &[gate.buf, up.buf]);
        let tile = swiglu_2d_tile(width);
        self.with_kernel(
            &self.swiglu_planes,
            tile,
            "swiglu_2d",
            || swiglu_2d_src(tile),
            |module| {
                self.launch(
                    module,
                    "swiglu_2d",
                    &[
                        (
                            self.ptr(gate.buf, gate.offset)?,
                            [rows as i64, gate.pitch as i64],
                        ),
                        (self.ptr(up.buf, up.offset)?, [rows as i64, up.pitch as i64]),
                        (self.ptr(out, 0)?, [rows as i64, width as i64]),
                    ],
                    (rows as u32, (width / tile) as u32, 1),
                )
            },
        )
    }

    fn copy(
        &self,
        src: Buf,
        src_offset: usize,
        dst: Buf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.check_distinct("copy", dst, &[src]);
        self.pointwise("copy", &[(src, src_offset), (dst, dst_offset)], len)
    }
}
