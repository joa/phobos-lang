// The `Backend` impl. A trait impl cannot be split across files, so this
// is the whole surface here; method bodies live in the sibling modules.

use super::*;

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
        // Through the f32 readback: a word is two halves, low one first.
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
        let buf = self.alloc_written_now(len)?;
        let ptr = self.ptr(buf, 0)?;
        // Async on the stream, so it orders with the pass instead of forcing
        // a sync the way an upload's staging copy would.
        cuda_ok(
            // SAFETY: the allocation is at least `len` floats and the handle
            // holds it alive for the duration.
            unsafe { cust::sys::cuMemsetD32Async(ptr, 0, len, self.stream.as_inner()) },
            "zeroing a device allocation",
        )?;
        Ok(buf)
    }

    fn begin_pass(&self) -> Result<()> {
        self.trim_after_dense()?;
        self.mark_pass_vram();
        self.act_next.set(0);
        self.recorded_len.set(0);
        self.flushed.set(false);
        self.recording.set(true);
        if self.report_pass.get() != 0 {
            self.report.borrow_mut().clear();
        }
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        // A pass that had to flush is only partly recorded: issue the tail as
        // launches and leave the cached graph for the next whole pass to replace.
        if self.flushed.replace(false) {
            self.recording.set(false);
            return self.issue_recorded("issuing the tail of a flushed pass");
        }
        self.recording.set(false);
        self.replay()
    }

    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()> {
        // The only synchronization point in a block: everything queued since
        // the last read has to land before the host can look at it.
        self.stream.synchronize()?;
        let slots = self.slots.borrow();
        let Some(mem::Slot::Owned(buffer)) = slots.get(buf.0).and_then(Option::as_ref) else {
            bail!("reading a released handle, or a constant, which lives in the arena")
        };
        ensure!(
            buffer.len() >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            buffer.len()
        );
        // Through page-locked staging: straight into a Vec the driver bounces
        // the copy through its own pinned staging a page at a time, at about
        // a third of the rate, and the logits are a megabyte a token here.
        let mut staging = self.readback.borrow_mut();
        let too_small = staging.as_ref().is_none_or(|s| s.len() < out.len());
        if too_small {
            *staging = Some(LockedBuffer::new(&0.0f32, out.len())?);
        }
        let pinned = staging.as_mut().expect("filled above");
        buffer
            .index(0..out.len())
            .copy_to(&mut pinned.as_mut_slice()[..out.len()])?;
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
        self.check_distinct("matmul", out, &[a, w]);
        let (a_ptr, w_ptr, out_ptr) = (self.ptr(a, 0)?, self.ptr(w, 0)?, self.ptr(out, 0)?);

        // Decoding (m == 1) stays on the matvec specialization; anything
        // wider tiles. Both handle a ragged shape by masking the boundary tile.
        if m > 1 {
            let f32_bytes = size_of::<f32>() as u64;
            // The tensor-core kernel needs whole TC_TILE_M-row bands of a
            // whole TC_TILE_N-wide output (@aligned demands provably in-bounds
            // slices). Whatever it can't cover falls to the plain kernel below,
            // the same deepest-tile-first ladder project_q8 uses for Q8_0.
            let tc_rows = if n.is_multiple_of(TC_TILE_N) && k.is_multiple_of(TC_TILE_K) {
                m - m % TC_TILE_M
            } else {
                0
            };
            if tc_rows > 0 {
                self.launch(
                    &self.matmul_tc,
                    "matmul_tc",
                    &[
                        (a_ptr, [tc_rows as i64, k as i64]),
                        (w_ptr, [k as i64, n as i64]),
                        (out_ptr, [tc_rows as i64, n as i64]),
                    ],
                    ((tc_rows / TC_TILE_M) as u32, (n / TC_TILE_N) as u32, 1),
                )?;
            }

            let rows = m - tc_rows;
            if rows == 0 {
                return Ok(());
            }
            let a_row_ptr = a_ptr + (tc_rows * k) as u64 * f32_bytes;
            let out_row_ptr = out_ptr + (tc_rows * n) as u64 * f32_bytes;
            let tiles_evenly = rows.is_multiple_of(TILE_M) && n.is_multiple_of(TILE_N);
            return self.launch(
                self.matmul.pick(tiles_evenly),
                "matmul",
                &[
                    (a_row_ptr, [rows as i64, k as i64]),
                    (w_ptr, [k as i64, n as i64]),
                    (out_row_ptr, [rows as i64, n as i64]),
                ],
                (rows.div_ceil(TILE_M) as u32, n.div_ceil(TILE_N) as u32, 1),
            );
        }

        self.launch(
            self.matvec.pick(n.is_multiple_of(MV_TN)),
            "matvec",
            &[
                (a_ptr, [1, k as i64]),
                (w_ptr, [k as i64, n as i64]),
                (out_ptr, [1, n as i64]),
            ],
            (n.div_ceil(MV_TN) as u32, 1, 1),
        )
    }

    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf> {
        if let Some(&buf) = self.q_constants.borrow().get(key) {
            return Ok(buf);
        }
        let (k, n) = (packed.k(), packed.n());
        let planes = packed.planes()?;
        let blocks = k / Q8_BLOCK;
        let mut row_scales = vec![0.0f32; planes.scales.len()];
        for (b, row) in planes.scales.chunks_exact(n).enumerate() {
            for (j, &s) in row.iter().enumerate() {
                row_scales[j * blocks + b] = s;
            }
        }
        let uploaded = if self.arena_const {
            DeviceQuant {
                qs: self.hot.upload(&planes.qs)?,
                scales: self.hot.upload(&planes.scales)?,
                row_scales: self.hot.upload(&row_scales)?,
                n,
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
        // Batching only pays off once a dequant kernel amortizes the decode
        // over m rows (project_raw_dense's match table covers every raw
        // format except Q3_K, the LM head, which never reaches m > 1).
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
        // The fused projection first: it decodes inside the contraction, so it
        // neither writes an expanded weight nor leaves scratch behind. Only the
        // shapes it can take whole, see `raw_qmma_eligible`.
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
        // Only the single-row kernel accumulates; a prefill's tensor-core
        // path treats the residual add as noise, not a launch that matters.
        if m != 1 || !n.is_multiple_of(Q8_QDOT_TN) {
            let temp = self.alloc(m * n)?;
            self.matmul_quant_act(act, m, k, w, n, temp)?;
            self.add_into(out, temp)?;
            self.release(temp);
            return Ok(());
        }
        self.project_q8(act, m, k, w, n, out, true)
    }

    fn rms_norm(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<()> {
        self.check_distinct("rms_norm", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        self.with_kernel(
            &self.norms,
            (width, eps.to_bits()),
            "rms_norm",
            || rms_norm_src(width, eps, NormForm::Plain),
            |module| {
                self.launch(
                    module,
                    "rms_norm",
                    &[
                        (self.ptr(x, 0)?, [(rows as i64) * shape.0, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [(rows as i64) * shape.0, shape.1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
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

    fn rms_norm_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<QAct> {
        self.check_distinct("rms_norm_q", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        let (act, qa_ptr, das_ptr) = self.act_slot(rows, width)?;
        self.with_kernel(
            &self.quant_norms,
            (width, eps.to_bits()),
            "rms_norm_q",
            || rms_norm_src(width, eps, NormForm::Quantized),
            |module| {
                let rb = (rows as i64) * shape.0;
                self.launch(
                    module,
                    "rms_norm_q",
                    &[
                        (self.ptr(x, 0)?, [rb, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [rb, shape.1]),
                        (qa_ptr, [rb, shape.1]),
                        (das_ptr, [rb, 1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn fused_mlp(&self, mlp: FusedMlp) -> Result<bool> {
        self.launch_fused_mlp(mlp)
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
                            // kernel indexes the table by row rather than
                            // taking the absolute position as an argument.
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
        // The blocked kernel's online-softmax pass measures 3x cheaper than
        // the gemm path below at shapes both can take; a tensor-core variant
        // was tried and reverted as a net loss here.
        //
        // It masks its diagonal tile with `tril`, the causal mask only when
        // the tile starts where the query block does; a misaligned
        // continuation falls to the row kernel instead, which needs no alignment.
        let block = attention_block_tile(spec.head_dim);
        if spec.rows > 1 && spec.start_pos.is_multiple_of(block) {
            return self.attention_blocked(q, keys, values, spec, block, out);
        }
        // Whatever the blocked kernel declined still goes through the two
        // matmuls if it tiles evenly; see `attn_gemm_src`.
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

    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()> {
        self.check_distinct("delta_conv", packed, &[history, taps]);
        let channels = mix.channels();
        let (pr, c) = ((mix.pad() + mix.rows) as i64, channels as i64);
        // The packed destination is one row per (position, head): `batch`
        // positions of one head are then a column window of consecutive rows,
        // so the strided store is an ordinary tile.
        let batch = delta_conv_batch(mix.rows);
        let (planes, width) = ((3 * mix.rows) as i64, (mix.heads * mix.head_dim) as i64);
        // Both fused layouts space the planes evenly, so the second's offset
        // is the whole spacing.
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
            mix.rows,
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
                    mix.rows,
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
        // The five operands are one allocation, each a descriptor over its
        // own window of it.
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (r, d) = ((rows * heads) as i64, head_dim as i64);

        // A prompt goes through the chunked form (whole chunks, a state slice
        // dividing the head); everything else, including single-position
        // decode, falls through to the sequential kernel.
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
