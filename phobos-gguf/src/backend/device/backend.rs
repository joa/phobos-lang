// The `Backend` implementation. A trait impl cannot be split across
// files, so this is the whole surface in one place; the work each method
// does lives in the sibling modules.

use super::*;

impl Backend for DeviceBackend {
    fn alloc(&self, len: usize) -> Result<Buf> {
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
        if let Some(buffer) = taken {
            self.pool.put(buffer);
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
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("upload lost its buffer")?;
        buffer.index(0..data.len()).copy_from(data)?;
        Ok(buf)
    }

    fn zeroed(&self, len: usize) -> Result<Buf> {
        let buf = self.alloc_written_now(len)?;
        let ptr = self.ptr(buf, 0)?;
        // On the stream, so it orders with the pass rather than forcing the
        // synchronization an upload's staging copy would.
        cuda_ok(
            // SAFETY: the allocation is at least `len` floats and the handle
            // holds it alive for the duration.
            unsafe { cust::sys::cuMemsetD32Async(ptr, 0, len, self.stream.as_inner()) },
            "zeroing a device allocation",
        )?;
        Ok(buf)
    }

    fn begin_pass(&self) -> Result<()> {
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
        // A pass that had to flush is only partly recorded, so it goes out as
        // launches and leaves the cached graph alone. The next pass records
        // whole and replaces it.
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
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?;
        ensure!(
            buffer.len() >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            buffer.len()
        );
        // Through page-locked staging. Straight into a Vec the driver bounces
        // the copy through its own pinned staging a page at a time, at about a
        // third of the rate, and the logits are a megabyte a token here.
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
        let buf = self.upload(data)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        let buf = self.upload(&fill()?)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.check_distinct("matmul", out, &[a, w]);
        let (a_ptr, w_ptr, out_ptr) = (self.ptr(a, 0)?, self.ptr(w, 0)?, self.ptr(out, 0)?);

        // The tiled kernel rounds a single row up to a whole TILE_M tile, so
        // decoding stays on the matvec specialization and anything wider tiles.
        // Either handles a ragged shape by masking the boundary tile, so the
        // choice is only which does less arithmetic.
        if m > 1 {
            let tiles_evenly = m.is_multiple_of(TILE_M) && n.is_multiple_of(TILE_N);
            return self.launch(
                self.matmul.pick(tiles_evenly),
                "matmul",
                &[
                    (a_ptr, [m as i64, k as i64]),
                    (w_ptr, [k as i64, n as i64]),
                    (out_ptr, [m as i64, n as i64]),
                ],
                (m.div_ceil(TILE_M) as u32, n.div_ceil(TILE_N) as u32, 1),
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
        let uploaded = (
            DeviceBuffer::from_slice(&planes.qs)?,
            DeviceBuffer::from_slice(&planes.scales)?,
            DeviceBuffer::from_slice(&row_scales)?,
            n,
        );
        let mut quants = self.quants.borrow_mut();
        quants.push(uploaded);
        let buf = QBuf(quants.len() - 1);
        drop(quants);
        self.q_constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        let (act, qa_ptr, das_ptr) = self.act_slot(m, k)?;
        let blocks = k / Q8_BLOCK;
        let a_ptr = self.ptr(a, 0)?;
        let rows = m * blocks;
        let (module, tile) = if rows * Q8_BLOCK >= WIDE_FLOOR {
            (&self.quantize_wide, QUANT_TB_WIDE)
        } else {
            (&self.quantize, QUANT_TB)
        };
        self.launch(
            module,
            "quantize",
            &[
                (a_ptr, [rows as i64, Q8_BLOCK as i64]),
                (qa_ptr, [rows as i64, Q8_BLOCK as i64]),
                (das_ptr, [rows as i64, 1]),
            ],
            (rows.div_ceil(tile) as u32, 1, 1),
        )?;
        Ok(act)
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
        // Only the single-row kernel accumulates. A prefill goes through the
        // tensor cores, where the residual add is a rounding error on the pass
        // rather than a launch that matters.
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
        if !self.fused_mlp {
            return Ok(false);
        }
        // The chain is the whole of what this backend says about the MLP. Which
        // stages share a loop nest, what reaches memory, where the one barrier
        // goes and whether the shape can be fused at all are the pass's, and a
        // shape it declines takes the four launches.
        let chain = fuse::mlp_chain(
            mlp.x,
            mlp.gain,
            mlp.gate_up,
            mlp.down,
            mlp.d_model,
            mlp.d_ff,
            mlp.eps,
        );
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(false);
        };
        self.fused_launch(&chain, &key)?;
        Ok(true)
    }

    fn fused_project(&self, project: FusedProject) -> Result<Fused> {
        if !self.fused_project {
            return Ok(Fused::default());
        }
        // Dropping the tail here rather than at the frontend keeps the recording
        // side free of the gate: the chain is what the gate is about.
        let project = FusedProject {
            mix: project.mix.filter(|_| self.fused_mix),
            ..project
        };
        let Some(chain) = fuse::project_chain(&project) else {
            return Ok(Fused::default());
        };
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(Fused::default());
        };
        self.fused_launch(&chain, &key)?;
        Ok(Fused {
            project: true,
            mix: project.mix.is_some(),
        })
    }

    fn fused_attn_out(&self, out: FusedAttnOut) -> Result<bool> {
        if !self.fused_attn_out {
            return Ok(false);
        }
        let Some(chain) = fuse::attn_out_chain(out.x, out.w, out.dest, out.width, out.d_model)
        else {
            return Ok(false);
        };
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(false);
        };
        self.fused_launch(&chain, &key)?;
        Ok(true)
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
        // The blocked kernel goes first: one online-softmax pass that never
        // materializes the score matrix measures 3x cheaper than the
        // three-launch gemm path below at the one shape both can take. A
        // tensor-core f16 variant of it was tried and reverted, a net loss at
        // this shape -- a 16-row query tile against a 128-wide head is a small
        // matmul, dominated by the WMMA path's per-launch fragment staging
        // rather than by compute.
        //
        // It masks its one diagonal tile with `tril`, which is the causal mask
        // only when that tile starts where the query block does. A prompt into
        // an empty cache always does; a continuation whose cache is not a
        // whole number of blocks deep does not, and takes the row kernel,
        // which needs no alignment.
        let block = attention_block_tile(spec.head_dim);
        if spec.rows > 1 && spec.start_pos.is_multiple_of(block) {
            return self.attention_blocked(q, keys, values, spec, block, out);
        }
        // Whatever the blocked kernel declined (a misaligned continuation, or
        // a head dimension whose block tile does not divide 64) still goes
        // through the two matmuls if it tiles evenly; see `attn_gemm_src`.
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
        // The packed destination is one row per (position, head), the same
        // memory as one row per position with the heads side by side. Read that
        // way a program's `batch` positions of one head are a column window of
        // consecutive rows, so the strided store is an ordinary tile.
        let batch = delta_conv_batch(mix.rows);
        let (planes, width) = ((3 * mix.rows) as i64, (mix.heads * mix.head_dim) as i64);
        // Both fused layouts space the planes evenly, so the second's offset is
        // the whole spacing.
        let plane_stride = mix.planes[1];
        ensure!(
            mix.planes == [0, plane_stride, 2 * plane_stride],
            "delta_conv expects evenly spaced query, key and value planes"
        );
        let key = (
            mix.heads,
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
        // The five operands are one allocation, each a descriptor over its own
        // window of it, which is the point of packing them.
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (r, d) = ((rows * heads) as i64, head_dim as i64);

        // A prompt goes through the chunked form, which needs whole chunks and
        // a state slice dividing the head. Decoding is a single position, and
        // every other shape falls through to the sequential kernel.
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
