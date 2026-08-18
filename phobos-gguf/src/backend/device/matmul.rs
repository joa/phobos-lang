// The Q8_0 projection.

use super::*;

impl DeviceBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_q8(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
        accumulate: bool,
    ) -> Result<()> {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "matmul_quant needs k ({k}) to be a multiple of {Q8_BLOCK}"
        );
        let quants = self.quants.borrow();
        let (qs, scales, row_scales, stored_n) = quants
            .get(w.0)
            .context("use of an unknown quantized weight handle")?;
        ensure!(
            *stored_n == n,
            "quantized weight was uploaded with n = {stored_n}, used with n = {n}"
        );
        let (w_ptr, s_ptr) = (qs.as_device_ptr().as_raw(), scales.as_device_ptr().as_raw());
        let rs_ptr = row_scales.as_device_ptr().as_raw();
        let out_ptr = self.ptr(out, 0)?;
        let blocks = k / Q8_BLOCK;
        let (qa_ptr, das_ptr) = self.act_ptrs(act)?;

        let f32_bytes = size_of::<f32>() as u64;

        // Four kernels, deepest tile first, each taking the whole tiles it can
        // before handing the remainder on. `qmma_t` carries the scales itself,
        // so it needs no bounds mask on the weight and keeps its 64-wide column
        // tile; a width not divisible by 64 goes to the kernel below. It comes
        // in two depths because the deeper one is 30% faster and a prompt is
        // rarely a whole number of 128-row tiles. Leftover tensor-core tiles
        // take `q8_mma`, and single rows finish on the matvec, as does decoding.
        let mut qmma_rows = 0;
        if n.is_multiple_of(Q8_QMMA_TN) {
            let wide = qmma_width(n);
            for (module, depth, tn) in [
                (&self.q8_qmma_deep[&wide], Q8_QMMA_TM, wide),
                (&self.q8_qmma, Q8_QMMA_SHALLOW, Q8_QMMA_TN),
            ] {
                let left = m - qmma_rows;
                let rows = left - left % depth;
                if rows == 0 {
                    continue;
                }
                // Only the deep tile ever leaves the grid starved enough for
                // this to fire: see Q8_QMMA_SPLIT_THRESHOLD.
                let splits = if depth == Q8_QMMA_TM && self.qmma_split {
                    q8_qmma_splits(rows, n, k, wide)
                } else {
                    1
                };
                if depth == Q8_QMMA_TM
                    && self.qmma_narrow
                    && q8_qmma_narrow_eligible(rows, n, wide)
                {
                    self.launch_qmma_narrow(
                        rows, qmma_rows, k, blocks, n, qa_ptr, das_ptr, w_ptr, s_ptr, out_ptr,
                        f32_bytes,
                    )?;
                } else if splits > 1 {
                    self.launch_qmma_split(
                        rows, qmma_rows, k, blocks, n, wide, splits, qa_ptr, das_ptr, w_ptr,
                        s_ptr, out_ptr, f32_bytes,
                    )?;
                } else {
                    self.launch(
                        module,
                        "q8_qmma",
                        &[
                            (qa_ptr + (qmma_rows * k) as u64, [rows as i64, k as i64]),
                            (
                                das_ptr + (qmma_rows * blocks) as u64 * f32_bytes,
                                [rows as i64, blocks as i64],
                            ),
                            (w_ptr, [n as i64, k as i64]),
                            (s_ptr, [blocks as i64, n as i64]),
                            (
                                out_ptr + (qmma_rows * n) as u64 * f32_bytes,
                                [rows as i64, n as i64],
                            ),
                        ],
                        ((rows / depth) as u32, (n / tn) as u32, 1),
                    )?;
                }
                qmma_rows += rows;
            }
        }

        let left = m - qmma_rows;
        let mma_rows = qmma_rows + left - left % Q8_MMA_TM;
        if mma_rows > qmma_rows {
            let rows = mma_rows - qmma_rows;
            self.launch(
                // The rows tile evenly by construction; only n can be ragged.
                self.q8_mma.pick(n.is_multiple_of(Q8_MMA_TN)),
                "q8_mma",
                &[
                    (qa_ptr + (qmma_rows * k) as u64, [rows as i64, k as i64]),
                    (
                        das_ptr + (qmma_rows * blocks) as u64 * f32_bytes,
                        [rows as i64, blocks as i64],
                    ),
                    (w_ptr, [n as i64, k as i64]),
                    (s_ptr, [blocks as i64, n as i64]),
                    (
                        out_ptr + (qmma_rows * n) as u64 * f32_bytes,
                        [rows as i64, n as i64],
                    ),
                ],
                ((rows / Q8_MMA_TM) as u32, n.div_ceil(Q8_MMA_TN) as u32, 1),
            )?;
        }

        let tiles_evenly = n.is_multiple_of(Q8_TN);
        let grid_n = n.div_ceil(Q8_TN) as u32;
        // A single row leaves the grid as short as the projection is wide, a
        // fraction of the card. `qdot_t` fills it from the contraction, 32 lanes
        // to an output, and wants no split; the split kernels, which pay a pass
        // to sum partials, stay for the widths its tile does not divide.
        let splits = q8_splits(n, k);
        let qdot = n.is_multiple_of(Q8_QDOT_TN);
        for row in mma_rows..m {
            let a_row = qa_ptr + (row * k) as u64;
            let as_row = das_ptr + (row * blocks) as u64 * f32_bytes;
            let c_row = out_ptr + (row * n) as u64 * f32_bytes;
            if qdot && self.persist_qdot {
                self.qdot_persistent(
                    n,
                    accumulate,
                    &[
                        (a_row, [1, k as i64]),
                        (as_row, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (rs_ptr, [n as i64, blocks as i64]),
                        (c_row, [1, n as i64]),
                    ],
                )?;
                continue;
            }
            if qdot {
                let (module, name) = if accumulate {
                    (&self.q8_qdot_add, "q8_qdot_add")
                } else {
                    (&self.q8_qdot, "q8_qdot")
                };
                self.launch(
                    module,
                    name,
                    &[
                        (a_row, [1, k as i64]),
                        (as_row, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (rs_ptr, [n as i64, blocks as i64]),
                        (c_row, [1, n as i64]),
                    ],
                    (n.div_ceil(Q8_QDOT_TN) as u32, 1, 1),
                )?;
                continue;
            }
            if splits == 1 {
                self.launch(
                    self.q8_dp4a.pick(tiles_evenly),
                    "q8_dp4a",
                    &[
                        (a_row, [1, k as i64]),
                        (as_row, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (s_ptr, [blocks as i64, n as i64]),
                        (c_row, [1, n as i64]),
                    ],
                    (grid_n, 1, 1),
                )?;
                continue;
            }
            let partials = self.split_partials(splits * n)?;
            let module = self.q8_split.pick(tiles_evenly);
            self.launch(
                module,
                "q8_split",
                &[
                    (a_row, [1, k as i64]),
                    (as_row, [1, blocks as i64]),
                    (w_ptr, [n as i64, k as i64]),
                    (s_ptr, [blocks as i64, n as i64]),
                    (partials, [splits as i64, n as i64]),
                ],
                (grid_n, splits as u32, 1),
            )?;
            self.launch(
                module,
                "q8_reduce",
                &[
                    (partials, [splits as i64, n as i64]),
                    (c_row, [1, n as i64]),
                ],
                (n.div_ceil(Q8_REDUCE_TN) as u32, 1, 1),
            )?;
        }
        Ok(())
    }

    /// The starved-grid path for `q8_qmma`'s deep tile: `splits` copies of the
    /// same `[Q8_QMMA_TM, wide]` patch, one per slice of `k`, landing in a
    /// `splits * rows * n` scratch that a second launch reduces into `out`.
    /// See `kernels::q8_qmma_split_src`'s doc comment for why the split index
    /// gets its own output operand instead of an offset computed from it.
    #[allow(clippy::too_many_arguments)]
    fn launch_qmma_split(
        &self,
        rows: usize,
        row_off: usize,
        k: usize,
        blocks: usize,
        n: usize,
        wide: usize,
        splits: usize,
        qa_ptr: u64,
        das_ptr: u64,
        w_ptr: u64,
        s_ptr: u64,
        out_ptr: u64,
        f32_bytes: u64,
    ) -> Result<()> {
        let key = (wide, k, splits);
        if !self.q8_qmma_split.borrow().contains_key(&key) {
            let split_src = q8_qmma_split_src(Q8_QMMA_CTA, k, splits);
            let reduce_src = q8_qmma_reduce_src(Q8_QMMA_CTA, splits);
            let split_mod = compile(
                &split_src,
                &[("TM", Q8_QMMA_TM), ("TN", wide)],
                "q8_qmma_split",
            )?;
            let reduce_mod = compile(&reduce_src, &[("TN", wide)], "q8_qmma_reduce")?;
            self.q8_qmma_split
                .borrow_mut()
                .insert(key, (split_mod, reduce_mod));
        }
        let cache = self.q8_qmma_split.borrow();
        let (split_mod, reduce_mod) = &cache[&key];

        let partials = self.split_partials(splits * rows * n)?;
        let plane_bytes = (rows * n) as u64 * f32_bytes;

        let mut operands = vec![
            (qa_ptr + (row_off * k) as u64, [rows as i64, k as i64]),
            (
                das_ptr + (row_off * blocks) as u64 * f32_bytes,
                [rows as i64, blocks as i64],
            ),
            (w_ptr, [n as i64, k as i64]),
            (s_ptr, [blocks as i64, n as i64]),
        ];
        for i in 0..splits {
            operands.push((partials + i as u64 * plane_bytes, [rows as i64, n as i64]));
        }
        self.launch(
            split_mod,
            "q8_qmma_split",
            &operands,
            (
                (rows / Q8_QMMA_TM) as u32,
                (n / wide) as u32,
                splits as u32,
            ),
        )?;

        let mut reduce_operands: Vec<(u64, [i64; 2])> = (0..splits)
            .map(|i| (partials + i as u64 * plane_bytes, [rows as i64, n as i64]))
            .collect();
        reduce_operands.push((
            out_ptr + (row_off * n) as u64 * f32_bytes,
            [rows as i64, n as i64],
        ));
        self.launch(
            reduce_mod,
            "q8_qmma_reduce",
            &reduce_operands,
            (rows as u32, (n / wide) as u32, 1),
        )
    }

    /// The narrow-CTA path for `q8_qmma`'s deep tile: the same `qmma_t`
    /// kernel at half the threads and half the column tile
    /// ([`Q8_QMMA_NARROW_CTA`], [`Q8_QMMA_NARROW_TN`]), which `qmma_patch`
    /// resolves to the *same* per-warp patch as the shipped 128-wide config
    /// (see that constant's doc comment) -- so this doubles the grid on a
    /// starved shape at unchanged tensor-core intensity, one launch, no
    /// reduction pass and no scratch buffer, unlike [`Self::launch_qmma_split`].
    #[allow(clippy::too_many_arguments)]
    fn launch_qmma_narrow(
        &self,
        rows: usize,
        row_off: usize,
        k: usize,
        blocks: usize,
        n: usize,
        qa_ptr: u64,
        das_ptr: u64,
        w_ptr: u64,
        s_ptr: u64,
        out_ptr: u64,
        f32_bytes: u64,
    ) -> Result<()> {
        if self.q8_qmma_narrow.borrow().is_none() {
            let src = q8_qmma_src(Q8_QMMA_NARROW_CTA);
            let module = compile(
                &src,
                &[("TM", Q8_QMMA_TM), ("TN", Q8_QMMA_NARROW_TN)],
                "q8_qmma",
            )?;
            *self.q8_qmma_narrow.borrow_mut() = Some(module);
        }
        let cache = self.q8_qmma_narrow.borrow();
        let module = cache.as_ref().expect("just compiled above");
        self.launch(
            module,
            "q8_qmma",
            &[
                (qa_ptr + (row_off * k) as u64, [rows as i64, k as i64]),
                (
                    das_ptr + (row_off * blocks) as u64 * f32_bytes,
                    [rows as i64, blocks as i64],
                ),
                (w_ptr, [n as i64, k as i64]),
                (s_ptr, [blocks as i64, n as i64]),
                (
                    out_ptr + (row_off * n) as u64 * f32_bytes,
                    [rows as i64, n as i64],
                ),
            ],
            (
                (rows / Q8_QMMA_TM) as u32,
                (n / Q8_QMMA_NARROW_TN) as u32,
                1,
            ),
        )
    }
}
