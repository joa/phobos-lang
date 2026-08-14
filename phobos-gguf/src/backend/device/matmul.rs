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
}
