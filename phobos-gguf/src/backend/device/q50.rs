// Q5_0 weights: uploaded as their blocks without the scale, 20 bytes per 32
// elements, beside the two scale planes a Q8_0 weight has. `q50_qdot_t` and
// `q50_qmma_t` widen each quant to `q - 16` as they load it, so the weight
// costs 0.625 bytes an element on the device instead of the one byte the
// Q8_0 planes take.

use super::*;

impl DeviceBackend {
    /// Whether `w` is a Q5_0 weight, which only [`Self::project_q50`] and no
    /// fused chain reads.
    pub(super) fn is_q50(&self, w: QBuf) -> bool {
        self.quants.borrow().get(w.0).is_some_and(|q| q.q50)
    }

    /// [`Self::project_q8`] for a Q5_0 weight: whole tiles of rows on the
    /// tensor cores, the rest a matvec per row. Only a single row
    /// accumulates, as [`Backend::matmul_quant_add`] asks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_q50(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        (qs, scales, row_scales): (u64, u64, u64),
        n: usize,
        out: Buf,
        accumulate: bool,
    ) -> Result<()> {
        ensure!(
            n.is_multiple_of(Q8_QDOT_TN),
            "a Q5_0 projection needs n ({n}) to be a multiple of {Q8_QDOT_TN}"
        );
        ensure!(!accumulate || m == 1, "a Q5_0 projection accumulates one row at a time");
        let blocks = k / Q8_BLOCK;
        let kw = blocks * Quant::Q5_0.device_block_bytes();
        let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
        let out_ptr = self.ptr(out, 0)?;
        let f32_bytes = size_of::<f32>() as u64;
        let at = |row: usize| {
            (
                qa_ptr + (row * k) as u64,
                das_ptr + (row * blocks) as u64 * f32_bytes,
                out_ptr + (row * n) as u64 * f32_bytes,
            )
        };

        let mut done = 0;
        if m > 1 && qmma_takes(n) {
            for (depth, tn) in [(Q8_QMMA_TM, qmma_width(n)), (Q8_QMMA_SHALLOW, Q8_QMMA_TN)] {
                let left = m - done;
                let rows = left - left % depth;
                if rows == 0 || !n.is_multiple_of(tn) {
                    continue;
                }
                let (a_rows, as_rows, c_rows) = at(done);
                self.with_kernel(
                    &self.q50_kernels,
                    (depth, tn),
                    "q50_qmma",
                    || q50_qmma_src(Q8_QMMA_CTA, depth, tn),
                    |module| {
                        self.launch(
                            module,
                            "q50_qmma",
                            &[
                                (a_rows, [rows as i64, k as i64]),
                                (as_rows, [rows as i64, blocks as i64]),
                                (qs, [n as i64, kw as i64]),
                                (scales, [blocks as i64, n as i64]),
                                (c_rows, [rows as i64, n as i64]),
                            ],
                            ((rows / depth) as u32, (n / tn) as u32, 1),
                        )
                    },
                )?;
                done += rows;
            }
        }

        let name = if accumulate { "q50_qdot_add" } else { "q50_qdot" };
        for row in done..m {
            let (a_row, as_row, c_row) = at(row);
            // Keyed apart from every qmma tile, whose depth is never zero.
            self.with_kernel(
                &self.q50_kernels,
                (0, usize::from(accumulate)),
                "q50_qdot",
                || q50_qdot_src(accumulate),
                |module| {
                    self.launch(
                        module,
                        name,
                        &[
                            (a_row, [1, k as i64]),
                            (as_row, [1, blocks as i64]),
                            (qs, [n as i64, kw as i64]),
                            (row_scales, [n as i64, blocks as i64]),
                            (c_row, [1, n as i64]),
                        ],
                        ((n / Q8_QDOT_TN) as u32, 1, 1),
                    )
                },
            )?;
        }
        Ok(())
    }
}
