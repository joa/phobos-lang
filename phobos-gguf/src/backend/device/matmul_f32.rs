// The f32 projection, for a weight held as f32.

use super::*;

impl DeviceBackend {
    /// [`Backend::matmul`].
    pub(super) fn matmul_dense(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
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

        let module = self.matvec.pick(n.is_multiple_of(MV_TN));
        let splits = mv_splits(n, k);
        if splits == 1 {
            return self.launch(
                module,
                "matvec",
                &[
                    (a_ptr, [1, k as i64]),
                    (w_ptr, [k as i64, n as i64]),
                    (out_ptr, [1, n as i64]),
                ],
                (n.div_ceil(MV_TN) as u32, 1, 1),
            );
        }
        let partials = self.split_partials(splits * n)?;
        self.launch(
            module,
            "matvec_split",
            &[
                (a_ptr, [1, k as i64]),
                (w_ptr, [k as i64, n as i64]),
                (partials, [splits as i64, n as i64]),
            ],
            (n.div_ceil(MV_TN) as u32, splits as u32, 1),
        )?;
        self.launch(
            self.q8_split.pick(n.is_multiple_of(Q8_REDUCE_TN)),
            "q8_reduce",
            &[(partials, [splits as i64, n as i64]), (out_ptr, [1, n as i64])],
            (n.div_ceil(Q8_REDUCE_TN) as u32, 1, 1),
        )
    }

    /// [`Backend::matmul_rows`]: a decode row by [`matvec_blocks_src`] where
    /// the row fits, otherwise a band of whole [`ROWS_TM`]-row programs, `k`
    /// split across programs where they are few, and then the rest a row a
    /// program.
    pub(super) fn matmul_rows_dense(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.check_distinct("matmul_rows", out, &[a, w]);
        ensure!(
            crate::backend::takes_rows(k, n),
            "a [{k}, {n}] weight is not one the row-major contraction takes"
        );
        let f32_bytes = size_of::<f32>() as u64;
        let (a_ptr, w_ptr, out_ptr) = (self.ptr(a, 0)?, self.ptr(w, 0)?, self.ptr(out, 0)?);
        if m == 1 && k <= BLOCKS_MAX_K {
            let tn = if n.is_multiple_of(2) { 2 } else { 1 };
            let kb = (k / 32) as i64;
            let operands = [(a_ptr, [kb, 32]), (w_ptr, [n as i64 * kb, 32]), (out_ptr, [n as i64, 1])];
            let kernel = RowsKernel::Blocks { k, tn };
            return self.with_kernel(
                &self.rows_matmuls,
                kernel,
                "matvec_blocks",
                || matvec_blocks_src(k, tn),
                |module| self.launch(module, "matvec_blocks", &operands, ((n / tn) as u32, 1, 1)),
            );
        }
        let band = m - m % ROWS_TM;
        let tn = (1..=ROWS_TN).rev().find(|t| n.is_multiple_of(*t)).unwrap_or(1);
        for (from, rows, tm) in [(0, band, ROWS_TM), (band, m - band, 1)] {
            if rows == 0 {
                continue;
            }
            let (a_at, out_at) = (a_ptr + (from * k) as u64 * f32_bytes, out_ptr + (from * n) as u64 * f32_bytes);
            let grid = ((rows / tm) as u32, (n / tn) as u32);
            let mut splits = (ROWS_SPLIT_TARGET / (grid.0 * grid.1) as usize).clamp(1, ROWS_MAX_SPLITS);
            while splits > 1 && !k.is_multiple_of(splits * 16) {
                splits -= 1;
            }
            if tm > 1 && splits > 1 {
                let partials = self.split_partials(splits * rows * n)?;
                let operands = [
                    (a_at, [rows as i64, k as i64]),
                    (w_ptr, [n as i64, k as i64]),
                    (partials, [(splits * rows) as i64, n as i64]),
                ];
                self.with_kernel(
                    &self.rows_matmuls,
                    RowsKernel::Tiles { tm, tn },
                    "matmul_rows",
                    || matmul_rows_src(tm, tn) + &matmul_rows_split_src(tm, tn),
                    |module| self.launch(module, "matmul_rows_split", &operands, (grid.0, grid.1, splits as u32)),
                )?;
                // The band's outputs are contiguous, so the sum runs over
                // them as one row.
                let len = rows * n;
                self.launch(
                    self.q8_split.pick(len.is_multiple_of(Q8_REDUCE_TN)),
                    "q8_reduce",
                    &[(partials, [splits as i64, len as i64]), (out_at, [1, len as i64])],
                    (len.div_ceil(Q8_REDUCE_TN) as u32, 1, 1),
                )?;
                continue;
            }
            let operands = [
                (a_at, [rows as i64, k as i64]),
                (w_ptr, [n as i64, k as i64]),
                (out_at, [rows as i64, n as i64]),
            ];
            self.with_kernel(
                &self.rows_matmuls,
                RowsKernel::Tiles { tm, tn },
                "matmul_rows",
                || matmul_rows_src(tm, tn) + &matmul_rows_split_src(tm, tn),
                |module| self.launch(module, "matmul_rows", &operands, (grid.0, grid.1, 1)),
            )?;
        }
        Ok(())
    }
}
