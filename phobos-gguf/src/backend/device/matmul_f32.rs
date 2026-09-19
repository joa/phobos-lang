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
}
