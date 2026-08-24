// The Q8_0 projection.

use super::*;

/// Scratch cap for [`DeviceBackend::project_raw_dense`]'s dequantized weight
/// strip, bounded well under card memory instead of dequantizing a whole
/// `[K, N]` tensor at once.
const RAW_DEQUANT_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// The five buffers every batched Q8_0 kernel contracts over. The weight
/// pair is whole for every launch; the rest are offset to the row band
/// a launch covers.
struct Q8Tiles {
    qa_ptr: u64,
    das_ptr: u64,
    w_ptr: u64,
    s_ptr: u64,
    out_ptr: u64,
    k: usize,
    blocks: usize,
    n: usize,
}

impl Q8Tiles {
    /// Operands for a launch covering `rows` rows from `row_off`.
    fn band(&self, row_off: usize, rows: usize) -> [(u64, [i64; 2]); 5] {
        let f32_bytes = size_of::<f32>() as u64;
        let (rows, k, blocks, n) = (
            rows as i64,
            self.k as i64,
            self.blocks as i64,
            self.n as i64,
        );
        [
            (self.qa_ptr + (row_off * self.k) as u64, [rows, k]),
            (
                self.das_ptr + (row_off * self.blocks) as u64 * f32_bytes,
                [rows, blocks],
            ),
            (self.w_ptr, [n, k]),
            (self.s_ptr, [blocks, n]),
            (
                self.out_ptr + (row_off * self.n) as u64 * f32_bytes,
                [rows, n],
            ),
        ]
    }
}

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
        let tiles = Q8Tiles {
            qa_ptr,
            das_ptr,
            w_ptr,
            s_ptr,
            out_ptr,
            k,
            blocks,
            n,
        };

        // Four kernels, deepest tile first, each taking the whole tiles it
        // can before handing the remainder on: qmma_t (in two depths, the
        // deeper one 30% faster), then q8_mma for the leftover tensor-core
        // tiles, then the matvec for single rows and decoding.
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
                    self.launch_qmma_narrow(&tiles, qmma_rows, rows)?;
                } else if splits > 1 {
                    self.launch_qmma_split(&tiles, qmma_rows, rows, wide, splits)?;
                } else {
                    self.launch(
                        module,
                        "q8_qmma",
                        &tiles.band(qmma_rows, rows),
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
                &tiles.band(qmma_rows, rows),
                ((rows / Q8_MMA_TM) as u32, n.div_ceil(Q8_MMA_TN) as u32, 1),
            )?;
        }

        let tiles_evenly = n.is_multiple_of(Q8_TN);
        let grid_n = n.div_ceil(Q8_TN) as u32;
        // A single row leaves the grid as short as the projection is wide.
        // qdot_t fills it from the contraction and wants no split; the split
        // kernels, which pay a pass to sum partials, cover the widths its
        // tile does not divide.
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
    /// same `[Q8_QMMA_TM, wide]` patch, one per slice of `k`, reduced from a
    /// `splits * rows * n` scratch into `out` by a second launch.
    fn launch_qmma_split(
        &self,
        tiles: &Q8Tiles,
        row_off: usize,
        rows: usize,
        wide: usize,
        splits: usize,
    ) -> Result<()> {
        let (k, n) = (tiles.k, tiles.n);
        let key = (wide, k, splits);
        if !self.q8_qmma_split.borrow().contains_key(&key) {
            let split_mod = compile(
                &q8_qmma_split_src(Q8_QMMA_CTA, k, splits),
                &[("TM", Q8_QMMA_TM), ("TN", wide)],
                "q8_qmma_split",
            )?;
            let reduce_mod = compile(
                &q8_qmma_reduce_src(Q8_QMMA_CTA, splits),
                &[("TN", wide)],
                "q8_qmma_reduce",
            )?;
            self.q8_qmma_split
                .borrow_mut()
                .insert(key, (split_mod, reduce_mod));
        }
        let cache = self.q8_qmma_split.borrow();
        let (split_mod, reduce_mod) = &cache[&key];

        let band = tiles.band(row_off, rows);
        let partials = self.split_partials(splits * rows * n)?;
        let plane_bytes = (rows * n) as u64 * size_of::<f32>() as u64;
        let plane = |i: usize| (partials + i as u64 * plane_bytes, [rows as i64, n as i64]);

        // The four inputs, then one output plane per split.
        let mut operands = band[..4].to_vec();
        operands.extend((0..splits).map(plane));
        self.launch(
            split_mod,
            "q8_qmma_split",
            &operands,
            ((rows / Q8_QMMA_TM) as u32, (n / wide) as u32, splits as u32),
        )?;

        let mut reduce_operands: Vec<_> = (0..splits).map(plane).collect();
        reduce_operands.push(band[4]);
        self.launch(
            reduce_mod,
            "q8_qmma_reduce",
            &reduce_operands,
            (rows as u32, (n / wide) as u32, 1),
        )
    }

    /// A raw-block weight's projection: one kernel, no split, no tensor
    /// cores, decoding straight from the file's bytes. The module, tile
    /// width and block size come from the weight's stored format.
    pub(super) fn project_raw(
        &self,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let raws = self.raw_quants.borrow();
        let (bytes, d, dmin, stored_n, nb, quant) = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        ensure!(
            *stored_n == n,
            "raw weight was uploaded with n = {stored_n}, used with n = {n}"
        );
        // The qdot_t kernels need their qb/d slices provably in bounds
        // (@aligned(N = TN)), which only a whole number of tiles gives them;
        // a ragged n falls back to the masked-tolerant body.
        let iq1s_qdot_eligible = *quant == Quant::IQ1_S && m == 1 && n.is_multiple_of(IQ1S_TN);
        let iq2xxs_qdot_eligible =
            *quant == Quant::IQ2_XXS && m == 1 && n.is_multiple_of(IQ2XXS_TN);
        let iq1m_qdot_eligible = *quant == Quant::IQ1_M && m == 1 && n.is_multiple_of(IQ1M_TN);
        let iq2s_qdot_eligible = *quant == Quant::IQ2_S && m == 1 && n.is_multiple_of(IQ2S_TN);
        let iq2xs_qdot_eligible = *quant == Quant::IQ2_XS && m == 1 && n.is_multiple_of(IQ2XS_TN);
        let iq3xxs_qdot_eligible =
            *quant == Quant::IQ3_XXS && m == 1 && n.is_multiple_of(IQ3XXS_TN);
        let iq3s_qdot_eligible = *quant == Quant::IQ3_S && m == 1 && n.is_multiple_of(IQ3S_TN);
        let iq4xs_qdot_eligible = *quant == Quant::IQ4_XS && m == 1 && n.is_multiple_of(IQ4XS_TN);
        let q2k_qdot_eligible = *quant == Quant::Q2_K && m == 1 && n.is_multiple_of(Q2K_TN);
        let q3k_qdot_eligible = *quant == Quant::Q3_K && m == 1 && n.is_multiple_of(Q3K_TN);
        let (module, name, block_bytes, tn) = if iq1s_qdot_eligible {
            (&self.iq1s_qdot_matvec, "iq1s_qdot_matvec", 50, IQ1S_TN)
        } else if iq2xxs_qdot_eligible {
            (&self.iq2xxs_qdot_matvec, "iq2xxs_qdot_matvec", 66, IQ2XXS_TN)
        } else if iq1m_qdot_eligible {
            (&self.iq1m_qdot_matvec, "iq1m_qdot_matvec", 56, IQ1M_TN)
        } else if iq2s_qdot_eligible {
            (&self.iq2s_qdot_matvec, "iq2s_qdot_matvec", 82, IQ2S_TN)
        } else if iq2xs_qdot_eligible {
            (&self.iq2xs_qdot_matvec, "iq2xs_qdot_matvec", 74, IQ2XS_TN)
        } else if iq3xxs_qdot_eligible {
            (&self.iq3xxs_qdot_matvec, "iq3xxs_qdot_matvec", 98, IQ3XXS_TN)
        } else if iq3s_qdot_eligible {
            (&self.iq3s_qdot_matvec, "iq3s_qdot_matvec", 110, IQ3S_TN)
        } else if iq4xs_qdot_eligible {
            (&self.iq4xs_qdot_matvec, "iq4xs_qdot_matvec", 136, IQ4XS_TN)
        } else if q2k_qdot_eligible {
            (&self.q2k_qdot_matvec, "q2k_qdot_matvec", 84, Q2K_TN)
        } else if q3k_qdot_eligible {
            (&self.q3k_qdot_matvec, "q3k_qdot_matvec", 110, Q3K_TN)
        } else {
            match quant {
                Quant::Q2_K => (&self.q2k_matvec, "q2k_matvec", 84, Q2K_TN),
                Quant::Q3_K => (&self.q3k_matvec, "q3k_matvec", 110, Q3K_TN),
                Quant::IQ1_S => (&self.iq1s_matvec, "iq1s_matvec", 50, IQ1S_TN),
                Quant::IQ2_XXS => (&self.iq2xxs_matvec, "iq2xxs_matvec", 66, IQ2XXS_TN),
                Quant::IQ1_M => (&self.iq1m_matvec, "iq1m_matvec", 56, IQ1M_TN),
                Quant::IQ2_S => (&self.iq2s_matvec, "iq2s_matvec", 82, IQ2S_TN),
                Quant::IQ2_XS => (&self.iq2xs_matvec, "iq2xs_matvec", 74, IQ2XS_TN),
                Quant::IQ3_XXS => (&self.iq3xxs_matvec, "iq3xxs_matvec", 98, IQ3XXS_TN),
                Quant::IQ3_S => (&self.iq3s_matvec, "iq3s_matvec", 110, IQ3S_TN),
                Quant::IQ4_XS => (&self.iq4xs_matvec, "iq4xs_matvec", 136, IQ4XS_TN),
                other => anyhow::bail!("no raw kernel launches {}", other.name()),
            }
        };
        let rb = nb * block_bytes;
        let (bytes_ptr, d_ptr) = (bytes.as_device_ptr().as_raw(), d.as_device_ptr().as_raw());
        let a_ptr = self.ptr(a, 0)?;
        let out_ptr = self.ptr(out, 0)?;
        let mut operands = vec![
            (a_ptr, [m as i64, k as i64]),
            (bytes_ptr, [n as i64, rb as i64]),
            (d_ptr, [n as i64, *nb as i64]),
        ];
        if let Some(dmin) = dmin {
            operands.push((dmin.as_device_ptr().as_raw(), [n as i64, *nb as i64]));
        }
        match quant {
            Quant::IQ1_S if iq1s_qdot_eligible => {
                // iq1s_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq1s_matvec's gather-based body.
                operands.push((
                    self.iq1s_grid.as_device_ptr().as_raw(),
                    [1, IQ1S_GRID_LEN as i64],
                ));
            }
            Quant::IQ1_M if iq1m_qdot_eligible => {
                // iq1m_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq1m_matvec's gather-based body.
                operands.push((
                    self.iq1s_grid.as_device_ptr().as_raw(),
                    [1, IQ1S_GRID_LEN as i64],
                ));
            }
            Quant::IQ1_S | Quant::IQ1_M => {
                operands.push((
                    self.iq1s_grid.as_device_ptr().as_raw(),
                    [1, IQ1S_GRID_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ2_XXS if iq2xxs_qdot_eligible => {
                // iq2xxs_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq2xxs_matvec's gather-based body.
                operands.push((
                    self.iq2xxs_grid.as_device_ptr().as_raw(),
                    [1, IQ2XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
            }
            Quant::IQ2_XXS => {
                operands.push((
                    self.iq2xxs_grid.as_device_ptr().as_raw(),
                    [1, IQ2XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ2_S if iq2s_qdot_eligible => {
                // iq2s_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq2s_matvec's gather-based body.
                operands.push((
                    self.iq2s_grid.as_device_ptr().as_raw(),
                    [1, IQ2S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs.as_device_ptr().as_raw(),
                    [1, IQ2S_SIGNS_LEN as i64],
                ));
            }
            Quant::IQ2_S => {
                operands.push((
                    self.iq2s_grid.as_device_ptr().as_raw(),
                    [1, IQ2S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs.as_device_ptr().as_raw(),
                    [1, IQ2S_SIGNS_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ2_XS if iq2xs_qdot_eligible => {
                // iq2xs_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq2xs_matvec's gather-based body.
                operands.push((
                    self.iq2xs_grid.as_device_ptr().as_raw(),
                    [1, IQ2XS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
            }
            Quant::IQ2_XS => {
                operands.push((
                    self.iq2xs_grid.as_device_ptr().as_raw(),
                    [1, IQ2XS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ3_XXS if iq3xxs_qdot_eligible => {
                // iq3xxs_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq3xxs_matvec's gather-based body.
                operands.push((
                    self.iq3xxs_grid.as_device_ptr().as_raw(),
                    [1, IQ3XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
            }
            Quant::IQ3_XXS => {
                operands.push((
                    self.iq3xxs_grid.as_device_ptr().as_raw(),
                    [1, IQ3XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs.as_device_ptr().as_raw(),
                    [1, IQ2XXS_SIGNS_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ3_S if iq3s_qdot_eligible => {
                // iq3s_qdot_matvec computes its own byte offsets; no iota8
                // operand, unlike iq3s_matvec's gather-based body.
                operands.push((
                    self.iq3s_grid.as_device_ptr().as_raw(),
                    [1, IQ3S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs.as_device_ptr().as_raw(),
                    [1, IQ2S_SIGNS_LEN as i64],
                ));
            }
            Quant::IQ3_S => {
                operands.push((
                    self.iq3s_grid.as_device_ptr().as_raw(),
                    [1, IQ3S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs.as_device_ptr().as_raw(),
                    [1, IQ2S_SIGNS_LEN as i64],
                ));
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            Quant::IQ4_XS => {
                operands.push((
                    self.iq4xs_codebook.as_device_ptr().as_raw(),
                    [1, IQ4XS_CODEBOOK_LEN as i64],
                ));
            }
            _ => {}
        }
        operands.push((out_ptr, [m as i64, n as i64]));
        self.launch(module, name, &operands, (n.div_ceil(tn) as u32, m as u32, 1))
    }

    /// Whether `w`'s format has a dequant kernel in [`Self::project_raw_dense`].
    pub(super) fn raw_quant_is(&self, w: RawBuf, quant: Quant) -> bool {
        self.raw_quants
            .borrow()
            .get(w.0)
            .is_some_and(|(_, _, _, _, _, q)| *q == quant)
    }

    /// [`Self::project_raw`], but for `m > 1`: dequantizes each output-column
    /// strip of the weight into an `f32` `[K, strip]` scratch once and runs
    /// `Backend::matmul` against it for all `m` rows, instead of redoing the
    /// decode per `(output tile, row)`. Strip-sized to stay under
    /// [`RAW_DEQUANT_BUDGET_BYTES`] rather than dequantizing the whole `[K, N]`
    /// weight at once.
    pub(super) fn project_raw_dense(
        &self,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let raws = self.raw_quants.borrow();
        let (bytes, d, dmin, stored_n, nb, quant) = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        ensure!(
            *stored_n == n,
            "raw weight was uploaded with n = {stored_n}, used with n = {n}"
        );

        // Byte width, tile size and extra grid/signs/iota operands mirror
        // project_raw's own match table.
        let (module, name, block_bytes, tn) = match quant {
            Quant::IQ1_S => (&self.iq1s_dequant, "iq1s_dequant", 50, IQ1S_TN),
            Quant::IQ2_XXS => (&self.iq2xxs_dequant, "iq2xxs_dequant", 66, IQ2XXS_TN),
            Quant::IQ1_M => (&self.iq1m_dequant, "iq1m_dequant", 56, IQ1M_TN),
            Quant::IQ2_S => (&self.iq2s_dequant, "iq2s_dequant", 82, IQ2S_TN),
            Quant::IQ2_XS => (&self.iq2xs_dequant, "iq2xs_dequant", 74, IQ2XS_TN),
            Quant::IQ3_XXS => (&self.iq3xxs_dequant, "iq3xxs_dequant", 98, IQ3XXS_TN),
            Quant::IQ3_S => (&self.iq3s_dequant, "iq3s_dequant", 110, IQ3S_TN),
            Quant::IQ4_XS => (&self.iq4xs_dequant, "iq4xs_dequant", 136, IQ4XS_TN),
            Quant::Q2_K => (&self.q2k_dequant, "q2k_dequant", 84, Q2K_TN),
            other => anyhow::bail!("project_raw_dense has no dequant kernel for {}", other.name()),
        };
        let rb = nb * block_bytes;
        let (bytes_ptr, d_ptr) = (bytes.as_device_ptr().as_raw(), d.as_device_ptr().as_raw());
        let f16_bytes = size_of::<u16>() as u64;

        let strip = (RAW_DEQUANT_BUDGET_BYTES / (k * size_of::<f32>())).clamp(1, n);
        let mut n0 = 0;
        while n0 < n {
            let cur = strip.min(n - n0);
            let scratch_w = self.alloc(k * cur)?;
            let mut operands = vec![
                (bytes_ptr + (n0 * rb) as u64, [cur as i64, rb as i64]),
                (d_ptr + (n0 * nb) as u64 * f16_bytes, [cur as i64, *nb as i64]),
            ];
            if *quant == Quant::Q2_K {
                let dmin_ptr = dmin
                    .as_ref()
                    .context("Q2_K raw weight is missing its dmin plane")?
                    .as_device_ptr()
                    .as_raw();
                operands.push((dmin_ptr + (n0 * nb) as u64 * f16_bytes, [cur as i64, *nb as i64]));
            }
            match quant {
                Quant::IQ1_S | Quant::IQ1_M => {
                    operands.push((self.iq1s_grid.as_device_ptr().as_raw(), [1, IQ1S_GRID_LEN as i64]));
                }
                Quant::IQ2_XXS => {
                    operands.push((self.iq2xxs_grid.as_device_ptr().as_raw(), [1, IQ2XXS_GRID_LEN as i64]));
                    operands.push((self.iq2xxs_signs.as_device_ptr().as_raw(), [1, IQ2XXS_SIGNS_LEN as i64]));
                }
                Quant::IQ2_S => {
                    operands.push((self.iq2s_grid.as_device_ptr().as_raw(), [1, IQ2S_GRID_LEN as i64]));
                    operands.push((self.iq2s_signs.as_device_ptr().as_raw(), [1, IQ2S_SIGNS_LEN as i64]));
                }
                Quant::IQ2_XS => {
                    operands.push((self.iq2xs_grid.as_device_ptr().as_raw(), [1, IQ2XS_GRID_LEN as i64]));
                    operands.push((self.iq2xxs_signs.as_device_ptr().as_raw(), [1, IQ2XXS_SIGNS_LEN as i64]));
                }
                Quant::IQ3_XXS => {
                    operands.push((self.iq3xxs_grid.as_device_ptr().as_raw(), [1, IQ3XXS_GRID_LEN as i64]));
                    operands.push((self.iq2xxs_signs.as_device_ptr().as_raw(), [1, IQ2XXS_SIGNS_LEN as i64]));
                }
                Quant::IQ3_S => {
                    operands.push((self.iq3s_grid.as_device_ptr().as_raw(), [1, IQ3S_GRID_LEN as i64]));
                    operands.push((self.iq2s_signs.as_device_ptr().as_raw(), [1, IQ2S_SIGNS_LEN as i64]));
                }
                Quant::IQ4_XS => {
                    operands.push((
                        self.iq4xs_codebook.as_device_ptr().as_raw(),
                        [1, IQ4XS_CODEBOOK_LEN as i64],
                    ));
                }
                // Q2_K decodes from static offsets: no grid, no signs.
                Quant::Q2_K => {}
                _ => unreachable!("checked in the match above"),
            }
            // IQ4_XS and Q2_K need no gather-index tile (fixed codebook,
            // static-offset decode); every other kernel here does.
            if *quant != Quant::IQ4_XS && *quant != Quant::Q2_K {
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            operands.push((self.ptr(scratch_w, 0)?, [k as i64, cur as i64]));
            self.launch(module, name, &operands, (cur.div_ceil(tn) as u32, 1, 1))?;

            let scratch_out = self.alloc(m * cur)?;
            self.matmul(a, m, k, scratch_w, cur, scratch_out)?;
            self.copy_2d(
                Plane { buf: scratch_out, offset: 0, pitch: cur },
                Plane { buf: out, offset: n0, pitch: n },
                m,
                cur,
            )?;

            self.release(scratch_out);
            self.release(scratch_w);
            n0 += cur;
        }
        Ok(())
    }

    /// The narrow-CTA path for `q8_qmma`'s deep tile: the same `qmma_t` kernel
    /// at half the threads and column tile ([`Q8_QMMA_NARROW_CTA`],
    /// [`Q8_QMMA_NARROW_TN`]), doubling the grid on a starved shape with one
    /// launch and no scratch, unlike [`Self::launch_qmma_split`].
    fn launch_qmma_narrow(&self, tiles: &Q8Tiles, row_off: usize, rows: usize) -> Result<()> {
        if self.q8_qmma_narrow.borrow().is_none() {
            let module = compile(
                &q8_qmma_src(Q8_QMMA_NARROW_CTA),
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
            &tiles.band(row_off, rows),
            (
                (rows / Q8_QMMA_TM) as u32,
                (tiles.n / Q8_QMMA_NARROW_TN) as u32,
                1,
            ),
        )
    }
}
