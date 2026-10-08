// The Q8_0 projection.

use super::*;

/// A dp4a decode matvec picked for one weight: the module, its name, the
/// output tile, and the lookup tables it takes after the block operands.
/// A dp4a decode matvec and the lookup tables it reads, as (address,
/// length) pairs.
type I8Kernel<'a> = (&'a formats::Kernel, Vec<(u64, i64)>);


/// Blocks below which a decode matvec takes its narrow tile; see `i8_pick`.
const WIDE_TILE_MIN_BLOCKS: usize = 8;

/// The five buffers every batched Q8_0 kernel contracts over. The weight
/// pair is passed whole. The others are offset to each launch's row band.
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
        let q = quants
            .get(w.0)
            .context("use of an unknown quantized weight handle")?;
        ensure!(
            q.n == n,
            "quantized weight was uploaded with n = {}, used with n = {n}",
            q.n
        );
        let (w_ptr, s_ptr, rs_ptr) = (q.qs, q.scales, q.row_scales);
        if q.q50 {
            drop(quants);
            return self.project_q50(act, m, k, (w_ptr, s_ptr, rs_ptr), n, out, accumulate);
        }
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

        // Deepest tile first. Each kernel takes the whole tiles it can and
        // passes the remaining rows on: qmma_t at two depths, then q8_mma,
        // then a matvec per leftover row.
        let mut qmma_rows = 0;
        if qmma_takes(n) {
            let wide = qmma_width(n);
            for (module, depth, tn) in [
                (&self.q8_qmma_deep[&wide], Q8_QMMA_TM, wide),
                (&self.q8_qmma, Q8_QMMA_SHALLOW, Q8_QMMA_TN),
            ] {
                let left = m - qmma_rows;
                let rows = left - left % depth;
                if rows == 0 || !n.is_multiple_of(tn) {
                    continue;
                }
                // Only the deep tile splits. A grid of one or two blocks
                // always splits, whatever the flag says.
                let starved = (rows / Q8_QMMA_TM) * (n / wide) <= 2;
                let splits = if depth == Q8_QMMA_TM && (self.qmma_split || starved) {
                    q8_qmma_splits(rows, n, k, wide)
                } else {
                    1
                };
                if depth == Q8_QMMA_TM && self.qmma_narrow && q8_qmma_narrow_eligible(rows, n, wide)
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
                // The rows tile evenly by construction. Only n can be ragged.
                self.q8_mma.pick(n.is_multiple_of(Q8_MMA_TN)),
                "q8_mma",
                &tiles.band(qmma_rows, rows),
                ((rows / Q8_MMA_TM) as u32, n.div_ceil(Q8_MMA_TN) as u32, 1),
            )?;
        }

        let tiles_evenly = n.is_multiple_of(Q8_TN);
        let grid_n = n.div_ceil(Q8_TN) as u32;
        // Per-row path. qdot_t needs no split. When its tile does not divide
        // n, the split kernels spread k across the grid and sum the partials
        // in a second launch.
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

    /// Split-K path for `q8_qmma`'s deep tile on a small grid. Each of
    /// `splits` programs covers one slice of `k` and writes a partial plane.
    /// A second launch sums the planes into `out`.
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

    /// Projection through a raw-block weight, decoded straight from the
    /// file's bytes by one kernel. The weight's format picks the kernel and
    /// tile width.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_raw(
        &self,
        act: Option<QAct>,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let raws = self.raw_quants.borrow();
        let raw = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        let (stored_n, nb, quant) = (&raw.n, &raw.nb, &raw.quant);
        ensure!(
            *stored_n == n,
            "raw weight was uploaded with n = {stored_n}, used with n = {n}"
        );
        let kernels = self.format_kernels(*quant)?;
        // The qdot_t kernels are `@aligned(N = TN)`, so they need n to be a
        // whole number of tiles. A ragged n takes the masked body.
        let qdot = kernels
            .qdot
            .as_ref()
            .filter(|qdot| m == 1 && n.is_multiple_of(qdot.tn));
        let qdot_eligible = qdot.is_some();

        // The dp4a decode matvecs. The format picks the output tile and the
        // lookup tables. It quantizes the activation, which the host
        // reference does not.
        let i8_pick = if self.iq1s_dp4a.get() && m == 1 {
            self.pick_qdot_i8(kernels, *quant, n)?
        } else {
            None
        };
        if let Some((kernel, tables)) = i8_pick {
            let tn = kernel.tn;
            let (bytes_ptr, d_ptr) = (raw.bytes, raw.d);
            let rb = *nb * quant.device_block_bytes();
            let (nb, n_blocks) = (*nb as i64, k / Q8_BLOCK);
            drop(raws);
            // Use the caller's quantized activation if it has one.
            let act = act.map_or_else(|| self.quantize_act(a, 1, k), Ok)?;
            let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
            // The tile is `@aligned` and stores whole tiles. A ragged n
            // writes a padded scratch row and copies the first n values out.
            // The upload already padded the weight to match.
            let n_pad = n.next_multiple_of(tn);
            let dest = if n_pad == n {
                out
            } else {
                self.dense_scratch(1, n_pad)?
            };
            let mut operands = vec![
                (qa_ptr, [1, k as i64]),
                (das_ptr, [1, n_blocks as i64]),
                (bytes_ptr, [n_pad as i64, rb as i64]),
                (d_ptr, [n_pad as i64, nb]),
            ];
            operands.extend(tables.into_iter().map(|(ptr, len)| (ptr, [1, len])));
            operands.push((self.ptr(dest, 0)?, [1, n_pad as i64]));
            self.launch(&kernel.module, kernel.name, &operands, ((n_pad / tn) as u32, 1, 1))?;
            if n_pad != n {
                let src = Plane {
                    buf: dest,
                    offset: 0,
                    pitch: n_pad,
                };
                let dst = Plane {
                    buf: out,
                    offset: 0,
                    pitch: n,
                };
                self.copy_2d(src, dst, 1, n)?;
            }
            return Ok(());
        }
        let kernel = match qdot {
            Some(qdot) => qdot,
            // A grouped format has no masked body; a ragged width goes dense.
            None if quant.grouped_rows() => return self.project_raw_dense(a, m, k, w, n, out),
            None => kernels
                .matvec
                .as_ref()
                .with_context(|| format!("no raw kernel launches {}", quant.name()))?,
        };
        let rb = nb * quant.device_block_bytes();
        let (bytes_ptr, d_ptr) = (raw.bytes, raw.d);
        let a_ptr = self.ptr(a, 0)?;
        let out_ptr = self.ptr(out, 0)?;
        let mut operands = vec![
            (a_ptr, [m as i64, k as i64]),
            (bytes_ptr, [n as i64, rb as i64]),
            (d_ptr, [n as i64, *nb as i64]),
        ];
        if let Some(dmin) = raw.dmin {
            operands.push((dmin, [n as i64, *nb as i64]));
        }
        match quant {
            Quant::IQ1_S if qdot_eligible => {
                // The qdot bodies compute their own byte offsets and take
                // no iota8 operand.
                operands.push((
                    self.iq1s_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ1S_GRID_LEN as i64],
                ));
            }
            Quant::IQ1_M if qdot_eligible => {
                operands.push((
                    self.iq1s_grid_packed.as_device_ptr().as_raw(),
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
            Quant::IQ2_XXS if qdot_eligible => {
                operands.push((
                    self.iq2xxs_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ2XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs_packed.as_device_ptr().as_raw(),
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
            Quant::IQ2_S if qdot_eligible => {
                operands.push((
                    self.iq2s_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ2S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs_packed.as_device_ptr().as_raw(),
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
            Quant::IQ2_XS if qdot_eligible => {
                operands.push((
                    self.iq2xs_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ2XS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs_packed.as_device_ptr().as_raw(),
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
            Quant::IQ3_XXS if qdot_eligible => {
                operands.push((
                    self.iq3xxs_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ3XXS_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2xxs_signs_packed.as_device_ptr().as_raw(),
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
            Quant::IQ3_S if qdot_eligible => {
                operands.push((
                    self.iq3s_grid_packed.as_device_ptr().as_raw(),
                    [1, IQ3S_GRID_LEN as i64],
                ));
                operands.push((
                    self.iq2s_signs_packed.as_device_ptr().as_raw(),
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
        self.launch(
            &kernel.module,
            kernel.name,
            &operands,
            (n.div_ceil(kernel.tn) as u32, m as u32, 1),
        )
    }

    /// The dp4a decode matvec for an `n`-wide projection in `quant`, if the
    /// format has one that fits.
    fn pick_qdot_i8<'a>(
        &self,
        kernels: &'a formats::FormatKernels,
        quant: Quant,
        n: usize,
    ) -> Result<Option<I8Kernel<'a>>> {
        let [Some(wide), Some(narrow)] = &kernels.qdot_i8 else {
            return Ok(None);
        };
        // The K-quants have no other path, so a ragged n runs padded.
        let padded = matches!(quant, Quant::Q4_K | Quant::Q5_K | Quant::Q6_K | Quant::PTQ1_0);
        if !padded && !n.is_multiple_of(narrow.tn) {
            return Ok(None);
        }
        // The wide tile whenever it divides n into enough blocks. A
        // projection only a few wide tiles across, such as a 256-wide key,
        // takes the narrow one at four times the blocks.
        let kernel = if n.is_multiple_of(wide.tn) && n / wide.tn >= WIDE_TILE_MIN_BLOCKS {
            wide
        } else {
            narrow
        };
        let grid = |b: &DeviceBuffer<i8>, len: usize| (b.as_device_ptr().as_raw(), len as i64);
        // The mask half sits one table past the +/-1 one.
        let mask =
            |b: &DeviceBuffer<i8>, len: usize| (b.as_device_ptr().as_raw() + len as u64, len as i64);
        let tables = match quant {
            Quant::IQ1_S | Quant::IQ1_M => vec![(self.qgemm.grid4()?, IQ1_GRID4_LEN as i64)],
            Quant::IQ2_XXS => vec![
                grid(&self.iq2xxs_grid_packed, IQ2XXS_GRID_LEN),
                mask(&self.iq2xxs_signs_packed, IQ2XXS_SIGNS_LEN),
            ],
            Quant::IQ2_S => vec![
                grid(&self.iq2s_grid_packed, IQ2S_GRID_LEN),
                mask(&self.iq2s_signs_packed, IQ2S_SIGNS_LEN),
            ],
            Quant::IQ2_XS => vec![
                grid(&self.iq2xs_grid_packed, IQ2XS_GRID_LEN),
                mask(&self.iq2xxs_signs_packed, IQ2XXS_SIGNS_LEN),
            ],
            Quant::IQ3_XXS => vec![
                grid(&self.iq3xxs_grid_packed, IQ3XXS_GRID_LEN),
                mask(&self.iq2xxs_signs_packed, IQ2XXS_SIGNS_LEN),
            ],
            Quant::IQ3_S => vec![
                grid(&self.iq3s_grid_packed, IQ3S_GRID_LEN),
                mask(&self.iq2s_signs_packed, IQ2S_SIGNS_LEN),
            ],
            _ => Vec::new(),
        };
        Ok(Some((kernel, tables)))
    }

    /// Whether `w`'s format has a dequant kernel in [`Self::project_raw_dense`].
    pub(super) fn raw_quant_is(&self, w: RawBuf, quant: Quant) -> bool {
        self.raw_quants
            .borrow()
            .get(w.0)
            .is_some_and(|raw| raw.quant == quant)
    }

    /// Narrow-CTA path for `q8_qmma`'s deep tile. The same kernel at half the
    /// threads and half the column tile, which doubles the grid in one launch
    /// with no scratch.
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
