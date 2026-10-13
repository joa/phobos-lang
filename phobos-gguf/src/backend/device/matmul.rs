// The Q8_0 projection.

use super::*;

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
        // passes the remaining rows on: q8_qmma at two depths, then q8_mma,
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
                    q8_qmma_splits(rows, n, k, wide, self.sms)
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

        // The int8 decode matvecs, written with lanes. They quantize the
        // activation, which the host reference does not.
        if self.iq1s_dp4a.get() && m == 1 && lanes_kernel(*quant).is_some() {
            let (bytes_ptr, d_ptr, nb, quant) = (raw.bytes, raw.d, *nb as i64, *quant);
            drop(raws);
            // Use the caller's quantized activation if it has one.
            let act = act.map_or_else(|| self.quantize_act(a, 1, k), Ok)?;
            let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
            let groups = n.div_ceil(8) as i64;
            let mut operands = vec![
                (qa_ptr, [1, k as i64]),
                (das_ptr, [1, (k / Q8_BLOCK) as i64]),
                (bytes_ptr, [groups, nb * 8 * quant.device_block_bytes() as i64]),
            ];
            // Q6_K's and the IQ formats' `d` is the grouped scale plane.
            if !matches!(quant, Quant::Q4_K | Quant::Q5_K) {
                operands.push((d_ptr, [groups, nb * 8]));
            }
            if let Some(grid) = self.lanes_grid(quant)? {
                operands.push(grid);
            }
            operands.push((self.ptr(out, 0)?, [1, n as i64]));
            return self.launch_lanes(quant, &operands, (n.div_ceil(Q4K_LANES_TN) as u32, 1, 1));
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
        // A row-major format's single row always takes its lanes kernel, a
        // warp a column, eight a CTA.
        if m == 1 && f32_lanes_kernel(*quant).is_some() {
            return self.launch_lanes(*quant, &operands, (n.div_ceil(8) as u32, 1, 1));
        }
        self.launch(
            &kernel.module,
            kernel.name,
            &operands,
            (n.div_ceil(kernel.tn) as u32, m as u32, 1),
        )
    }

    /// The packed grid an IQ format's lanes matvec stages, as (address,
    /// length). `None` for a format that reads no table.
    fn lanes_grid(&self, quant: Quant) -> Result<Option<(u64, [i64; 2])>> {
        let grid = |b: &DeviceBuffer<i8>, len: usize| Some((b.as_device_ptr().as_raw(), [1, len as i64]));
        Ok(match quant {
            Quant::IQ1_S | Quant::IQ1_M => Some((self.qgemm.grid4()?, [1, IQ1_GRID4_LEN as i64])),
            Quant::IQ2_XXS => grid(&self.iq2xxs_grid_packed, IQ2XXS_GRID_LEN),
            Quant::IQ2_XS => grid(&self.iq2xs_grid_packed, IQ2XS_GRID_LEN),
            Quant::IQ2_S => grid(&self.iq2s_grid_packed, IQ2S_GRID_LEN),
            Quant::IQ3_XXS => grid(&self.iq3xxs_grid_packed, IQ3XXS_GRID_LEN),
            Quant::IQ3_S => grid(&self.iq3s_grid_packed, IQ3S_GRID_LEN),
            _ => None,
        })
    }

    /// Whether `w`'s format has a dequant kernel in [`Self::project_raw_dense`].
    pub(super) fn raw_quant_is(&self, w: RawBuf, quant: Quant) -> bool {
        self.raw_quants
            .borrow()
            .get(w.0)
            .is_some_and(|raw| raw.quant == quant)
    }

    /// A raw format's decode matvec written with lanes, compiled on first use.
    fn launch_lanes(&self, quant: Quant, operands: &[(u64, [i64; 2])], grid: (u32, u32, u32)) -> Result<()> {
        let (src, name) = lanes_kernel(quant).or_else(|| f32_lanes_kernel(quant)).expect("checked by the caller");
        if !self.lanes_kernels.borrow().contains_key(&quant) {
            let module = compile(&src(), &[], name)?;
            self.lanes_kernels.borrow_mut().insert(quant, module);
        }
        let cache = self.lanes_kernels.borrow();
        self.launch(&cache[&quant], name, operands, grid)
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

/// A lanes kernel's source builder and kernel name.
type LanesKernel = (fn() -> String, &'static str);

/// A raw format's decode matvec written with lanes: its source and kernel
/// name. Each covers [`Q4K_LANES_TN`] columns a CTA over the grouped weight.
fn lanes_kernel(quant: Quant) -> Option<LanesKernel> {
    match quant {
        Quant::Q4_K => Some((q4k_qdot_lanes_src, "q4k_qdot_lanes")),
        Quant::Q5_K => Some((q5k_qdot_lanes_src, "q5k_qdot_lanes")),
        Quant::Q6_K => Some((q6k_qdot_lanes_src, "q6k_qdot_lanes")),
        Quant::IQ2_XXS => Some((iq2xxs_qdot_lanes_src, "iq2xxs_qdot_lanes")),
        Quant::IQ2_XS => Some((iq2xs_qdot_lanes_src, "iq2xs_qdot_lanes")),
        Quant::IQ2_S => Some((iq2s_qdot_lanes_src, "iq2s_qdot_lanes")),
        Quant::IQ3_XXS => Some((iq3xxs_qdot_lanes_src, "iq3xxs_qdot_lanes")),
        Quant::IQ3_S => Some((iq3s_qdot_lanes_src, "iq3s_qdot_lanes")),
        Quant::IQ1_S => Some((iq1s_qdot_lanes_src, "iq1s_qdot_lanes")),
        Quant::IQ1_M => Some((iq1m_qdot_lanes_src, "iq1m_qdot_lanes")),
        Quant::PTQ1_0 => Some((ptq1_qdot_lanes_src, "ptq1_qdot_lanes")),
        _ => None,
    }
}

/// A row-major format's decode matvec written with lanes, against an `f32`
/// activation: its source and kernel name.
fn f32_lanes_kernel(quant: Quant) -> Option<LanesKernel> {
    match quant {
        Quant::Q2_K => Some((q2k_qdot_lanes_src, "q2k_qdot_lanes")),
        Quant::Q3_K => Some((q3k_qdot_lanes_src, "q3k_qdot_lanes")),
        Quant::IQ4_XS => Some((iq4xs_qdot_lanes_src, "iq4xs_qdot_lanes")),
        _ => None,
    }
}

