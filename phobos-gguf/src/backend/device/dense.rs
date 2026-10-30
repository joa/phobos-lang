// The dense fallback for a raw format: decode the weight into an `f32`
// strip, then run the plain matmul. Used for shapes no fused path takes.

use super::*;

/// Default scratch cap for [`DeviceBackend::project_raw_dense`]'s
/// dequantized weight strip.
const RAW_DEQUANT_BUDGET_BYTES: usize = 32 * 1024 * 1024;

/// [`RAW_DEQUANT_BUDGET_BYTES`], or `PHOBOS_DEQUANT_MIB` if set.
///
/// The scratch stays live for the whole prompt pass. A smaller budget frees
/// memory but needs more launches per weight.
fn raw_dequant_budget() -> usize {
    std::env::var("PHOBOS_DEQUANT_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mib| mib > 0)
        .map_or(RAW_DEQUANT_BUDGET_BYTES, |mib| mib * 1024 * 1024)
}

impl DeviceBackend {
    /// [`Self::project_raw`] for the shapes no matvec takes, such as
    /// `m > 1`. Dequantizes one output-column strip of the weight at a time
    /// into a `[K, strip]` scratch, then runs `Backend::matmul` against it
    /// for all `m` rows. The strip width keeps the scratch under the dequant
    /// budget.
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
        let raw = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        let (stored_n, nb, quant) = (&raw.n, &raw.nb, &raw.quant);
        ensure!(
            *stored_n == n,
            "raw weight was uploaded with n = {stored_n}, used with n = {n}"
        );

        let kernels = self.format_kernels(*quant)?;
        // A grouped format expands with its `_qdecode`, a row-major one with
        // its masked `_dequant`. An f16 strip is only used when the matmul is
        // entirely tensor-core. The plain tile contracts in f32 and would lose
        // precision on an f16 weight.
        let (wide, half) = match &kernels.qdecode {
            [Some(wide), Some(half)] => (wide, Some(half)),
            _ => (formats::need(&kernels.dequant, *quant, "dequant")?, None),
        };
        let tn = wide.tn;
        let all_tc = m.is_multiple_of(TC_TILE_M) && k.is_multiple_of(TC_TILE_K);
        let rb = nb * quant.device_block_bytes();
        let (bytes_ptr, d_ptr) = (raw.bytes, raw.d);
        let f16_bytes = size_of::<u16>() as u64;
        let tables = self.raw_tables(*quant);

        // Round down to a whole TC_TILE_N so `Backend::matmul` can use the
        // tensor cores. Only the last strip is then ragged.
        let strip = (raw_dequant_budget() / (k * size_of::<f32>())).clamp(1, n);
        let strip = if strip >= TC_TILE_N {
            strip - strip % TC_TILE_N
        } else {
            strip
        };
        // One shared buffer serves every strip of every weight; a narrower
        // weight uses a prefix. Pool buffers per shape would pile up, since
        // the pool keys by exact length. See `residency.rs`.
        //
        // A grouped format is padded to whole tiles, so a ragged strip
        // decodes `dec` columns and copies out `cur`.
        let grouped = quant.grouped_rows();
        let round_up = |x: usize| if grouped { x.div_ceil(tn) * tn } else { x };
        let widest = k * round_up(strip);
        self.note_dense_pass();
        let scratch_w = self.dense_scratch(0, widest)?;
        let scratch_out = self.dense_scratch(1, m * round_up(strip))?;
        let mut n0 = 0;
        while n0 < n {
            let cur = strip.min(n - n0);
            let dec = round_up(cur);
            let narrow = half.is_some() && all_tc && dec.is_multiple_of(TC_TILE_N);
            let mut operands = vec![
                (bytes_ptr + (n0 * rb) as u64, [dec as i64, rb as i64]),
                (
                    d_ptr + (n0 * nb) as u64 * f16_bytes,
                    [dec as i64, *nb as i64],
                ),
            ];
            if *quant == Quant::Q2_K {
                let dmin_ptr = raw
                    .dmin
                    .context("Q2_K raw weight is missing its dmin plane")?;
                operands.push((
                    dmin_ptr + (n0 * nb) as u64 * f16_bytes,
                    [cur as i64, *nb as i64],
                ));
            }
            let kernel = match half {
                Some(half) if narrow => half,
                _ => wide,
            };
            operands.extend_from_slice(&tables);
            operands.push((self.ptr(scratch_w, 0)?, [k as i64, dec as i64]));
            let grid = (dec.div_ceil(tn) as u32, 1, 1);
            self.launch(&kernel.module, kernel.name, &operands, grid)?;

            if narrow {
                self.matmul_f16_weight(
                    self.ptr(a, 0)?,
                    m,
                    k,
                    self.ptr(scratch_w, 0)?,
                    dec,
                    self.ptr(scratch_out, 0)?,
                )?;
            } else {
                self.matmul(a, m, k, scratch_w, dec, scratch_out)?;
            }
            self.copy_2d(
                Plane {
                    buf: scratch_out,
                    offset: 0,
                    pitch: dec,
                },
                Plane {
                    buf: out,
                    offset: n0,
                    pitch: n,
                },
                m,
                cur,
            )?;

            n0 += cur;
        }
        if !self.dense_scratch_shared {
            self.release(scratch_out);
            self.release(scratch_w);
        }
        Ok(())
    }
    /// [`super::DeviceBackend::matmul`]'s ladder over an f16 weight. The
    /// tensor-core arm matches the f32 one bit for bit. Callers keep
    /// remainder rows off this path, see `narrow` in `project_raw_dense`.
    fn matmul_f16_weight(
        &self,
        a_ptr: u64,
        m: usize,
        k: usize,
        w_ptr: u64,
        n: usize,
        out_ptr: u64,
    ) -> Result<()> {
        let f32_bytes = size_of::<f32>() as u64;
        let tc_rows = if n.is_multiple_of(TC_TILE_N) && k.is_multiple_of(TC_TILE_K) {
            m - m % TC_TILE_M
        } else {
            0
        };
        if tc_rows > 0 {
            self.launch(
                &self.matmul_tc_f16w,
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
        self.launch(
            self.matmul_f16w.pick(tiles_evenly),
            "matmul",
            &[
                (a_row_ptr, [rows as i64, k as i64]),
                (w_ptr, [k as i64, n as i64]),
                (out_row_ptr, [rows as i64, n as i64]),
            ],
            (rows.div_ceil(TILE_M) as u32, n.div_ceil(TILE_N) as u32, 1),
        )
    }
}
