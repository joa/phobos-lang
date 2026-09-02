// The dense fallback for a raw format: the weight decoded to an `f32`
// strip, then the plain matmul, for the shapes no fused or decode-in-kernel
// path takes.

use super::*;

/// Scratch cap for [`DeviceBackend::project_raw_dense`]'s dequantized weight
/// strip, bounded well under card memory instead of dequantizing a whole
/// `[K, N]` tensor at once.
const RAW_DEQUANT_BUDGET_BYTES: usize = 32 * 1024 * 1024;

/// [`RAW_DEQUANT_BUDGET_BYTES`], or what `PHOBOS_DEQUANT_MIB` overrides it to.
///
/// The scratch is live for the whole prompt pass, and it is what keeps the
/// 521 MiB output head paged. Traced, the same kernel at the same shape reads
/// **3.03 ms a token in a decode that follows no prompt pass and 47.19 ms in
/// one that does**, which is 50% of a decode step and 11.6 GB/s, the bus
/// rather than the card.
///
/// Shrinking it buys that back and costs launches. The trade was steep when
/// two formats were fused and most of a pass went through here; with four it
/// is not, and 32 MiB is where it lands. Each row twice, `-p 128 -n 128 -r 1`:
///
/// | budget | pp128 | tg128 |
/// | ---: | ---: | ---: |
/// | 128 MiB | 204.4, 208.6 | 10.06, 11.33 |
/// | 64 MiB | 202.2, 206.2 | 11.31, 12.93 |
/// | **32 MiB** | **192.9, 193.0** | **18.03, 18.03** |
/// | 16 MiB | 150.7, 2.6 | 15.08, 5.79 |
/// | 8 MiB | 88.8, 87.1 | 18.03, 12.90 |
///
/// Below 32 the strip stops covering a whole tensor in few enough launches and
/// the prompt pass falls apart; the two rounds at 16 MiB disagree by 58x, which
/// is the shape of a pass that has started thrashing rather than a measurement.
/// The way out is still no scratch at all: a format with a fused projection
/// never allocates one, and IQ1_M, IQ3_XXS and IQ3_S are what is left.
fn raw_dequant_budget() -> usize {
    std::env::var("PHOBOS_DEQUANT_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mib| mib > 0)
        .map_or(RAW_DEQUANT_BUDGET_BYTES, |mib| mib * 1024 * 1024)
}

impl DeviceBackend {
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
        let raw = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        let (stored_n, nb, quant) = (&raw.n, &raw.nb, &raw.quant);
        ensure!(
            *stored_n == n,
            "raw weight was uploaded with n = {stored_n}, used with n = {n}"
        );

        // Tile size and extra grid/signs/iota operands mirror project_raw's
        // own match table.
        let (dequant, dequant_name, tn) = match quant {
            Quant::IQ1_S => (&self.iq1s_dequant, "iq1s_dequant", IQ1S_TN),
            Quant::IQ2_XXS => (&self.iq2xxs_dequant, "iq2xxs_dequant", IQ2XXS_TN),
            Quant::IQ1_M => (&self.iq1m_dequant, "iq1m_dequant", IQ1M_TN),
            Quant::IQ2_S => (&self.iq2s_dequant, "iq2s_dequant", IQ2S_TN),
            Quant::IQ2_XS => (&self.iq2xs_dequant, "iq2xs_dequant", IQ2XS_TN),
            Quant::IQ3_XXS => (&self.iq3xxs_dequant, "iq3xxs_dequant", IQ3XXS_TN),
            Quant::IQ3_S => (&self.iq3s_dequant, "iq3s_dequant", IQ3S_TN),
            Quant::IQ4_XS => (&self.iq4xs_dequant, "iq4xs_dequant", IQ4XS_TN),
            Quant::Q2_K => (&self.q2k_dequant, "q2k_dequant", Q2K_TN),
            other => anyhow::bail!(
                "project_raw_dense has no dequant kernel for {}",
                other.name()
            ),
        };
        // `_dequant` above stays the masked fallback, for a ragged last strip
        // and for the formats with no `_qdecode`. The f16 strip is only sound
        // where the matmul reading it is entirely tensor-core: the plain tile
        // contracts in f32, and an f16 weight there costs real precision.
        let all_tc = m.is_multiple_of(TC_TILE_M) && k.is_multiple_of(TC_TILE_K);
        let qdecode = match quant {
            Quant::IQ1_S => Some((&self.iq1s_qdecode, &self.iq1s_qdecode_f16, "iq1s_qdecode")),
            Quant::IQ2_XXS => Some((
                &self.iq2xxs_qdecode,
                &self.iq2xxs_qdecode_f16,
                "iq2xxs_qdecode",
            )),
            Quant::IQ1_M => Some((&self.iq1m_qdecode, &self.iq1m_qdecode_f16, "iq1m_qdecode")),
            Quant::IQ2_S => Some((&self.iq2s_qdecode, &self.iq2s_qdecode_f16, "iq2s_qdecode")),
            Quant::IQ2_XS => Some((
                &self.iq2xs_qdecode,
                &self.iq2xs_qdecode_f16,
                "iq2xs_qdecode",
            )),
            Quant::IQ3_XXS => Some((
                &self.iq3xxs_qdecode,
                &self.iq3xxs_qdecode_f16,
                "iq3xxs_qdecode",
            )),
            Quant::IQ3_S => Some((&self.iq3s_qdecode, &self.iq3s_qdecode_f16, "iq3s_qdecode")),
            // Q2_K and IQ4_XS decode on a different lane geometry and are 1%
            // of a prompt pass between them; they keep `_dequant`.
            _ => None,
        };
        let rb = nb * quant.device_block_bytes();
        let (bytes_ptr, d_ptr) = (raw.bytes, raw.d);
        let f16_bytes = size_of::<u16>() as u64;

        // Rounded to a whole TC_TILE_N: a budget-shaped strip is a multiple of
        // 64 for no reason, and `Backend::matmul` needs one to reach the
        // tensor cores. Only the last strip is then ragged.
        let strip = (raw_dequant_budget() / (k * size_of::<f32>())).clamp(1, n);
        let strip = if strip >= TC_TILE_N {
            strip - strip % TC_TILE_N
        } else {
            strip
        };
        // One buffer for every strip of every weight in the model, not one per
        // distinct width. A kernel operand is a pointer and a shape, so a
        // narrower weight simply uses a prefix, and `k * strip` is bounded by
        // RAW_DEQUANT_BUDGET_BYTES by construction. Taken from the pool per
        // weight instead, the exact-length keying leaves one entry per shape:
        // 741 to 886 MiB on the 27B, which is what pages its output head out.
        // See `residency.rs`.
        // A grouped format is padded to `RAW_GROUP_PAD` columns, so a ragged
        // strip decodes to the next whole tile, `dec` wide, and copies `cur`.
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
            let expand = qdecode.filter(|_| dec.is_multiple_of(tn));
            let narrow = expand.is_some() && all_tc && dec.is_multiple_of(TC_TILE_N);
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
            let (module, name) = match expand {
                Some((wide, half, name)) => (if narrow { half } else { wide }, name),
                None => (dequant, dequant_name),
            };
            // A `_qdecode` indexes its tables itself and reads the packed i8
            // ones; a `_dequant` gathers against the i32 ones. Same slot count,
            // so only the pointer changes.
            let packed = expand.is_some();
            let table = |i32_buf: &DeviceBuffer<i32>, i8_buf: &DeviceBuffer<i8>, len: usize| {
                let ptr = if packed {
                    i8_buf.as_device_ptr().as_raw()
                } else {
                    i32_buf.as_device_ptr().as_raw()
                };
                (ptr, [1i64, len as i64])
            };
            match quant {
                Quant::IQ1_S | Quant::IQ1_M => {
                    let t = table(&self.iq1s_grid, &self.iq1s_grid_packed, IQ1S_GRID_LEN);
                    operands.push(t);
                }
                Quant::IQ2_XXS => {
                    let g = table(&self.iq2xxs_grid, &self.iq2xxs_grid_packed, IQ2XXS_GRID_LEN);
                    let v = table(
                        &self.iq2xxs_signs,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    );
                    operands.push(g);
                    operands.push(v);
                }
                Quant::IQ2_S => {
                    let g = table(&self.iq2s_grid, &self.iq2s_grid_packed, IQ2S_GRID_LEN);
                    let v = table(&self.iq2s_signs, &self.iq2s_signs_packed, IQ2S_SIGNS_LEN);
                    operands.push(g);
                    operands.push(v);
                }
                Quant::IQ2_XS => {
                    let g = table(&self.iq2xs_grid, &self.iq2xs_grid_packed, IQ2XS_GRID_LEN);
                    let v = table(
                        &self.iq2xxs_signs,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    );
                    operands.push(g);
                    operands.push(v);
                }
                Quant::IQ3_XXS => {
                    let g = table(&self.iq3xxs_grid, &self.iq3xxs_grid_packed, IQ3XXS_GRID_LEN);
                    let v = table(
                        &self.iq2xxs_signs,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    );
                    operands.push(g);
                    operands.push(v);
                }
                Quant::IQ3_S => {
                    let g = table(&self.iq3s_grid, &self.iq3s_grid_packed, IQ3S_GRID_LEN);
                    let v = table(&self.iq2s_signs, &self.iq2s_signs_packed, IQ2S_SIGNS_LEN);
                    operands.push(g);
                    operands.push(v);
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
            // Only a `gather`-based body needs the iota tile.
            if expand.is_none() && *quant != Quant::IQ4_XS && *quant != Quant::Q2_K {
                operands.push((self.iota8.as_device_ptr().as_raw(), [1, 8]));
            }
            operands.push((self.ptr(scratch_w, 0)?, [k as i64, dec as i64]));
            self.launch(module, name, &operands, (dec.div_ceil(tn) as u32, 1, 1))?;

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
    /// tensor-core arm is bit-identical to the f32 one, since `stage_to_f16`
    /// truncates a weight operand either way; callers keep the remainder rows
    /// off this path (see `narrow` in `project_raw_dense`).
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
