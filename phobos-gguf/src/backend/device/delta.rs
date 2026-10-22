// The prompt's delta rule: the register-state scan and the chunked form.

use super::*;

impl DeviceBackend {
    /// The gated delta rule over a prompt with the state in registers; see
    /// [`delta_scan_reg_src`].
    pub(super) fn delta_scan_reg(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (n, width, h) = (rows as i64, (heads * head_dim) as i64, heads as i64);
        self.with_kernel(
            &self.delta_reg,
            head_dim,
            "delta_scan_reg",
            || delta_scan_reg_src(head_dim),
            |module| {
                self.launch(
                    module,
                    "delta_scan_reg",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 2 * span)?, [n, width]),
                        (self.ptr(packed, 3 * span)?, [n, h]),
                        (self.ptr(packed, 3 * span + gates)?, [n, h]),
                        (self.ptr(state, 0)?, [width, head_dim as i64]),
                        (self.ptr(out, 0)?, [n, width]),
                    ],
                    (heads as u32, (head_dim / DELTA_REG_COLS) as u32, 1),
                )
            },
        )
    }

    /// The chunked gated delta rule, in two passes; see [`delta_wy_src`].
    /// With `tc`, the scan is [`delta_scan_tc_src`] at its chunk, else
    /// [`delta_scan_src`].
    ///
    /// The packed operands are read as one row per position with the heads
    /// side by side. A chunk of one head is then an ordinary tile. The gates
    /// and the first pass's three `[C, C]` matrices use the same layout.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn delta_chunked(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
        tc: bool,
    ) -> Result<()> {
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (n, width) = (rows as i64, (heads * head_dim) as i64);
        let (chunk, tn, scan) = match tc {
            true => (DELTA_TC_CHUNK, DELTA_TC_TN, "delta_scan_tc"),
            false => (DELTA_CHUNK, DELTA_CHUNK_TN, "delta_scan"),
        };
        let wy_width = heads * 4 * chunk;
        let eye = self.identity(chunk)?;
        let wy = self.alloc(rows * wy_width)?;
        let result = self.with_kernel(
            &self.chunks,
            (heads, head_dim, tc),
            "delta_chunk",
            || {
                let mut source = delta_wy_src(heads, head_dim, chunk);
                source.push_str(&match tc {
                    true => delta_scan_tc_src(heads, head_dim),
                    false => delta_scan_src(heads, head_dim, tn, chunk),
                });
                source
            },
            |module| {
                let wy_ptr = self.ptr(wy, 0)?;
                let wy_dims = [n, wy_width as i64];
                self.launch(
                    module,
                    "delta_wy",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 3 * span)?, [n, heads as i64]),
                        (self.ptr(packed, 3 * span + gates)?, [n, heads as i64]),
                        (eye, [chunk as i64, chunk as i64]),
                        (wy_ptr, wy_dims),
                    ],
                    (heads as u32, (rows / chunk) as u32, 1),
                )?;
                self.launch(
                    module,
                    scan,
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 2 * span)?, [n, width]),
                        (wy_ptr, wy_dims),
                        (
                            self.ptr(state, 0)?,
                            [(heads * head_dim) as i64, head_dim as i64],
                        ),
                        (self.ptr(out, 0)?, [n, width]),
                    ],
                    (heads as u32, (head_dim / tn) as u32, 1),
                )
            },
        );
        self.release(wy);
        result
    }
}
