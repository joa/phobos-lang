// The chunked delta rule.

use super::*;

impl DeviceBackend {
    /// The chunked gated delta rule, in two passes; see [`delta_wy_src`].
    ///
    /// The packed operands are read as one row per position with the heads
    /// side by side. A chunk of one head is then an ordinary tile. The gates
    /// and the first pass's three `[C, C]` matrices use the same layout.
    pub(super) fn delta_chunked(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (n, width) = (rows as i64, (heads * head_dim) as i64);
        // Experiment only: PHOBOS_EXP_DELTA=tn,c.
        let (chunk_tn, chunk) = std::env::var("PHOBOS_EXP_DELTA")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((DELTA_CHUNK_TN, DELTA_CHUNK));
        let wy_width = heads * 4 * chunk;
        let eye = self.identity(chunk)?;
        let wy = self.alloc(rows * wy_width)?;
        let result = self.with_kernel(
            &self.chunks,
            (heads, head_dim * 1_000_000 + chunk_tn * 1000 + chunk),
            "delta_chunk",
            || {
                let mut source = delta_wy_src(heads, head_dim, chunk);
                source.push_str(&delta_scan_src(heads, head_dim, chunk_tn, chunk));
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
                    "delta_scan",
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
                    (heads as u32, (head_dim / chunk_tn) as u32, 1),
                )
            },
        );
        self.release(wy);
        result
    }
}
