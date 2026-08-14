// The four attention paths: gemm, blocked, decode and row-at-a-time.

use super::*;

impl DeviceBackend {
    /// Causal attention for a prompt, one head at a time, as two matmuls with
    /// the scores materialized in between. The gathers and the key transpose are
    /// what let those matmuls be clean; see [`attn_gemm_src`].
    pub(super) fn attention_gemm(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
    ) -> Result<()> {
        let (rows, dim, nk) = (spec.rows, spec.head_dim, spec.total());
        let (qw, kw) = ((spec.n_head * dim) as i64, spec.kv_width());
        let tile = ATTN_GEMM_TILE;

        let head = self.alloc(rows * dim)?;
        let transposed = self.alloc(dim * nk)?;
        let value = self.alloc(nk * dim)?;
        let scores = self.alloc(rows * nk)?;
        let sums = self.alloc(rows)?;
        let mixed = self.alloc(rows * dim)?;

        let result = self.with_kernel(
            &self.attn_gemm,
            dim,
            "attn_gemm",
            || attn_gemm_src(dim, tile),
            |module| {
                let (r, d, n) = (rows as i64, dim as i64, nk as i64);
                for h in 0..spec.n_head {
                    // Consecutive query heads share a key head, so the
                    // transpose and the value gather only redo on a change.
                    if h.is_multiple_of(spec.group()) {
                        let at = h / spec.group() * dim;
                        self.launch(
                            module,
                            "attn_kt",
                            &[
                                (self.hptr(keys, at)?, [n, kw as i64]),
                                (self.ptr(transposed, 0)?, [d, n]),
                            ],
                            (nk.div_ceil(ATTN_KT_ROWS) as u32, 1, 1),
                        )?;
                        // The prompt's two matmuls contract in f32, so the head
                        // widens once here rather than on every tile the mix
                        // stages. The transpose above widens for the same
                        // reason.
                        self.strided(
                            Strided::Load,
                            (self.hptr(values, at)?, kw),
                            (self.ptr(value, 0)?, dim),
                            nk,
                            dim,
                        )?;
                    }
                    self.copy_2d(
                        Plane {
                            buf: q,
                            offset: h * dim,
                            pitch: spec.n_head * dim,
                        },
                        Plane {
                            buf: head,
                            offset: 0,
                            pitch: dim,
                        },
                        rows,
                        dim,
                    )?;
                    self.launch(
                        module,
                        "attn_scores",
                        &[
                            (self.ptr(head, 0)?, [r, d]),
                            (self.ptr(transposed, 0)?, [d, n]),
                            (self.ptr(scores, 0)?, [r, n]),
                        ],
                        ((rows / tile) as u32, nk.div_ceil(tile) as u32, 1),
                    )?;
                    self.launch(
                        module,
                        "attn_softmax",
                        &[(self.ptr(scores, 0)?, [r, n]), (self.ptr(sums, 0)?, [r, 1])],
                        ((rows / ATTN_SOFT_TILE) as u32, 1, 1),
                    )?;
                    self.launch(
                        module,
                        "attn_mix",
                        &[
                            (self.ptr(scores, 0)?, [r, n]),
                            (self.ptr(value, 0)?, [n, d]),
                            (self.ptr(sums, 0)?, [r, 1]),
                            (self.ptr(mixed, 0)?, [r, d]),
                        ],
                        ((rows / ATTN_SOFT_TILE) as u32, (dim / tile) as u32, 1),
                    )?;
                    self.copy_2d(
                        Plane {
                            buf: mixed,
                            offset: 0,
                            pitch: dim,
                        },
                        Plane {
                            buf: out,
                            offset: h * dim,
                            pitch: qw as usize,
                        },
                        rows,
                        dim,
                    )?;
                }
                Ok(())
            },
        );

        for buf in [head, transposed, value, scores, sums, mixed] {
            self.release(buf);
        }
        result
    }

    /// Causal attention for a prompt, one query block per program and its whole
    /// attention inside it; see [`attention_block_src`]. `block` is the query
    /// rows one program covers, and the caller has checked that the cache is a
    /// whole number of them deep.
    pub(super) fn attention_blocked(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        block: usize,
        out: Buf,
    ) -> Result<()> {
        let (rows, qw) = (spec.rows as i64, (spec.n_head * spec.head_dim) as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        self.with_kernel(
            &self.blocked,
            (spec.n_head, spec.group(), spec.head_dim),
            "attention_block",
            || attention_block_src(spec.n_head, spec.group(), spec.head_dim, block),
            |module| {
                self.launch(
                    module,
                    "attention_block",
                    &[
                        (self.ptr(q, 0)?, [rows, qw]),
                        (self.hptr(keys, 0)?, [nk, kw]),
                        (self.hptr(values, 0)?, [nk, kw]),
                        (self.ptr(out, 0)?, [rows, qw]),
                    ],
                    (spec.rows.div_ceil(block) as u32, spec.n_head as u32, 1),
                )
            },
        )
    }

    /// Attention for a decode step, the key axis split across the grid and a
    /// merge folding the pieces back together; see [`attention_split_src`].
    ///
    /// The row kernel would leave this a grid of `n_head` blocks, a sixth of
    /// this card, each walking the whole cache. Splitting the keys fills it.
    pub(super) fn attention_decode(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
    ) -> Result<()> {
        // One query row, so the query and the output are a head to a row.
        let (r, d) = (spec.n_head as i64, spec.head_dim as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        // Carrying `qgroup` heads to a program divides the grid, so the split
        // count multiplies by it to leave the block count unchanged.
        let qgroup = attention_qgroup(spec.group());
        let splits = ATTN_SPLITS * qgroup;
        let (part, ml) = self.attn_scratch(splits * spec.n_head, spec.head_dim)?;
        // The same two buffers under the shape each kernel wants: the partials
        // are written `[S * NH, D]` and read `[S, NH * D]`, the maxima and sums
        // are `[NH, 2 * S]` to both.
        let tall = [(splits * spec.n_head) as i64, d];
        let wide = [splits as i64, spec.n_head as i64 * d];
        let stats = [spec.n_head as i64, 2 * splits as i64];
        self.with_kernel(
            &self.split_attn,
            (spec.n_head, spec.group(), spec.head_dim),
            "attention_split",
            || {
                attention_split_src(
                    spec.n_head,
                    spec.group(),
                    spec.head_dim,
                    attention_tile(spec.head_dim),
                    qgroup,
                    splits,
                )
            },
            |module| {
                self.launch(
                    module,
                    "attention_split",
                    &[
                        (self.ptr(q, 0)?, [r, d]),
                        (self.hptr(keys, 0)?, [nk, kw]),
                        (self.hptr(values, 0)?, [nk, kw]),
                        (part, tall),
                        (ml, stats),
                    ],
                    ((spec.n_head / qgroup) as u32, splits as u32, 1),
                )?;
                self.launch(
                    module,
                    "attention_merge",
                    &[(part, wide), (ml, stats), (self.ptr(out, 0)?, [r, d])],
                    (spec.n_head as u32, 1, 1),
                )
            },
        )
    }

    /// Causal attention one query row per program, over the whole cache; see
    /// [`attention_src`]. What the dispatch above falls through to, since this
    /// one needs no alignment of any extent.
    pub(super) fn attention_rows(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
    ) -> Result<()> {
        let (r, d) = ((spec.rows * spec.n_head) as i64, spec.head_dim as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        self.with_kernel(
            &self.attentions,
            (spec.n_head, spec.group(), spec.head_dim),
            "attention",
            || {
                attention_src(
                    spec.n_head,
                    spec.group(),
                    spec.head_dim,
                    attention_tile(spec.head_dim),
                )
            },
            |module| {
                self.launch(
                    module,
                    "attention",
                    &[
                        (self.ptr(q, 0)?, [r, d]),
                        (self.hptr(keys, 0)?, [nk, kw]),
                        (self.hptr(values, 0)?, [nk, kw]),
                        (self.ptr(out, 0)?, [r, d]),
                    ],
                    (spec.rows as u32, spec.n_head as u32, 1),
                )
            },
        )
    }
}
