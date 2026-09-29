// The four attention paths: gemm, blocked, decode and row-at-a-time.

use super::*;

impl DeviceBackend {
    /// Causal attention for a prompt, one head at a time, as two matmuls with
    /// the scores materialized in between. See [`attn_gemm_src`].
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
                    // transpose and value gather only rerun when it changes.
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
                        // Both matmuls contract in f32, so the values widen
                        // once here, as the keys do in the transpose above.
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

    /// Causal attention for a prompt, one query block per program. See
    /// [`attention_block_src`].
    ///
    /// `block` is the query rows per program. The caller ensures the cache
    /// depth is a multiple of it.
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

    /// Attention for a decode step. The key axis is split across the grid
    /// and a merge folds the pieces back together. See
    /// [`attention_split_src`]. Splitting fills the card, where one block per
    /// head would not.
    ///
    /// Tries [`Self::attention_persist`] first, which does the split and
    /// merge in one `@persistent` kernel. `PHOBOS_ATTN_PERSIST=0` disables
    /// it. [`Self::attn_persist_plan`] declines any shape whose grid barrier
    /// could deadlock.
    pub(super) fn attention_decode(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
    ) -> Result<()> {
        // Each program carries `qgroup` heads, so the split count scales by
        // `qgroup` to keep the block count unchanged.
        let qgroup = attention_qgroup(spec.group());
        if self.attn_persist {
            let key = (spec.n_head, spec.group(), spec.head_dim, qgroup);
            if let Some((blocks, splits)) = self.attn_persist_plan(key, spec, qgroup)? {
                return self.attention_persist(q, keys, values, spec, out, splits, key, blocks);
            }
        }
        let splits = ATTN_SPLITS * qgroup;
        // One query row, so the query and the output are a head to a row.
        let (r, d) = (spec.n_head as i64, spec.head_dim as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        let (part, ml) = self.attn_scratch(splits * spec.n_head, spec.head_dim)?;
        // The partials are written as `[S * NH, D]` and read as `[S, NH * D]`.
        // The maxima and sums are `[NH, 2 * S]` in both kernels.
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
                    qgroup,
                    splits,
                    ATTN_WARP_SPLITS,
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

    /// [`Self::attention_decode`]'s split and merge as one `@persistent`
    /// kernel. See [`attention_persist_src`] for the phases and the barrier.
    /// Requires [`Self::attn_persist_plan`] to have settled `blocks` for
    /// `key`.
    ///
    /// The partials buffer is bound twice, once under each phase's shape.
    #[allow(clippy::too_many_arguments)]
    fn attention_persist(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
        splits: usize,
        key: AttnPersistKey,
        blocks: u32,
    ) -> Result<()> {
        let (r, d) = (spec.n_head as i64, spec.head_dim as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        let (part, ml) = self.attn_scratch(splits * spec.n_head, spec.head_dim)?;
        let tall = [(splits * spec.n_head) as i64, d];
        let wide = [splits as i64, spec.n_head as i64 * d];
        let stats = [spec.n_head as i64, 2 * splits as i64];
        let bar = self.fused_bar()?;
        let modules = self.attn_persist_modules.borrow();
        let module = &modules
            .get(&key)
            .and_then(Option::as_ref)
            .expect("attn_persist_plan already settled and cached this key")
            .0;
        self.launch(
            module,
            "attention_persist",
            &[
                (self.ptr(q, 0)?, [r, d]),
                (self.hptr(keys, 0)?, [nk, kw]),
                (self.hptr(values, 0)?, [nk, kw]),
                (part, tall),
                (part, wide),
                (ml, stats),
                (self.ptr(out, 0)?, [r, d]),
                (bar, [2, 1]),
            ],
            (blocks, 1, 1),
        )
    }

    /// Picks the block count and split count for [`attention_persist_src`],
    /// compiling it on first use. The decision is cached either way.
    ///
    /// Stage one settles `blocks` from an occupancy query. Stage two sets
    /// `splits = blocks / groups`. Settling both in one loop would feed the
    /// guess back on itself and settle too low.
    ///
    /// `None` means the grid cannot hold one unit per group
    /// (`blocks < groups`).
    fn attn_persist_plan(
        &self,
        key: AttnPersistKey,
        spec: Attn,
        qgroup: usize,
    ) -> Result<Option<(u32, usize)>> {
        if let Some(cached) = self.attn_persist_modules.borrow().get(&key) {
            return Ok(cached.as_ref().map(|&(_, blocks, splits)| (blocks, splits)));
        }
        let groups = spec.n_head / qgroup;
        let probe_splits = ATTN_SPLITS * qgroup;
        let device = cust::device::Device::get_device(0)?;
        let sms = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as u32;
        let per_sm = device
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerMultiprocessor)?
            as u32
            / CTA_THREADS;
        let mut blocks = per_sm * sms;
        let mut settled = None;
        for _ in 0..FUSED_GRID_TRIES {
            let src = attention_persist_src(
                spec.n_head,
                spec.group(),
                spec.head_dim,
                qgroup,
                probe_splits,
                ATTN_WARP_SPLITS,
                blocks,
            );
            let module = self.compile_dynamic(&src, "attention_persist")?;
            let func = module.get_function("attention_persist")?.to_raw();
            // The kernel is `@dynshared`, so the occupancy query needs its
            // launch-time shared bytes. Without them it undercounts and picks
            // a grid that deadlocks `grid_barrier`.
            let dynamic_shared = self.shared_of(func) as usize;
            // SAFETY: func belongs to a module alive for this call.
            let (allowed, _) = unsafe { persistent_grid(func, CTA_THREADS, dynamic_shared)? };
            if allowed >= blocks {
                settled = Some((module, blocks));
                break;
            }
            // `func_shared` is keyed by function address, which a later
            // compile may reuse once this module unloads. Drop the stale
            // entry.
            self.func_shared.borrow_mut().remove(&(func as usize));
            blocks = allowed;
        }
        let (probe_module, blocks) = match settled {
            Some(pair) => pair,
            None => {
                let src = attention_persist_src(
                    spec.n_head,
                    spec.group(),
                    spec.head_dim,
                    qgroup,
                    probe_splits,
                    ATTN_WARP_SPLITS,
                    blocks,
                );
                (self.compile_dynamic(&src, "attention_persist")?, blocks)
            }
        };
        // Stage two: as many whole units as the grid holds, so every block
        // has work in phase one.
        let splits = ((blocks as usize) / groups).max(1);
        let units1 = (groups * splits) as u32;
        if blocks < units1 {
            // blocks < groups.
            self.attn_persist_modules.borrow_mut().insert(key, None);
            return Ok(None);
        }
        let (module, splits) = if splits == probe_splits {
            (probe_module, splits)
        } else {
            let src = attention_persist_src(
                spec.n_head,
                spec.group(),
                spec.head_dim,
                qgroup,
                splits,
                ATTN_WARP_SPLITS,
                blocks,
            );
            let module = self.compile_dynamic(&src, "attention_persist")?;
            let func = module.get_function("attention_persist")?.to_raw();
            let dynamic_shared = self.shared_of(func) as usize;
            // SAFETY: func belongs to a module alive for this call.
            let (allowed, _) = unsafe { persistent_grid(func, CTA_THREADS, dynamic_shared)? };
            if allowed >= blocks {
                self.func_shared
                    .borrow_mut()
                    .remove(&(probe_module.get_function("attention_persist")?.to_raw() as usize));
                (module, splits)
            } else {
                // More splits need more shared memory than the grid allows.
                // Fall back to the probe module, which is safe for
                // `grid_barrier` but leaves some blocks idle in phase one.
                self.func_shared.borrow_mut().remove(&(func as usize));
                (probe_module, probe_splits)
            }
        };
        let decision = Some((module, blocks, splits));
        let result = decision
            .as_ref()
            .map(|&(_, blocks, splits)| (blocks, splits));
        self.attn_persist_modules.borrow_mut().insert(key, decision);
        Ok(result)
    }

    /// Causal attention, one query row per program over the whole cache. See
    /// [`attention_src`]. The fallback path, since it needs no alignment.
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

    /// [`Backend::rope_gather`]'s device implementation.
    pub(super) fn rope_gather_impl(
        &self,
        src: Plane,
        rows: usize,
        table: Buf,
        spec: Rope,
        dest: Buf,
    ) -> Result<()> {
        self.check_distinct("rope_gather", dest, &[src.buf]);
        // The fused kernel needs a pitch that is a whole number of heads.
        // Otherwise fall back to a copy and a separate rope.
        if !src.pitch.is_multiple_of(spec.head_dim) {
            let width = spec.heads * spec.head_dim;
            self.copy_2d(
                src,
                Plane {
                    buf: dest,
                    offset: 0,
                    pitch: width,
                },
                rows,
                width,
            )?;
            return self.rope(dest, rows, table, spec);
        }
        let stride_heads = src.pitch / spec.head_dim;
        let half = spec.rope_dim / 2;
        let (r, d) = ((rows * spec.heads) as i64, spec.head_dim as i64);
        self.with_kernel(
            &self.rope_gathers,
            (spec.heads, half, stride_heads, spec.head_dim),
            "rope_gather",
            || rope_gather_src(spec.heads, half, stride_heads, spec.head_dim),
            |module| {
                self.launch(
                    module,
                    "rope_gather",
                    &[
                        (
                            self.ptr(src.buf, src.offset)?,
                            [(rows as i64) * stride_heads as i64, d],
                        ),
                        (
                            self.ptr(table, spec.start_pos * spec.rope_dim)?,
                            [rows as i64, spec.rope_dim as i64],
                        ),
                        (self.ptr(dest, 0)?, [r, d]),
                    ],
                    (r as u32, 1, 1),
                )
            },
        )
    }
}
