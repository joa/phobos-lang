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
    ///
    /// `PHOBOS_ATTN_PERSIST` tries [`Self::attention_persist`] first, which
    /// does the same split and merge inside one `@persistent` kernel and a
    /// `grid_barrier()` rather than two launches; see that function,
    /// [`Self::attn_persist_plan`] and [`attention_persist_src`]. Opt-in
    /// rather than the default while it is new: a grid barrier deadlocks if
    /// co-residency does not hold, which is asked of the driver rather than
    /// assumed but is still a sharper failure mode than a launched kernel's.
    ///
    /// The split count handed to the persistent path is not this function's
    /// `ATTN_SPLITS`-derived one -- see [`Self::attn_persist_plan`], which
    /// picks its own so the split phase's `groups * splits` units land
    /// exactly on the settled grid rather than leaving a remainder of
    /// resident blocks with nothing assigned. `attn_persist_plan` declines a
    /// shape only if the settled grid cannot even hold one unit per group
    /// (`blocks < groups`), which neither model's shape reaches on this card.
    pub(super) fn attention_decode(
        &self,
        q: Buf,
        keys: HBuf,
        values: HBuf,
        spec: Attn,
        out: Buf,
    ) -> Result<()> {
        // Carrying `qgroup` heads to a program divides the grid, so the split
        // count multiplies by it to leave the block count unchanged.
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

    /// [`Self::attention_decode`]'s split-plus-merge, as one `@persistent`
    /// kernel instead of two launches and a scratch round-trip through a
    /// second kernel's parameters; see [`attention_persist_src`] for the
    /// phase split and why the barrier makes it safe. Only reached once
    /// [`Self::attn_persist_plan`] has already settled `blocks` for `key`.
    ///
    /// `P` is bound twice, once per shape the two phases read and write it
    /// under, the same way the launched pair already passed one buffer to two
    /// kernels; a persistent kernel does it with two parameters in one launch
    /// instead of two launches.
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

    /// Settles the block count and the persist-specific split count, and
    /// compiles [`attention_persist_src`] for one attention shape, on first
    /// use, caching the decision either way so a declined shape is not
    /// recompiled every call.
    ///
    /// A grid barrier deadlocks unless every block of the launch is resident
    /// at once, so the grid comes from
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` rather than a constant,
    /// iterated the same way `fused_plan` settles the wider megakernel's grid:
    /// a candidate block count changes the compiled trip counts, which can in
    /// principle change what the driver allows, so the answer is asked again
    /// at whatever the driver returned until it stops shrinking.
    ///
    /// The split count is settled here too, not carried in from
    /// [`Self::attention_decode`]'s `ATTN_SPLITS`: that constant is tuned for
    /// the *launched* kernel's own grid (blocks = groups * splits exactly),
    /// which has no reason to land on this kernel's occupancy-settled grid,
    /// and measurably did not -- `ncu` on minicpm's shape found 42.7% of
    /// stall cycles at `grid_barrier()`, and a discriminating profile against
    /// the launched kernel (same combine, no grid barrier) pinned the cost to
    /// the barrier itself rather than the combine. The mechanism: phase one
    /// hands one unit of `[lo, hi)` key range to each of `groups * splits`
    /// blocks, grid-strided over the settled `blocks`; with `ATTN_SPLITS`
    /// fixed at 8 regardless of the settled grid, minicpm's shape measured
    /// 128 units against a settled 192 blocks (a third of the grid idle,
    /// nothing assigned, arriving at the barrier immediately and stalling
    /// there for the entire split-phase duration) and Qwen's measured 64
    /// against 144 (over half idle). Picking `splits = blocks / groups`
    /// (floored) instead makes `groups * splits` land on `blocks` exactly
    /// whenever it divides evenly -- both shapes do on this card (192 = 8 *
    /// 24, 144 = 4 * 36) -- so every resident block gets a unit and none
    /// idles at the barrier from the very start of the phase.
    ///
    /// Settling blocks and picking splits are kept as two separate stages
    /// rather than one loop that resettles both together every candidate.
    /// The first version of this function did that and it back-fired: a
    /// wide first-try `blocks` guess picks a wide `splits` too (`blocks /
    /// groups`), which grows phase two's `mv`/`c` tiles enough to shrink
    /// what the driver allows back down; the *next* candidate then reads
    /// that shrunk `blocks`, picks a *narrower* `splits` for it, and the
    /// occupancy query on that narrower kernel comfortably allows the
    /// original wide grid again -- but the loop only ever compares `allowed`
    /// against its own shrunk candidate, never revisits the wider one, so it
    /// settles on the first accidentally-small `blocks` it lands on instead
    /// of the true ceiling. Measured on Qwen's shape: the loop version
    /// settled at 48 blocks (1/SM) where the fixed-splits probe below still
    /// finds 144 (3/SM) is genuinely resident. Splitting the two stages
    /// removes the feedback path: stage one's probe `splits` never depends
    /// on the `blocks` candidate it is helping to settle, so there is
    /// nothing for a candidate to spiral against.
    ///
    /// Stage one settles `blocks` exactly as before this beam (the loop
    /// `[[flash-attention-decode]]`'s pooling fix already validated), using
    /// `ATTN_SPLITS * qgroup` as a stand-in `splits` purely to measure
    /// phase one's footprint, which is what the shared-memory budget is
    /// actually dominated by (`wacc` alone is `QW * D` f32s; phase two's
    /// `mv`/`c` are `S` f32s each, negligible next to it at any `S` either
    /// stage picks). Stage two, once `blocks` is stable, computes the real
    /// `splits = blocks / groups` (floored) and recompiles once more at that
    /// `S` -- if the small `mv`/`c` growth this causes ever does push the
    /// kernel's real footprint over what the settled `blocks` needs, the
    /// occupancy check below still verifies it and falls back to the probe
    /// module rather than risk a barrier deadlock from an unverified grid.
    ///
    /// `None` means the settled grid cannot hold even one unit per group
    /// (`blocks < groups`), which would need a second grid-strided pass over
    /// the whole key axis -- measured worse than the launch and scratch
    /// round-trip it would have removed, back when a fixed `ATTN_SPLITS`
    /// could actually produce that case (Qwen's pre-`@dynshared`-fix shape).
    /// Neither shipped model shape reaches it any more.
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
            // attention_persist_src is now `@dynshared`, so its shared
            // footprint is a launch-time byte count rather than baked into
            // the compiled function's static attribute; the occupancy query
            // needs that count or it undercounts the kernel's real
            // footprint and settles a grid wider than actually fits,
            // deadlocking grid_barrier. compile_dynamic already recorded it
            // in func_shared as a side effect of compiling above.
            let dynamic_shared = self.shared_of(func) as usize;
            // SAFETY: func belongs to a module alive for this call.
            let (allowed, _) = unsafe { persistent_grid(func, CTA_THREADS, dynamic_shared)? };
            if allowed >= blocks {
                settled = Some((module, blocks));
                break;
            }
            // This candidate is being discarded and its module is about to
            // unload, which can hand its CUfunction's address to a later,
            // wholly unrelated compile -- shared_of's cache is keyed on that
            // address, so a stale entry left behind would hand the next
            // kernel to reuse someone else's dynamic-shared byte count. Not
            // reachable in practice with a fixed probe_splits (this stage's
            // footprint no longer varies candidate to candidate), kept as a
            // defensive habit rather than removed with the mechanism that
            // made it necessary.
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
        // Stage two: as many whole units as the settled grid holds exactly,
        // so no resident block starts phase one with nothing assigned.
        let splits = ((blocks as usize) / groups).max(1);
        let units1 = (groups * splits) as u32;
        if blocks < units1 {
            // blocks < groups: not reached by either shipped shape, kept as
            // the same decline path the pre-existing code used.
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
                // The wider S's mv/c growth pushed this shape's footprint
                // past what the settled grid allows -- fall back to the
                // probe module verbatim rather than risk an unverified
                // grid; that leaves some idle blocks in phase one but stays
                // provably safe against grid_barrier's co-residency need.
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
