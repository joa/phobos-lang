// The megakernel path: planning a chain, sizing its grid, launching it.

use super::*;

pub(super) const FUSED_KERNEL: &str = "fused";

/// Times to try settling a fused kernel's grid against the occupancy API. The
/// answer depends on the compiled code and the compiled code on the answer, so
/// it is iterated; two passes is the most that has ever been needed.
pub(super) const FUSED_GRID_TRIES: usize = 4;

/// The pass emits a contraction that accumulates in tiles of
/// [`fuse::OUT_TILE`], which is this backend's own tile under another name
/// because the pass cannot see it.
const _: () = assert!(Q8_QDOT_TN == fuse::OUT_TILE);

/// Whether one stage of a decode step goes through the fusion pass.
///
/// Switch off via `PHOBOS_FUSED=0`.
pub(super) fn fused_stage(var: &str) -> bool {
    fn asked(var: &str) -> Option<bool> {
        let want = std::env::var(var).ok()?;
        Some(!matches!(want.trim(), "0" | "off" | "no" | "false"))
    }
    asked(var).or_else(|| asked("PHOBOS_FUSED")).unwrap_or(true)
}

impl DeviceBackend {
    /// Run `f` against a module compiled under `key` on first use.
    ///
    /// Several kernels take a tile extent that has to be a compile-time
    /// constant, so they are generated per shape. Every shape a model uses is
    /// fixed at load, so each cache holds a handful of entries and stops growing
    /// once decoding starts.
    /// The matvec on a fixed, card-sized grid. See [`q8_qdot_persist_src`]: this
    /// exists to be measured against the launched kernel, not to be the default.
    ///
    /// The grid is the driver's answer for a kernel already compiled at it, so the
    /// first projection compiles twice: once at a provisional 4 blocks per SM to
    /// have something to ask about, then at whatever came back. Both are cached.
    pub(super) fn qdot_persistent(
        &self,
        n: usize,
        accumulate: bool,
        operands: &[(u64, [i64; 2])],
    ) -> Result<()> {
        if self.persist_blocks.get() == 0 {
            // A narrower grid than the occupancy answer stays co-resident, so
            // PHOBOS_PERSIST_BLOCKS can ask what a matvec loses at a block count
            // a fused kernel would be stuck with.
            if let Some(forced) = std::env::var("PHOBOS_PERSIST_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| v > 0)
            {
                self.persist_blocks.set(forced);
            }
        }
        if self.persist_blocks.get() == 0 {
            let sms = cust::device::Device::get_device(0)?
                .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)?
                as u32;
            let provisional = q8_qdot_persist_src(1, 4 * sms, false);
            let module = self.compile_dynamic(&provisional, "q8_qdot_persist probe")?;
            let func = module.get_function("q8_qdot_persist")?.to_raw();
            // SAFETY: the function belongs to a module alive for this call.
            let (grid, _) = unsafe { persistent_grid(func, CTA_THREADS, 0)? };
            self.persist_blocks.set(grid);
        }
        let blocks = self.persist_blocks.get();
        let stride = blocks as usize * Q8_QDOT_TN;
        let iters = n.div_ceil(stride);
        let name = if accumulate {
            "q8_qdot_persist_add"
        } else {
            "q8_qdot_persist"
        };
        self.with_kernel(
            &self.q8_qdot_persist,
            (iters, accumulate),
            name,
            || q8_qdot_persist_src(iters, blocks, accumulate),
            |module| self.launch(module, name, operands, (blocks, 1, 1)),
        )
    }

    /// Runs the pass over a chain, compiles what it emits, and settles the grid
    /// the result must be launched with. `None` means the pass declined the
    /// chain and the caller should run its stages as separate launches.
    ///
    /// A grid barrier makes the block count part of what the kernel means:
    /// `BLOCKS` is compiled in, so the launch has to use exactly it, and every
    /// block of it has to be resident or the barrier waits for an arrival that
    /// never comes. The occupancy answer is a property of the compiled code and
    /// the compiled code carries the block count, so the two are settled by
    /// starting at the thread ceiling and shrinking to what the driver allows.
    pub(super) fn fused_plan(&self, chain: &Chain) -> Result<Option<ChainKey>> {
        let settled = self.fused_blocks.get();
        if settled != 0 {
            let key = chain.key(settled);
            if self.fused_plans.borrow().contains_key(&key) {
                return Ok(Some(key));
            }
        }

        let device = cust::device::Device::get_device(0)?;
        let sms = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as u32;
        let per_sm = device
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerMultiprocessor)?
            as u32
            / CTA_THREADS;
        let mut blocks = if settled != 0 { settled } else { per_sm * sms };
        for _ in 0..FUSED_GRID_TRIES {
            let key = chain.key(blocks);
            let Some(plan) = key.plan()? else {
                return Ok(None);
            };
            let module = self.compile_dynamic(&plan.source, FUSED_KERNEL)?;
            let func = module.get_function(FUSED_KERNEL)?.to_raw();
            // SAFETY: the function belongs to a module alive for this call.
            let (allowed, _) = unsafe { persistent_grid(func, CTA_THREADS, 0)? };
            if allowed >= blocks {
                // What the fusion collapsed, what it still pays and what it had
                // to put in memory anyway: the numbers that say whether it was
                // worth emitting.
                phobos_base::phdebug!(
                    "fused kernel: stages={} blocks={blocks} barriers={} published={} held={}",
                    key.stages(),
                    plan.barriers,
                    plan.scratch.len(),
                    plan.held
                );
                self.fused_blocks.set(blocks);
                self.fused_plans
                    .borrow_mut()
                    .insert(key.clone(), (module, plan));
                return Ok(Some(key));
            }
            blocks = allowed;
        }
        // Declining costs launches; failing would cost the model. Fusion is what
        // a step does by default, so a card whose occupancy answer never settles
        // falls back rather than taking the pass down with it.
        phobos_base::phinfo!(
            "fused kernel: no co-resident grid after {FUSED_GRID_TRIES} tries, not fusing"
        );
        Ok(None)
    }

    /// A fused kernel's barrier state, zeroed on first use. Every barrier leaves
    /// the counter and the generation as it found them, so one pair serves every
    /// launch of every layer.
    pub(super) fn fused_bar(&self) -> Result<u64> {
        if self.fused_barrier.borrow().is_none() {
            self.flush_pending()?;
            *self.fused_barrier.borrow_mut() = Some(DeviceBuffer::from_slice(&[0i32; 2])?);
        }
        let bar = self.fused_barrier.borrow();
        Ok(bar.as_ref().expect("filled above").as_device_ptr().as_raw())
    }

    /// Binds a plan's slots and launches it.
    pub(super) fn fused_launch(&self, chain: &Chain, key: &ChainKey) -> Result<()> {
        let plans = self.fused_plans.borrow();
        let (module, plan) = &plans[key];
        self.grow_scratch(&plan.scratch)?;

        let quants = self.quants.borrow();
        let pool = self.fused_scratch.borrow();
        let mut operands = self.fused_operands.borrow_mut();
        operands.clear();
        for slot in &plan.slots {
            let ptr = match slot.bound {
                Bound::Given(val) => self.ptr(chain.buf(val)?, 0)?,
                Bound::WeightQs(val) | Bound::WeightScales(val) => {
                    let w = chain.weight_of(val)?;
                    let q = quants
                        .get(w.0)
                        .context("use of an unknown quantized weight handle")?;
                    ensure!(
                        q.n as i64 == slot.dims[0],
                        "a fused weight went up with n = {}, used with n = {}",
                        q.n,
                        slot.dims[0]
                    );
                    match slot.bound {
                        Bound::WeightQs(_) => q.qs,
                        _ => q.row_scales,
                    }
                }
                Bound::ScratchQs(at) => pool[at].0.as_device_ptr().as_raw(),
                Bound::ScratchScales(at) => pool[at].1.as_device_ptr().as_raw(),
                Bound::Barrier => self.fused_bar()?,
            };
            operands.push((ptr, slot.dims));
        }
        self.launch(module, FUSED_KERNEL, &operands, (plan.blocks, 1, 1))
    }
}
