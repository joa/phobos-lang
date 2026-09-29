// The megakernel path: planning a chain, sizing its grid, launching it.

use super::*;

pub(super) const FUSED_KERNEL: &str = "fused";

/// Attempts to settle a fused kernel's grid against the occupancy API. The
/// grid size is compiled into the kernel and the occupancy depends on the
/// compiled kernel, so it is iterated.
pub(super) const FUSED_GRID_TRIES: usize = 4;

/// [`fuse::OUT_TILE`] must match this backend's matvec tile, which the pass
/// cannot see.
const _: () = assert!(Q8_QDOT_TN == fuse::OUT_TILE);

/// Whether one stage of a decode step goes through the fusion pass.
///
/// Switch off via `PHOBOS_FUSED=0`.
pub(super) fn fused_stage(var: &str) -> bool {
    use phobos_base::env::flag_set;
    flag_set(var).or_else(|| flag_set("PHOBOS_FUSED")).unwrap_or(true)
}

impl DeviceBackend {
    /// The matvec on a fixed, card-sized grid, see [`q8_qdot_persist_src`].
    /// An experimental alternative to the launched kernel, not the default.
    ///
    /// The first call compiles a probe at 4 blocks per SM to query the
    /// occupancy, then compiles at the answer. Both are cached.
    pub(super) fn qdot_persistent(
        &self,
        n: usize,
        accumulate: bool,
        operands: &[(u64, [i64; 2])],
    ) -> Result<()> {
        if self.persist_blocks.get() == 0 {
            // PHOBOS_PERSIST_BLOCKS forces a block count. A grid narrower than
            // the occupancy answer is still co-resident.
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

    /// Runs the pass over a chain, compiles the result, and settles its grid.
    /// `None` means the pass declined the chain, and the caller should run
    /// the stages as separate launches.
    ///
    /// The block count is compiled in, and every block must be resident or
    /// the grid barrier deadlocks. So the grid starts at the thread ceiling
    /// and shrinks until the driver's occupancy answer allows it.
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
        // Fall back to separate launches rather than fail the pass.
        phobos_base::phinfo!(
            "fused kernel: no co-resident grid after {FUSED_GRID_TRIES} tries, not fusing"
        );
        Ok(None)
    }

    /// A fused kernel's barrier state, zeroed on first use. Every barrier
    /// restores the counter and generation, so one pair serves every launch.
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
                Bound::RawBytes(val) | Bound::RawD(val) => {
                    let w = chain.raw_of(val)?;
                    let raws = self.raw_quants.borrow();
                    let r = raws.get(w.0).context("use of an unknown raw weight handle")?;
                    ensure!(
                        r.n.next_multiple_of(crate::quant::grouped::RAW_GROUP_PAD) as i64 == slot.dims[0],
                        "a fused raw weight went up with n = {}, used with {} padded rows",
                        r.n,
                        slot.dims[0]
                    );
                    match slot.bound {
                        Bound::RawBytes(_) => r.bytes,
                        _ => r.d,
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

impl DeviceBackend {
    // The `Backend` entry points of the fused path. This backend builds the
    // chain; the pass decides everything else. `false` means the caller runs
    // the separate launches.
    pub(super) fn launch_fused_mlp(&self, mlp: FusedMlp) -> Result<bool> {
        if !self.fused_mlp {
            return Ok(false);
        }
        let chain = fuse::mlp_chain(
            mlp.x,
            mlp.gain,
            mlp.gate_up,
            mlp.down,
            mlp.d_model,
            mlp.d_ff,
            mlp.eps,
        );
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(false);
        };
        self.fused_launch(&chain, &key)?;
        Ok(true)
    }

    pub(super) fn launch_fused_mlp_raw(&self, mlp: FusedMlpRaw) -> Result<bool> {
        if !self.fused_mlp {
            return Ok(false);
        }
        let Some(chain) = fuse::mlp_chain_raw(
            mlp.x,
            mlp.gain,
            mlp.gate,
            mlp.up,
            mlp.down,
            mlp.d_model,
            mlp.d_ff,
            mlp.eps,
        ) else {
            return Ok(false);
        };
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(false);
        };
        self.fused_launch(&chain, &key)?;
        Ok(true)
    }

    pub(super) fn launch_fused_project(&self, project: FusedProject) -> Result<Fused> {
        if !self.fused_project {
            return Ok(Fused::default());
        }
        // The mix tail is gated separately.
        let project = FusedProject {
            mix: project.mix.filter(|_| self.fused_mix),
            ..project
        };
        let Some(chain) = fuse::project_chain(&project) else {
            return Ok(Fused::default());
        };
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(Fused::default());
        };
        self.fused_launch(&chain, &key)?;
        Ok(Fused {
            project: true,
            mix: project.mix.is_some(),
        })
    }

    pub(super) fn launch_fused_attn_out(&self, out: FusedAttnOut) -> Result<bool> {
        if !self.fused_attn_out {
            return Ok(false);
        }
        let Some(chain) = fuse::attn_out_chain(out.x, out.w, out.dest, out.width, out.d_model)
        else {
            return Ok(false);
        };
        let Some(key) = self.fused_plan(&chain)? else {
            return Ok(false);
        };
        self.fused_launch(&chain, &key)?;
        Ok(true)
    }
}
