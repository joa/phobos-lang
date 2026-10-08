// Compiling kernels, resolving them in a module, and issuing or recording
// launches.

use super::*;

impl DeviceBackend {
    /// Launches a kernel over tensor operands given as pointer and extents,
    /// or records it while a pass is recording.
    ///
    /// Reuses its buffers instead of allocating, since a pass issues
    /// hundreds of these.
    pub(super) fn launch(
        &self,
        module: &Module,
        name: &'static str,
        operands: &[(u64, [i64; 2])],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let func = self.function(module, name)?;

        if self.recording.get() {
            let mut pending = self.pending.borrow_mut();
            let at = self.recorded_len.get();
            if at == pending.len() {
                pending.push(Recorded::default());
            }
            let slot = &mut pending[at];
            slot.func = func;
            slot.grid = grid;
            slot.shared = self.shared_of(func);
            slot.threads = self.threads_of(func)?;
            slot.slots.clear();
            for &(ptr, dims) in operands {
                push_descriptor(&mut slot.slots, ptr, dims);
            }
            self.recorded_len.set(at + 1);
            if self.report_pass.get() != 0 {
                self.report.borrow_mut().push(PassOp {
                    name,
                    func,
                    blocks: grid.0 * grid.1 * grid.2,
                    threads: slot.threads,
                    shared: slot.shared,
                });
            }
            return Ok(());
        }

        let mut eager = self.eager.borrow_mut();
        eager.func = func;
        eager.grid = grid;
        eager.shared = self.shared_of(func);
        eager.threads = self.threads_of(func)?;
        eager.slots.clear();
        for &(ptr, dims) in operands {
            push_descriptor(&mut eager.slots, ptr, dims);
        }
        self.issue(&eager, name)
    }

    /// The kernel's dynamic shared bytes. Zero unless it is `@dynshared`.
    #[inline]
    pub(super) fn shared_of(&self, func: cust::sys::CUfunction) -> u32 {
        self.func_shared
            .borrow()
            .get(&(func as usize))
            .copied()
            .unwrap_or(0)
    }

    /// The kernel's block width from its `maxntid`, queried once per kernel.
    #[inline]
    pub(super) fn threads_of(&self, func: cust::sys::CUfunction) -> Result<u32> {
        if let Some(&threads) = self.func_threads.borrow().get(&(func as usize)) {
            return Ok(threads);
        }
        let mut threads = 0i32;
        cuda_ok(
            // SAFETY: `func` came from cuModuleGetFunction and the module is
            // held for the backend's lifetime.
            unsafe {
                cust::sys::cuFuncGetAttribute(
                    &mut threads,
                    cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    func,
                )
            },
            "asking a kernel how wide a block it wants",
        )?;
        let threads = (threads as u32).min(CTA_THREADS);
        self.func_threads
            .borrow_mut()
            .insert(func as usize, threads);
        Ok(threads)
    }

    /// The kernel handle for `name`, resolved once and cached.
    ///
    /// The cache key is the module and name addresses, not the name text.
    /// Every caller passes a literal, so the pointer identifies it.
    #[inline]
    pub(super) fn function(
        &self,
        module: &Module,
        name: &'static str,
    ) -> Result<cust::sys::CUfunction> {
        let key = (module as *const Module as usize, name.as_ptr() as usize);
        if let Some(&func) = self.functions.borrow().get(&key) {
            return Ok(func);
        }
        let func = module.get_function(name)?.to_raw();
        self.functions.borrow_mut().insert(key, func);
        Ok(func)
    }

    /// Runs `f` against the module cached under `key`, compiling it on first
    /// use. Kernels with compile-time shapes are generated per shape, and a
    /// model uses only a few.
    pub(super) fn with_kernel<K: Copy + Eq + std::hash::Hash>(
        &self,
        cache: &RefCell<HashMap<K, Module>>,
        key: K,
        what: &'static str,
        source: impl FnOnce() -> String,
        f: impl FnOnce(&Module) -> Result<()>,
    ) -> Result<()> {
        if cache.borrow().contains_key(&key) {
            self.kernels_reused.set(self.kernels_reused.get() + 1);
        } else {
            let module = self.compile_dynamic(&source(), what)?;
            cache.borrow_mut().insert(key, module);
            self.kernels_compiled.set(self.kernels_compiled.get() + 1);
        }
        let modules = cache.borrow();
        f(&modules[&key])
    }

    /// Compiles a module whose kernels may use dynamic shared memory. Records
    /// each kernel's need and raises the limit for those past 48 KB.
    pub(super) fn compile_dynamic(&self, source: &str, what: &'static str) -> Result<Module> {
        self.compile_dynamic_within(source, what)?
            .with_context(|| format!("{what} needs more shared memory per block than the card allows"))
    }

    /// [`Self::compile_dynamic`], or `None` when a kernel needs more shared
    /// memory per block than the card allows. A kernel sized to the card,
    /// such as a persistent grid's, takes that as a size to back off from.
    pub(super) fn compile_dynamic_within(&self, source: &str, what: &'static str) -> Result<Option<Module>> {
        let (module, shared) = compile_shared(source, &[], what)?;
        let ceiling_bytes = shared_ceiling_bytes()?;
        if shared.iter().any(|&(_, bytes)| bytes > ceiling_bytes) {
            return Ok(None);
        }
        for (name, bytes) in shared {
            let func = module.get_function(&name)?.to_raw();
            if bytes > STATIC_SHARED_LIMIT {
                // SAFETY: func comes from the module just loaded and outlives
                // this call; the attribute takes a plain byte count.
                cuda_ok(
                    unsafe {
                        cust::sys::cuFuncSetAttribute(
                            func,
                            cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes as i32,
                        )
                    },
                    "raising a kernel's dynamic shared memory ceiling",
                )?;
            }
            self.func_shared
                .borrow_mut()
                .insert(func as usize, bytes as u32);
        }
        Ok(Some(module))
    }
}

/// The most shared memory one block may opt into on this card.
fn shared_ceiling_bytes() -> Result<usize> {
    let device = cust::device::Device::get_device(0)?;
    let mut bytes = 0;
    // SAFETY: bytes outlives the call and device is a live handle.
    cuda_ok(
        unsafe {
            cust::sys::cuDeviceGetAttribute(
                &mut bytes,
                cust::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                device.as_raw(),
            )
        },
        "reading the card's shared memory ceiling",
    )?;
    Ok(bytes.max(0) as usize)
}
