use anyhow::{Result, bail};

#[inline]
pub fn cuda_ok(status: cust::sys::CUresult, what: &str) -> Result<()> {
    match status {
        cust::sys::CUresult::CUDA_SUCCESS => Ok(()),
        other => bail!("{what}: {other:?}"),
    }
}

#[inline]
pub fn push_descriptor(slots: &mut Vec<u64>, ptr: u64, dims: [i64; 2]) {
    let word = |v: i64| v as i32 as u32 as u64;
    slots.extend_from_slice(&[ptr, ptr, 0, word(dims[0]), word(dims[1]), word(dims[1]), 1]);
}

pub const STATIC_SHARED_LIMIT: usize = 48 * 1024;

pub const CTA_THREADS: u32 = 256;

pub const WARP_THREADS: usize = 32;

/// The largest grid every block of which is resident at once: what a
/// `@persistent` kernel using `grid_barrier` must be launched with, since a
/// block still waiting for an SM never arrives and the barrier deadlocks.
///
/// Asked of the driver rather than assumed, because the answer is a property of
/// the compiled kernel: shared memory, not registers, is what bounds it in
/// practice, and it is the widest fused stage that sets the figure. Returns the
/// block count and the blocks per SM behind it.
///
/// # Safety
///
/// `func` must have come from `cuModuleGetFunction` on a module that is still
/// loaded.
pub unsafe fn persistent_grid(
    func: cust::sys::CUfunction,
    threads: u32,
    dynamic_shared: usize,
) -> Result<(u32, u32)> {
    let mut per_sm = 0i32;
    // SAFETY: the caller guarantees func names a function of a loaded module.
    cuda_ok(
        unsafe {
            cust::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut per_sm,
                func,
                threads as i32,
                dynamic_shared,
            )
        },
        "querying occupancy for a persistent grid",
    )?;
    if per_sm < 1 {
        bail!(
            "a persistent kernel needs at least one resident block per SM, \
             but this one fits none at {threads} threads and {dynamic_shared} \
             bytes of dynamic shared memory"
        );
    }
    let sms = cust::device::Device::get_device(0)?
        .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)?;
    Ok((per_sm as u32 * sms as u32, per_sm as u32))
}
