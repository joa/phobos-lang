// A block's experts in pinned host memory, in the grouped layout the slabs
// hold, so a miss is one `cuMemcpyHtoDAsync` from here into its slot.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::Result;
use phobos_kernels::cuda_ok;

use crate::experts::ExpertSet;

/// The three stacks of one block, each `count` experts of
/// [`ExpertStack::grouped_bytes`] back to back, then the three scale planes
/// the same way, all page-locked.
pub(in crate::backend::device) struct Mirror {
    host: *mut c_void,
    bytes: usize,
    /// Byte offsets of the six regions: gate, up and down blocks, then
    /// gate, up and down scale planes.
    at: [usize; 6],
    /// Bytes an expert takes in each region.
    per: [usize; 6],
}

// SAFETY: the mapping is owned here and only read after it is built.
unsafe impl Send for Mirror {}

impl Mirror {
    pub(super) fn build(set: &Arc<ExpertSet>) -> Result<Mirror> {
        let stacks = [&set.gate, &set.up, &set.down];
        let mut per = [0usize; 6];
        for (i, stack) in stacks.iter().enumerate() {
            per[i] = stack.grouped_bytes();
            per[3 + i] = stack.grouped_scales() * 2;
        }
        let mut at = [0usize; 6];
        let mut bytes = 0;
        for i in 0..6 {
            at[i] = bytes;
            bytes += stacks[i % 3].count() * per[i];
        }
        let mut host: *mut c_void = std::ptr::null_mut();
        // Pinned so the copies out of it are asynchronous; not mapped into
        // the device, since no kernel reads it in place.
        cuda_ok(
            unsafe { cust::sys::cuMemHostAlloc(&mut host, bytes, 0) },
            "pinning an expert mirror",
        )?;
        let mirror = Mirror { host, bytes, at, per };
        // SAFETY: the allocation is `bytes` long and nothing else holds it.
        let whole = unsafe { std::slice::from_raw_parts_mut(host.cast::<u8>(), bytes) };
        let mut regions: Vec<&mut [u8]> = Vec::with_capacity(6);
        let mut rest = whole;
        for i in 0..6 {
            let (region, tail) = rest.split_at_mut(stacks[i % 3].count() * per[i]);
            regions.push(region);
            rest = tail;
        }
        std::thread::scope(|scope| {
            for (i, region) in regions.into_iter().enumerate() {
                let stack = stacks[i % 3];
                let per = per[i];
                // A few threads a region: the regroup is a memcpy in a
                // different order, and a block's experts are half a gigabyte.
                let lanes = 4;
                let chunk = stack.count().div_ceil(lanes);
                for (lane, part) in region.chunks_mut(chunk * per).enumerate() {
                    scope.spawn(move || {
                        for (j, out) in part.chunks_mut(per).enumerate() {
                            let e = lane * chunk + j;
                            if i < 3 {
                                stack.grouped_into(e, out);
                            } else {
                                // SAFETY: the region is u16-aligned (every
                                // region before it is a multiple of 16
                                // bytes long) and `out` is whole halves.
                                let halves = unsafe {
                                    std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u16>(), out.len() / 2)
                                };
                                stack.grouped_scales_into(e, halves);
                            }
                        }
                    });
                }
            }
        });
        Ok(mirror)
    }

    /// Host addresses of expert `e`'s gate, up and down matrices and of
    /// their scale planes, in that order, with the bytes of each.
    pub(super) fn expert(&self, e: usize) -> [(*const c_void, usize); 6] {
        let base = self.host as usize;
        std::array::from_fn(|i| ((base + self.at[i] + e * self.per[i]) as *const c_void, self.per[i]))
    }

    /// Bytes pinned for this block.
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        // SAFETY: allocated by cuMemHostAlloc above, freed once.
        unsafe {
            cust::sys::cuMemFreeHost(self.host);
        }
    }
}
