use anyhow::Result;
use cust::memory::{DeviceBuffer, DeviceCopy};
use phobos_kernels::cuda_ok;
use std::cell::{Cell, RefCell};

/// Default slab size. Bigger slabs mean fewer allocations but more space
/// wasted in the tails tensors do not fill.
const SLAB_BYTES: usize = 128 * 1024 * 1024;

/// [`SLAB_BYTES`], or `PHOBOS_SLAB_MIB` if set.
///
/// A tensor larger than a slab gets its own allocation, which lets the
/// driver evict it on its own. So the size decides whether the output head
/// is separate or shares a slab with weights read every token.
fn slab_bytes() -> usize {
    match std::env::var("PHOBOS_SLAB_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(mib) if mib > 0 => mib * 1024 * 1024,
        _ => SLAB_BYTES,
    }
}

/// Slab size for small, hot constants: Q8_0 scale planes and f32 tensors.
///
/// In a bulk slab, a plane read every token would keep the whole slab
/// resident. One allocation each would push the allocation count too high.
/// Small slabs of their own avoid both.
pub(super) const HOT_SLAB_BYTES: usize = 4 * 1024 * 1024;

/// Alignment of every region. 256 bytes keeps each region on a sector
/// boundary, so a neighbour never costs a tensor an extra sector.
const ALIGN: usize = 256;

/// Slabs the weights are bump-allocated from, so the model lives in a few
/// large allocations. Never freed: a weight lives as long as the backend.
///
/// WDDM manages residency per allocation. Many allocations with a large
/// total footprint stop staying resident.
#[derive(Default)]
pub(super) struct Arena {
    /// Bytes taken from the driver per slab.
    slab: Cell<usize>,
    /// Each slab and how many bytes of it are handed out. Allocation is
    /// first fit across all slabs, not only the newest.
    slabs: RefCell<Vec<(DeviceBuffer<u8>, usize)>>,
    /// Bytes handed out across every slab, for the report.
    handed: Cell<usize>,
}

impl Arena {
    /// An arena whose slabs are `slab` bytes. Zero means [`SLAB_BYTES`].
    pub(super) fn with_slab(slab: usize) -> Arena {
        let arena = Arena::default();
        arena.slab.set(slab);
        arena
    }

    /// Copies `data` onto the device and returns its address.
    ///
    /// A region larger than a slab gets a slab of its own.
    pub(super) fn upload<T: DeviceCopy>(&self, data: &[T]) -> Result<u64> {
        let bytes = std::mem::size_of_val(data);
        let want = bytes.next_multiple_of(ALIGN);
        let mut slabs = self.slabs.borrow_mut();
        let at = match slabs.iter().position(|(s, used)| s.len() - used >= want) {
            Some(at) => at,
            None => {
                let slab = match self.slab.get() {
                    0 => slab_bytes(),
                    n => n,
                };
                // SAFETY: nothing reads a slab before a weight is copied in.
                slabs.push((unsafe { DeviceBuffer::uninitialized(want.max(slab))? }, 0));
                slabs.len() - 1
            }
        };
        let (slab, used) = &mut slabs[at];
        let at = slab.as_device_ptr().as_raw() + *used as u64;
        *used += want;
        self.handed.set(self.handed.get() + want);
        cuda_ok(
            // SAFETY: `at` is `want >= bytes` of live slab, and the copy is
            // synchronous, so `data` outlives it.
            unsafe { cust::sys::cuMemcpyHtoD_v2(at, data.as_ptr().cast(), bytes) },
            "uploading a weight into the arena",
        )?;
        Ok(at)
    }

    /// Bytes the slabs occupy and bytes handed out. The difference is the
    /// unused tails.
    pub(super) fn bytes(&self) -> (usize, usize) {
        let held = self.slabs.borrow().iter().map(|(s, _)| s.len()).sum();
        (held, self.handed.get())
    }

    /// Number of slabs, which is the number of allocations.
    pub(super) fn slabs(&self) -> usize {
        self.slabs.borrow().len()
    }

    /// Frees every slab. Only valid once no region is in use any more.
    pub(super) fn reset(&self) {
        self.slabs.borrow_mut().clear();
        self.handed.set(0);
    }
}
