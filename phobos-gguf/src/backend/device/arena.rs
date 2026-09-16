//! The model's weights in a handful of large allocations rather than one
//! each.
//!
//! WDDM manages residency per allocation, and past a point it stops being
//! able to keep a large set of them resident even at a constant total size.
//! The cliff needs both a high allocation count and a large footprint.

use anyhow::Result;
use cust::memory::{DeviceBuffer, DeviceCopy};
use phobos_kernels::cuda_ok;
use std::cell::{Cell, RefCell};

/// Slab size. A slab only ever wastes the tail a tensor does not fill, so
/// the choice trades allocation count against wasted space: bigger slabs
/// mean fewer allocations and more waste in tails, smaller slabs the
/// opposite.
const SLAB_BYTES: usize = 128 * 1024 * 1024;

/// [`SLAB_BYTES`], or what `PHOBOS_SLAB_MIB` overrides it to. The size
/// decides whether the output head gets a slab to itself: a tensor larger
/// than a slab does, and its own allocation lets the driver single it out
/// for eviction. Above it, the head shares with weights that are read every
/// token and have to be resident anyway.
fn slab_bytes() -> usize {
    match std::env::var("PHOBOS_SLAB_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(mib) if mib > 0 => mib * 1024 * 1024,
        _ => SLAB_BYTES,
    }
}

/// Slab size for the small, hot constants: the Q8_0 scale planes and the f32
/// tensors. They cannot share the bulk slabs, since a small plane read every
/// token would drag a whole bulk slab resident with it, but leaving them one
/// allocation each keeps the total over the count the driver falls off at.
/// They get slabs of their own instead, small enough that dragging one is
/// cheap.
pub(super) const HOT_SLAB_BYTES: usize = 4 * 1024 * 1024;

/// What every region is aligned to. The widest load a decode kernel issues is
/// eight bytes; 256 keeps regions on a sector boundary as well, so a slab
/// neighbour cannot cost a tensor an extra sector on its first block.
const ALIGN: usize = 256;

/// Slabs the weights are bump-allocated out of. Never freed: a weight lives as
/// long as the backend does.
#[derive(Default)]
pub(super) struct Arena {
    /// How much is taken from the driver at a time.
    slab: Cell<usize>,
    /// Each slab and how much of it has been handed out. First fit rather
    /// than only the newest: a tensor that will not fit the slab being
    /// filled still fits an earlier one instead of wasting it.
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

    /// Copy `data` onto the device and return where it landed.
    ///
    /// A region larger than a slab gets one of its own, so the output head is
    /// still a single allocation rather than a special case.
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

    /// Bytes the slabs occupy, which is what the weights cost the card, and
    /// the bytes actually handed out, whose difference is the tails.
    pub(super) fn bytes(&self) -> (usize, usize) {
        let held = self.slabs.borrow().iter().map(|(s, _)| s.len()).sum();
        (held, self.handed.get())
    }

    /// How many allocations that is.
    pub(super) fn slabs(&self) -> usize {
        self.slabs.borrow().len()
    }

    /// Give every slab back. Only for an arena whose regions are all dead: a
    /// caller that hands out regions individually has to count them itself.
    pub(super) fn reset(&self) {
        self.slabs.borrow_mut().clear();
        self.handed.set(0);
    }
}
