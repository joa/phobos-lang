//! The model's weights in a handful of large allocations rather than one each.
//!
//! WDDM manages residency per allocation, and past a point it stops being able
//! to keep a large set of them resident at once. Measured with
//! `resident_probe`, which holds ballast and writes it between launches, then
//! times the output head's matvec at the shape this model uses:
//!
//! | ballast | allocations | head matvec |
//! | ---: | ---: | ---: |
//! | 6000 MiB | 1 | 2.96 ms, 188 GB/s |
//! | 6000 MiB | 256 | 2.94 ms, 189 GB/s |
//! | 6000 MiB | 320 | 24.88 ms, 22 GB/s |
//! | 6000 MiB | 400 | 47.03 ms, 12 GB/s |
//! | 6000 MiB | 900 | **96.40 ms, 5.8 GB/s** |
//!
//! The total is identical in every row; only the number of allocations it is
//! split across changes. 96.40 ms is what a prompt pass traced for
//! `q3k_qdot_matvec`, and Qwen3.8-27B has 851 tensors going up as two or three
//! buffers apiece, so that is the row phobos was on.
//!
//! Note the cliff is not visible at 5500 MiB, where 1 and 400 allocations are
//! within 6% of each other. It takes both the footprint *and* the count, which
//! is why measuring one at a time said there was nothing here.

use anyhow::Result;
use cust::memory::{DeviceBuffer, DeviceCopy};
use phobos_kernels::cuda_ok;
use std::cell::{Cell, RefCell};

/// Slab size. The cliff above is between 256 and 320 allocations, and a slab
/// only ever wastes the tail a tensor would not fit in, so the choice is how
/// much of both to spend. At 512 MiB a 6 GiB model wasted 668 MiB in tails,
/// which is more than the output head; at 128 it takes about 50 allocations,
/// a fifth of the budget, and wastes a fifth as much.
const SLAB_BYTES: usize = 128 * 1024 * 1024;

/// [`SLAB_BYTES`], or what `PHOBOS_SLAB_MIB` overrides it to. The size decides
/// whether the 521 MiB output head gets a slab to itself: a tensor larger than
/// a slab does, and being its own allocation is exactly what lets the driver
/// single it out. Above it, the head shares with weights that are read every
/// token, which have to be resident anyway.
fn slab_bytes() -> usize {
    match std::env::var("PHOBOS_SLAB_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(mib) if mib > 0 => mib * 1024 * 1024,
        _ => SLAB_BYTES,
    }
}

/// Slab size for the small, hot constants: the Q8_0 scale planes and the f32
/// tensors, 200 MiB across 644 allocations. They cannot share the bulk slabs
/// -- a 0.1 MiB plane read every token drags a whole 128 MiB slab resident
/// with it, and doing that cost tg128 9.74 to 7.82. But leaving them one
/// allocation each keeps the total over the count the driver falls off at, so
/// they get slabs of their own, small enough that dragging one is cheap.
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
    /// Each slab and how much of it has been handed out. First fit rather than
    /// only the newest: a tensor that will not fit the slab being filled still
    /// fits an earlier one, and packing 14 MiB tensors into the newest slab
    /// alone left 540 MiB in tails.
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

    /// How many allocations that is, the number the table above is about.
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
