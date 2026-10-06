// Sizes the expert cache to the memory the rest of the card leaves it.
//
// The cache is sized from free memory when it is laid out. The attention
// caches, a longer prompt's scratch, the desktop and other programs all grow
// later. On this driver an over-subscribed allocation still succeeds by
// paging other buffers to the host, and reading a paged buffer costs far more
// than the cache slots save, so the cache shrinks. When the memory comes back,
// say because another program exited, it grows again, up to what its budget
// bought at load.
//
// Every resize empties the cache and drops the recorded pass graphs, so the
// thresholds are far apart: a grown cache leaves more free than a shrink
// restores, and a grow has to be worth a fair share of a slab. Free memory
// read just after another program exits overstates what stays free by a few
// hundred MiB, so a grow to the shrink's headroom would only shrink again.
//
// Free memory is the whole card's where NVML is there, since the process's
// own figure does not see another program fill the card. That figure stops
// a few hundred MiB short of zero once the card is over-subscribed, so a
// reading near the floor says the card is full but not by how much. A shrink
// is therefore followed by another check on the very next pass, and a few
// steps in a row restore the headroom an unknown shortfall took.

use anyhow::Result;

use super::super::DeviceBackend;
use super::super::kernels::MOE_USED;
use super::{Experts, NONE, Slab};

/// When free memory drops below this, the cache shrinks. It sits above the
/// few hundred MiB the driver keeps unoccupied on a full card, so a full card
/// always reads as short.
const LOW_BYTES: usize = 512 << 20;

/// Free memory a shrink restores, room for later growth.
const HEADROOM_BYTES: usize = 1 << 30;

/// Free memory a grow leaves.
const GROWN_HEADROOM_BYTES: usize = 1536 << 20;

/// The least a grow may add. Free memory has to exceed
/// [`GROWN_HEADROOM_BYTES`] by this much before the cache grows.
const GROW_BYTES: usize = 512 << 20;

/// Decode passes between free memory checks.
const HEADROOM_EVERY: usize = 16;

impl Experts {
    /// Drops the slabs and everything that points into them. The next `moe`
    /// lays them out again with at most `per_block` slots per block.
    fn drop_slabs(&mut self, per_block: usize) {
        self.slabs = None;
        self.grouped = None;
        self.refills.clear();
        self.per_block_cap = Some(per_block);
        for b in &mut self.blocks {
            b.slot_of.fill(NONE);
            b.fetched = None;
        }
    }
}

impl DeviceBackend {
    /// Shrinks the cache when free memory drops below [`LOW_BYTES`], and
    /// grows it back when [`GROW_BYTES`] more than [`GROWN_HEADROOM_BYTES`]
    /// is free. Call between passes.
    ///
    /// Querying free memory is not free, so decode checks every
    /// [`HEADROOM_EVERY`] passes, and a prompt pass and the pass after a
    /// resize check every time.
    pub(in super::super) fn keep_headroom(&self, rows: usize) -> Result<()> {
        let mut experts = self.experts.borrow_mut();
        experts.since_checked += 1;
        if rows == 1 && experts.since_checked < HEADROOM_EVERY {
            return Ok(());
        }
        experts.since_checked = 0;
        let Some(slabs) = experts.slabs.as_ref() else {
            return Ok(());
        };
        let free = free_bytes()?;
        let slab_bytes: usize = slabs.values().map(Slab::held_bytes).sum();
        // The bytes one more slot in every block costs.
        let per_row = slab_bytes / (experts.per_block + 1);
        let per_block = if free < LOW_BYTES {
            let fewer = (HEADROOM_BYTES - free).div_ceil(per_row);
            experts.per_block.saturating_sub(fewer).max(MOE_USED)
        } else if free >= GROWN_HEADROOM_BYTES + GROW_BYTES {
            let more = (free - GROWN_HEADROOM_BYTES) / per_row;
            let grown = (experts.per_block + more).min(experts.most_per_block);
            if (grown - experts.per_block) * per_row < GROW_BYTES {
                return Ok(());
            }
            grown
        } else {
            return Ok(());
        };
        if per_block == experts.per_block {
            return Ok(());
        }
        let verb = if per_block < experts.per_block { "shrinking" } else { "growing" };
        phobos_base::log::emit(
            phobos_base::log::Level::Info,
            format_args!(
                "expert cache: {} MiB of the card left free; {verb} from {} to {per_block} slots a block",
                free >> 20,
                experts.per_block
            ),
        );
        // Nothing may still read or fill a slot, and the cached pass graphs
        // point into the slabs, so drop them too.
        experts.join_started()?;
        self.stream.synchronize()?;
        self.copy_stream.synchronize()?;
        self.pass.borrow_mut().clear();
        experts.drop_slabs(per_block);
        experts.since_checked = HEADROOM_EVERY;
        Ok(())
    }
}

/// Free device memory, the whole card's where NVML can say.
pub(in super::super) fn free_bytes() -> Result<usize> {
    let (own, _) = cust::memory::mem_get_info()?;
    Ok(phobos_kernels::nvml::card_free_bytes().map_or(own, |card| card.min(own)))
}
