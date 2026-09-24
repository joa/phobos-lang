// The expert cache giving memory back as the rest of a session grows.
//
// The cache is sized from free memory when it is laid out, and the
// attention caches, a longer prompt's scratch and the desktop all grow
// after. On this driver an allocation past the card's memory succeeds by
// evicting others to the host, and a decode step that then reads one costs
// several times what the slots it would have taken save.

use anyhow::Result;

use super::super::DeviceBackend;
use super::super::kernels::MOE_USED;
use super::{Experts, NONE, Slab};

/// Device memory the cache leaves free for what grows after it is laid
/// out. Below half of it at a pass boundary the cache is laid out again
/// smaller, to leave all of it.
const HEADROOM_BYTES: usize = 512 << 20;

/// Decode passes between two looks at free memory.
const HEADROOM_EVERY: usize = 16;

impl Experts {
    /// The slabs dropped, for the next `moe` to lay out again with at most
    /// `per_block` slots a block, and everything that pointed into them
    /// forgotten.
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
    /// The cache laid out again smaller when the card has come within half
    /// of [`HEADROOM_BYTES`] of full, for the next `moe` to fill. Called
    /// between passes. It only ever shrinks, so a session settles at the
    /// size its longest context needed. Asking the driver costs most of a
    /// hundred microseconds, so a decode step, which grows the caches a
    /// position at a time, asks every [`HEADROOM_EVERY`] passes, and a
    /// prompt pass always.
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
        let (free, _) = cust::memory::mem_get_info()?;
        if free >= HEADROOM_BYTES / 2 || experts.per_block <= MOE_USED {
            return Ok(());
        }
        let slab_bytes: usize = slabs.values().map(Slab::held_bytes).sum();
        let blocks = experts.blocks.len();
        let per_slot = slab_bytes / ((experts.per_block + 1) * blocks);
        let fewer = (HEADROOM_BYTES - free).div_ceil(per_slot * blocks);
        let per_block = experts.per_block.saturating_sub(fewer).max(MOE_USED);
        phobos_base::log::emit(
            phobos_base::log::Level::Info,
            format_args!(
                "expert cache: {} MiB of the card left free; shrinking from {} to {per_block} slots a block",
                free >> 20,
                experts.per_block
            ),
        );
        // Nothing may still read or fill a slot, and the cached pass graphs
        // point into the slabs about to go.
        experts.join_started()?;
        self.stream.synchronize()?;
        self.copy_stream.synchronize()?;
        self.pass.borrow_mut().clear();
        experts.drop_slabs(per_block);
        Ok(())
    }
}
