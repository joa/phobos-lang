// Admission to device memory: an ALLOC waits for room in the arena, a FREE
// for the peers still to fetch its tile, and a peer's fetch for the tile to
// become resident.

use anyhow::Result;
use tokio::sync::oneshot;

use phobos_base::cli::format_size;
use phobos_base::phinfo;
use phobos_cluster::isa::InstrId;
use phobos_cluster::tile::TileId;

use super::Engine;

impl Engine {
    pub(super) fn try_alloc(&mut self, iid: InstrId, tile: TileId, shape: Vec<u64>) -> Result<bool> {
        if !self.store.can_alloc(&shape) {
            self.deferred_allocs.push((iid, tile, shape));
            return Ok(false);
        }

        self.store.alloc(tile, shape)?;
        self.complete(iid)?;

        Ok(true)
    }

    pub(super) fn try_free(&mut self, iid: InstrId, tile: TileId) -> Result<bool> {
        if !self.store.serves_satisfied(tile) {
            self.deferred_frees.push((iid, tile));
            return Ok(false);
        }

        self.store.free(tile)?;
        self.complete(iid)?;

        Ok(true)
    }

    pub(super) fn on_resident(&mut self, tile: TileId) -> Result<()> {
        if let Some(waiters) = self.parked_serves.remove(&tile) {
            for reply in waiters {
                self.serve_now(tile, reply)?;
            }
        }
        Ok(())
    }

    pub(super) fn on_serve(&mut self, tile: TileId, reply: oneshot::Sender<Option<Vec<f32>>>) -> Result<()> {
        if self.store.is_resident(tile) {
            self.serve_now(tile, reply)?;
        } else {
            self.parked_serves.entry(tile).or_default().push(reply);
        }
        Ok(())
    }

    fn serve_now(&mut self, tile: TileId, reply: oneshot::Sender<Option<Vec<f32>>>) -> Result<()> {
        let data = self.store.d2h(tile)?;
        phinfo!(
            "node{}: SERVE tile={:#x} {} -> peer (D2H)",
            self.node_id,
            tile.0,
            format_size(data.len() as u64 * 4),
        );
        let _ = reply.send(Some(data));
        self.store.record_serve(tile)?;
        self.retry_frees();
        Ok(())
    }

    fn retry_frees(&mut self) {
        let mut freed_any = false;

        for (iid, tile) in std::mem::take(&mut self.deferred_frees) {
            match self.try_free(iid, tile) {
                Ok(freed) => freed_any |= freed,
                Err(e) => self.fail(iid, format!("deferred free: {e:#}")),
            }
        }

        if freed_any {
            self.retry_allocs();
        }
    }

    pub(super) fn retry_allocs(&mut self) {
        for (iid, tile, shape) in std::mem::take(&mut self.deferred_allocs) {
            if let Err(e) = self.try_alloc(iid, tile, shape) {
                self.fail(iid, format!("alloc retry: {e:#}"));
            }
        }
    }

    pub(super) fn check_alloc_stall(&mut self) {
        if self.deferred_allocs.is_empty()
            || self.outstanding_io != 0
            || !self.deferred_frees.is_empty()
            || !self.fetching_kernel.is_empty()
        {
            return;
        }

        let stuck = std::mem::take(&mut self.deferred_allocs);
        for (iid, tile, shape) in stuck {
            let bytes = shape.iter().product::<u64>() * 4;
            self.fail(
                iid,
                format!(
                    "device arena exhausted: tile {:#x} needs {} but the resident working \
                     set already fills the arena; raise the node's --arena, or lower \
                     SUPER_*/the scheduler memory budget",
                    tile.0,
                    format_size(bytes),
                ),
            );
        }
    }
}
