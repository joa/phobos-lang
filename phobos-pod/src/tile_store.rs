use std::collections::HashMap;

use anyhow::{Result, bail};
use cust::memory::DeviceBuffer;
use cust::memory::{CopyDestination, DevicePointer, DeviceSlice};
use phobos_cluster::tile::TileId;

#[derive(Clone, Copy)]
pub struct Slab {
    pub addr: u64, // CUdeviceptr
    pub class: usize,
}

pub struct DeviceArena {
    _arena: DeviceBuffer<u8>,
    base: u64,
    cap: usize,
    bump: usize,
    free: HashMap<usize, Vec<u64>>,
}

const ALIGN: usize = 256;
const MIN_CLASS: usize = 256;

fn size_class(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN).saturating_mul(ALIGN).max(MIN_CLASS)
}

impl DeviceArena {
    pub fn new(bytes: usize) -> Result<DeviceArena> {
        // SAFETY: contents are fully written by LOAD/FETCH/COMPUTE before read
        let arena = unsafe { DeviceBuffer::<u8>::uninitialized(bytes)? };
        let base = arena.as_device_ptr().as_raw();

        Ok(DeviceArena {
            _arena: arena,
            base,
            cap: bytes,
            bump: 0,
            free: HashMap::new(),
        })
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<Slab> {
        let class = size_class(bytes);

        if let Some(addr) = self.free.get_mut(&class).and_then(Vec::pop) {
            return Ok(Slab { addr, class });
        }

        let off = (self.bump + ALIGN - 1) & !(ALIGN - 1);
        if off + class > self.cap {
            bail!(
                "device arena exhausted ({} MiB cap): need {} more bytes",
                self.cap >> 20,
                class
            );
        }

        self.bump = off + class;

        Ok(Slab {
            addr: self.base + off as u64,
            class,
        })
    }

    pub fn dealloc(&mut self, slab: Slab) {
        self.free.entry(slab.class).or_default().push(slab.addr);
    }

    pub fn can_alloc(&self, bytes: usize) -> bool {
        let class = size_class(bytes);

        if self.free.get(&class).is_some_and(|v| !v.is_empty()) {
            return true;
        }

        let off = (self.bump + ALIGN - 1) & !(ALIGN - 1);

        off + class <= self.cap
    }
}

/// resident or in-flight supertile buffer
pub struct TileBuf {
    pub slab: Slab,
    pub shape: Vec<u64>,
    pub elems: usize,
    pub resident: bool,
    /// How often this tile was served to a peer.
    pub serves: u32,
    /// The number of serves required before FREE may release it.
    pub expected_serves: u32,
}

pub struct TileStore {
    arena: DeviceArena,
    tiles: HashMap<TileId, TileBuf>,
}

impl TileStore {
    pub fn new(arena_bytes: usize) -> Result<TileStore> {
        Ok(TileStore {
            arena: DeviceArena::new(arena_bytes)?,
            tiles: HashMap::new(),
        })
    }

    /// Reserve an f32 buffer (4b) and register the tile.
    pub fn alloc(&mut self, tile: TileId, shape: Vec<u64>) -> Result<()> {
        let elems: usize = shape.iter().product::<u64>() as usize;
        let slab = self.arena.alloc(elems * 4)?;
        self.tiles.insert(
            tile,
            TileBuf {
                slab,
                shape,
                elems,
                resident: false,
                serves: 0,
                expected_serves: 0,
            },
        );
        Ok(())
    }

    /// Whether an ALLOC for this shape at f32 would succeed.
    pub fn can_alloc(&self, shape: &[u64]) -> bool {
        let elems: usize = shape.iter().product::<u64>() as usize;
        self.arena.can_alloc(elems * 4)
    }

    pub fn is_resident(&self, tile: TileId) -> bool {
        self.tiles.get(&tile).is_some_and(|t| t.resident)
    }

    pub fn addr(&self, tile: TileId) -> Result<u64> {
        Ok(self.get(tile)?.slab.addr)
    }

    pub fn shape(&self, tile: TileId) -> Result<Vec<u64>> {
        Ok(self.get(tile)?.shape.clone())
    }

    fn get(&self, tile: TileId) -> Result<&TileBuf> {
        self.tiles
            .get(&tile)
            .ok_or_else(|| anyhow::anyhow!("unknown tile {:#x}", tile.0))
    }

    /// Host -> device marks the tile resident
    pub fn h2d(&mut self, tile: TileId, data: &[f32]) -> Result<()> {
        let t = self
            .tiles
            .get_mut(&tile)
            .ok_or_else(|| anyhow::anyhow!("h2d to unknown tile {:#x}", tile.0))?;
        if data.len() != t.elems {
            bail!("h2d length {} != tile elems {}", data.len(), t.elems);
        }

        // SAFETY: slab owns elems f32 of valid device memory
        let mut slice = unsafe {
            DeviceSlice::from_raw_parts(DevicePointer::<f32>::from_raw(t.slab.addr), t.elems)
        };

        slice.copy_from(data)?;
        t.resident = true;
        Ok(())
    }

    /// Device -> host (does not change residency).
    pub fn d2h(&self, tile: TileId) -> Result<Vec<f32>> {
        let t = self.get(tile)?;
        let mut out = vec![0f32; t.elems];
        let slice = unsafe {
            DeviceSlice::from_raw_parts(DevicePointer::<f32>::from_raw(t.slab.addr), t.elems)
        };
        slice.copy_to(&mut out)?;
        Ok(out)
    }

    /// A COMPUTE just wrote tile in place; it is now resident
    pub fn mark_resident(&mut self, tile: TileId) -> Result<()> {
        self.tiles
            .get_mut(&tile)
            .ok_or_else(|| anyhow::anyhow!("mark_resident on unknown tile {:#x}", tile.0))?
            .resident = true;
        Ok(())
    }

    pub fn set_expected_serves(&mut self, tile: TileId, n: u32) {
        if let Some(t) = self.tiles.get_mut(&tile) {
            t.expected_serves = n;
        }
    }

    /// Record one peer serve; returns the new serve count
    pub fn record_serve(&mut self, tile: TileId) -> Result<u32> {
        let t = self
            .tiles
            .get_mut(&tile)
            .ok_or_else(|| anyhow::anyhow!("serve of unknown tile {:#x}", tile.0))?;
        t.serves += 1;
        Ok(t.serves)
    }

    pub fn serves_satisfied(&self, tile: TileId) -> bool {
        self.tiles
            .get(&tile)
            .is_some_and(|t| t.serves >= t.expected_serves)
    }

    pub fn free(&mut self, tile: TileId) -> Result<()> {
        let t = self
            .tiles
            .remove(&tile)
            .ok_or_else(|| anyhow::anyhow!("free of unknown tile {:#x}", tile.0))?;
        self.arena.dealloc(t.slab);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_class_is_exact_fit_not_power_of_two() {
        // A 400 MiB supertile must claim ~400 MiB, not round to a 512 MiB slab:
        // three of them have to fit a 1200 MiB arena.
        let tile = 10240 * 10240 * 4; // 400 MiB, already 256-aligned
        assert_eq!(size_class(tile), tile);
        assert_eq!(size_class(tile).next_power_of_two(), 512 << 20);
        assert!(3 * size_class(tile) <= 1200 << 20);

        // Align up, with the floor for tiny allocations.
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(1), MIN_CLASS);
        assert_eq!(size_class(0), MIN_CLASS);
    }
}
