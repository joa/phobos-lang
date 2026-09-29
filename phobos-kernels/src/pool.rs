use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use anyhow::Result;
use cust::memory::DeviceBuffer;

/// Released allocations, handed out again rather than going back to the driver.
#[derive(Default)]
pub struct Pool {
    free: RefCell<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
    /// How many [`Pool::take`] calls were served from the free list, and how
    /// many had to allocate.
    reused: Cell<u64>,
    allocated: Cell<u64>,
    /// Bytes in every buffer the pool allocated and still accounts for, and
    /// the part of those waiting on the free list.
    owned_bytes: Cell<u64>,
    idle_bytes: Cell<u64>,
}

impl Pool {
    pub fn new() -> Pool {
        Pool::default()
    }

    /// A buffer of exactly `len` elements, reusing a released one if there is
    /// one. Contents are undefined, so the caller must overwrite the whole
    /// buffer before reading it.
    ///
    /// Not for storage written immediately rather than from the stream; see
    /// [`Pool::take_fresh`].
    pub fn take(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        if let Some(pooled) = self.free.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            self.reused.set(self.reused.get() + 1);
            self.idle_bytes
                .set(self.idle_bytes.get() - bytes_of(&pooled));
            return Ok(pooled);
        }
        self.allocated.set(self.allocated.get() + 1);
        self.take_fresh(len)
    }

    /// Buffers handed back off the free list, and buffers allocated because
    /// nothing of that size was on it.
    ///
    /// In a steady state nearly everything is reused, since every pass asks
    /// for the same shapes. Allocations that keep climbing after decoding
    /// settles mean something asks for a size nothing returns.
    /// [`Pool::take_fresh`] is not counted.
    pub fn reuse_counts(&self) -> (u64, u64) {
        (self.reused.get(), self.allocated.get())
    }

    /// Bytes in buffers handed out and not yet back, and bytes on the free
    /// list. Idle bytes that keep growing point to the same leak as climbing
    /// allocations.
    pub fn byte_counts(&self) -> (u64, u64) {
        let idle_bytes = self.idle_bytes.get();
        (self.owned_bytes.get() - idle_bytes, idle_bytes)
    }

    /// A buffer of exactly `len` elements that the pool has never handed out.
    /// Use it for storage the caller writes immediately rather than from the
    /// stream.
    ///
    /// A released buffer may still be read by launches recorded earlier in the
    /// pass but not yet run. An immediate write would land before them and
    /// overwrite their input. Only the caller knows whether it is recording,
    /// so only the caller can choose between this and [`Pool::take`].
    pub fn take_fresh(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: no caller reads before writing, see above.
        let buf = unsafe { DeviceBuffer::uninitialized(len)? };
        self.owned_bytes
            .set(self.owned_bytes.get() + bytes_of(&buf));
        Ok(buf)
    }

    /// Hand a buffer back. It is filed under its exact length and only comes
    /// out again for an allocation of that many elements.
    ///
    /// Each length's buffers are kept in address order, lowest out first. So
    /// which buffer an allocation gets depends only on what is free, not on
    /// release order. Otherwise a recorded pass that releases in a different
    /// order than it allocates would swap buffers every step, and every
    /// launch reading them would need its graph node patched.
    pub fn put(&self, buf: DeviceBuffer<f32>) {
        self.idle_bytes.set(self.idle_bytes.get() + bytes_of(&buf));
        let mut free = self.free.borrow_mut();
        let list = free.entry(buf.len()).or_default();
        let at = buf.as_device_ptr().as_raw();
        let slot = list.partition_point(|b| b.as_device_ptr().as_raw() > at);
        list.insert(slot, buf);
    }

    /// Frees every held buffer back to the driver and returns how many bytes
    /// that was. The stream must be idle first (see [`Pool::take_fresh`]).
    ///
    /// Worth calling when the pass shape changes. The pool keys on exact
    /// length, so a prompt pass's scratch never serves a decode step.
    pub fn trim(&self) -> usize {
        let mut free = self.free.borrow_mut();
        let bytes = free
            .iter()
            .map(|(len, bufs)| len * bufs.len() * size_of::<f32>())
            .sum();
        free.clear();
        self.owned_bytes.set(self.owned_bytes.get() - bytes as u64);
        self.idle_bytes.set(0);
        bytes
    }

    /// Gives a buffer the pool handed out back to the driver instead of the
    /// free list. Dropping it directly works too, but leaves it counted.
    pub fn forget(&self, buf: DeviceBuffer<f32>) {
        self.owned_bytes
            .set(self.owned_bytes.get() - bytes_of(&buf));
    }
}

fn bytes_of(buf: &DeviceBuffer<f32>) -> u64 {
    (buf.len() * size_of::<f32>()) as u64
}
