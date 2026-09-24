use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use anyhow::Result;
use cust::memory::DeviceBuffer;

/// Released allocations, handed out again rather than going back to the driver.
#[derive(Default)]
pub struct Pool {
    free: RefCell<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
    /// Buffers [`Pool::take`] found on the free list, and buffers it had to
    /// allocate. Counters only: nothing here reads them back.
    reused: Cell<u64>,
    allocated: Cell<u64>,
    /// Bytes of every buffer the pool allocated and still accounts for, in
    /// use or not, and of those the ones waiting on the free list.
    owned_bytes: Cell<u64>,
    idle_bytes: Cell<u64>,
}

impl Pool {
    pub fn new() -> Pool {
        Pool::default()
    }

    /// A buffer of exactly `len` elements, reusing a released one if there is
    /// one. Contents are undefined: the caller must overwrite the whole buffer
    /// before reading it. Not for storage written immediately rather than from
    /// the stream; see [`Pool::take_fresh`].
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
    /// A steady state reuses nearly everything: the shapes a pass asks for are
    /// the same every pass. Allocations still climbing once decoding has
    /// settled mean something is asking for a size nothing returns.
    /// [`Pool::take_fresh`] is deliberately not counted, being a request for a
    /// new buffer rather than a lookup that missed.
    pub fn reuse_counts(&self) -> (u64, u64) {
        (self.reused.get(), self.allocated.get())
    }

    /// Bytes in buffers handed out and not yet back, and bytes on the free
    /// list. Idle bytes that keep growing are the same leak as allocations
    /// that keep climbing, measured in what it costs.
    pub fn byte_counts(&self) -> (u64, u64) {
        let idle_bytes = self.idle_bytes.get();
        (self.owned_bytes.get() - idle_bytes, idle_bytes)
    }

    /// A buffer of exactly `len` elements that the pool has never handed out,
    /// for storage the caller writes immediately rather than from the stream:
    /// a released buffer can still be read by launches recorded earlier in the
    /// same pass but not yet run, and an immediate write would land before them
    /// and overwrite their input. Only the caller knows whether it is
    /// recording, so only the caller can choose between this and [`Pool::take`].
    pub fn take_fresh(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: no caller reads before writing, see above.
        let buf = unsafe { DeviceBuffer::uninitialized(len)? };
        self.owned_bytes
            .set(self.owned_bytes.get() + bytes_of(&buf));
        Ok(buf)
    }

    /// Hand a buffer back. It is filed under its own length, so it only comes
    /// out again for an allocation of exactly that many elements.
    ///
    /// Each length's buffers are kept in address order, lowest out first, so
    /// which buffer an allocation gets depends only on what is free and not
    /// on the order it was released in. A recorded pass that releases in a
    /// different order than it allocates would otherwise swap its buffers
    /// every step, and every launch reading them would need its graph node
    /// patched.
    pub fn put(&self, buf: DeviceBuffer<f32>) {
        self.idle_bytes.set(self.idle_bytes.get() + bytes_of(&buf));
        let mut free = self.free.borrow_mut();
        let list = free.entry(buf.len()).or_default();
        let at = buf.as_device_ptr().as_raw();
        let slot = list.partition_point(|b| b.as_device_ptr().as_raw() > at);
        list.insert(slot, buf);
    }

    /// Frees everything held back to the driver and returns how many bytes that
    /// was. The caller must ensure the stream is idle first (see
    /// [`Pool::take_fresh`]). Worth calling when the pass shape changes, since
    /// the pool keys on exact length and a prompt pass's scratch can never
    /// serve a decode step.
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
