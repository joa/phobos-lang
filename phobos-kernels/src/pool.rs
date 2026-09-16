use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;
use cust::memory::DeviceBuffer;

/// Released allocations, handed out again rather than going back to the driver.
#[derive(Default)]
pub struct Pool {
    free: RefCell<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
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
            return Ok(pooled);
        }
        self.take_fresh(len)
    }

    /// A buffer of exactly `len` elements that the pool has never handed out,
    /// for storage the caller writes immediately rather than from the stream:
    /// a released buffer can still be read by launches recorded earlier in the
    /// same pass but not yet run, and an immediate write would land before them
    /// and overwrite their input. Only the caller knows whether it is
    /// recording, so only the caller can choose between this and [`Pool::take`].
    pub fn take_fresh(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: no caller reads before writing, see above.
        Ok(unsafe { DeviceBuffer::uninitialized(len)? })
    }

    /// Hand a buffer back. It is filed under its own length, so it only comes
    /// out again for an allocation of exactly that many elements.
    pub fn put(&self, buf: DeviceBuffer<f32>) {
        self.free
            .borrow_mut()
            .entry(buf.len())
            .or_default()
            .push(buf);
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
        bytes
    }
}
