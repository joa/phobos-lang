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

    /// A buffer of exactly `len` elements, reusing a released one when there is
    /// one. The contents are undefined: every caller either copies the whole
    /// buffer from the host or has a kernel write all of it before reading.
    ///
    /// Not for storage written immediately rather than from the stream; see
    /// [`Pool::take_fresh`].
    pub fn take(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        if let Some(pooled) = self.free.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            return Ok(pooled);
        }
        self.take_fresh(len)
    }

    /// A buffer of exactly `len` elements that the pool has never handed out,
    /// for storage the caller writes immediately rather than from the stream.
    ///
    /// The free list is only free in the sense that no handle points at it.
    /// While a caller is recording a pass, a buffer released earlier in that
    /// same pass is still read by launches recorded and not yet run. An
    /// immediate write lands before those launches do, so a pooled buffer would
    /// overwrite their input. A fresh allocation cannot collide, since the pool
    /// holds every released buffer alive.
    ///
    /// The caller decides which of the two it needs, because only the caller
    /// knows whether it is recording. Two reached for it in `phobos-gguf` and
    /// both were wrong before the distinction existed: a constant that grows
    /// mid-pass, as the rotary table does when a sequence passes its length,
    /// and a zero fill whose memset goes straight to the stream while the pass
    /// around it is only recorded.
    pub fn take_fresh(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: no caller reads before writing, see above.
        Ok(unsafe { DeviceBuffer::uninitialized(len)? })
    }

    /// Hand a buffer back. It is filed under its own length, so it can only
    /// come out again for an allocation of exactly that many elements.
    pub fn put(&self, buf: DeviceBuffer<f32>) {
        self.free
            .borrow_mut()
            .entry(buf.len())
            .or_default()
            .push(buf);
    }
}
