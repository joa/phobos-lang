// Pinned host memory that kernels access in place, for the small tables a
// decode step passes between host and device.

use anyhow::Result;
use cust::memory::{DeviceCopy, LockedBuffer};
use phobos_kernels::cuda_ok;

/// Pinned host memory the device reads and writes in place, through its
/// mapped address.
///
/// This avoids DMA, because on this driver any transfer queues behind the
/// expert copies in flight, while kernel loads and stores do not. The host
/// may read device writes after the next sync point. The device sees host
/// writes in any launch made after them.
pub(super) struct Mapped<T: DeviceCopy> {
    host: LockedBuffer<T>,
    dev: u64,
}

impl<T: DeviceCopy + Default> Mapped<T> {
    pub(super) fn new(len: usize) -> Result<Mapped<T>> {
        let mut host = LockedBuffer::new(&T::default(), len)?;
        let mut dev = 0;
        // SAFETY: the buffer is page-locked and outlives its mapping.
        cuda_ok(unsafe { cust::sys::cuMemHostGetDevicePointer_v2(&mut dev, host.as_mut_slice().as_mut_ptr().cast(), 0) }, "mapping a pinned buffer")?;
        Ok(Mapped { host, dev })
    }

    pub(super) fn dev(&self) -> u64 {
        self.dev
    }

    pub(super) fn host(&self) -> &[T] {
        self.host.as_slice()
    }

    pub(super) fn host_mut(&mut self) -> &mut [T] {
        self.host.as_mut_slice()
    }

    pub(super) fn host_ptr(&mut self) -> *mut T {
        self.host.as_mut_slice().as_mut_ptr()
    }

    /// Writes `values` at `at` and returns their device address. No earlier
    /// launch may still be waiting to read the region.
    pub(super) fn put(&mut self, at: usize, values: &[T]) -> u64 {
        self.host.as_mut_slice()[at..at + values.len()].copy_from_slice(values);
        self.dev + (at * size_of::<T>()) as u64
    }
}
