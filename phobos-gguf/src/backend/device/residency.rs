//! Handing a prompt pass's scratch back before the decode steps run.
//!
//! Dequantizing a raw weight needs a strip of `f32` big enough to matter next
//! to the weights themselves, and the pool keys on exact length, so nothing a
//! prompt pass leaves behind can ever serve a decode step. Held anyway it is
//! memory only the weights could have used, and on a card where they already
//! nearly fill it the driver answers by paging the largest one out. That costs
//! far more than the scratch saves: an output head read across PCIe instead of
//! from VRAM is two orders of magnitude slower, once per token, forever.

use anyhow::Result;
use cust::memory::mem_get_info;

use super::DeviceBackend;

impl DeviceBackend {
    /// Record that this pass dequantized a raw weight into pooled scratch.
    pub(super) fn note_dense_pass(&self) {
        self.dense_pass.set(true);
    }

    /// Give that scratch back, once, at the start of the pass after it.
    ///
    /// The stream has to be idle first: a buffer released while a pass was
    /// being recorded is still read by launches that have not run yet.
    pub(super) fn trim_after_dense(&self) -> Result<()> {
        if !self.dense_pass.replace(false) {
            return Ok(());
        }
        self.stream.synchronize()?;
        let freed = self.pool.trim();
        if std::env::var_os("PHOBOS_VRAM").is_some() {
            let (free, total) = mem_get_info()?;
            let mib = |bytes: usize| bytes as f64 / (1 << 20) as f64;
            eprintln!(
                "[vram] trimmed {:.0} MiB of prompt scratch, {:.0} of {:.0} MiB free",
                mib(freed),
                mib(free),
                mib(total),
            );
        }
        Ok(())
    }
}
