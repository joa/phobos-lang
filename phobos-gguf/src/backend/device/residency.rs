//! The one scratch a prompt pass expands a raw weight into, and what the card
//! is holding besides the weights.
//!
//! Dequantizing a raw weight needs a strip of `f32` big enough to matter next
//! to the weights themselves, and the pool keys on exact length. A pair taken
//! and released per weight therefore leaves one entry per distinct shape, not
//! one entry: 741 to 886 MiB measured on Qwen3.8-27B, against 6.22 GiB of
//! weights on a card with 6.95 GiB free. The driver answers by paging out the
//! largest cold allocation, which is the 521 MiB output head, and reading that
//! across PCIe once a token costs two orders of magnitude more than the
//! scratch saves.
//!
//! `k * strip` is bounded by `RAW_DEQUANT_BUDGET_BYTES` by construction, so one
//! buffer at the budget serves every weight in the model and a narrower one
//! uses a prefix: 131 MiB rather than 741.
//!
//! It is handed back at the first pass after a dense one, which is the only
//! point that is safe: a recorded pass graph is cached and replayed, and its
//! kernel nodes carry raw device pointers, so freeing a pooled buffer at any
//! later boundary can leave a replay reading memory the driver has taken back.
//! Doing exactly that at a *decode* boundary, after the decode graph had been
//! recorded, faults with `an illegal memory access was encountered`.
//!
//! Handing it back is what makes decode fast and what used to make prefill
//! slow, and it is *what* is handed back that reconciles them. Note the free
//! list itself is not where the memory is: measured, it holds **8 MiB across
//! two lengths**. What costs is the re-allocation the trim forces, which this
//! card does at roughly 70 ms a MiB. Measured at
//! `-p 128 -n 128 -r 2`, one session:
//!
//! | handed back at that boundary | pp128 | tg128 |
//! | --- | ---: | ---: |
//! | the whole free list, `Pool::trim` | 11.68 | 6.94 |
//! | nothing | 56.96 | 4.70 |
//!
//! The free list is hundreds of entries, so emptying it makes the next prompt
//! pass allocate them all again: that is the 40.5 s first rep. The scratch is
//! one buffer and one `cuMemAlloc`. So only the scratch goes back, and it goes
//! to the driver rather than to the pool, since a pooled buffer is still
//! reserved for a shape none of the decode steps ask for.
//!
//! Capping the free list on every `put` is not a substitute and is not safe: a
//! caller releases a buffer while the pass is still being recorded, and the
//! pool is what keeps it alive until those launches run. The cost of the trim is the
//! re-allocation the next prompt pass has to do, and that used to be 741 to
//! 886 MiB across seven pool entries: 30.2 s against 5.9 s for the pass after
//! it. Sized as one buffer it is a single 128 MiB `cuMemAlloc`. Held through
//! decode instead, a step runs against `live 336 MiB` and tg128 measures
//! 6.00 t/s against the 11.07 it should.

use super::*;

impl DeviceBackend {
    /// The dequant scratch, at least `len` elements of it. Grown rather than
    /// reallocated per weight, and held for the process.
    ///
    /// Growing has to drain the stream first: an earlier weight in this pass
    /// has launches recorded and not yet run that still read the old buffer.
    /// In practice the weight scratch reaches the budget on its first use and
    /// never grows again.
    pub(super) fn dense_scratch(&self, which: usize, len: usize) -> Result<Buf> {
        if !self.dense_scratch_shared {
            return self.alloc(len);
        }
        if let Some(buf) = self.dense_scratch[which].get() {
            if self.len_of(buf)? >= len {
                return Ok(buf);
            }
            self.stream.synchronize()?;
            self.release(buf);
        }
        let buf = self.alloc(len)?;
        self.dense_scratch[which].set(Some(buf));
        Ok(buf)
    }

    /// Record that this pass expanded a raw weight into the scratch.
    pub(super) fn note_dense_pass(&self) {
        self.drop_scratch.set(true);
    }

    /// Hand the scratch and the pool's free list back, once, at the start of
    /// the pass after the one that filled them. The stream has to be drained
    /// first: a buffer released while a pass was being recorded is still read
    /// by launches that have not run.
    pub(super) fn trim_after_dense(&self) -> Result<()> {
        if !self.drop_scratch.replace(false) {
            return Ok(());
        }
        self.stream.synchronize()?;
        // The quantized-activation slots always. A prompt pass takes one per
        // projection, `m * k` bytes each, and they are not pooled, so nothing
        // else ever gives them back: on this model they come to about 130 MiB
        // and that alone is tg128 5.69 t/s against 4.28. A decode step re-takes
        // the two or three it needs at one row, which costs nothing.
        self.act_scratch.borrow_mut().clear();
        // Freeing anything here frees memory the cached pass graph's kernel
        // nodes still point at, so it must not be replayed after this.
        self.pass.borrow_mut().take();
        if !self.trim_after_dense {
            return Ok(());
        }
        for slot in &self.dense_scratch {
            if let Some(buf) = slot.take() {
                self.discard(buf);
            }
        }
        self.pool.trim();
        Ok(())
    }

    /// Free memory at the first few pass boundaries, then every 32nd: the
    /// weights land on the first, so the drop across it is what they cost and
    /// what is left is what the output head has to sit in.
    pub(super) fn mark_pass_vram(&self) {
        let n = self.pass_marks.get();
        self.pass_marks.set(n + 1);
        if n < 4 || n.is_multiple_of(32) {
            vram_mark(&format!("pass {n}"));
            self.mark_buffers();
            self.mark_alloc_hist();
        }
    }

    /// Count an allocation of `len` elements in or out, so the free list can be
    /// attributed to the shapes that make it up.
    pub(super) fn note_alloc(&self, len: usize, delta: isize) {
        if std::env::var_os("PHOBOS_VRAM").is_none() {
            return;
        }
        *self.alloc_hist.borrow_mut().entry(len).or_insert(0) += delta;
    }

    /// The shapes the pool's free list is made of: an entry is one distinct
    /// length, and a length released more often than taken is one sitting in
    /// the free list for a shape nothing is asking for.
    pub(super) fn mark_alloc_hist(&self) {
        if std::env::var_os("PHOBOS_VRAM").is_none() {
            return;
        }
        let hist = self.alloc_hist.borrow();
        let mut held: Vec<(usize, isize)> = hist
            .iter()
            .filter(|&(_, &n)| n < 0)
            .map(|(&len, &n)| (len, -n))
            .collect();
        held.sort_by_key(|&(len, n)| std::cmp::Reverse(len * n as usize));
        let total: usize = held.iter().map(|&(l, n)| l * n as usize * 4).sum();
        eprintln!(
            "[vram]   free list: {} distinct lengths, {:.0} MiB",
            held.len(),
            total as f64 / (1 << 20) as f64,
        );
        for &(len, n) in held.iter().take(8) {
            eprintln!(
                "[vram]     {n} x {len} elements = {:.1} MiB",
                (len * n as usize * 4) as f64 / (1 << 20) as f64
            );
        }
    }

    /// What the pass allocations come to, against the raw weights. Neither is
    /// what [`vram_mark`] reports: the difference between the two is the pool's
    /// free list plus whatever the driver is holding on its own account.
    pub(super) fn mark_buffers(&self) {
        if std::env::var_os("PHOBOS_VRAM").is_none() {
            return;
        }
        let mib = |bytes: usize| bytes as f64 / (1 << 20) as f64;
        let live: usize = self
            .slots
            .borrow()
            .iter()
            .flatten()
            .map(|s| s.len() * size_of::<f32>())
            .sum();
        let live_n = self.slots.borrow().iter().flatten().count();
        let mut big: Vec<usize> = self
            .slots
            .borrow()
            .iter()
            .flatten()
            .map(|s| s.len() * size_of::<f32>())
            .filter(|&b| b >= 1 << 20)
            .collect();
        big.sort_unstable_by_key(|&b| std::cmp::Reverse(b));
        eprintln!(
            "[vram]   live buffers over 1 MiB: {} of them, {:.0} MiB; largest {:?} MiB",
            big.len(),
            mib(big.iter().sum()),
            big.iter().take(6).map(|&b| mib(b) as usize).collect::<Vec<_>>(),
        );
        let (raw, handed) = self.arena.bytes();
        let pair = |v: &RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>| -> usize {
            v.borrow().iter().map(|(b, s)| b.len() + 4 * s.len()).sum()
        };
        let act = pair(&self.act_scratch);
        eprintln!(
            "[vram]   handles: {} constants, {} quants, {} raw, {} slabs, {} hot slabs",
            self.constants.borrow().len(),
            self.quants.borrow().len(),
            self.raw_quants.borrow().len(),
            self.arena.slabs(),
            self.hot.slabs(),
        );
        let fused = pair(&self.fused_scratch);
        eprintln!(
            "[vram]   live {live_n} bufs {:.0} MiB | arena {:.0} MiB ({:.0} used) in {} slabs (+{} hot) | act {:.0} | fused {:.0} MiB",
            mib(live),
            mib(raw),
            mib(handed),
            self.arena.slabs(),
            self.hot.slabs(),
            mib(act),
            mib(fused),
        );
    }
}

/// What the card has free at a named point, and what the step before it cost.
/// `PHOBOS_VRAM=1` only; this is how the headroom a 521 MiB output head has to
/// sit in gets attributed to the weights, the loaded modules and the context.
pub(super) fn vram_mark(label: &str) {
    use std::sync::atomic::{AtomicI64, Ordering};
    static LAST: AtomicI64 = AtomicI64::new(-1);
    if std::env::var_os("PHOBOS_VRAM").is_none() {
        return;
    }
    let Ok((free, total)) = cust::memory::mem_get_info() else {
        return;
    };
    let mib = |bytes: i64| bytes as f64 / (1 << 20) as f64;
    let free_mib = free as i64;
    let prev = LAST.swap(free_mib, Ordering::Relaxed);
    let step = if prev < 0 {
        String::new()
    } else {
        format!(", {:+.0} MiB since the last mark", mib(free_mib - prev))
    };
    eprintln!(
        "[vram] {label}: {:.0} of {:.0} MiB free{step}",
        mib(free_mib),
        mib(total as i64),
    );
}
