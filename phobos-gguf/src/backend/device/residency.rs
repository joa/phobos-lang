use super::*;

impl DeviceBackend {
    /// The one scratch a prompt pass expands a raw weight into, at least
    /// `len` elements of it. One buffer at `RAW_DEQUANT_BUDGET_BYTES`
    /// serves every weight in the model, grown rather than reallocated per
    /// weight, and held for the process.
    ///
    /// Growing has to drain the stream first: an earlier weight in this pass
    /// has launches recorded and not yet run that still read the old buffer.
    /// The scratch reaches the budget on its first use and does not grow
    /// again.
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

    /// Hand a prompt pass's buffers back once the pass shape moves on from
    /// it. The pool files a released buffer under its exact length and the
    /// activation slots only grow, so a server whose prompts end in a ragged
    /// batch of a new length every request strands a whole pass's worth of
    /// `rows x width` buffers per length, gigabytes within a few dozen
    /// requests. Same-length passes, a long prompt's full batches or a
    /// repeated benchmark, keep theirs: they are the next pass's buffers.
    pub(super) fn trim_after_prompt(&self, rows: usize) -> Result<()> {
        let last = self.last_rows.replace(rows);
        if last <= 1 || last == rows {
            return Ok(());
        }
        self.stream.synchronize()?;
        self.act_scratch.borrow_mut().clear();
        // Freeing anything frees memory the cached pass graph points at.
        self.pass.borrow_mut().take();
        self.pool.trim();
        Ok(())
    }

    /// Hand the scratch and the pool's free list back, once, at the start
    /// of the pass after the one that filled them. This is the only safe
    /// point: a cached pass graph replays with raw device pointers in its
    /// kernel nodes, so freeing any later leaves it reading memory the
    /// driver already took back. The bytes go to the driver, not the pool,
    /// since a pooled buffer stays reserved for a shape decode never asks
    /// for. The stream has to be drained first: a buffer released while a
    /// pass was being recorded is still read by launches that have not
    /// run.
    pub(super) fn trim_after_dense(&self) -> Result<()> {
        if !self.drop_scratch.replace(false) {
            return Ok(());
        }
        self.stream.synchronize()?;
        // The quantized-activation slots always: a prompt pass takes one per
        // projection, `m * k` bytes each, and they are not pooled, so
        // nothing else ever gives them back. A decode step re-takes the two
        // or three it needs at one row, which costs nothing.
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
        if !vram_report() {
            return;
        }
        *self.alloc_hist.borrow_mut().entry(len).or_insert(0) += delta;
    }

    /// The shapes the pool's free list is made of: an entry is one distinct
    /// length, and a length released more often than taken is one sitting in
    /// the free list for a shape nothing is asking for.
    pub(super) fn mark_alloc_hist(&self) {
        if !vram_report() {
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
        if !vram_report() {
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
        let mut named: Vec<(usize, usize)> = self
            .slots
            .borrow()
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (s.len(), i)))
            .filter(|&(len, _)| len * 4 >= 8 << 20)
            .collect();
        named.sort_unstable_by_key(|&(len, _)| std::cmp::Reverse(len));
        eprintln!(
            "[vram]   slots over 8 MiB: {:?}",
            named
                .iter()
                .take(8)
                .map(|&(len, i)| format!("#{i} {len} f32 = {:.0} MiB", mib(len * 4)))
                .collect::<Vec<_>>()
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
        let (state_held, state_used) = self.state_arena.bytes();
        let (hot_held, hot_used) = self.hot.bytes();
        let owned: usize = self
            .owned_quants
            .borrow()
            .iter()
            .map(|(q, s, r)| q.len() + 4 * s.len() + 4 * r.len())
            .sum();
        eprintln!(
            "[vram]   state {:.0} MiB ({:.0} used) in {} slabs | hot {:.0} ({:.0}) | owned quants {:.0} MiB | accounted {:.0} MiB",
            mib(state_held),
            mib(state_used),
            self.state_arena.slabs(),
            mib(hot_held),
            mib(hot_used),
            mib(owned),
            mib(raw + state_held + hot_held + owned + act + fused + live),
        );
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

/// The same for a named model constant: names what is big rather than where.
pub(super) fn note_big_const(key: &str, len: usize) {
    if len * size_of::<f32>() < (8 << 20) || !vram_report() {
        return;
    }
    eprintln!(
        "[vram] constant {key:?} {:.0} MiB ({len} f32)",
        (len * size_of::<f32>()) as f64 / (1 << 20) as f64
    );
}

/// Where a large pass buffer is asked for. `PHOBOS_VRAM=1` only: the call
/// site is the only thing that names it, and one buffer nobody remembers
/// asking for can be the difference between the output head staying
/// resident or not.
#[track_caller]
pub(super) fn note_big_alloc(len: usize) {
    if len * size_of::<f32>() < (8 << 20) || !vram_report() {
        return;
    }
    eprintln!(
        "[vram] alloc {:.0} MiB ({len} f32) at {}",
        (len * size_of::<f32>()) as f64 / (1 << 20) as f64,
        std::panic::Location::caller()
    );
}

/// What the card has free at a named point, and what the step before it
/// cost. `PHOBOS_VRAM=1` only; this is how the output head's headroom gets
/// attributed to the weights, the loaded modules and the context.
pub(super) fn vram_mark(label: &str) {
    use std::sync::atomic::{AtomicI64, Ordering};
    static LAST: AtomicI64 = AtomicI64::new(-1);
    if !vram_report() {
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

/// Whether `PHOBOS_VRAM` is set, read once: every allocation and release
/// asks, and reading the environment takes a process-wide lock.
pub(super) fn vram_report() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PHOBOS_VRAM").is_some())
}
