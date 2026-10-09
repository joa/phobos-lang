use super::*;

impl DeviceBackend {
    /// The shared scratch a prompt pass expands a raw weight into, holding at
    /// least `len` elements. One buffer serves every weight in the model and
    /// only ever grows.
    ///
    /// Growing drains the stream first, because launches already recorded in
    /// this pass still read the old buffer.
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

    /// Records that this pass used prompt-only scratch, so the next pass
    /// frees it.
    pub(super) fn note_dense_pass(&self) {
        self.drop_scratch.set(true);
    }

    /// Frees a prompt pass's buffers when the next pass has a different
    /// row count.
    ///
    /// The pool keys free buffers by exact length. Without this, every new
    /// prompt length would keep a full pass's worth of buffers alive.
    /// Passes of the same length keep theirs for reuse.
    pub(super) fn trim_after_prompt(&self, rows: usize) -> Result<()> {
        let last = self.last_rows.replace(rows);
        if last <= 1 || last == rows {
            return Ok(());
        }
        self.stream.synchronize()?;
        self.clear_act_slots();
        // Cached pass graphs point into this memory, so drop them too.
        self.pass.borrow_mut().clear();
        self.pool.trim();
        Ok(())
    }

    /// Drops every quantized-activation slot and the arena under them. The
    /// stream must be idle.
    fn clear_act_slots(&self) {
        self.act_scratch.borrow_mut().clear();
        self.act_arena.reset();
    }

    /// Frees the prompt scratch and the pool's free list at the start of the
    /// pass after the one that used them.
    ///
    /// Cached pass graphs hold raw pointers, so they are dropped here too.
    /// The memory goes back to the driver, not the pool. The stream is
    /// drained first, since recorded launches may still read these buffers.
    ///
    /// Another prompt pass of the same length, the next chunk of a long
    /// prompt, keeps them: it takes the same slots, and
    /// [`DeviceBackend::trim_after_prompt`] has already freed them if the
    /// length changed.
    pub(super) fn trim_after_dense(&self, rows: usize) -> Result<()> {
        if rows > 1 || !self.drop_scratch.replace(false) {
            return Ok(());
        }
        self.stream.synchronize()?;
        // Always free the quantized-activation slots. A prompt pass takes one
        // per projection.
        self.clear_act_slots();
        // Cached pass graphs point into this memory, so drop them too.
        self.pass.borrow_mut().clear();
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

    /// Reports free memory at the first few pass boundaries, then every
    /// 32nd. Only with `PHOBOS_VRAM`.
    pub(super) fn mark_pass_vram(&self) {
        let n = self.pass_marks.get();
        self.pass_marks.set(n + 1);
        if n < 4 || n.is_multiple_of(32) {
            vram_mark(&format!("pass {n}"));
            self.mark_buffers();
            self.mark_alloc_hist();
        }
    }

    /// Counts an allocation of `len` elements in or out, so the free list can
    /// be broken down by length.
    pub(super) fn note_alloc(&self, len: usize, delta: isize) {
        if !vram_report() {
            return;
        }
        *self.alloc_hist.borrow_mut().entry(len).or_insert(0) += delta;
    }

    /// Prints the pool's free list by length. A length released more often
    /// than taken is sitting idle in the free list.
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

    /// Prints the live buffers and the weight storage. The gap to
    /// [`vram_mark`]'s figure is the pool's free list plus what the driver
    /// holds itself.
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
        let act = self.act_arena.bytes().0;
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

/// Reports a large named model constant. Only with `PHOBOS_VRAM`.
pub(super) fn note_big_const(key: &str, len: usize) {
    if len * size_of::<f32>() < (8 << 20) || !vram_report() {
        return;
    }
    eprintln!(
        "[vram] constant {key:?} {:.0} MiB ({len} f32)",
        (len * size_of::<f32>()) as f64 / (1 << 20) as f64
    );
}

/// Reports the call site of a large pass buffer allocation. Only with
/// `PHOBOS_VRAM`.
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

/// Reports free device memory at a named point, and the change since the
/// last mark. Only with `PHOBOS_VRAM`.
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

/// Whether `PHOBOS_VRAM` is set. Read once, since every allocation asks.
pub(super) fn vram_report() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| phobos_base::env::flag("PHOBOS_VRAM"))
}
