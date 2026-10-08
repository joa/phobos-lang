// Device buffers, the pointers handed to kernels, and the scratch a
// pass grows into.

use super::*;

/// Free and total bytes on the card. Free is the whole card's where NVML can
/// say, since the process's own figure does not see another program fill it.
/// Under WDDM it stops a few hundred MiB short of zero once the card is
/// over-subscribed, so a reading near that floor means full.
pub(super) fn card_memory() -> Result<(usize, usize)> {
    let (own_free, total) = cust::memory::mem_get_info()?;
    let free = phobos_kernels::nvml::card_free_bytes().map_or(own_free, |card| card.min(own_free));
    Ok((free, total))
}

/// What a [`Buf`] handle points at: an owned buffer that goes back to the
/// pool, or a region of an arena.
///
/// A constant is never released, so it lives in the arena rather than in an
/// allocation of its own. See `arena.rs`.
pub(super) enum Slot {
    Owned(DeviceBuffer<f32>),
    Const {
        at: u64,
        len: usize,
    },
    /// A sequence's recurrent state. Arena-backed like a constant, but
    /// released when the sequence ends. The state arena counts its live
    /// regions and resets once none are left.
    State {
        at: u64,
        len: usize,
    },
}

impl Slot {
    fn base(&self) -> u64 {
        match self {
            Slot::Owned(b) => b.as_device_ptr().as_raw(),
            Slot::Const { at, .. } | Slot::State { at, .. } => *at,
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Slot::Owned(b) => b.len(),
            Slot::Const { len, .. } | Slot::State { len, .. } => *len,
        }
    }
}


/// Quantized-activation slots in the transient ring, at the front of a pass.
pub(super) const ACT_RING: usize = 4;

/// Slots in the shared ring, after the transient one.
const ACT_SHARED: usize = 2;

/// The first slot a pass hands out exclusively.
const ACT_EXCLUSIVE: usize = ACT_RING + ACT_SHARED;

impl DeviceBackend {
    /// Uploads a constant into the arena. It is never released or read back.
    pub(super) fn store_const(&self, data: &[f32]) -> Result<Buf> {
        if !self.arena_const {
            return self.upload(data);
        }
        let at = self.hot.upload(data)?;
        Ok(self.store_slot(Slot::Const {
            at,
            len: data.len(),
        }))
    }

    /// A zeroed buffer in the state arena that lives as long as the
    /// sequence. See `arena.rs` for why it is not its own allocation.
    pub(super) fn zeroed_state_buf(&self, len: usize) -> Result<Buf> {
        let at = self.state_arena.upload(&vec![0.0f32; len])?;
        self.state_live.set(self.state_live.get() + 1);
        Ok(self.store_slot(Slot::State { at, len }))
    }

    pub(super) fn store(&self, buffer: DeviceBuffer<f32>) -> Buf {
        self.store_slot(Slot::Owned(buffer))
    }

    fn store_slot(&self, slot_value: Slot) -> Buf {
        if let Some(slot) = self.free_slots.borrow_mut().pop() {
            self.slots.borrow_mut()[slot] = Some(slot_value);
            return Buf(slot);
        }
        let mut slots = self.slots.borrow_mut();
        slots.push(Some(slot_value));
        Buf(slots.len() - 1)
    }

    /// Frees a buffer to the driver instead of returning it to the pool.
    /// Only safe where [`Pool::trim`] is. See `residency.rs`.
    pub(super) fn discard(&self, buf: Buf) {
        let taken = self
            .slots
            .borrow_mut()
            .get_mut(buf.0)
            .and_then(Option::take);
        if let Some(slot) = taken {
            if let Slot::Owned(buffer) = slot {
                self.pool.forget(buffer);
            }
            self.free_slots.borrow_mut().push(buf.0);
        }
    }

    /// The device pointer behind a handle, offset by `elements`.
    pub(super) fn ptr(&self, buf: Buf, elements: usize) -> Result<u64> {
        let slots = self.slots.borrow();
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?;
        ensure!(
            elements <= buffer.len(),
            "offset {elements} is past the buffer"
        );
        Ok(buffer.base() + (elements * size_of::<f32>()) as u64)
    }

    /// The same for f16 storage, with the offset counted in halves.
    ///
    /// An [`HBuf`] lives in the same slot table as a [`Buf`], backed by half
    /// as many f32 words, so the pool serves both. Only this file treats the
    /// two handle types as interchangeable.
    pub(super) fn hptr(&self, buf: HBuf, elements: usize) -> Result<u64> {
        let slots = self.slots.borrow();
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?;
        ensure!(
            elements <= 2 * buffer.len(),
            "offset {elements} is past the buffer"
        );
        Ok(buffer.base() + (elements * size_of::<u16>()) as u64)
    }

    /// The f32 words behind f16 storage. Only valid for an even-length run,
    /// as [`Backend::copy_h`] guarantees. See [`DeviceBackend::hptr`].
    pub(super) fn words(buf: HBuf) -> Buf {
        Buf(buf.0)
    }

    pub(super) fn len_of(&self, buf: Buf) -> Result<usize> {
        let slots = self.slots.borrow();
        Ok(slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?
            .len())
    }

    /// Panics if the destination is one of the sources, which no kernel
    /// tolerates. Only checked under `PHOBOS_CHECK_BUFS`.
    pub(super) fn check_distinct(&self, what: &str, dst: Buf, sources: &[Buf]) {
        // Cached: nearly every launch calls this, and reading the
        // environment takes a process-wide lock.
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| phobos_base::env::flag("PHOBOS_CHECK_BUFS")) {
            return;
        }
        for &src in sources {
            assert_ne!(dst.0, src.0, "{what}: destination aliases a source");
        }
    }

    /// Scratch for `rows` split-attention partials of `head_dim` each, plus
    /// their maxima and sums.
    pub(super) fn attn_scratch(&self, rows: usize, head_dim: usize) -> Result<(u64, u64)> {
        let too_small = self
            .attn_partials
            .borrow()
            .as_ref()
            .is_none_or(|(p, ml)| p.len() < rows * head_dim || ml.len() < rows * 2);
        if too_small {
            self.flush_pending()?;
            // SAFETY: the split kernel writes every row before the merge reads it.
            let grown = unsafe {
                (
                    DeviceBuffer::uninitialized(rows * head_dim)?,
                    DeviceBuffer::uninitialized(rows * 2)?,
                )
            };
            *self.attn_partials.borrow_mut() = Some(grown);
        }
        let scratch = self.attn_partials.borrow();
        let (p, ml) = scratch.as_ref().expect("filled above");
        Ok((p.as_device_ptr().as_raw(), ml.as_device_ptr().as_raw()))
    }

    /// The row count and width of the reshaped normalization tile.
    pub(super) fn norm_shape(&self, width: usize) -> Result<(i64, i64)> {
        ensure!(
            width.is_multiple_of(RMS_LANE),
            "a normalization needs a width ({width}) that is a multiple of {RMS_LANE}"
        );
        Ok(((width / RMS_LANE) as i64, RMS_LANE as i64))
    }

    /// [`Backend::quantize_act`] into a slot the caller already picked.
    pub(super) fn quantize_act_into(
        &self,
        slot: (QAct, u64, u64),
        a: Buf,
        m: usize,
        k: usize,
    ) -> Result<QAct> {
        let (act, qa_ptr, das_ptr) = slot;
        let blocks = k / Q8_BLOCK;
        let a_ptr = self.ptr(a, 0)?;
        let rows = m * blocks;
        let (module, tile) = if rows * Q8_BLOCK >= WIDE_FLOOR {
            (&self.quantize_wide, QUANT_TB_WIDE)
        } else {
            (&self.quantize, QUANT_TB)
        };
        self.launch(
            module,
            "quantize",
            &[
                (a_ptr, [rows as i64, Q8_BLOCK as i64]),
                (qa_ptr, [rows as i64, Q8_BLOCK as i64]),
                (das_ptr, [rows as i64, 1]),
            ],
            (rows.div_ceil(tile) as u32, 1, 1),
        )?;
        Ok(act)
    }

    /// [`Self::quantize_act_into`] with a ring slot, for an activation read
    /// only by the projection that asked for it.
    pub(super) fn quantize_act_transient(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        self.quantize_act_into(self.act_slot_transient(m, k)?, a, m, k)
    }

    /// [`Self::act_slot`] for an activation read once, by the projection that
    /// asked for it, and never again.
    ///
    /// Normally each `quantize_act` gets a fresh slot, because a caller such
    /// as `rms_norm_q` may hold the handle across several projections. A
    /// single-use activation can instead reuse a small ring of slots. The
    /// recorded pass runs in issue order, so a slot is never overwritten
    /// before its reader runs.
    pub(super) fn act_slot_transient(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        let at = self.act_ring.get();
        self.act_ring.set((at + 1) % ACT_RING);
        self.act_next.set(self.act_next.get().max(ACT_EXCLUSIVE));
        self.act_slot_at(at, m, k)
    }

    /// [`Self::act_slot`] for the Hadamard transform's quantized copy.
    ///
    /// Only the projections sharing that input read it, and it is dead
    /// before the next one is made. At most one is live, so a pair of slots
    /// serves the whole pass.
    pub(super) fn act_slot_shared(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        let at = self.act_shared.get();
        self.act_shared.set(ACT_RING + (at + 1 - ACT_RING) % ACT_SHARED);
        self.act_next.set(self.act_next.get().max(ACT_EXCLUSIVE));
        self.act_slot_at(at, m, k)
    }

    /// This pass's next quantized-activation slot, big enough for `m` rows
    /// of `k`, with its device pointers.
    pub(super) fn act_slot(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        // The two rings reserve the first slots, so exclusive ones start
        // after them.
        let at = self.act_next.get().max(ACT_EXCLUSIVE);
        self.act_next.set(at + 1);
        self.act_slot_at(at, m, k)
    }

    /// The slot at `at`, grown to `m` by `k` if it is not already that big.
    fn act_slot_at(&self, at: usize, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "a quantized activation needs k ({k}) to be a multiple of {Q8_BLOCK}"
        );
        let blocks = k / Q8_BLOCK;
        let too_small = self
            .act_scratch
            .borrow()
            .get(at)
            .is_none_or(|(q, s)| q.len() < m * k || s.len() < m * blocks);
        if too_small {
            // Growing frees a buffer recorded launches point at, so flush
            // them first.
            self.flush_pending()?;
            // SAFETY: whatever fills the slot writes every element before the
            // projection reads it.
            let grown = unsafe {
                (
                    DeviceBuffer::uninitialized(m * k)?,
                    DeviceBuffer::uninitialized(m * blocks)?,
                )
            };
            let mut scratch = self.act_scratch.borrow_mut();
            while scratch.len() < at {
                // SAFETY: as above. A gap slot is grown by the `too_small`
                // path before anything reads it.
                scratch.push(unsafe {
                    (
                        DeviceBuffer::uninitialized(1)?,
                        DeviceBuffer::uninitialized(1)?,
                    )
                });
            }
            if at == scratch.len() {
                scratch.push(grown);
            } else {
                scratch[at] = grown;
            }
        }
        let (q, s) = self.act_ptrs(QAct(at))?;
        Ok((QAct(at), q, s))
    }

    /// Device pointers to the quantized activation behind a handle.
    pub(super) fn act_ptrs(&self, act: QAct) -> Result<(u64, u64)> {
        let scratch = self.act_scratch.borrow();
        let (q, s) = scratch
            .get(act.0)
            .context("use of an unknown quantized activation handle")?;
        Ok((q.as_device_ptr().as_raw(), s.as_device_ptr().as_raw()))
    }

    /// Grows the scratch a plan wants for the values it found crossing a nest.
    pub(super) fn grow_scratch(&self, need: &[Scratch]) -> Result<()> {
        let fits = |pool: &[(DeviceBuffer<i8>, DeviceBuffer<f32>)], at: usize, n: &Scratch| {
            pool.get(at)
                .is_some_and(|(b, s)| b.len() >= n.bytes && s.len() >= n.scales)
        };
        if need
            .iter()
            .enumerate()
            .all(|(at, n)| fits(&self.fused_scratch.borrow(), at, n))
        {
            return Ok(());
        }

        // Growing frees buffers recorded launches point at, so flush them
        // first.
        self.flush_pending()?;
        let mut pool = self.fused_scratch.borrow_mut();
        for (at, n) in need.iter().enumerate() {
            if fits(&pool, at, n) {
                continue;
            }
            // SAFETY: the plan's nest and barrier analysis guarantees every
            // byte is written by a stage before any stage reads it.
            let fresh = unsafe {
                (
                    DeviceBuffer::uninitialized(n.bytes)?,
                    DeviceBuffer::uninitialized(n.scales)?,
                )
            };
            match pool.get_mut(at) {
                Some(slot) => *slot = fresh,
                None => pool.push(fresh),
            }
        }
        Ok(())
    }

    /// An allocation that is written immediately rather than by a recorded
    /// launch.
    ///
    /// While a pass is recording, a pooled buffer may still be read by a
    /// pending launch, so this takes a fresh one instead. See
    /// [`Pool::take_fresh`]. Used for constants that grow mid-pass, like the
    /// rotary table, and for zero fills.
    #[track_caller]
    pub(super) fn alloc_written_now(&self, len: usize) -> Result<Buf> {
        if !self.recording.get() {
            return self.alloc(len);
        }
        Ok(self.store(self.pool.take_fresh(len)?))
    }

    /// A device pointer to an `n` by `n` identity, uploaded once.
    pub(super) fn identity(&self, n: usize) -> Result<u64> {
        if !self.identities.borrow().contains_key(&n) {
            let mut host = vec![0.0f32; n * n];
            for i in 0..n {
                host[i * n + i] = 1.0;
            }
            let buf = DeviceBuffer::from_slice(&host)?;
            self.identities.borrow_mut().insert(n, buf);
        }
        let identities = self.identities.borrow();
        Ok(identities[&n].as_device_ptr().as_raw())
    }

    /// A device pointer to scratch big enough for `len` split-K partial sums.
    pub(super) fn split_partials(&self, len: usize) -> Result<u64> {
        if self
            .split_scratch
            .borrow()
            .as_ref()
            .is_none_or(|p| p.len() < len)
        {
            self.flush_pending()?;
            // SAFETY: q8_split writes every row before q8_reduce reads it.
            let grown = unsafe { DeviceBuffer::uninitialized(len)? };
            *self.split_scratch.borrow_mut() = Some(grown);
        }
        let scratch = self.split_scratch.borrow();
        Ok(scratch
            .as_ref()
            .expect("just filled")
            .as_device_ptr()
            .as_raw())
    }
}
