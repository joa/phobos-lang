// Device buffers, the pointers handed to kernels, and the scratch a
// pass grows into.

use super::*;

/// What a [`Buf`] handle points at: scratch this backend owns and can hand
/// back to the pool, or a constant living in the arena.
///
/// A constant is uploaded once and never released, so it does not need its own
/// allocation, and giving it one costs residency: see `arena.rs`. 356 of the
/// 488 live handles on Qwen3.8-27B are constants.
pub(super) enum Slot {
    Owned(DeviceBuffer<f32>),
    Const {
        at: u64,
        len: usize,
    },
    /// A sequence's recurrent state: arena-backed like a constant, but
    /// released when the sequence ends, so the arena is counted and reset.
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

impl DeviceBackend {
    /// A handle onto `data` in the arena, for a constant the model never
    /// releases and never reads back.
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

    /// A zeroed handle onto the state arena, for a buffer that lives as long as
    /// the sequence does. 48 of these, three megabytes each, and giving each
    /// its own allocation is what leaves `delta_rule` reading its state at
    /// 2 ms a launch where a resident one takes 15 us: see `arena.rs`.
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

    /// Give a buffer back to the driver rather than to the pool, so the memory
    /// becomes the weights' again instead of staying reserved for a shape that
    /// only a prompt pass asks for. Only safe where [`Pool::trim`] is: see
    /// `residency.rs`.
    pub(super) fn discard(&self, buf: Buf) {
        let taken = self
            .slots
            .borrow_mut()
            .get_mut(buf.0)
            .and_then(Option::take);
        if taken.is_some() {
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

    /// The same for f16 storage, whose offset is counted in halves.
    ///
    /// An [`HBuf`] is a slot of the one table, holding a buffer of half as many
    /// f32 words: the pool then serves both kinds, and a cache released at the
    /// end of a sequence comes back as ordinary scratch. Nothing else about the
    /// two handles is interchangeable, which is why they are separate types
    /// above; only this file knows they share a table.
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

    /// The f32 words behind f16 storage. Only safe for a whole even run, which
    /// [`Backend::copy_h`] promises; see [`DeviceBackend::hptr`].
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

    /// The kernels do not tolerate a destination sharing storage with a
    /// source, so fail loudly on one.
    pub(super) fn check_distinct(&self, what: &str, dst: Buf, sources: &[Buf]) {
        if std::env::var_os("PHOBOS_CHECK_BUFS").is_none() {
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

    /// This pass's next quantized-activation slot, big enough for `m` rows of
    /// `k`, with its device pointers.
    /// [`Backend::quantize_act`] into a slot the caller picked.
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

    /// [`Self::quantize_act_into`] with a slot from the ring, for an
    /// activation the projection that asked for it is the only reader of.
    pub(super) fn quantize_act_transient(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        self.quantize_act_into(self.act_slot_transient(m, k)?, a, m, k)
    }

    /// [`Self::act_slot`] for an activation nothing outlives: quantized, read
    /// by the one projection that asked for it, and never referred to again.
    ///
    /// A pass takes a fresh slot per `quantize_act` because a caller may hold
    /// a handle across many projections, which is what `rms_norm_q` does for
    /// QKV and for gate and up. The fused projection has one caller that does
    /// not -- the down projection, whose input is the SwiGLU output -- and at
    /// 128 rows of a 17408-wide FFN that is 2.2 MiB a layer, 143 MiB across a
    /// 64-layer model, which is most of what the fused path costs a decode
    /// step in residency.
    ///
    /// Those can share a handful of slots. A recorded pass is instantiated as
    /// a chain of kernel nodes, in issue order, so the quantize that overwrites
    /// a ring slot cannot run before the projection that read it: `RING` only
    /// has to exceed how many are live at once, which is one.
    pub(super) fn act_slot_transient(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        const RING: usize = 4;
        let at = self.act_ring.get();
        self.act_ring.set((at + 1) % RING);
        let base = self.act_next.get();
        self.act_next.set(base.max(RING));
        self.act_slot_at(at, m, k)
    }

    pub(super) fn act_slot(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        // The ring reserves the first `RING` slots of every pass, so an
        // exclusive one starts past them.
        let at = self.act_next.get().max(4);
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
            // Growing frees the buffer the recorded launches point at, so the
            // recording has to be spent first.
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
                // SAFETY: as above; a gap is only ever written before it is
                // read, by the `too_small` path that fills it.
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

        // Growing frees what the recorded launches point at, so the recording
        // has to be spent first.
        self.flush_pending()?;
        let mut pool = self.fused_scratch.borrow_mut();
        for (at, n) in need.iter().enumerate() {
            if fits(&pool, at, n) {
                continue;
            }
            // SAFETY: a plan only publishes a value whose every byte one of its
            // stages writes before any stage reads it, which is what the nest
            // and barrier analysis establishes.
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

    /// An allocation for something that writes it immediately rather than from
    /// the stream, so it must not come out of the pool mid-pass. This is the
    /// only place that knows both halves: the pool cannot see whether a pass is
    /// recording, and the recorder does not allocate. See [`Pool::take_fresh`].
    ///
    /// Two callers need it: a constant that grows mid-pass, as the rotary table
    /// does when a sequence passes its length, and a zero fill, whose memset
    /// goes straight to the stream while the pass around it is only recorded.
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
