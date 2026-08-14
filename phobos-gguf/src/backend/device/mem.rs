// Device buffers, the pointers handed to kernels, and the scratch a
// pass grows into.

use super::*;

impl DeviceBackend {
    pub(super) fn store(&self, buffer: DeviceBuffer<f32>) -> Buf {
        if let Some(slot) = self.free_slots.borrow_mut().pop() {
            self.slots.borrow_mut()[slot] = Some(buffer);
            return Buf(slot);
        }
        let mut slots = self.slots.borrow_mut();
        slots.push(Some(buffer));
        Buf(slots.len() - 1)
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
        Ok(buffer.as_device_ptr().as_raw() + (elements * size_of::<f32>()) as u64)
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
        Ok(buffer.as_device_ptr().as_raw() + (elements * size_of::<u16>()) as u64)
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
    pub(super) fn act_slot(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "a quantized activation needs k ({k}) to be a multiple of {Q8_BLOCK}"
        );
        let blocks = k / Q8_BLOCK;
        let at = self.act_next.get();
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
            if at == scratch.len() {
                scratch.push(grown);
            } else {
                scratch[at] = grown;
            }
        }
        self.act_next.set(at + 1);
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
