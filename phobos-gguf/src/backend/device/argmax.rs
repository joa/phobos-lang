// Device-side argmax over the logits row, so greedy decode reads back one
// token id instead of the whole vocab.

use super::*;

impl DeviceBackend {
    /// [`Backend::argmax`]'s device override.
    ///
    /// Runs outside the recorded pass, after `end_pass` has replayed the
    /// graph that produced `buf`. It is two small eager launches,
    /// `argmax_reduce` then `argmax_finish`.
    pub(super) fn device_argmax(&self, buf: Buf, len: usize) -> Result<i64> {
        let w = argmax_chunk_width(len);
        let s = ARGMAX_SPLITS;

        if !self.argmax_iota.borrow().contains_key(&w) {
            let iota: Vec<f32> = (0..w as i64).map(|i| i as f32).collect();
            let uploaded = DeviceBuffer::from_slice(&iota)?;
            self.argmax_iota.borrow_mut().insert(w, uploaded);
        }

        {
            let mut scratch = self.argmax_scratch.borrow_mut();
            let short = scratch.as_ref().is_none_or(|(p, _)| p.len() < 2 * s);
            if short {
                let p = DeviceBuffer::from_slice(&vec![0.0f32; 2 * s])?;
                let out = DeviceBuffer::from_slice(&[0.0f32; 2])?;
                *scratch = Some((p, out));
            }
        }

        let a_ptr = self.ptr(buf, 0)?;
        let iotas = self.argmax_iota.borrow();
        let io_ptr = iotas[&w].as_device_ptr().as_raw();
        let scratch = self.argmax_scratch.borrow();
        let (p, out) = scratch.as_ref().expect("filled above");
        let (p_ptr, out_ptr) = (p.as_device_ptr().as_raw(), out.as_device_ptr().as_raw());

        self.with_kernel(
            &self.argmax_reduce,
            w,
            "argmax_reduce",
            || argmax_reduce_src(w),
            |module| {
                self.launch(
                    module,
                    "argmax_reduce",
                    &[
                        (a_ptr, [1, len as i64]),
                        (io_ptr, [1, w as i64]),
                        (p_ptr, [2, s as i64]),
                    ],
                    (s as u32, 1, 1),
                )
            },
        )?;

        self.launch(
            &self.argmax_finish,
            "argmax_finish",
            &[(p_ptr, [2, s as i64]), (out_ptr, [1, 2])],
            (1, 1, 1),
        )?;

        // Reads back two floats: the max and its index.
        self.stream.synchronize()?;
        let mut winner = [0.0f32; 2];
        out.copy_to(&mut winner)?;
        Ok(winner[1].round() as i64)
    }
}
