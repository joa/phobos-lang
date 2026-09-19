// The RMS normalizations ahead of a projection.

use super::*;

impl DeviceBackend {
    /// [`Backend::rms_norm`].
    pub(super) fn rms_norm_rows(&self, x: Buf, rows: usize, width: usize, gain: Buf, eps: f32, out: Buf) -> Result<()> {
        self.check_distinct("rms_norm", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        self.with_kernel(
            &self.norms,
            (width, eps.to_bits()),
            "rms_norm",
            || rms_norm_src(width, eps, NormForm::Plain),
            |module| {
                self.launch(
                    module,
                    "rms_norm",
                    &[
                        (self.ptr(x, 0)?, [(rows as i64) * shape.0, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [(rows as i64) * shape.0, shape.1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
    }

    /// [`Backend::rms_norm_q`] into a slot the caller picked.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rms_norm_q_into(
        &self,
        slot: (QAct, u64, u64),
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<QAct> {
        self.check_distinct("rms_norm_q", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        let (act, qa_ptr, das_ptr) = slot;
        self.with_kernel(
            &self.quant_norms,
            (width, eps.to_bits()),
            "rms_norm_q",
            || rms_norm_src(width, eps, NormForm::Quantized),
            |module| {
                let rb = (rows as i64) * shape.0;
                self.launch(
                    module,
                    "rms_norm_q",
                    &[
                        (self.ptr(x, 0)?, [rb, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [rb, shape.1]),
                        (qa_ptr, [rb, shape.1]),
                        (das_ptr, [rb, 1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }
}
