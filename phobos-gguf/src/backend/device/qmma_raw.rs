//! A prompt pass's projection against a raw quantized weight, with no
//! expanded weight in between.
//!
//! [`super::DeviceBackend::project_raw_dense`] decodes a strip of the weight
//! into `f32` or `f16` scratch and runs a dense matmul over it. Measured on
//! Qwen3.8-27B that costs 872 ms of `_qdecode` and 529 ms of `matmul_tc` in a
//! 1582 ms pass, and the reason is not the decode: `iq1s_qdecode` and
//! `iq1s_qdot_matvec` are the same size instruction for instruction and move
//! DRAM at the same 67 GB/s. The expansion moves **11.2x the bytes** -- 23.8 GB
//! written and read back against the 2.3 GB IQ1_S actually is -- and leaves
//! 128 MiB of scratch behind on a card whose weights are 6.37 GiB of 8.
//!
//! Contracting out of the registers the decode already lands in pays neither.

use anyhow::{Context, Result, ensure};

use super::*;

impl DeviceBackend {
    /// Whether `w` at this shape can take the fused projection.
    ///
    /// The kernel is `@aligned`, so every slice it takes has to be whole: the
    /// output tile both ways, and `k` a whole number of 256-element blocks
    /// because a lane indexes the block bytes itself. A shape that misses goes
    /// back to `project_raw_dense`, which masks.
    pub(super) fn raw_qmma_eligible(&self, w: RawBuf, m: usize, k: usize, n: usize) -> bool {
        self.raw_quant_is(w, Quant::IQ1_S)
            && m.is_multiple_of(IQ1S_QMMA_TM)
            && n.is_multiple_of(IQ1S_QMMA_TN)
            && k.is_multiple_of(256)
    }

    /// `out[m, n] = a[m, k] . w[n, k]` with the IQ1_S decode folded into the
    /// integer tensor-core contraction.
    ///
    /// The activation is quantized once for the whole weight, as it is for
    /// [`Self::project_q8`]: a Q8_0 block is 32 elements and so is an IQ1_S
    /// scale group, which is what lets both scales land on the same k step.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_raw_qmma(
        &self,
        act: Option<QAct>,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let raws = self.raw_quants.borrow();
        let raw = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        ensure!(
            raw.n == n,
            "raw weight was uploaded with n = {}, used with n = {n}",
            raw.n
        );
        ensure!(
            raw.quant == Quant::IQ1_S,
            "project_raw_qmma is IQ1_S only, got {}",
            raw.quant.name()
        );
        let (nb, rb) = (raw.nb, raw.nb * raw.quant.device_block_bytes());
        let (bytes_ptr, d_ptr) = (raw.bytes, raw.d);
        drop(raws);

        // The caller's copy where there is one: `rms_norm_q` already left the
        // rows quantized, and taking a slot per weight to redo it is what the
        // fused path costs a decode step. The slots it does take are a prompt
        // pass's and no use to a decode step either, hence `note_dense_pass`.
        let act = match act {
            Some(act) => act,
            None => {
                self.note_dense_pass();
                self.quantize_act(a, m, k)?
            }
        };
        let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
        let operands = [
            (qa_ptr, [m as i64, k as i64]),
            (das_ptr, [m as i64, (k / Q8_BLOCK) as i64]),
            (bytes_ptr, [n as i64, rb as i64]),
            (d_ptr, [n as i64, nb as i64]),
            (
                self.iq1s_signed_grid.as_device_ptr().as_raw(),
                [1, IQ1S_SIGNED_GRID_LEN as i64],
            ),
            (self.ptr(out, 0)?, [m as i64, n as i64]),
        ];
        self.launch(
            &self.iq1s_qmma,
            "iq1s_qmma",
            &operands,
            ((m / IQ1S_QMMA_TM) as u32, (n / IQ1S_QMMA_TN) as u32, 1),
        )
    }
}
