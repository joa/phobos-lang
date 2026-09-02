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

use anyhow::{Context, Result, bail, ensure};

use super::*;

/// The staged projections (`<fmt>_qgemm_t`), built on first use;
/// `PHOBOS_QGEMM=0` turns them off.
pub(super) struct Qgemm {
    on: bool,
    kernels: RefCell<HashMap<Quant, Module>>,
    /// The ternary grid at two bits a lane and at a nibble a lane.
    grid2: RefCell<Option<DeviceBuffer<i8>>>,
    grid4: RefCell<Option<DeviceBuffer<i8>>>,
}

impl Qgemm {
    pub(super) fn from_env() -> Qgemm {
        Qgemm {
            on: env_flag_on("PHOBOS_QGEMM"),
            kernels: RefCell::new(HashMap::new()),
            grid2: RefCell::new(None),
            grid4: RefCell::new(None),
        }
    }

    /// Whether the staged kernel takes this shape: a format it decodes, its
    /// tile both ways, and `k` a whole number of blocks.
    fn takes(&self, quant: Quant, m: usize, k: usize, n: usize) -> bool {
        self.on
            && qgemm_name(quant).is_some()
            && m.is_multiple_of(QGEMM_TM)
            && n.is_multiple_of(QGEMM_TN)
            && k.is_multiple_of(256)
    }

    fn module(&self, quant: Quant) -> Result<std::cell::Ref<'_, Module>> {
        if !self.kernels.borrow().contains_key(&quant) {
            let src = qgemm_src(quant).context("no staged kernel for this format")?;
            let name = qgemm_kernel(quant).expect("the source exists, so does the name");
            let module = compile(&src, &[("TM", QGEMM_TM), ("TN", QGEMM_TN)], name)?;
            self.kernels.borrow_mut().insert(quant, module);
        }
        Ok(std::cell::Ref::map(self.kernels.borrow(), |m| &m[&quant]))
    }

    pub(super) fn grid4(&self) -> Result<u64> {
        if self.grid4.borrow().is_none() {
            *self.grid4.borrow_mut() = Some(DeviceBuffer::from_slice(&crate::quant::iq1s_grid4())?);
        }
        Ok(self
            .grid4
            .borrow()
            .as_ref()
            .expect("filled just above")
            .as_device_ptr()
            .as_raw())
    }

    pub(super) fn grid2(&self) -> Result<u64> {
        if self.grid2.borrow().is_none() {
            *self.grid2.borrow_mut() = Some(DeviceBuffer::from_slice(&crate::quant::iq1s_grid2())?);
        }
        Ok(self
            .grid2
            .borrow()
            .as_ref()
            .expect("filled just above")
            .as_device_ptr()
            .as_raw())
    }
}

impl DeviceBackend {
    /// Whether `w` at this shape can take the fused projection.
    ///
    /// The kernel is `@aligned`, so every slice it takes has to be whole: the
    /// output tile both ways, and `k` a whole number of 256-element blocks
    /// because a lane indexes the block bytes itself. A shape that misses goes
    /// back to `project_raw_dense`, which masks.
    pub(super) fn raw_qmma_eligible(&self, w: RawBuf, m: usize, k: usize, n: usize) -> bool {
        const FUSED: [Quant; 6] = [
            Quant::IQ1_S,
            Quant::IQ2_XXS,
            Quant::IQ2_S,
            Quant::IQ2_XS,
            Quant::IQ3_XXS,
            Quant::IQ3_S,
        ];
        // IQ1_M has only the staged kernel; the rest have both.
        if self.raw_quant_is(w, Quant::IQ1_M) {
            return self.qgemm.takes(Quant::IQ1_M, m, k, n);
        }
        FUSED
            .iter()
            .filter(|&&q| self.raw_qmma_formats.contains(&q))
            .any(|&q| self.raw_quant_is(w, q))
            && {
                let (tm, tn, _) = qmma_tile();
                m.is_multiple_of(tm) && n.is_multiple_of(tn)
            }
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
        let (qtm, qtn, _) = qmma_tile();
        let raws = self.raw_quants.borrow();
        let raw = raws
            .get(w.0)
            .context("use of an unknown raw weight handle")?;
        ensure!(
            raw.n == n,
            "raw weight was uploaded with n = {}, used with n = {n}",
            raw.n
        );
        let (nb, rb) = (raw.nb, raw.nb * raw.quant.device_block_bytes());
        let (bytes_ptr, d_ptr, quant) = (raw.bytes, raw.d, raw.quant);
        drop(raws);
        if self.qgemm.takes(quant, m, k, n) {
            let act = match act {
                Some(act) => act,
                None => {
                    self.note_dense_pass();
                    self.quantize_act_transient(a, m, k)?
                }
            };
            let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
            let mut operands = vec![
                (qa_ptr, [m as i64, k as i64]),
                (das_ptr, [m as i64, (k / Q8_BLOCK) as i64]),
                (bytes_ptr, [n as i64, rb as i64]),
                (d_ptr, [n as i64, nb as i64]),
            ];
            operands.extend(self.qgemm_tables(quant)?);
            operands.push((self.ptr(out, 0)?, [m as i64, n as i64]));
            let module = self.qgemm.module(quant)?;
            let name = qgemm_kernel(quant).expect("takes() checked the format");
            return self.launch(
                &module,
                name,
                &operands,
                ((m / QGEMM_TM) as u32, (n / QGEMM_TN) as u32, 1),
            );
        }
        let (module, name) = match quant {
            Quant::IQ1_S => (&self.iq1s_qmma, "iq1s_qmma"),
            Quant::IQ2_XXS => (&self.iq2xxs_qmma, "iq2xxs_qmma"),
            Quant::IQ2_S => (&self.iq2s_qmma, "iq2s_qmma"),
            Quant::IQ2_XS => (&self.iq2xs_qmma, "iq2xs_qmma"),
            Quant::IQ3_XXS => (&self.iq3xxs_qmma, "iq3xxs_qmma"),
            Quant::IQ3_S => (&self.iq3s_qmma, "iq3s_qmma"),
            other => bail!("project_raw_qmma has no fused kernel for {}", other.name()),
        };

        // The caller's copy where there is one: `rms_norm_q` already left the
        // rows quantized, and taking a slot per weight to redo it is what the
        // fused path costs a decode step. The slots it does take are a prompt
        // pass's and no use to a decode step either, hence `note_dense_pass`.
        let act = match act {
            Some(act) => act,
            None => {
                // Nothing outlives this one: the down projection is its only
                // reader, so it takes a ring slot rather than a slot of its
                // own. At 128 rows of a 17408-wide FFN a slot apiece is 2.2 MiB
                // a layer and 143 MiB across the model, which is most of what
                // the fused path costs a decode step.
                self.note_dense_pass();
                self.quantize_act_transient(a, m, k)?
            }
        };
        let (qa_ptr, das_ptr) = self.act_ptrs(act)?;
        let mut operands = vec![
            (qa_ptr, [m as i64, k as i64]),
            (das_ptr, [m as i64, (k / Q8_BLOCK) as i64]),
            (bytes_ptr, [n as i64, rb as i64]),
            (d_ptr, [n as i64, nb as i64]),
        ];
        // IQ1_S folds its delta and its sign into one table; IQ2_XXS keeps
        // magnitudes and signs apart, as its decode does everywhere else.
        match quant {
            Quant::IQ1_S => operands.push((
                self.iq1s_signed_grid.as_device_ptr().as_raw(),
                [1, IQ1S_SIGNED_GRID_LEN as i64],
            )),
            // IQ2_XS shares IQ2_XXS's sign table, as its decode does
            // everywhere else in this backend.
            other => {
                let (grid, glen, signs, slen) = match other {
                    Quant::IQ2_S => (
                        &self.iq2s_grid_packed,
                        IQ2S_GRID_LEN,
                        &self.iq2s_signs_packed,
                        IQ2S_SIGNS_LEN,
                    ),
                    Quant::IQ2_XS => (
                        &self.iq2xs_grid_packed,
                        IQ2XS_GRID_LEN,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    ),
                    Quant::IQ3_XXS => (
                        &self.iq3xxs_grid_packed,
                        IQ3XXS_GRID_LEN,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    ),
                    Quant::IQ3_S => (
                        &self.iq3s_grid_packed,
                        IQ3S_GRID_LEN,
                        &self.iq2s_signs_packed,
                        IQ2S_SIGNS_LEN,
                    ),
                    _ => (
                        &self.iq2xxs_grid_packed,
                        IQ2XXS_GRID_LEN,
                        &self.iq2xxs_signs_packed,
                        IQ2XXS_SIGNS_LEN,
                    ),
                };
                operands.push((grid.as_device_ptr().as_raw(), [1, glen as i64]));
                operands.push((signs.as_device_ptr().as_raw(), [1, slen as i64]));
            }
        }
        operands.push((self.ptr(out, 0)?, [m as i64, n as i64]));
        self.launch(
            module,
            name,
            &operands,
            ((m / qtm) as u32, (n / qtn) as u32, 1),
        )
    }

    /// The table operands a format's staged kernel reads, in declaration
    /// order: the packed magnitude grid, then the 0/-1 sign masks.
    fn qgemm_tables(&self, quant: Quant) -> Result<Vec<(u64, [i64; 2])>> {
        let lens = qgemm_tables(quant).context("no staged kernel for this format")?;
        let raw = |b: &DeviceBuffer<i8>| b.as_device_ptr().as_raw();
        let ptrs: Vec<u64> = match quant {
            Quant::IQ1_S | Quant::IQ1_M => vec![self.qgemm.grid2()?],
            Quant::IQ2_XXS => vec![
                raw(&self.iq2xxs_grid_packed),
                raw(&self.iq2xxs_signs_packed) + IQ2XXS_SIGNS_LEN as u64,
            ],
            Quant::IQ2_XS => vec![
                raw(&self.iq2xs_grid_packed),
                raw(&self.iq2xxs_signs_packed) + IQ2XXS_SIGNS_LEN as u64,
            ],
            Quant::IQ2_S => vec![
                raw(&self.iq2s_grid_packed),
                raw(&self.iq2s_signs_packed) + IQ2S_SIGNS_LEN as u64,
            ],
            Quant::IQ3_XXS => vec![
                raw(&self.iq3xxs_grid_packed),
                raw(&self.iq2xxs_signs_packed) + IQ2XXS_SIGNS_LEN as u64,
            ],
            Quant::IQ3_S => vec![
                raw(&self.iq3s_grid_packed),
                raw(&self.iq2s_signs_packed) + IQ2S_SIGNS_LEN as u64,
            ],
            other => bail!("no staged kernel for {}", other.name()),
        };
        Ok(ptrs
            .into_iter()
            .zip(lens)
            .map(|(p, &len)| (p, [1, len as i64]))
            .collect())
    }
}
