// Split-K for the K-quant decode matvecs whose grid is too narrow for the
// card: a projection to the model width contracts a long `k` in a few dozen
// programs, each a long serial walk, while most multiprocessors idle.

use super::*;

/// Most slices a split takes.
const QDOT_SPLIT_MAX: usize = 8;

/// Shortest `k` worth splitting. A shorter contraction already keeps its
/// programs brief, and the partials would cost more than they save.
const QDOT_SPLIT_MIN_K: usize = 4096;

/// The slice count for a matvec `tiles` programs wide over `k`: the most
/// slices, up to [`QDOT_SPLIT_MAX`], that divide `k`'s 256-element blocks
/// and keep the grid within four programs a multiprocessor. `None` when the
/// plain grid already gives two programs a multiprocessor, or `k` is short.
pub(super) fn qdot_splits(tiles: usize, k: usize, sms: usize) -> Option<usize> {
    if tiles >= 2 * sms || k < QDOT_SPLIT_MIN_K || !k.is_multiple_of(256) {
        return None;
    }
    let blocks = k / 256;
    (2..=QDOT_SPLIT_MAX)
        .rev()
        .find(|&s| blocks.is_multiple_of(s) && tiles * s <= 4 * sms)
}

/// A split matvec's operands: the quantized activation and the raw weight.
pub(super) struct SplitOperands {
    pub(super) qa: u64,
    pub(super) das: u64,
    pub(super) bytes: u64,
    pub(super) d: u64,
}

impl DeviceBackend {
    /// `out[n] = w[n, k] . a` as `s` slices of `k` into partials and a sum,
    /// or with `add` `out[n] += ..`; see [`kquant_qdot_i8_split_src`].
    /// Returns `false`, launching nothing, when the format has no split form
    /// or the shape does not take one.
    pub(super) fn project_raw_split(
        &self,
        quant: Quant,
        ops: SplitOperands,
        (nb, k, n): (usize, usize, usize),
        (out, add): (Buf, bool),
    ) -> Result<bool> {
        let tn = Q4K_I8_TN;
        if !self.qdot_split_on || !n.is_multiple_of(tn) {
            return Ok(false);
        }
        let Some(s) = qdot_splits(n / tn, k, self.sms) else {
            return Ok(false);
        };
        let key = (quant, n, k, s, add);
        if !self.qdot_split_mods.borrow().contains_key(&key) {
            let Some(src) = kquant_qdot_i8_split_src(quant, n, k, s) else {
                return Ok(false);
            };
            let split = compile(&src, &[("TN", tn)], kquant_qdot_i8_split_name(quant))?;
            let reduce = compile(&qgemm_reduce_src(n, s, add), &[("TN", QGEMM_TN)], "qgemm_reduce")?;
            self.qdot_split_mods.borrow_mut().insert(key, (split, reduce));
        }
        let mods = self.qdot_split_mods.borrow();
        let (split, reduce) = &mods[&key];
        let rb = nb * quant.device_block_bytes();
        let partials = self.split_partials(s * n)?;
        let plane = (partials, [1, (s * n) as i64]);
        let operands = [
            (ops.qa, [1, k as i64]),
            (ops.das, [1, (k / Q8_BLOCK) as i64]),
            (ops.bytes, [n as i64, rb as i64]),
            (ops.d, [n as i64, nb as i64]),
            plane,
        ];
        let name = kquant_qdot_i8_split_name(quant);
        self.launch(split, name, &operands, ((n / tn) as u32, s as u32, 1))?;
        let into = [plane, (self.ptr(out, 0)?, [1, n as i64])];
        self.launch(reduce, "qgemm_reduce", &into, (1, (n / QGEMM_TN) as u32, 1))?;
        Ok(true)
    }

    /// `dest += act . w` for a raw weight where a kernel can add in the
    /// projection: the split decode matvec for one row, the split prompt
    /// projection for more. `false` where neither takes the shape.
    pub(super) fn raw_act_add(&self, act: QAct, (m, k): (usize, usize), w: RawBuf, n: usize, dest: Buf) -> Result<bool> {
        if m == 1 {
            return self.project_raw_split_add(act, k, w, n, dest);
        }
        Ok(self.raw_qmma.get() && self.raw_qmma_eligible(w, m, k, n) && self.project_raw_qmma_add(act, (m, k), w, n, dest)?)
    }

    /// `dest[n] += w[n, k] . act` for one row, split as
    /// [`Self::project_raw_split`] splits it, so the sum lands on `dest`
    /// with no separate add. `false` when it does not split.
    pub(super) fn project_raw_split_add(&self, act: QAct, k: usize, w: RawBuf, n: usize, dest: Buf) -> Result<bool> {
        if !self.iq1s_dp4a.get() {
            return Ok(false);
        }
        let raws = self.raw_quants.borrow();
        let raw = raws.get(w.0).context("use of an unknown raw weight handle")?;
        let (quant, nb, bytes, d) = (raw.quant, raw.nb, raw.bytes, raw.d);
        drop(raws);
        let (qa, das) = self.act_ptrs(act)?;
        let ops = SplitOperands { qa, das, bytes, d };
        self.project_raw_split(quant, ops, (nb, k, n), (dest, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_narrow_matvec_splits_until_the_card_is_covered() {
        // The 4B's down projection on 170 SMs: 40 tiles over 36 blocks.
        assert_eq!(qdot_splits(40, 9216, 170), Some(6));
        // Its output projection, 16 blocks of k.
        assert_eq!(qdot_splits(40, 4096, 170), Some(8));
        // A short k, and a grid already two programs a multiprocessor.
        assert_eq!(qdot_splits(40, 2560, 170), None);
        assert_eq!(qdot_splits(400, 9216, 170), None);
    }
}
