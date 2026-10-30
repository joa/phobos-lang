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

impl DeviceBackend {
    /// `dest += act . w` for a raw weight where a kernel can add in the
    /// projection: the split decode matvec for one row, the split prompt
    /// projection for more. `false` where neither takes the shape.
    pub(super) fn raw_act_add(&self, act: QAct, (m, k): (usize, usize), w: RawBuf, n: usize, dest: Buf) -> Result<bool> {
        if m == 1 {
            return self.project_raw_split_add(act, k, w, n, dest);
        }
        Ok(self.raw_qmma.get() && self.raw_qmma_eligible(w, m, k, n) && self.project_raw_qmma_add(act, (m, k), w, n, dest)?)
    }

    /// `dest[n] += w[n, k] . act` for one row as `s` slices of `k` into
    /// partials and a sum onto `dest`, so the residual needs no separate add;
    /// see [`kquant_lanes_split_src`]. Returns `false`, launching nothing,
    /// when the format has no split form or the shape does not take one.
    pub(super) fn project_raw_split_add(&self, act: QAct, k: usize, w: RawBuf, n: usize, dest: Buf) -> Result<bool> {
        let tn = Q4K_LANES_TN;
        if !self.iq1s_dp4a.get() || !self.qdot_split_on || !n.is_multiple_of(tn) {
            return Ok(false);
        }
        let Some(s) = qdot_splits(n / tn, k, self.sms) else {
            return Ok(false);
        };
        let raws = self.raw_quants.borrow();
        let raw = raws.get(w.0).context("use of an unknown raw weight handle")?;
        let (quant, nb, bytes, d) = (raw.quant, raw.nb, raw.bytes, raw.d);
        drop(raws);
        let key = (quant, n, k, s);
        if !self.qdot_split_mods.borrow().contains_key(&key) {
            let Some(src) = kquant_lanes_split_src(quant, n, k, s) else {
                return Ok(false);
            };
            let split = compile(&src, &[], kquant_lanes_split_name(quant))?;
            let reduce = compile(&qgemm_reduce_src(n, s, true), &[("TN", QGEMM_TN)], "qgemm_reduce")?;
            self.qdot_split_mods.borrow_mut().insert(key, (split, reduce));
        }
        let (qa, das) = self.act_ptrs(act)?;
        let mods = self.qdot_split_mods.borrow();
        let (split, reduce) = &mods[&key];
        let groups = (n / 8) as i64;
        let partials = self.split_partials(s * n)?;
        let plane = (partials, [1, (s * n) as i64]);
        let mut operands = vec![
            (qa, [1, k as i64]),
            (das, [1, (k / Q8_BLOCK) as i64]),
            (bytes, [groups, (nb * 8 * quant.device_block_bytes()) as i64]),
        ];
        // Q6_K's `d` is the grouped scale plane.
        if quant == Quant::Q6_K {
            operands.push((d, [groups, (nb * 8) as i64]));
        }
        operands.push(plane);
        let name = kquant_lanes_split_name(quant);
        self.launch(split, name, &operands, ((n / tn) as u32, s as u32, 1))?;
        let into = [plane, (self.ptr(dest, 0)?, [1, n as i64])];
        self.launch(reduce, "qgemm_reduce", &into, (1, (n / QGEMM_TN) as u32, 1))?;
        Ok(true)
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
