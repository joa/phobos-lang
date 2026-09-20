// The Hadamard transform ahead of a folded weight; the kernel is
// `kernels/hadamard.rs`.

use super::*;

/// The operands a [`HadamardForm`] adds to the transform's own.
pub(super) struct HadamardExtra {
    pub(super) form: HadamardForm,
    /// The gain and the plain normalized row, for [`HadamardForm::Normed`].
    pub(super) norm: Option<(Buf, Buf)>,
}

impl DeviceBackend {
    /// The transform, and the quantized copy of its output where `extra` asks
    /// for one.
    ///
    /// The width has to be a whole number of Hadamard blocks, and a `perm` has to
    /// regroup the row in whole tiles. Both are checked here rather than left to
    /// the kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hadamard_rows(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        signs: Buf,
        perm: Option<HeadPerm>,
        out: Buf,
        extra: HadamardExtra,
    ) -> Result<Option<QAct>> {
        let HadamardExtra { form, norm } = extra;
        self.check_distinct("hadamard", out, &[x, signs]);
        ensure!(
            width.is_multiple_of(HADAMARD_BLOCK),
            "a {width}-wide row is not a whole number of Hadamard blocks"
        );
        if let Some(p) = perm {
            ensure!(
                p.head_dim.is_multiple_of(HADAMARD_SIDE)
                    && HADAMARD_BLOCK.is_multiple_of(p.head_dim)
                    && p.head_dim * p.groups * p.repeat == width,
                "{p:?} does not regroup a {width}-wide row in whole tiles"
            );
        }
        ensure!(
            matches!(form, HadamardForm::Normed(_)) == norm.is_some()
                && (norm.is_none() || (perm.is_none() && width <= HADAMARD_NORM_MAX_WIDTH)),
            "a {width}-wide Hadamard transform cannot take {form:?}"
        );
        let h = self.constant("hadamard.h32", &hadamard_matrix())?;
        let side = HADAMARD_SIDE as i64;
        let tiles = (rows * width / HADAMARD_SIDE) as i64;
        let sign_rows = (width / HADAMARD_SIDE) as i64;
        let blocks = (rows * width / HADAMARD_BLOCK) as u32;
        let mut operands = vec![
            (self.ptr(x, 0)?, [tiles, side]),
            (self.ptr(signs, 0)?, [sign_rows, side]),
            (self.ptr(h, 0)?, [side, side]),
            (self.ptr(out, 0)?, [tiles, side]),
        ];
        if let Some((gain, normed)) = norm {
            self.check_distinct("hadamard", normed, &[x, gain, out]);
            operands.push((self.ptr(gain, 0)?, [sign_rows, side]));
            operands.push((self.ptr(normed, 0)?, [tiles, side]));
        }
        let act = match form {
            HadamardForm::Plain => None,
            _ => {
                let (act, qa_ptr, das_ptr) = self.act_slot_shared(rows, width)?;
                operands.push((qa_ptr, [tiles, side]));
                operands.push((das_ptr, [tiles, 1]));
                Some(act)
            }
        };
        self.with_kernel(
            &self.hadamards,
            (width, perm, form),
            form.kernel(),
            || hadamard_src(width, perm, form),
            |module| self.launch(module, form.kernel(), &operands, (blocks, 1, 1)),
        )?;
        Ok(act)
    }
}
