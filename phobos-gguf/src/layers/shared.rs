// One input read by several projections, made ready once for all of them.

use anyhow::{Context, Result, ensure};

use crate::backend::{Backend, Buf, HeadPerm, QAct};

use super::{Fold, Linear};

/// An input several projections read, from [`Linear::share`]: the
/// activation and its quantized copy, both after the Hadamard transform when
/// the weights are folded, so the transform and the quantization run once
/// rather than per weight.
#[derive(Clone, Copy)]
pub(crate) struct Shared {
    pub(super) x: Buf,
    pub(super) act: Option<QAct>,
    rotation: Option<(usize, Option<HeadPerm>)>,
}

impl Shared {
    /// The quantized copy, if it is of the row as it came in rather than of
    /// its transform.
    pub(crate) fn plain_act(&self) -> Option<QAct> {
        self.act.filter(|_| self.rotation.is_none())
    }

    /// Hands back the transformed copy, the only buffer this owns.
    pub(crate) fn release(self, backend: &dyn Backend) {
        if self.rotation.is_some() {
            backend.release(self.x);
        }
    }
}

impl Linear {
    /// This weight's sign vector, for a folded input weight.
    fn signs(&self, backend: &dyn Backend) -> Result<Option<Buf>> {
        let Some(Fold::Input { signs, .. }) = self.fold.as_deref() else {
            return Ok(None);
        };
        let key = format!("prism.hadamard.signs.{}", self.in_dim);
        Ok(Some(backend.constant(&key, signs)?))
    }

    /// The transform this weight's input goes through: its width and head
    /// regrouping, or `None` for a weight that reads its input as it is.
    pub(super) fn rotation(&self) -> Option<(usize, Option<HeadPerm>)> {
        match self.fold.as_deref() {
            Some(Fold::Input { perm, .. }) => Some((self.in_dim, *perm)),
            _ => None,
        }
    }

    /// `x`, and `act` its quantized copy where the caller has one, made ready
    /// once for this weight and every other one reading the same input
    /// through the same transform. See [`Linear::forward_shared`].
    pub(crate) fn share(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: Option<QAct>,
        rows: usize,
    ) -> Result<Shared> {
        let (Some(signs), Some((width, perm))) = (self.signs(backend)?, self.rotation()) else {
            return Ok(Shared { x, act, rotation: None });
        };
        let out = backend.alloc(rows * width)?;
        let act = match self.is_quantized() || self.is_raw() {
            true => backend.hadamard_q(x, rows, width, signs, perm, out).map(Some),
            false => backend.hadamard(x, rows, width, signs, perm, out).map(|()| None),
        }
        .with_context(|| format!("Hadamard transform ahead of '{}'", self.key))?;
        Ok(Shared { x: out, act, rotation: Some((width, perm)) })
    }

    /// The normalization of `x` into `normed` ahead of this weight and every
    /// other one reading the same row, shared as [`Linear::share`] does. A
    /// folded weight has it normalized, transformed and quantized in one
    /// operation; `normed` holds the plain row either way.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn share_norm(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        gain: Buf,
        eps: f32,
        normed: Buf,
    ) -> Result<Shared> {
        let width = self.in_dim;
        match (self.signs(backend)?, self.rotation()) {
            (Some(signs), Some((_, None))) => {
                let out = backend.alloc(rows * width)?;
                let act = backend
                    .rms_norm_hadamard_q(x, rows, width, gain, eps, signs, normed, out)
                    .with_context(|| format!("Hadamard transform ahead of '{}'", self.key))?;
                Ok(Shared { x: out, act: Some(act), rotation: Some((width, None)) })
            }
            (Some(_), _) => {
                backend.rms_norm(x, rows, width, gain, eps, normed)?;
                self.share(backend, normed, None, rows)
            }
            (None, _) => {
                let act = backend.rms_norm_q(x, rows, width, gain, eps, normed)?;
                Ok(Shared { x: normed, act: Some(act), rotation: None })
            }
        }
    }

    /// [`Linear::forward_act`] on an input [`Linear::share`] prepared.
    pub(crate) fn forward_shared(&self, backend: &dyn Backend, input: Shared, rows: usize) -> Result<Buf> {
        let out = backend.alloc(rows * self.out_dim)?;
        self.project_into_shared(backend, input, rows, out)?;
        Ok(out)
    }

    /// [`Linear::forward_shared`] into a destination the caller owns.
    pub(crate) fn project_into_shared(&self, backend: &dyn Backend, input: Shared, rows: usize, out: Buf) -> Result<()> {
        // An input rotated for another weight, or not at all, still projects
        // to fluent-looking output, so a mismatch is refused here.
        ensure!(
            input.rotation == self.rotation(),
            "'{}' reads its input through {:?}, but it was prepared for {:?}",
            self.key,
            self.rotation(),
            input.rotation
        );
        self.project_plain(backend, input.x, input.act, rows, out)
    }
}
