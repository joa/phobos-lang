// A region of a tensor or a shared buffer as the interpreter tracks it:
// offsets and sizes in the root's own coordinates, and where masking clips
// each dimension.

use std::collections::HashMap;

use super::model::Access;
use super::place::Placement;
use crate::ir::{Ir, TileType, Type, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Root {
    /// A tensor parameter, by its entry argument.
    Tensor(ValueId),
    /// A shared buffer, by the value that made it.
    Buffer(ValueId),
}

#[derive(Clone, Debug)]
pub(super) struct View {
    pub(super) root: Root,
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    /// The index each dimension stops at, from a bound mask or the
    /// tensor's extent.
    limits: Vec<Option<i64>>,
}

impl View {
    pub(super) fn tensor(arg: ValueId, dims: &[i64]) -> View {
        View {
            root: Root::Tensor(arg),
            offsets: vec![0; dims.len()],
            sizes: dims.to_vec(),
            limits: dims.iter().map(|&d| Some(d)).collect(),
        }
    }

    pub(super) fn buffer(value: ValueId, t: &TileType) -> View {
        let sizes = t
            .shape
            .iter()
            .map(|e| e.fixed().unwrap_or(0))
            .collect::<Vec<_>>();
        View {
            root: Root::Buffer(value),
            offsets: vec![0; sizes.len()],
            limits: vec![None; sizes.len()],
            sizes,
        }
    }

    /// The region `offsets :+ sizes` of this one, a dimension with a mask
    /// extent stopping there.
    pub(super) fn slice(&self, offsets: &[i64], sizes: &[i64], masks: &[Option<i64>]) -> View {
        let rank = self.offsets.len().min(offsets.len());
        let mut out = self.clone();
        for d in 0..rank {
            let base = self.offsets[d];
            out.offsets[d] = base + offsets[d];
            out.sizes[d] = sizes[d];
            let mask = masks.get(d).copied().flatten().map(|e| base + e);
            out.limits[d] = match (self.limits[d], mask) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        out
    }

    /// The single element at `slot` of the view's row-major order.
    pub(super) fn flat_point(&self, slot: i64) -> View {
        let mut at = vec![0; self.sizes.len()];
        let mut rest = slot;
        for d in (0..self.sizes.len()).rev() {
            let n = self.sizes[d].max(1);
            at[d] = rest % n;
            rest /= n;
        }
        self.slice(&at, &vec![1; at.len()], &vec![None; at.len()])
    }

    pub(super) fn access(
        &self,
        ir: &Ir,
        place: &Placement,
        tensors: &HashMap<ValueId, (String, Vec<i64>)>,
    ) -> Access {
        let region: Vec<[i64; 2]> = (0..self.offsets.len())
            .map(|d| {
                let lo = self.offsets[d];
                let hi = lo + self.sizes[d];
                let clipped = self.limits[d].map_or(hi, |l| hi.min(l)).max(lo);
                [lo, clipped]
            })
            .collect();
        let full: i64 = self.sizes.iter().product();
        let kept: i64 = region.iter().map(|[lo, hi]| hi - lo).product();
        let masked = (full > kept).then_some(full - kept);
        match self.root {
            Root::Tensor(arg) => Access {
                tensor: tensors.get(&arg).map(|(name, _)| name.clone()),
                buffer: None,
                region,
                bytes: None,
                masked_elems: masked,
            },
            Root::Buffer(value) => {
                let id = place
                    .of_value
                    .get(&value)
                    .and_then(|ids| ids.first().copied());
                let bytes = id.and_then(|id| {
                    let base = place.buffers[id].offset_bytes?;
                    let Type::Tile(t) = ir.ty(value) else {
                        return None;
                    };
                    let (first, last) = elem_span(t, &region)?;
                    let eb = t.elem.bytes()?;
                    Some([base + first * eb, base + (last + 1) * eb])
                });
                Access {
                    tensor: None,
                    buffer: id,
                    region,
                    bytes,
                    masked_elems: masked,
                }
            }
        }
    }
}

/// The first and last element a region touches in the buffer's physical
/// row-major order, padding included.
fn elem_span(t: &TileType, region: &[[i64; 2]]) -> Option<(i64, i64)> {
    let shape = t.static_shape()?;
    if region.iter().any(|[lo, hi]| hi <= lo) || shape.len() != region.len() {
        return None;
    }
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        let inner = if d + 1 == shape.len() - 1 {
            t.layout.row_stride.unwrap_or(shape[d + 1])
        } else {
            shape[d + 1]
        };
        strides[d] = strides[d + 1] * inner;
    }
    let first = region.iter().zip(&strides).map(|([lo, _], s)| lo * s).sum();
    let last = region
        .iter()
        .zip(&strides)
        .map(|([_, hi], s)| (hi - 1) * s)
        .sum();
    Some((first, last))
}

/// The smallest region holding all of `accesses`, which share one root.
pub(super) fn union<'a>(mut accesses: impl Iterator<Item = &'a Access>) -> Option<Access> {
    let mut out = accesses.next()?.clone();
    for a in accesses {
        for (r, s) in out.region.iter_mut().zip(&a.region) {
            r[0] = r[0].min(s[0]);
            r[1] = r[1].max(s[1]);
        }
        out.bytes = match (out.bytes, a.bytes) {
            (Some(x), Some(y)) => Some([x[0].min(y[0]), x[1].max(y[1])]),
            _ => None,
        };
        out.masked_elems = None;
    }
    Some(out)
}
