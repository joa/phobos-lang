// `extent(x, d)`: the extent of axis `d` of `x`, so a function can size its
// loops and slices from the tiles and tensors it is handed instead of taking
// every dimension as a parameter.

use anyhow::{Result, bail};

use super::{Binding, Build, DYN};
use crate::ast::{Expr, ExprKind};
use crate::ir::ValueId;

pub(crate) const EXTENT: &str = "extent";

pub(crate) enum Extent {
    /// Known when the kernel builds.
    Static(i64),
    /// A dynamic axis `d` of the tensor or view `v`, read at run time.
    Dynamic(ValueId, usize),
}

impl Build {
    /// What `extent(args)` stands for. `x` is a name bound to a tile, a view
    /// or a tensor, a slice of one with a static shape, or an elementwise
    /// expression over such, whose shape is their broadcast.
    pub(crate) fn extent_of(&self, args: &[Expr]) -> Result<Extent> {
        let [x, d] = args else {
            bail!("`extent` takes a tile or a tensor and an axis, as `extent(t, 0)`");
        };
        let Some(d) = self.const_fold(d) else {
            bail!("`extent`'s axis must be a constant");
        };
        let (shape, src) = match &x.kind {
            ExprKind::Var(name) => match self.lookup(name) {
                Some(Binding::Tensor(v) | Binding::View(v) | Binding::Tile(v) | Binding::Regs(v)) => {
                    (self.shape(v), Some(v))
                }
                Some(Binding::Generated(g)) => (g.shape.to_vec(), None),
                Some(_) => bail!("`extent` takes a tile or a tensor, and `{name}` is a scalar"),
                None => bail!("`extent` of `{name}`, which is not defined here"),
            },
            ExprKind::Index { .. } => match self.slice_shape_of(x, true) {
                Some(shape) => (shape, None),
                None => bail!("`extent` takes a slice only of a static shape"),
            },
            _ => match self.fused_decl_type(x) {
                Some((_, shape)) => (shape, None),
                None => bail!("`extent` takes a tile, a tensor, a slice of one or an elementwise expression over them"),
            },
        };
        let rank = shape.len();
        let Some(&n) = usize::try_from(d).ok().and_then(|d| shape.get(d)) else {
            bail!("`extent` axis {d} of a rank-{rank} value");
        };
        match (n, src) {
            (DYN, Some(v)) => Ok(Extent::Dynamic(v, d as usize)),
            (DYN, None) => bail!("`extent` axis {d} is dynamic"),
            (n, _) => Ok(Extent::Static(n)),
        }
    }

    /// `extent(args)` folded, when it is known as the kernel builds.
    pub(crate) fn extent_fold(&self, args: &[Expr]) -> Option<i64> {
        match self.extent_of(args).ok()? {
            Extent::Static(n) => Some(n),
            Extent::Dynamic(..) => None,
        }
    }

    /// The largest known divisor of `extent(args)`.
    pub(crate) fn extent_div(&self, args: &[Expr]) -> Option<i64> {
        match self.extent_of(args).ok()? {
            Extent::Static(n) => Some(n),
            Extent::Dynamic(v, d) => Some(self.div_of(v, d)),
        }
    }

    pub(crate) fn emit_extent(&mut self, args: &[Expr]) -> Result<ValueId> {
        Ok(match self.extent_of(args)? {
            Extent::Static(n) => self.const_index(n),
            Extent::Dynamic(v, d) => self.dim_of(v, d),
        })
    }
}
