// The integer arithmetic of `arith`, at the width of its type, and the
// address arithmetic of a view.

use super::exec::Mem;
use super::program::Arith;
use crate::shape::{DYN, row_major_strides};

/// [`row_major_strides`] over sizes that may be unknown: a stride past an
/// unknown size is unknown.
pub(super) fn row_major(sizes: &[Option<i64>]) -> Vec<Option<i64>> {
    let shape: Vec<i64> = sizes.iter().map(|s| s.unwrap_or(DYN)).collect();
    row_major_strides(&shape).into_iter().map(|s| (s != DYN).then_some(s)).collect()
}

/// The element an index lands on, counted from the view's base.
pub(super) fn linear(m: &Mem, idx: &[Option<i64>]) -> Option<i64> {
    let mut at = m.offset?;
    for (i, s) in idx.iter().zip(&m.strides) {
        at += i.as_ref()? * s.as_ref()?;
    }
    Some(at)
}

/// The bytes a view can reach, from its base.
pub(super) fn extent(m: &Mem) -> Option<(i64, i64)> {
    let lo = m.base + m.offset? * m.elem;
    let mut last = 0;
    for (size, stride) in m.sizes.iter().zip(&m.strides) {
        last += (size.as_ref()? - 1).max(0) * stride.as_ref()?;
    }
    Some((lo, lo + (last + 1) * m.elem))
}

/// `x` as an unsigned `bits`-wide integer.
pub(super) fn unsigned_of(x: i64, bits: u32) -> u64 {
    if bits >= 64 {
        x as u64
    } else {
        (x as u64) & ((1u64 << bits) - 1)
    }
}

/// `x` cut to `bits` and sign-extended back.
pub(super) fn wrap(x: i64, bits: u32) -> i64 {
    if bits >= 64 {
        x
    } else {
        let shift = 64 - bits;
        (x << shift) >> shift
    }
}

pub(super) fn arith(a: Arith, x: i64, y: i64, bits: u32) -> Option<i64> {
    let (ux, uy) = (unsigned_of(x, bits), unsigned_of(y, bits));
    let r = match a {
        Arith::Add => x.wrapping_add(y),
        Arith::Sub => x.wrapping_sub(y),
        Arith::Mul => x.wrapping_mul(y),
        Arith::DivU => ux.checked_div(uy)? as i64,
        Arith::RemU => ux.checked_rem(uy)? as i64,
        Arith::DivS => x.checked_div(y)?,
        Arith::RemS => x.checked_rem(y)?,
        Arith::And => x & y,
        Arith::Or => x | y,
        Arith::Xor => x ^ y,
        Arith::Shl => x.checked_shl(u32::try_from(y).ok()?)?,
        Arith::ShrU => ux.checked_shr(u32::try_from(y).ok()?)? as i64,
        Arith::ShrS => x.checked_shr(u32::try_from(y).ok()?)?,
        Arith::MinS => x.min(y),
        Arith::MaxS => x.max(y),
        Arith::MinU => ux.min(uy) as i64,
        Arith::MaxU => ux.max(uy) as i64,
    };
    Some(wrap(r, bits))
}

/// `arith.cmpi`'s predicates, by number: eq, ne, slt, sle, sgt, sge, ult,
/// ule, ugt, uge.
pub(super) fn compare(pred: i64, x: i64, y: i64, bits: u32) -> Option<bool> {
    let (ux, uy) = (unsigned_of(x, bits), unsigned_of(y, bits));
    Some(match pred {
        0 => x == y,
        1 => x != y,
        2 => x < y,
        3 => x <= y,
        4 => x > y,
        5 => x >= y,
        6 => ux < uy,
        7 => ux <= uy,
        8 => ux > uy,
        9 => ux >= uy,
        _ => return None,
    })
}
