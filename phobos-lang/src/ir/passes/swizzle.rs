// Which shared tiles take an XOR-swizzled layout.
//
// A tile whose rows are a multiple of 128 bytes puts the same 16-byte
// granule of every row in the same banks, so the eight rows a tensor-core
// fragment load reads at once conflict eight ways. Permuting each row's
// granules by its row index spreads them, with no padding. A tile is
// swizzled when the scaled int8 dot reads it, which is what pays, and every
// access to it can apply the permutation: the dot's own loads, directly or
// through slices; `store` and `word` runs inside one granule; and copies
// into it, a pipelined stage's refills included.

use anyhow::{Result, bail};

use crate::ir::build::alloc_align_div;
use crate::ir::{AtOp, Ir, OpId, OpKind, Space, Swizzle, Type, ValueId};

/// Swizzles the tiles that pay and allow it. `mode` is the kernel's
/// `@swizzle`: None decides alone, Some(true) fails on a tile that would pay
/// but cannot be swizzled, Some(false) swizzles nothing.
pub(crate) fn decide(ir: &mut Ir, mode: Option<bool>) -> Result<()> {
    if mode == Some(false) {
        return Ok(());
    }
    let mut declined = Vec::new();
    let mut chosen = Vec::new();
    for op in ir.all_ops() {
        if !matches!(ir.kind(op), OpKind::Alloc | OpKind::Stage(_)) {
            continue;
        }
        let tile = ir.result(op);
        let Some(swizzle) = shape_swizzle(ir, tile) else { continue };
        let mut uses = Uses::default();
        walk(ir, tile, &mut uses);
        if !uses.fragments {
            continue;
        }
        match uses.blocker {
            None => chosen.push((op, tile, swizzle)),
            Some(blocker) => declined.push((tile, blocker)),
        }
    }
    if mode == Some(true) {
        if let Some((tile, first)) = declined.first() {
            let mut e = AtOp::unplaced(format!("`@swizzle` cannot swizzle {}: {}", describe(ir, *tile), first.why))
                .label("asserted here");
            for (tile, b) in &declined {
                e = e.with(Some(b.op), format!("{} {}", describe(ir, *tile), b.label));
            }
            return Err(e.into());
        }
        if chosen.is_empty() {
            bail!("`@swizzle`: no shared tile here is read as tensor-core fragments with rows a multiple of 128 bytes");
        }
    }
    for (op, tile, swizzle) in chosen {
        let Type::Tile(mut t) = ir.ty(tile).clone() else { continue };
        t.layout.swizzle = Some(swizzle);
        // A swizzle spreads the rows; a pad would only cost bytes.
        t.layout.row_stride = None;
        t.layout.align_div = alloc_align_div(&t.static_shape().unwrap_or_default());
        ir.set_ty(tile, Type::Tile(t));
        if let OpKind::Stage(stage) = ir.kind_mut(op) {
            stage.pad = false;
        }
    }
    Ok(())
}

/// The permutation for a rank-2 static shared tile whose rows hold a
/// multiple of eight 16-byte granules, as `alloc_tile_swizzled` lays it out.
fn shape_swizzle(ir: &Ir, tile: ValueId) -> Option<Swizzle> {
    let Type::Tile(t) = ir.ty(tile) else { return None };
    let shape = t.static_shape()?;
    let (&[_, cols], Some(bytes)) = (&shape[..], t.elem.bytes()) else { return None };
    // A wider element would split a copy's 16-byte chunk across granules.
    if t.space != Space::Shared || t.layout.swizzle.is_some() || bytes > 4 || cols * bytes % 128 != 0 {
        return None;
    }
    let granule = (16 / bytes).max(1);
    Some(Swizzle {
        bits: 3,
        shift: 0,
        elem_log: granule.trailing_zeros(),
    })
}

/// What a tile's uses allow: whether the scaled dot reads it, and the first
/// use that cannot apply a swizzle.
#[derive(Default)]
struct Uses {
    fragments: bool,
    blocker: Option<Blocker>,
}

/// A use that cannot apply a swizzle: why, and what to label it with.
struct Blocker {
    op: OpId,
    why: &'static str,
    label: &'static str,
}

fn walk(ir: &Ir, value: ValueId, uses: &mut Uses) {
    for u in ir.uses(value) {
        let (op, index) = (u.op, u.index);
        match ir.kind(op) {
            OpKind::FragDotScaled if index == 1 || index == 2 => uses.fragments = true,
            // Only the scaled dot reads through a slice, from the tile
            // itself at the slice's start.
            OpKind::Slice(_) if index == 0 => {
                let slice = ir.result(op);
                if ir.uses(slice).iter().all(|s| matches!(ir.kind(s.op), OpKind::FragDotScaled) && (s.index == 1 || s.index == 2)) {
                    uses.fragments |= !ir.uses(slice).is_empty();
                } else {
                    uses.blocker.get_or_insert(Blocker {
                        op,
                        why: "a slice of it is read by something other than the scaled int8 dot",
                        label: "is sliced here for something else",
                    });
                }
            }
            OpKind::LaneStore { len, align, dim, .. } | OpKind::LaneLoad { len, align, dim } if index == 0 => {
                if !run_in_granule(ir, value, *len, *align, *dim) {
                    uses.blocker.get_or_insert(Blocker {
                        op,
                        why: "a thread's run of it crosses a 16-byte granule",
                        label: "is read here in a run that crosses a granule",
                    });
                }
            }
            kind if kind.writes(ir.operands(op).len()).contains(&index) && is_copy(kind) => {}
            _ => {
                uses.blocker.get_or_insert(Blocker {
                    op,
                    why: "something reads it that cannot apply the permutation",
                    label: "is read here without the permutation",
                });
            }
        }
    }
}

/// Whether a copy writes the tile, which honors a swizzle.
fn is_copy(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Copy { .. })
}

/// Whether a run of `len` elements along `dim`, aligned to `align` bytes,
/// stays inside one 16-byte granule of its row.
fn run_in_granule(ir: &Ir, tile: ValueId, len: i64, align: i64, dim: usize) -> bool {
    let Type::Tile(t) = ir.ty(tile) else { return false };
    let bytes = t.elem.bytes().unwrap_or(16);
    let run = len * bytes;
    dim == 1 && (len == 1 || (run <= 16 && align >= run && 16 % run == 0))
}

/// A tile as an error message names it.
fn describe(ir: &Ir, tile: ValueId) -> String {
    match ir.name(tile) {
        Some(name) => format!("tile `{name}`"),
        None => format!("a staged {}", ir.ty(tile)),
    }
}
