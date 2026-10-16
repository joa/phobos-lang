// Where a `@persistent` kernel's scratch tensors live.
//
// Held, in shared memory, when every write to it is one every block makes
// alike (`owner`'s `Every`): each block then has its own complete copy and
// reads only that, at any index, so the value never leaves the block. The
// held ones fit the static shared budget together, in declaration order.
// Everything else is published: device memory the launch supplies, a
// parameter after the declared ones, in declaration order, before the
// barrier word.

use std::collections::BTreeSet;

use super::owner::{Owner, Owners};
use crate::ir::build::alloc_align_div;
use crate::ir::{Ir, Layout, OpId, OpKind, Space, TileType, Type, ValueId};

/// What held scratch may take of a block's shared memory, the static limit.
const HELD_BUDGET_BYTES: i64 = 48 * 1024;

/// The scratch markers of `ir`, in declaration order.
pub(super) fn marks(ir: &Ir) -> Vec<OpId> {
    ir.all_ops().into_iter().filter(|&op| matches!(ir.kind(op), OpKind::Scratch)).collect()
}

/// The scratch every block writes alike, as far as the budget takes it, but
/// none `@global` names.
pub(super) fn held(ir: &Ir, owners: &Owners<'_>, global: &[String]) -> BTreeSet<ValueId> {
    let writes: Vec<_> = ir.all_ops().into_iter().flat_map(|op| owners.accesses(op)).filter(|a| a.write).collect();
    let mut out = BTreeSet::new();
    let mut bytes = 0;
    for mark in marks(ir) {
        let root = ir.result(mark);
        if ir.name(root).is_some_and(|n| global.iter().any(|g| g == n)) {
            continue;
        }
        let alike = writes.iter().filter(|a| a.root == root).all(|a| a.owner == Owner::Every);
        let size = tile_bytes(ir.ty(root));
        if alike && size.is_some_and(|s| bytes + s <= HELD_BUDGET_BYTES) {
            bytes += size.unwrap_or_default();
            out.insert(root);
        }
    }
    out
}

/// Fails when a scratch `@shared` names is not held, or is no scratch.
pub(super) fn check_shared(ir: &Ir, held: &BTreeSet<ValueId>, shared: &[String]) -> anyhow::Result<()> {
    for name in shared {
        let mark = marks(ir).into_iter().find(|&m| ir.name(ir.result(m)) == Some(name.as_str()));
        let Some(mark) = mark else {
            anyhow::bail!("`@shared({name})` names no scratch tensor of the kernel");
        };
        if !held.contains(&ir.result(mark)) {
            anyhow::bail!(
                "`@shared({name})`: `{name}` cannot live in shared memory, since some block writes part of it, or the held scratch would outgrow {HELD_BUDGET_BYTES} bytes"
            );
        }
    }
    Ok(())
}

fn tile_bytes(ty: &Type) -> Option<i64> {
    let Type::Tensor(t) = ty else { return None };
    let elems: Option<i64> = t.shape.iter().map(|e| e.fixed()).product();
    Some(elems? * t.elem.bytes()?)
}

/// Holds the scratch in `held` as shared tiles and publishes the rest.
pub(super) fn place(ir: &mut Ir, held: &BTreeSet<ValueId>) {
    let entry = ir.entry();
    let barrier = usize::from(ir.kernel.grid_barrier);
    for mark in marks(ir) {
        let value = ir.result(mark);
        if held.contains(&value) {
            hold(ir, mark);
            continue;
        }
        let at = ir.args(entry).len() - barrier;
        let arg = ir.insert_arg(entry, at, ir.ty(value).clone());
        if let Some(name) = ir.name(value).map(str::to_owned) {
            ir.set_name(arg, name);
        }
        ir.replace_all_uses(value, arg);
        ir.erase_op(mark);
    }
}

/// A scratch marker as the shared tile a `var t: tile<..>` declares, its
/// alignment promise dropped and its views retyped to match: a flat view as
/// the tile's, a slice in shared memory with the layout it had, which over
/// a static shape is the same either way.
fn hold(ir: &mut Ir, mark: OpId) {
    let value = ir.result(mark);
    let Type::Tensor(t) = ir.ty(value).clone() else { unreachable!("scratch is a tensor") };
    let shape: Vec<i64> = t.shape.iter().filter_map(|e| e.fixed()).collect();
    let tile = TileType {
        elem: t.elem,
        shape: t.shape.clone(),
        layout: Layout {
            row_stride: None,
            swizzle: None,
            align_div: alloc_align_div(&shape),
        },
        space: Space::Shared,
    };
    *ir.kind_mut(mark) = OpKind::Alloc;
    ir.set_ty(value, Type::Tile(tile.clone()));
    for u in ir.uses(value).to_vec() {
        if matches!(ir.kind(u.op), OpKind::AssumeAlign) {
            let r = ir.result(u.op);
            ir.set_ty(r, Type::Tile(tile.clone()));
            ir.replace_all_uses(r, value);
            ir.erase_op(u.op);
        }
    }
    retype_views(ir, value, tile.layout.align_div);
}

fn retype_views(ir: &mut Ir, v: ValueId, align_div: i64) {
    for u in ir.uses(v).to_vec() {
        let ty = match ir.kind(u.op) {
            OpKind::Flat => {
                let Type::Tensor(f) = ir.ty(ir.result(u.op)).clone() else { continue };
                Type::Tile(TileType {
                    elem: f.elem,
                    shape: f.shape,
                    layout: Layout {
                        row_stride: None,
                        swizzle: None,
                        align_div,
                    },
                    space: Space::Shared,
                })
            }
            OpKind::Slice(_) if u.index == 0 => {
                let Type::Tile(t) = ir.ty(ir.result(u.op)).clone() else { continue };
                Type::Tile(TileType { space: Space::Shared, ..t })
            }
            _ => continue,
        };
        let r = ir.result(u.op);
        ir.set_ty(r, ty);
        retype_views(ir, r, align_div);
    }
}
