// Grid barriers where a `@persistent` kernel needs them: `membar` one level
// up, over device memory, with blocks for threads.
//
// The kernel's top-level statements run in order on every block, so a grid
// barrier can only go between two of them, never inside a unit loop. An
// access conflicts with one an earlier statement made, since the last grid
// barrier, when another block may touch an element it does and one of the
// two writes it (`owner`). The barrier goes before the statement making the
// later access, and one barrier serves every conflict it orders. A loop
// every block runs alike is walked twice, so what one turn writes at its
// bottom and the next reads at its top is ordered too.

use std::collections::BTreeSet;

use super::owner::{Access, Owners, grid_uniform};
use crate::diag::Span;
use crate::ir::uniform::Uniformity;
use crate::ir::{At, BlockId, Extent, Ir, OpId, OpKind, Scalar, TensorType, Type, ValueId};

/// Where the barriers go: before each of these ops, with the access each
/// orders and the earlier one it conflicts with.
pub(super) struct Placement {
    pub before: Vec<(OpId, Access, Access)>,
}

/// Plans the barriers of `ir`, ignoring accesses to the buffers in
/// `private`: held scratch, a copy per block.
pub(super) fn plan(ir: &Ir, owners: &Owners<'_>, grid: &Uniformity, private: &BTreeSet<ValueId>) -> Placement {
    let mut walk = Walk { ir, owners, grid, private, before: Vec::new() };
    let mut pending = Vec::new();
    walk.block(ir.entry(), &mut pending);
    Placement { before: walk.before }
}

struct Walk<'a> {
    ir: &'a Ir,
    owners: &'a Owners<'a>,
    grid: &'a Uniformity,
    private: &'a BTreeSet<ValueId>,
    before: Vec<(OpId, Access, Access)>,
}

impl Walk<'_> {
    fn block(&mut self, block: BlockId, pending: &mut Vec<Access>) {
        for &op in self.ir.ops(block) {
            if matches!(self.ir.kind(op), OpKind::GridBarrier) {
                pending.clear();
                continue;
            }
            if self.is_layer_loop(op) {
                let body = self.ir.blocks_of(op)[0];
                // Twice: the second turn meets what the first left pending.
                for _ in 0..2 {
                    self.block(body, pending);
                }
                continue;
            }
            let made = self.accesses(op);
            let clash = made.iter().find_map(|a| pending.iter().find(|p| p.conflicts(a)).map(|p| (a.clone(), p.clone())));
            if let Some((later, earlier)) = clash {
                if !self.before.iter().any(|(at, ..)| *at == op) {
                    self.before.push((op, later, earlier));
                }
                pending.clear();
            }
            pending.extend(made);
        }
    }

    /// Every access `op` and the ops in its blocks make, but to private
    /// buffers.
    fn accesses(&self, op: OpId) -> Vec<Access> {
        let mut out = Vec::new();
        self.ir.walk_from(op, &mut |o| {
            out.extend(self.owners.accesses(o).into_iter().filter(|a| !self.private.contains(&a.root)));
        });
        out
    }

    /// A loop every block runs alike whose turns may need a barrier between
    /// them: its body is under grid-uniform control, and its variable hands
    /// out no units.
    fn is_layer_loop(&self, op: OpId) -> bool {
        let OpKind::For(_) = self.ir.kind(op) else { return false };
        let body = self.ir.blocks_of(op)[0];
        grid_uniform(self.grid, body) && !self.owners.hands_out_units(self.ir.args(body)[0])
    }
}

/// Inserts the barriers `placement` names, appending the barrier word when
/// the kernel takes none yet. Returns how many it inserted.
pub(super) fn insert(ir: &mut Ir, placement: &Placement, spans: &mut std::collections::BTreeMap<OpId, Span>) -> usize {
    if placement.before.is_empty() {
        return 0;
    }
    let entry = ir.entry();
    let word = if ir.kernel.grid_barrier {
        *ir.args(entry).last().expect("the barrier word is the last parameter")
    } else {
        let ty = Type::Tensor(TensorType {
            elem: Scalar::I32,
            shape: vec![Extent::Fixed(2), Extent::Fixed(1)],
            div: vec![1, 1],
        });
        let at = ir.args(entry).len();
        let word = ir.insert_arg(entry, at, ty);
        ir.set_name(word, "grid_barrier");
        ir.kernel.grid_barrier = true;
        word
    };
    for (op, ..) in &placement.before {
        // The barrier stands for the access it orders.
        let outer = ir.set_span(ir.span(*op));
        let made = ir.create_op(At::Before(*op), OpKind::GridBarrier, &[word], vec![Type::INDEX], Vec::new());
        ir.set_span(outer);
        if let Some(&span) = spans.get(op) {
            spans.insert(made, span);
        }
    }
    placement.before.len()
}
