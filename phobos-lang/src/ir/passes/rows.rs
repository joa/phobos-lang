// Fragment-row columns. A `[rows, 1]` f32 tile that row reductions of f16
// fragments write, and that only fills, elementwise column updates, copies,
// accumulations, row scalings, fragment maps and stores out touch, as flash attention's
// running max and sum are, becomes registers in the lanes that hold those
// rows of the fragments (`RegLayout::MmaRows`). Each write yields a new value and each
// loop that writes one live before it carries it, so its reads need no
// shared memory and no barrier.
//
// The columns go together or not at all: an update reads several, so one
// that must stay a tile keeps every column it shares an update with.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::BinOp;
use crate::ir::{At, BlockId, DistType, ForInfo, Ir, OpId, OpKind, RegLayout, Scalar, Space, Tree, Type, ValueId};

pub(super) fn promote(ir: &mut Ir) {
    let Some((tiles, ty)) = find(ir) else { return };
    let mut p = Promote { tiles, ty, env: BTreeMap::new() };
    p.block(ir, ir.entry());
    // The allocations last, once nothing names the tiles.
    for &t in &p.tiles {
        if let Some(op) = ir.def_op(t)
            && matches!(ir.kind(op), OpKind::Alloc)
        {
            ir.erase_op(op);
        }
    }
}

/// The columns to promote and the register type they take, or None.
fn find(ir: &Ir) -> Option<(BTreeSet<ValueId>, Type)> {
    let mut tiles: BTreeSet<ValueId> = ir
        .all_ops()
        .into_iter()
        .filter(|&op| matches!(ir.kind(op), OpKind::Alloc | OpKind::FragReduce(_)))
        .map(|op| ir.result(op))
        .filter(|&v| column(ir, v))
        .collect();
    loop {
        let bad: Vec<ValueId> = tiles.iter().copied().filter(|&t| !uses_allowed(ir, t, &tiles)).collect();
        if bad.is_empty() {
            break;
        }
        for t in bad {
            tiles.remove(&t);
        }
    }
    if tiles.is_empty() {
        return None;
    }
    // Every fragment op reading or writing one shares one warp grid whose
    // warps own whole rows, so a lane holds the same rows in all of them.
    let mut grid = BTreeSet::new();
    for &t in &tiles {
        if let Some(op) = ir.def_op(t)
            && matches!(ir.kind(op), OpKind::FragReduce(_))
        {
            grid.insert(rows_grid(ir.ty(ir.operand(op, 0)))?);
        }
        for u in ir.uses(t) {
            match ir.kind(u.op) {
                OpKind::FragScale(_) => grid.insert(rows_grid(ir.ty(ir.operand(u.op, 0)))?),
                OpKind::FragMap(_) => grid.insert(rows_grid(ir.ty(ir.result(u.op)))?),
                _ => false,
            };
        }
    }
    let [(m, wm)] = grid.into_iter().collect::<Vec<_>>()[..] else { return None };
    if tiles.iter().any(|&t| ir.ty(t).shape().and_then(|s| s[0].fixed()) != Some(m)) {
        return None;
    }
    // Only loops may lie between a column and its uses, and every read
    // must see a value.
    let structured = tiles.iter().all(|&t| {
        ir.uses(t)
            .iter()
            .chain(ir.def_op(t).map(|op| crate::ir::Use { op, index: 0 }).as_ref())
            .all(|u| plain_loops(ir, ir.parent_block(u.op)))
    });
    if !structured || !defined(ir, ir.entry(), &tiles, &mut BTreeSet::new()) {
        return None;
    }
    let ty = Type::Dist(DistType { elem: Scalar::F32, shape: vec![m, 1], layout: RegLayout::MmaRows { wm } });
    Some((tiles, ty))
}

/// Whether `v` is a static `[rows, 1]` f32 shared tile.
fn column(ir: &Ir, v: ValueId) -> bool {
    matches!(ir.ty(v), Type::Tile(t)
        if t.space == Space::Shared && t.elem == Scalar::F32 && t.layout.row_stride.is_none()
            && t.static_shape().is_some_and(|s| s.len() == 2 && s[1] == 1))
}

/// `(rows, warps)` of f16 fragments whose warps own whole rows.
fn rows_grid(ty: &Type) -> Option<(i64, i64)> {
    let Type::Dist(d) = ty else { return None };
    let acc = d.mma_acc().filter(|a| !a.imma && a.wn == 1)?;
    Some((acc.m, acc.wm))
}

/// Whether every use of `t` is one the registers hold, with any other
/// column it reads or writes beside it in `tiles`.
fn uses_allowed(ir: &Ir, t: ValueId, tiles: &BTreeSet<ValueId>) -> bool {
    ir.uses(t).iter().all(|u| {
        let operands = ir.operands(u.op);
        let others_ok = || operands.iter().all(|&v| tiles.contains(&v) || matches!(ir.ty(v), Type::Scalar(_)));
        match ir.kind(u.op) {
            OpKind::Fill => u.index == 1,
            // A store of the column to a tile outside the group.
            OpKind::Copy { .. } if u.index == 0 && !tiles.contains(&operands[1]) => {
                matches!(ir.ty(operands[1]), Type::Tile(_))
            }
            OpKind::Fused(_) | OpKind::Copy { .. } | OpKind::Accumulate => others_ok(),
            OpKind::FragScale(_) => u.index == 1,
            OpKind::FragMap(_) => true,
            _ => false,
        }
    })
}

/// Whether every op enclosing `block` is a loop with one body.
fn plain_loops(ir: &Ir, block: BlockId) -> bool {
    ir.ancestors(block).iter().all(|&(op, _)| matches!(ir.kind(op), OpKind::For(info) if !info.ragged))
}

/// The columns `op` reads, among `tiles`.
fn reads(ir: &Ir, op: OpId, tiles: &BTreeSet<ValueId>) -> Vec<ValueId> {
    let operands = ir.operands(op);
    let read: &[ValueId] = match ir.kind(op) {
        OpKind::Fused(_) => operands.split_last().map_or(&[], |(_, leaves)| leaves),
        OpKind::Copy { .. } => &operands[..1],
        OpKind::Accumulate => operands,
        OpKind::FragScale(_) => &operands[1..],
        OpKind::FragMap(_) => operands,
        _ => &[],
    };
    read.iter().copied().filter(|v| tiles.contains(v)).collect()
}

/// The column `op` writes, among `tiles`.
fn writes(ir: &Ir, op: OpId, tiles: &BTreeSet<ValueId>) -> Option<ValueId> {
    let v = match ir.kind(op) {
        OpKind::Fill | OpKind::Copy { .. } | OpKind::Accumulate => ir.operands(op).get(1).copied(),
        OpKind::Fused(_) => ir.operands(op).last().copied(),
        OpKind::FragReduce(_) => Some(ir.result(op)),
        _ => None,
    }?;
    tiles.contains(&v).then_some(v)
}

/// Whether every read in `block` sees a value written before it, given the
/// columns `known` holds at its start. A column first written in a loop
/// counts as written after it only if it was before.
fn defined(ir: &Ir, block: BlockId, tiles: &BTreeSet<ValueId>, known: &mut BTreeSet<ValueId>) -> bool {
    for &op in ir.ops(block) {
        if !reads(ir, op, tiles).iter().all(|v| known.contains(v)) {
            return false;
        }
        if let Some(v) = writes(ir, op, tiles) {
            known.insert(v);
        }
        for &body in ir.blocks_of(op) {
            if !defined(ir, body, tiles, &mut known.clone()) {
                return false;
            }
        }
    }
    true
}

/// The columns written anywhere inside `op`'s blocks.
fn written_in(ir: &Ir, op: OpId, tiles: &BTreeSet<ValueId>, out: &mut BTreeSet<ValueId>) {
    for &body in ir.blocks_of(op) {
        for &inner in ir.ops(body) {
            out.extend(writes(ir, inner, tiles));
            written_in(ir, inner, tiles, out);
        }
    }
}

struct Promote {
    tiles: BTreeSet<ValueId>,
    ty: Type,
    /// Each column's value at the point the walk has reached.
    env: BTreeMap<ValueId, ValueId>,
}

impl Promote {
    fn value(&self, v: ValueId) -> ValueId {
        if self.tiles.contains(&v) { self.env[&v] } else { v }
    }

    /// A register op standing for `like`, before it.
    fn make(&self, ir: &mut Ir, like: OpId, kind: OpKind, operands: &[ValueId]) -> ValueId {
        let outer = ir.set_span(ir.span(like));
        let op = ir.create_op(At::Before(like), kind, operands, vec![self.ty.clone()], Vec::new());
        ir.set_span(outer);
        ir.result(op)
    }

    fn block(&mut self, ir: &mut Ir, block: BlockId) {
        for op in ir.ops(block).to_vec() {
            if !ir.is_alive(op) {
                continue;
            }
            let operands = ir.operands(op).to_vec();
            match ir.kind(op).clone() {
                OpKind::Fill if self.tiles.contains(&operands[1]) => {
                    let v = self.make(ir, op, OpKind::RegsFill, &operands[..1]);
                    self.env.insert(operands[1], v);
                    ir.erase_op(op);
                }
                OpKind::FragReduce(_) if self.tiles.contains(&ir.result(op)) => {
                    let v = ir.result(op);
                    ir.set_ty(v, self.ty.clone());
                    self.env.insert(v, v);
                }
                OpKind::Fused(tree) if self.tiles.contains(operands.last().expect("a destination")) => {
                    let (&dst, leaves) = operands.split_last().expect("a destination");
                    let leaves: Vec<ValueId> = leaves.iter().map(|&l| self.value(l)).collect();
                    let v = self.make(ir, op, OpKind::RegsFused(tree), &leaves);
                    self.env.insert(dst, v);
                    ir.erase_op(op);
                }
                OpKind::Accumulate if self.tiles.contains(&operands[1]) => {
                    let sum = Tree::Binary(BinOp::Add, Box::new(Tree::Leaf(0)), Box::new(Tree::Leaf(1)));
                    let leaves = [self.value(operands[1]), self.value(operands[0])];
                    let v = self.make(ir, op, OpKind::RegsFused(sum), &leaves);
                    self.env.insert(operands[1], v);
                    ir.erase_op(op);
                }
                OpKind::Copy { .. } if self.tiles.contains(&operands[0]) && !self.tiles.contains(&operands[1]) => {
                    let outer = ir.set_span(ir.span(op));
                    ir.create_op(At::Before(op), OpKind::RegsStore, &[self.env[&operands[0]], operands[1]], Vec::new(), Vec::new());
                    ir.set_span(outer);
                    ir.erase_op(op);
                }
                OpKind::Copy { .. } if self.tiles.contains(&operands[1]) => {
                    self.env.insert(operands[1], self.value(operands[0]));
                    ir.erase_op(op);
                }
                OpKind::FragScale(_) | OpKind::FragMap(_) => {
                    for (i, &v) in operands.iter().enumerate() {
                        if self.tiles.contains(&v) {
                            ir.set_operand(op, i, self.env[&v]);
                        }
                    }
                }
                OpKind::For(info) => self.carry(ir, op, info),
                _ => {}
            }
        }
    }

    /// Walks loop `old`, rebuilt to carry the columns it writes that hold a
    /// value before it.
    fn carry(&mut self, ir: &mut Ir, old: OpId, info: ForInfo) {
        let mut written = BTreeSet::new();
        written_in(ir, old, &self.tiles, &mut written);
        let carried: Vec<ValueId> = written.into_iter().filter(|t| self.env.contains_key(t)).collect();
        let body = ir.blocks_of(old)[0];
        if carried.is_empty() {
            let outer = self.env.clone();
            self.block(ir, body);
            self.env = outer;
            return;
        }

        let first = info.bound_operands();
        let operands = ir.operands(old).to_vec();
        let mut new_operands = operands[..first + info.carried].to_vec();
        new_operands.extend(carried.iter().map(|t| self.env[t]));
        new_operands.extend_from_slice(&operands[first + info.carried..]);

        let mut arg_types: Vec<Type> = ir.args(body).iter().map(|&a| ir.ty(a).clone()).collect();
        arg_types.extend(carried.iter().map(|_| self.ty.clone()));
        let block = ir.new_block(&arg_types);
        let (old_args, new_args) = (ir.args(body).to_vec(), ir.args(block).to_vec());
        for (&from, &to) in old_args.iter().zip(&new_args) {
            if let Some(name) = ir.name(from).map(str::to_owned) {
                ir.set_name(to, name);
            }
            ir.replace_all_uses(from, to);
        }
        let mut finals = Vec::new();
        for op in ir.ops(body).to_vec() {
            if ir.kind(op).is_terminator() {
                finals = ir.operands(op).to_vec();
                ir.erase_op(op);
            } else {
                ir.move_op(op, At::End(block));
            }
        }

        let outer = self.env.clone();
        for (&t, &arg) in carried.iter().zip(&new_args[old_args.len()..]) {
            self.env.insert(t, arg);
        }
        self.block(ir, block);
        finals.extend(carried.iter().map(|t| self.env[t]));
        let span = ir.set_span(ir.span(old));
        ir.create_op(At::End(block), OpKind::Yield, &finals, Vec::new(), Vec::new());

        let mut types: Vec<Type> = ir.results(old).iter().map(|&r| ir.ty(r).clone()).collect();
        types.extend(carried.iter().map(|_| self.ty.clone()));
        let kind = OpKind::For(ForInfo { carried: info.carried + carried.len(), ..info });
        let new = ir.create_op(At::Before(old), kind, &new_operands, types, vec![block]);
        ir.set_span(span);
        for (&from, &to) in ir.results(old).to_vec().iter().zip(&ir.results(new).to_vec()) {
            ir.replace_all_uses(from, to);
        }
        ir.erase_op(old);

        self.env = outer;
        let results = ir.results(new).to_vec();
        for (&t, &r) in carried.iter().zip(&results[info.carried..]) {
            self.env.insert(t, r);
        }
    }
}
