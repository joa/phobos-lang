// Fragments read and rewritten element by element, on the f16 path. A lane
// holds, of each m16n8 D tile, rows `g` and `g + 8` at columns `2t` and
// `2t + 1`, with `g = lane / 4` and `t = lane % 4` (`for_each_dfrag`), so
// every element knows its row and column, and a row's columns lie in the
// four lanes of one quad.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use melior::dialect::{arith, memref, scf};
use melior::ir::{Block, BlockLike, Region, RegionLike, Type, Value};

use super::Lowered;
use super::regs::RegTile;
use crate::codegen::elemwise::Fused;
use crate::codegen::frag::row_slot;
use crate::codegen::{Codegen, FragAcc, MemVal, Reduce};
use crate::ir::{self, Ir, OpId, Tree};

impl<'c> Codegen<'c> {
    /// `frag_map`: the tree per element of the result's fragments.
    pub(super) fn frag_map(&mut self, block: &Block<'c>, ir: &Ir, op: OpId, tree: &Tree) -> Result<FragAcc<'c>> {
        let Some(acc) = (match ir.ty(ir.result(op)) {
            ir::Type::Dist(d) => d.mma_acc(),
            _ => None,
        }) else {
            bail!("frag_map of a non-fragment");
        };
        let mut fa = FragAcc {
            frags: Vec::new(),
            m: acc.m,
            n: acc.n,
            wm: acc.wm,
            wn: acc.wn,
            imma: false,
        };
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        // Each leaf once: fragments and fragment-row columns by register,
        // tiles as views.
        let mut frags = BTreeMap::new();
        let mut rows = BTreeMap::new();
        let mut tiles = BTreeMap::new();
        for (i, &v) in ir.operands(op).iter().enumerate() {
            match ir.ty(v) {
                t @ ir::Type::Dist(_) if t.mma_rows().is_some() => {
                    rows.insert(i, self.regs_ir(v)?.vals);
                }
                ir::Type::Dist(_) => {
                    frags.insert(i, self.frags_ir(v)?.frags);
                }
                ir::Type::Tile(_) => {
                    tiles.insert(i, self.tile_ir(ir, v)?);
                }
                _ => {}
            }
        }
        let zero = self.zero_scalar(block, self.f32_t)?;
        let seed = self.vec_broadcast(block, zero, Type::vector(&[2, 2], self.f32_t))?;
        let mut out = vec![seed; (fm * fnn * 2) as usize];
        let shape = [fa.m, fa.n];
        let vec_t = Type::vector(&[4], self.f32_t);
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for (at, idx) in elems {
                let leaves = Leaves { frags: &frags, rows: &rows, tiles: &tiles };
                let fused = cg.frag_tree(block, ir, op, tree, &leaves, i, row_slot(i, fnn, at[0]), *at)?;
                let v = cg.eval_fused(block, &fused, idx, &shape, 1, vec_t)?;
                let v = cg.coerce(block, v, cg.f32_t)?;
                out[i] = cg.vec_insert(block, v, out[i], at)?;
            }
            Ok(())
        })?;
        for tile in tiles.values() {
            self.release(tile);
        }
        fa.frags = out;
        Ok(fa)
    }

    /// `tree` at one element: each fragment leaf its value there, each
    /// fragment-row leaf its value at the element's row slot, each tile leaf
    /// the tile, read where the element sits.
    #[allow(clippy::too_many_arguments)]
    fn frag_tree(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        tree: &Tree,
        leaves: &Leaves<'_, 'c>,
        frag: usize,
        slot: usize,
        at: [i64; 2],
    ) -> Result<Fused<'c>> {
        let go = |cg: &mut Self, t: &Tree| -> Result<Box<Fused<'c>>> {
            Ok(Box::new(cg.frag_tree(block, ir, op, t, leaves, frag, slot, at)?))
        };
        Ok(match tree {
            Tree::Leaf(i) => match (leaves.frags.get(i), leaves.rows.get(i), leaves.tiles.get(i)) {
                (Some(f), ..) => Fused::Scalar(self.vec_extract(block, f[frag], &at, self.f32_t)?),
                (None, Some(r), _) => Fused::Scalar(r[slot]),
                (None, None, Some(t)) => Fused::Tile(t.clone()),
                (None, None, None) => bail!("frag_map leaf {i} is neither fragments nor a tile"),
            },
            Tree::Scalar(i) => Fused::Scalar(self.scalar_ir(ir.operand(op, *i))?),
            Tree::Unary(step, x) => Fused::Unary(self.ir_elem_step(*step), go(self, x)?),
            Tree::Binary(bop, a, b) => Fused::Binary(*bop, go(self, a)?, go(self, b)?),
            Tree::Max(a, b) => Fused::Max(go(self, a)?, go(self, b)?),
            Tree::Iota(d) => Fused::Iota(*d),
            Tree::Select(c, a, b) => Fused::Select(go(self, c)?, go(self, a)?, go(self, b)?),
        })
    }

    /// `frag_reduce`: each row folded in the lanes that hold it, the result
    /// a shared `[rows, 1]` column, or with `to_regs` the lanes' own rows
    /// in registers. Every warp owns whole rows, so a row's elements are
    /// this lane's and its quad's: a fold over the lane's own columns, then
    /// two xor shuffles within the quad, which leave every lane of the quad
    /// with the row's value.
    pub(super) fn frag_reduce(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        how: ir::Reduce,
        to_regs: bool,
    ) -> Result<Lowered<'c>> {
        let kind = match how {
            ir::Reduce::RowSum => Reduce::Sum,
            ir::Reduce::RowMax => Reduce::Max,
            other => bail!("{other:?} of fragments"),
        };
        if fa.imma || fa.wn != 1 {
            bail!("internal: a row reduction of fragments whose rows span warps");
        }
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        // The lane's rows: two per fragment row, each folded over its columns.
        let rows_per_lane = (fm * 2) as usize;
        let mut parts: Vec<Option<(Value<'c, 'c>, Value<'c, 'c>)>> = vec![None; rows_per_lane];
        let f32_t = self.f32_t;
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            let fi = i / (fnn as usize * 2);
            for ([di, dj], [row, _]) in elems {
                let e = cg.vec_extract(block, fa.frags[i], &[*di, *dj], f32_t)?;
                let slot = &mut parts[fi * 2 + *di as usize];
                *slot = Some(match *slot {
                    Some((acc, r)) => (cg.fold(block, kind, f32_t, acc, e)?, r),
                    None => (e, *row),
                });
            }
            Ok(())
        })?;
        let mut folded = Vec::with_capacity(rows_per_lane);
        for part in parts {
            let Some((mut acc, row)) = part else { bail!("internal: a lane row no fragment holds") };
            for mask in [1, 2] {
                let other = self.shfl_xor(block, acc, mask)?;
                acc = self.fold(block, kind, f32_t, acc, other)?;
            }
            folded.push((acc, row));
        }
        if to_regs {
            let vals = folded.into_iter().map(|(v, _)| v).collect();
            return Ok(Lowered::Regs(RegTile { vals, elem: f32_t, shape: vec![fa.m, 1], rows_wm: Some(fa.wm) }));
        }
        let out = self.alloc_tile_shaped(block, f32_t, &[fa.m, 1])?;
        let tid = self.thread_id(block)?;
        let four = self.const_index(block, 4)?;
        let in_quad = self.remui(block, tid, four)?;
        let zero = self.const_index(block, 0)?;
        let lead = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, in_quad, zero, self.loc),
        )?;
        for (acc, row) in folded {
            // The quad holds the row's value four times; one lane stores it.
            let then = Block::new(&[]);
            then.append_operation(memref::store(acc, out.mem, &[row, zero], self.loc));
            then.append_operation(scf::r#yield(&[], self.loc));
            let region = Region::new();
            region.append_block(then);
            block.append_operation(scf::r#if(lead, &[], region, Region::new(), self.loc));
        }
        self.barrier(block)?;
        Ok(Lowered::Tile(out))
    }
}

/// A `frag_map`'s leaves by operand: fragments and fragment-row columns as
/// a lane's registers, tiles as views.
struct Leaves<'a, 'c> {
    frags: &'a BTreeMap<usize, Vec<Value<'c, 'c>>>,
    rows: &'a BTreeMap<usize, Vec<Value<'c, 'c>>>,
    tiles: &'a BTreeMap<usize, MemVal<'c>>,
}
