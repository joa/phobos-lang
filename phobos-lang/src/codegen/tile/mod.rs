// Tile operations, the bulk of the emitter. Every file here reopens `impl
// Codegen` and groups one kind of work; `contract.rs` dispatches across the
// quantized contraction files, which are separate hardware paths.

use super::*;

mod alloc;
mod check;
mod contract;
mod dp4a;
mod elem;
mod gather;
mod imma;
mod iq1m;
mod iq1m_qdot;
mod iq1s;
mod iq1s_qdot;
mod iq1s_qmma;
mod iq2s;
mod iq2s_qdot;
mod iq2xs;
mod iq2xs_qdot;
mod iq2xxs;
mod iq2xxs_qdot;
mod iq3s;
mod iq3s_qdot;
mod iq3xxs;
mod iq3xxs_qdot;
mod iq4xs_qdot;
mod kquant;
mod kquant_qdot;
mod math;
mod norm;
mod q2k_qdot;
mod q3k_qdot;
mod qdecode;
mod qdot;
mod qdot_i8_reg;
mod qgemm;
mod qgemm_fmt;
mod qmma;
mod qmma_signed;
mod reduce;
mod vector;
mod warp_attn;

pub(in crate::codegen) use qdecode::QFormat;
pub(in crate::codegen) use qgemm::QgFormat;

/// Activations a lane loads at once in the quantized matvecs: a lane owns a
/// contiguous run, so a warp covers 1024 bytes in two loads instead of eight.
pub(super) const ACT_VEC: i64 = 4;

/// Elements an int8 activation shares one scale over; the `quantize` kernel
/// the Q8_0 path already uses emits this layout.
pub(super) const ACT_SCALE_BLOCK: i64 = 32;

/// A format's packed lookup tables, one `i8` a slot: a lane's whole entry is
/// one vector load. `signs` aliases `grid` for the one-table formats.
pub(super) struct QTables<'a, 'c> {
    pub(super) grid: &'a MemVal<'c>,
    pub(super) signs: &'a MemVal<'c>,
}

/// Columns to a group of the grouped raw layout: the payload is
/// `[N / 8][NB][8][block]` and the scale plane `[N / 8][NB][8]`, declared
/// `[N, RB]` and `[N, NB]`. The backend uploads the IQ formats so
/// (`Quant::grouped_rows`); every reader goes through
/// [`Codegen::raw_block_at`].
pub(super) const RAW_GROUP: i64 = 8;

/// Where one quantized block sits for a decoding thread: weight row, block
/// index, and the byte offset that index lands at.
pub(super) struct BlockAt<'c> {
    /// The row of `qb` the block bytes are read from, and the byte offset
    /// of the block within it. A staged decode points these at a
    /// shared-memory copy, so the row is then the stage's, not the weight's.
    pub(super) j: Value<'c, 'c>,
    pub(super) off: Value<'c, 'c>,
    /// Where the block's scale sits in the scale plane.
    pub(super) d_row: Value<'c, 'c>,
    pub(super) d_col: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// Column `j`'s block `blk` in the grouped layout: row `8 (j / 8)`,
    /// block `8 blk + j % 8` of it, and the scale at the same entry.
    pub(super) fn raw_block_at(
        &mut self,
        block: &Block<'c>,
        j: Value<'c, 'c>,
        blk: Value<'c, 'c>,
        blk_bytes: Value<'c, 'c>,
    ) -> Result<BlockAt<'c>> {
        let group_w = self.const_index(block, RAW_GROUP)?;
        let group = self.divui(block, j, group_w)?;
        let in_group = self.remui(block, j, group_w)?;
        let row = self.muli(block, group, group_w)?;
        let d_blk = self.muli(block, blk, group_w)?;
        let d_col = self.addi(block, d_blk, in_group)?;
        let off = self.muli(block, d_col, blk_bytes)?;
        Ok(BlockAt { j: row, off, d_row: row, d_col })
    }

    /// One raw block byte of a grouped weight, by column and block.
    pub(super) fn qbyte_grouped(
        &mut self,
        block: &Block<'c>,
        qb: &MemVal<'c>,
        j: Value<'c, 'c>,
        blk: Value<'c, 'c>,
        blk_bytes: Value<'c, 'c>,
        off: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let at = self.raw_block_at(block, j, blk, blk_bytes)?;
        self.qbyte(block, qb, at.j, at.off, off)
    }
}
