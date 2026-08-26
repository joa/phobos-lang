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
mod math;
mod q2k_qdot;
mod q3k_qdot;
mod qdecode;
mod qdot;
mod qmma;
mod reduce;
mod vector;
mod warp_attn;

pub(in crate::codegen) use qdecode::QFormat;

/// Activations a lane loads at once in the quantized matvecs: a lane owns a
/// contiguous run, so a warp covers 1024 bytes in two loads instead of eight.
pub(super) const ACT_VEC: i64 = 4;

/// A format's packed lookup tables, one `i8` a slot: a lane's whole entry is
/// one vector load. `signs` aliases `grid` for the one-table formats.
pub(super) struct QTables<'a, 'c> {
    pub(super) grid: &'a MemVal<'c>,
    pub(super) signs: &'a MemVal<'c>,
}

/// Where one quantized block sits for a decoding thread: weight row, block
/// index, and the byte offset that index lands at.
pub(super) struct BlockAt<'c> {
    pub(super) j: Value<'c, 'c>,
    pub(super) blk: Value<'c, 'c>,
    pub(super) off: Value<'c, 'c>,
}
