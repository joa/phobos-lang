// The register matmul: planning it and emitting the three paths.
//
// `gemm.rs` decides and drives the three thirds, then one of `reg.rs`,
// `wmma.rs` or `mma_sync.rs` emits each third; `plan.rs` holds the k-loop
// they share and `stage.rs` what all three share for moving operands.

use super::*;

pub(in crate::codegen) use gemm::{GemmAcc, GemmOperand, GemmPath, GemmPlan, GemmScale, GemmSource};

/// Width of the vector<Nxf16> global loads used when staging f16 operands
const HALF_VEC: i64 = 4;

/// How an epilogue drain moves half a slab row to C.
#[derive(Clone, Copy)]
enum DrainMode {
    /// 16-byte 4xf32 vectors: f32 slab to an aligned f32 C.
    VecF32,
    /// 8-byte 4xf16 vectors, widened to f32 for the scaling and rounded back:
    /// f16 slab to an aligned f16 C. Coalesced, where the scalar path only
    /// writes 2 bytes per sector.
    VecF16,
    /// Element-wise: widen to f32, scale, round to C's element type (a no-op
    /// when both are f32).
    Scalar,
}

/// The loop-invariant state of a tensor-core epilogue drain. Both the WMMA
/// and the mma.sync epilogues park each finished 16x16 tile in the warp's
/// shared slab, then the lanes copy it out to C between barriers: half a slab
/// row per lane (row = lane / 2, column = lane % 2 * 8), as two 4-vectors.
struct SlabDrain<'c> {
    slab: MemVal<'c>,
    view: MemVal<'c>,
    /// The lane's slab row (its warp's tile origin plus lrow).
    srow: Value<'c, 'c>,
    /// The lane's row and column offsets within a 16x16 tile.
    lrow: Value<'c, 'c>,
    lcol: Value<'c, 'c>,
    /// Precomputed GEMM scaling (see [`Codegen::epilogue_scaling`]).
    alpha: Option<Value<'c, 'c>>,
    beta: Option<Value<'c, 'c>>,
    mode: DrainMode,
}

mod gemm;
mod mma_sync;
mod plan;
mod reg;
mod stage;
mod wmma;
