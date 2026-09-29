// The register matmul: planning it and emitting the three paths.
//
// `gemm.rs` picks a path and drives its three parts (seed, loop, drain).
// `reg.rs`, `wmma.rs` and `mma_sync.rs` emit them for each path. `plan.rs`
// holds the shared k-loop, and `stage.rs` the shared operand staging.

use super::*;

pub(in crate::codegen) use gemm::{GemmAcc, GemmOperand, GemmPath, GemmPlan, GemmScale, GemmSource};

/// Width of the vector<Nxf16> global loads used when staging f16 operands.
const HALF_VEC: i64 = 4;

/// How an epilogue drain moves half a slab row to C.
#[derive(Clone, Copy)]
enum DrainMode {
    /// 16-byte 4xf32 vectors: f32 slab to an aligned f32 C.
    VecF32,
    /// 8-byte 4xf16 vectors: f16 slab to an aligned f16 C. Widened to f32 for
    /// the scaling and rounded back.
    VecF16,
    /// Element-wise: widen to f32, scale, round to C's element type.
    Scalar,
}

/// The loop-invariant state of a tensor-core epilogue drain.
///
/// Both tensor-core paths write each finished 16x16 tile into the warp's
/// shared slab. The lanes then copy it to C between barriers. Each lane moves
/// half a slab row (row = lane / 2, column = lane % 2 * 8) as two 4-vectors.
struct SlabDrain<'c> {
    slab: MemVal<'c>,
    view: MemVal<'c>,
    /// The lane's slab row (its warp's tile origin plus lrow).
    srow: Value<'c, 'c>,
    /// The lane's row and column offsets within a 16x16 tile.
    lrow: Value<'c, 'c>,
    lcol: Value<'c, 'c>,
    /// Precomputed GEMM scaling, see [`Codegen::epilogue_scaling`].
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
