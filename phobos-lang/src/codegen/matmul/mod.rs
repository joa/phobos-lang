// Matmul: recognising one, planning it, and emitting the three paths.
//
// `plan.rs` decides, then one of `reg.rs`, `wmma.rs` or `mma_sync.rs`
// emits; `stage.rs` is what all three share for moving operands.

use super::*;

/// Width of the vector<Nxf16> global loads used when staging f16 operands
const HALF_VEC: i64 = 4;

/// The matched register-accumulator GEMM pattern.
///
/// See also [`Codegen::matmul_candidate`].
///
/// ```plain
/// var acc: tile<f32>[M, N] = <scalar>
///
/// for kt in range(lo, hi, st) {
///     var a = A[<static f32 slice>] // let works too
///     var b = B[<static f32 slice>]
///     acc += dot(a, b)
/// }
///
/// // Optional GEMM epilogue (alpha/beta scaling with prev_load load):
/// [let prev_load = C[<slice>]] // same slice as store target
/// C[<slice>] = [alpha *] acc [+ beta * prev_load] // acc's last use
/// ```
pub struct MatmulFusion<'a> {
    pub dims: &'a [Dim],
    /// Element type of the accumulator
    pub acc_scalar: Scalar,
    pub init: &'a Expr,
    pub kt: &'a str,
    pub start: &'a Expr,
    pub end: &'a Expr,
    pub step: Option<&'a Expr>,
    pub a_slice: &'a Expr,
    pub b_slice: &'a Expr,
    pub out: &'a str,
    pub out_subs: &'a [Sub],
    /// Number of statements this fusion consumes (3 minimum, 4 with prev_load)
    pub consumed: usize,
    /// Coefficient for acc in the epilogue; None is identity (1.0)
    pub alpha: Option<&'a Expr>,
    /// Previous-C load for the GEMM epilogue; None = pure accumulation
    pub prev_load: Option<GemmPrevLoad<'a>>,
}

/// The prev_load term in a GEMM epilogue: beta*prev_load loaded from global C.
pub struct GemmPrevLoad<'a> {
    /// Coefficient for prev_load; None is identity (1.0)
    pub beta: Option<&'a Expr>,
}

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

mod mma_sync;
mod plan;
mod reg;
mod stage;
mod wmma;
