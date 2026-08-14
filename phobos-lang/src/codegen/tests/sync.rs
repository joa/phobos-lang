// `atomic_add` and `grid_barrier`.

use super::*;

#[test]
fn atomic_add_lowers_to_an_rmw() {
    let mlir = emit_mlir(
        "kernel tally(BAR: tensor<i32>[2], OUT: tensor<i32>[N]) {
            let i = program_id(0)
            OUT[i] = atomic_add(BAR, 0, 1)
        }",
    );
    assert_contains(&mlir, &["memref.atomic_rmw", "addi"]);
}

#[test]
fn atomic_add_rejects_a_float_tensor() {
    let err = std::panic::catch_unwind(|| {
        emit_mlir(
            "kernel tally(BAR: tensor<f32>[2], OUT: tensor<i32>[N]) {
                let i = program_id(0)
                OUT[i] = atomic_add(BAR, 0, 1)
            }",
        )
    });
    assert!(err.is_err(), "an f32 barrier tensor should not compile");
}

/// The two-phase barrier: a generation read, an arrival, a reset, a release
/// and a spin, bracketed by the CTA barriers that carry the release to the
/// rest of the block. See codegen/sync.rs.
#[test]
fn grid_barrier_lowers_to_arrive_and_spin() {
    let mlir = emit_mlir(
        "kernel staged(X: tensor<f32>[N], BAR: tensor<i32>[2]) {
            let i = program_id(0)
            X[i] = X[i] + 1.0
            grid_barrier(BAR)
            X[i] = X[i] * 2.0
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.atomic_rmw", // arrive, release and spin all go through it
            "scf.while",         // the spin
            "gpu.barrier",       // the CTA brackets
            "gpu.grid_dim",      // how many arrivals make a full barrier
        ],
    );
    // Five atomics: the generation read, the arrival, the counter reset, the
    // release, and the spin's read.
    assert_eq!(mlir.matches("memref.atomic_rmw").count(), 5);
}

/// The same barrier as a `[2, 1]` column, which is how a host whose launch
/// ABI passes rank-2 descriptors spells it. The slot indexes the leading
/// dimension and the atomic takes a second subscript, which the verifier
/// checks against the memref rank.
#[test]
fn grid_barrier_takes_a_rank_two_column() {
    let mlir = emit_mlir(
        "kernel staged(X: tensor<f32>[N], BAR: tensor<i32>[2, 1]) {
            let i = program_id(0)
            X[i] = X[i] + 1.0
            grid_barrier(BAR)
        }",
    );
    assert_contains(&mlir, &["memref.atomic_rmw", "scf.while"]);
    assert_eq!(mlir.matches("memref.atomic_rmw").count(), 5);
}

#[test]
fn grid_barrier_rejects_a_non_tensor() {
    let err = std::panic::catch_unwind(|| {
        emit_mlir(
            "kernel staged(X: tensor<f32>[N]) {
                var t: tile<i32>[1, 2] = 0
                grid_barrier(t)
            }",
        )
    });
    assert!(err.is_err(), "a tile barrier should not compile");
}
