
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
            "memref.atomic_rmw",
            "scf.while",
            "gpu.barrier",
            "gpu.grid_dim",
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

/// Two elementwise sweeps over one tile at one width read and write the
/// same elements from the same thread, so nothing separates them.
#[test]
fn same_thread_sweeps_need_no_barrier_between_them() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            t = t + 1.0
            C[0 :+ 8, 0 :+ 8] = t
        }",
    );
    // The staging copy moves 16 bytes a lane and the f32 sweeps four
    // elements, the same mapping; the store reads it the same way.
    assert_eq!(mlir.matches("gpu.barrier").count(), 0, "{mlir}");
}

/// A reduction reads every column of a row from one thread, so the sweep
/// that wrote the row has to publish it first.
#[test]
fn a_reduction_after_a_sweep_keeps_the_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, 1]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            var s: tile<f32>[8, 1] = rowsum(t)
            C[0 :+ 8, 0 :+ 1] = s
        }",
    );
    let barriers: Vec<usize> = mlir.match_indices("gpu.barrier").map(|(at, _)| at).collect();
    let sweep = mlir.find("arith.mulf").expect("the scaling sweep");
    let reduce = mlir.find("arith.addf").expect("the reduction");
    assert!(
        barriers.iter().any(|&b| sweep < b && b < reduce),
        "the scaling sweep must publish before the reduction reads:\n{mlir}"
    );
}

/// A sweep a loop carries to its next iteration is read by the same
/// threads again, so the loop runs with no barrier at all.
#[test]
fn a_loop_carried_sweep_needs_no_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            for i in range(0, n) {
                t = t * 2.0
            }
            C[0 :+ 8, 0 :+ 8] = t
        }",
    );
    assert_eq!(mlir.matches("gpu.barrier").count(), 0, "{mlir}");
}

/// A loop whose body reads its tile at another width than it wrote it has
/// to keep a barrier for what one iteration hands the next.
#[test]
fn a_loop_carried_hazard_keeps_a_barrier_in_the_body() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, 1], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            var s: tile<f32>[8, 1] = 0.0
            for i in range(0, n) {
                t = t * 2.0
                s = rowsum(t)
            }
            C[0 :+ 8, 0 :+ 1] = s
        }",
    );
    let body = mlir.find("scf.for").expect("the loop");
    let barriers = mlir.match_indices("gpu.barrier").filter(|(at, _)| *at > body).count();
    assert!(barriers >= 1, "{mlir}");
}

/// A per-thread element read of a tile another thread's sweep wrote needs
/// the sweep's barrier, and a per-thread store has no barrier site of its
/// own, so the pass keeps the barrier.
#[test]
fn an_element_read_after_a_sweep_keeps_the_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            let x = t[0, 1]
            C[0, 0] = x
        }",
    );
    let sweep = mlir.find("arith.mulf").expect("the scaling sweep");
    let load = mlir.rfind("memref.load").expect("the element read");
    assert!(
        mlir.match_indices("gpu.barrier").any(|(b, _)| sweep < b && b < load),
        "the sweep must publish before another thread reads an element:\n{mlir}"
    );
}

/// A hazard that only crosses the iteration boundary: the body's last op
/// writes the tile at one width and its first op reads it at another. The
/// write's own trailing barrier is the only site between them, and the
/// walk sees the conflict only on its second pass over the body.
#[test]
fn a_hazard_across_the_iteration_boundary_keeps_the_last_ops_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            var u: tile<f32>[8, 8] = 0.0
            for i in range(0, n) {
                u = exp(t)
                t = t * 2.0
            }
            C[0 :+ 8, 0 :+ 8] = u
        }",
    );
    // From the body's first op on: one barrier after the read for the write
    // that follows it in the same iteration, one after the write for the
    // read that opens the next, and none for the store after the loop.
    let read = mlir.find("ex2.approx").expect("the exp sweep");
    let barriers: Vec<usize> = mlir
        .match_indices("gpu.barrier")
        .map(|(at, _)| at)
        .filter(|&at| at > read)
        .collect();
    assert_eq!(barriers.len(), 2, "{mlir}");
    let write = mlir.rfind("arith.mulf").expect("the scaling sweep");
    assert!(barriers[1] > write, "the write's own barrier must stay:\n{mlir}");
}
