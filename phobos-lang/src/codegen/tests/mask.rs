// Slices, `@aligned`, bounds masks and the ragged loop split.

use super::*;

#[test]
fn range_and_full_slices() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[M, N], i: i32, j: i32) {
            let t = A[i : j, :]
            A[0 :+ 4, :] = 0.0
        }",
    );
    assert_contains(
        &mlir,
        &[
            // i:j -> dynamic size (subi), : on dynamic N -> memref.dim
            "arith.subi",
            "memref.dim",
            "memref.subview",
            // scalar store into a slice is a distributed fill
            "gpu.thread_id",
            "memref.store",
            "memref<4x?xf32, strided<[?, 1], offset: ?>, 1>",
        ],
    );
}

#[test]
fn partial_static_slice_is_masked() {
    // A 32-wide tile does not tile a 100-element tensor evenly, so the
    // last tile runs past the end. The read stages through a zero-filled
    // buffer (select) and the write skips the out-of-bounds lanes (scf.if
    // guarded by offset + index < extent).
    let mlir = emit_mlir(
        "kernel copy(A: tensor<f32>[100], B: tensor<f32>[100]) {
            let p = program_id(0)
            B[p * 32 :+ 32] = A[p * 32 :+ 32]
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.subview",
            "arith.cmpi ult", // offset + index < extent
            "arith.select",   // out-of-bounds reads fold to zero
            "scf.if",         // the store runs only in bounds
        ],
    );
}

#[test]
fn a_program_id_slice_of_a_dynamic_extent_is_masked() {
    // The extent is a runtime value, so nothing inside the kernel bounds
    // the grid: the last program id can address a tile that runs off the
    // end. Left unmasked this writes through the end of a row and into the
    // next one, which is a whole-tensor corruption rather than a bad tail.
    let mlir = emit_mlir(
        "kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
            let p = program_id(0)
            B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.dim",     // the extent is only known at runtime
            "arith.cmpi ult", // offset + index < extent
            "arith.select",   // out-of-bounds reads fold to zero
            "scf.if",         // the store runs only in bounds
        ],
    );
}

#[test]
fn an_aligned_declaration_drops_the_dynamic_bounds_mask() {
    // @aligned is the host promising the extent is a whole number of tiles,
    // which is what a general GEMM needs to keep the register-blocked and
    // tensor-core drains: those have no per-element store guard, so without
    // the promise they decline and the kernel falls back.
    let mlir = emit_mlir(
        "@aligned(N = 32)
        kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
            let p = program_id(0)
            B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
        }",
    );
    assert!(
        !mlir.contains("scf.if"),
        "unexpected bounds guard for a declared-aligned extent:
{mlir}"
    );
}

#[test]
fn an_aligned_declaration_must_cover_the_tile() {
    // Promising a coarser tiling than the slice takes proves nothing: 32
    // does not divide a multiple of 24, so the mask stays.
    let mlir = emit_mlir(
        "@aligned(N = 24)
        kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
            let p = program_id(0)
            B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
        }",
    );
    assert_contains(&mlir, &["arith.cmpi ult", "scf.if"]);
}

#[test]
fn an_unknown_aligned_constant_is_rejected() {
    let err = emit_err(
        "@aligned(N = TILE)
        kernel copy(A: tensor<f32>[M, N]) {
            let p = program_id(0)
            A[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
        }",
    );
    assert!(err.contains("unknown constant"), "unexpected error: {err}");
}

#[test]
fn aligned_static_slice_is_not_masked() {
    // A 32-wide tile tiles a 128-element tensor evenly, so no lane ever
    // leaves the tensor: no bounds mask, and the copy still vectorizes.
    let mlir = emit_mlir(
        "kernel copy(A: tensor<f32>[128], B: tensor<f32>[128]) {
            let p = program_id(0)
            B[p * 32 :+ 32] = A[p * 32 :+ 32]
        }",
    );
    assert_contains(&mlir, &["memref.subview", "vector<4xf32>"]);
    assert!(
        !mlir.contains("scf.if") && !mlir.contains("arith.select"),
        "unexpected bounds mask for an evenly tiled tensor:\n{mlir}"
    );
}

#[test]
fn partial_matmul_epilogue_is_masked() {
    // N = 100 is not a multiple of TILE_N = 32, so the fused register
    // matmul declines (its blocking has no bounds guard) and the generic
    // tiled path runs with a masked epilogue store into C.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [32], TILE_N in [32], TILE_K in [32])
        kernel matmul(A: tensor<f32>[96, 64], B: tensor<f32>[64, 100], C: tensor<f32>[96, 100]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, 64, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_contains(&mlir, &["memref.subview", "arith.cmpi ult", "scf.if"]);
}

#[test]
fn dot_directly_into_partial_slice_is_rejected() {
    // A dot result written straight into a partially out-of-bounds slice
    // has no accumulator tile to carry the bounds guard, so it is a
    // compile error rather than an out-of-bounds store.
    let err = emit_err(
        "kernel matmul(A: tensor<f32>[100, 64], B: tensor<f32>[64, 100], C: tensor<f32>[100, 100]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var a = A[pm * 32 :+ 32, 0 :+ 64]
            var b = B[0 :+ 64, pn * 32 :+ 32]
            C[pm * 32 :+ 32, pn * 32 :+ 32] = dot(a, b)
        }",
    );
    assert!(
        err.contains("partially out-of-bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn dynamic_extent_loop_splits_off_a_masked_remainder() {
    // N is dynamic, so nothing statically proves C tiles it evenly. The
    // loop is trimmed to (N / C) * C and the ragged remainder replays the
    // body once, guarded against the tensor's runtime memref.dim.
    let mlir = emit_mlir(
        "@autotune(D in [32], C in [32])
        kernel scale(X: tensor<f32>[N, D], O: tensor<f32>[N, D]) {
            for c in range(0, N, C) {
                var t: tile<f32>[C, D] = X[c :+ C, :]
                t = t * 2.0
                O[c :+ C, :] = t
            }
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.dim",     // the runtime extent
            "arith.divui",    // (N - 0) / C whole chunks
            "arith.cmpi ult", // offset + index < N inside the remainder
            "arith.select",   // out-of-bounds reads fold to zero
            "scf.if",         // the remainder runs only when N is ragged
        ],
    );
}

#[test]
fn static_extent_loop_does_not_split() {
    // A static extent the tile divides evenly is provably whole, so the
    // loop keeps its single unguarded form.
    let mlir = emit_mlir(
        "@autotune(D in [32], C in [32])
        kernel scale(X: tensor<f32>[128, D], O: tensor<f32>[128, D]) {
            for c in range(0, 128, C) {
                var t: tile<f32>[C, D] = X[c :+ C, :]
                t = t * 2.0
                O[c :+ C, :] = t
            }
        }",
    );
    assert!(
        !mlir.contains("arith.cmpi ult") && !mlir.contains("memref.dim"),
        "unexpected ragged split for an evenly tiled static extent:\n{mlir}"
    );
}

#[test]
fn split_main_loop_keeps_wmma_and_needs_no_mask() {
    // The trimmed main loop's slices are in bounds by construction, so it
    // keeps the tensor-core path and takes no bounds mask; only the
    // remainder pays for the guard.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [32], BC in [32])
        @tensorcore
        @launch(128)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], O: tensor<f32>[Nq, BC]) {
            let pid = program_id(0)
            let q = Q[pid * BR :+ BR, :]
            var acc: tile<f32>[BR, BC] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                acc += dot_t(q, k)
            }
            O[pid * BR :+ BR, :] = acc
        }",
    );

    let (_, loops) = split_at_kt_loop(&mlir);
    let at_remainder = loops
        .find("\n      scf.if ")
        .expect("no ragged remainder in module");
    let (main, remainder) = loops.split_at(at_remainder);
    // The remainder's `trim < extent` guard is emitted just before the
    // scf.if, so drop it before asserting the main loop carries no mask.
    let main = main.rfind("arith.cmpi ult").map_or(main, |at| &main[..at]);

    assert!(
        main.contains("gpu.subgroup_mma_compute"),
        "the trimmed main loop lost WMMA:\n{mlir}"
    );
    assert!(
        !main.contains("arith.cmpi ult"),
        "the trimmed main loop should need no bounds mask:\n{mlir}"
    );
    assert!(
        remainder.contains("arith.cmpi ult"),
        "the ragged remainder is unguarded:\n{mlir}"
    );
}
