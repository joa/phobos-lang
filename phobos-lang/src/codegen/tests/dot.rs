// The `dot` builtin, its staging, and fragment-resident accumulators.

use super::*;

#[test]
fn flash_accumulator_rides_in_fragments() {
    // The canonical fp16 flash kernel (examples/flash_attention_fp16.ph).
    // Every use of acc is fragment-representable, so it never exists in
    // shared memory: its per-lane fragments ride the kt loop as iter_args
    // and the epilogue scatters straight to O. With buffer pooling and the
    // fused in-place exp, the kernel carries 9 shared globals / ~15KB down
    // from 18 / 40KB, so four CTAs fit an sm_75 SM instead of one.
    let mlir = emit_mlir_sync(
        "@autotune(D in [64], BR in [32], BC in [32])
        @tensorcore
        @launch(128)
        @aligned(Nq = BR, Nk = BC)
        kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                               V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            var m: tile<f32>[BR, 1] = -65504.0
            var l: tile<f32>[BR, 1] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                let v = V[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                s = s * scale
                var mnew: tile<f32>[BR, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                s = exp(s - mnew)
                var corr: tile<f32>[BR, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(s)
                acc = acc * corr
                acc += dot(s, v)
                m = mnew
            }
            acc = acc / l
            O[row :+ BR, :] = acc
        }",
        "sm_75",
    );
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "nvgpu.ldmatrix",
            // fragments thread the kt loop as iter_args
            "iter_args",
            "vector<2x2xf32>",
        ],
    );
    // acc never materializes in shared memory.
    assert!(
        !mlir.contains("memref<32x64xf32, 3>"),
        "fragment accumulator materialized in shared memory:\n{mlir}"
    );
    let globals = mlir.matches("memref.global").count();
    assert!(
        globals <= 9,
        "expected at most 9 pooled shared buffers, got {globals}:\n{mlir}"
    );
    // One [BR, BC] f32 buffer serves scores and probabilities: the
    // fused exp(s - mnew) sweep rewrites it in place.
    let s_bufs = mlir
        .lines()
        .filter(|l| l.contains("memref.global") && l.contains("memref<32x32xf32, 3>"))
        .count();
    assert_eq!(
        s_bufs, 1,
        "expected the score tile to stay a single in-place buffer:\n{mlir}"
    );
}

#[test]
fn flash_q_staging_hoists_to_the_loop_preheader() {
    // q in dot_t(q, k) is a let-bound Q slice defined outside the kt
    // loop and nothing in the body stores to global memory, so its
    // global-to-shared f16 staging copy runs once in the preheader
    // instead of every iteration. q's subview is the first in the
    // kernel, so its loads print as "%subview[" exactly.
    let src = "@autotune(D in [64], BR in [32], BC in [32])
        @tensorcore
        @launch(128)
        @aligned(Nq = BR, Nk = BC)
        kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                               V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            var m: tile<f32>[BR, 1] = -65504.0
            var l: tile<f32>[BR, 1] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                let v = V[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                s = s * scale
                var mnew: tile<f32>[BR, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                s = exp(s - mnew)
                var corr: tile<f32>[BR, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(s)
                acc = acc * corr
                acc += dot(s, v)
                m = mnew
            }
            acc = acc / l
            O[row :+ BR, :] = acc
        }";
    // Both tensor-core paths hoist: mma.sync (frag-carried kt loop) and
    // the legacy WMMA fallback (plain kt loop).
    for mlir in [emit_mlir_sync(src, "sm_75"), emit_mlir(src)] {
        let (preheader, body) = split_at_kt_loop(&mlir);
        assert!(
            preheader.contains("load %subview["),
            "q staging not hoisted to the preheader:\n{mlir}"
        );
        assert!(
            !body.contains("load %subview["),
            "q still re-staged inside the kt loop:\n{mlir}"
        );
    }
}

#[test]
fn dot_staging_stays_in_loop_when_body_stores_global() {
    // The body stores s to O each iteration: a staged copy of q could
    // not see global writes, so the hoist must stand down and q stages
    // inside the loop as before.
    let mlir = emit_mlir_sync(
        "@autotune(D in [64], BR in [32], BC in [32])
        @tensorcore
        @launch(128)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], O: tensor<f32>[Nq, BC]) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                O[row :+ BR, :] = s
            }
        }",
        "sm_75",
    );
    let (preheader, body) = split_at_kt_loop(&mlir);
    assert!(
        !preheader.contains("load %subview["),
        "q staging hoisted past a global store:\n{mlir}"
    );
    assert!(
        body.contains("load %subview["),
        "q staging missing from the loop body:\n{mlir}"
    );
}

#[test]
fn frag_acc_falls_back_on_unsanctioned_reads() {
    // rowsum(o) reads the accumulator outside the fragment-representable
    // forms, so the candidate must reject it and o stays a shared tile.
    let mlir = emit_mlir_sync(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel pv(P: tensor<f16>[Nq, Nk], V: tensor<f16>[Nk, D],
                  O: tensor<f32>[Nq, D], R: tensor<f32>[Nq, 1]) {
            let pid = program_id(0)
            let row = pid * BR
            var o: tile<f32>[BR, D] = 0.0
            let p = P[row :+ BR, 0 :+ BC]
            let v = V[0 :+ BC, :]
            o += dot(p, v)
            var r: tile<f32>[BR, 1] = rowsum(o)
            O[row :+ BR, :] = o
            R[row :+ BR, :] = r
        }",
        "sm_75",
    );
    assert_contains(&mlir, &["memref<64x64xf32, 3>", "nvgpu.mma.sync"]);
}

#[test]
fn tensorcore_dot_t_loads_transposed_b() {
    // dot_t (Q @ K.T) on the tensor cores stages both operands in
    // their natural [rows, D] layout and loads the B (K) fragment
    // column-major (transpose), so no transposing staging pass is
    // needed.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D], S: tensor<f32>[Nq, Nk]) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            let k = K[0 :+ BC, :]
            var s: tile<f32>[BR, BC] = dot_t(q, k)
            S[row :+ BR, 0 :+ BC] = s
        }",
    );
    assert_contains(
        &mlir,
        &[
            // both operands staged f16 in their natural [rows, D] layout
            "memref<64x64xf16, 3>",
            "arith.truncf",
            // warp-collective fragment loads + computes, the B load
            // column-major (transpose) for the Q @ K.T contraction
            "gpu.subgroup_mma_load_matrix",
            "transpose",
            "gpu.subgroup_mma_compute",
            "gpu.subgroup_mma_store_matrix",
        ],
    );
    assert!(
        !mlir.contains("vector.contract"),
        "vector MACs left on the tensor-core dot_t path:\n{mlir}"
    );
}

#[test]
fn tensorcore_f16_dot_stages_vectorized() {
    // With f16 operands the WMMA staging is a plain copy (no truncf), and
    // it vectorizes as 8xf16, the same 16 bytes a lane an f32 tile moves
    // as 4xf32. Eight needs a 16-byte reach that the row-pitch ABI does not
    // promise on its own; here the head dimension is a static 64, so the
    // row pitch is 128 bytes and the proof comes from the shape.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], S: tensor<f32>[Nq, Nk]) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            let k = K[0 :+ BC, :]
            var s: tile<f32>[BR, BC] = dot_t(q, k)
            S[row :+ BR, 0 :+ BC] = s
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref<64x64xf16, 3>",
            // 16-byte 8xf16 staging loads and stores, not scalar
            "vector.load",
            "vector.store",
            "vector<8xf16>",
            "alignment = 16",
            "gpu.subgroup_mma_compute",
        ],
    );
    // f16 in, f16 staged: no rounding conversion on the staging path.
    assert!(
        !mlir.contains("arith.truncf"),
        "unexpected truncf staging f16 operands:\n{mlir}"
    );
}

#[test]
fn tensorcore_dot_uses_wmma() {
    // The plain dot (P @ V) path runs on the tensor cores with a
    // row-major (non-transposed) B load.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel pv(P: tensor<f32>[Nq, Nk], V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D]) {
            let pid = program_id(0)
            let row = pid * BR
            let p = P[row :+ BR, 0 :+ BC]
            let v = V[0 :+ BC, :]
            var o: tile<f32>[BR, D] = dot(p, v)
            O[row :+ BR, :] = o
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref<64x64xf16, 3>",
            "gpu.subgroup_mma_load_matrix",
            "gpu.subgroup_mma_compute",
            "gpu.subgroup_mma_store_matrix",
        ],
    );
    // NN contraction: the B fragment stays row-major, no transpose.
    assert!(
        !mlir.contains("transpose"),
        "unexpected transposed load on the NN dot path:\n{mlir}"
    );
    assert!(
        !mlir.contains("vector.contract"),
        "vector MACs left on the tensor-core dot path:\n{mlir}"
    );
}

#[test]
fn tensorcore_dot_t_uses_mma_sync() {
    // At 64-bit index dot_t (Q @ K.T) takes the default mma.sync path:
    // ldmatrix + nvgpu.mma.sync over swizzled f16 staging, no WMMA.
    let mlir = emit_mlir_sync(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], S: tensor<f32>[Nq, Nk]) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            let k = K[0 :+ BC, :]
            var s: tile<f32>[BR, BC] = dot_t(q, k)
            S[row :+ BR, 0 :+ BC] = s
        }",
        "sm_75",
    );
    assert_contains(
        &mlir,
        &[
            // unpadded, XOR-swizzled f16 staging read by ldmatrix
            "memref<64x64xf16, 3>",
            "arith.xori",
            "nvgpu.ldmatrix",
            "nvgpu.mma.sync",
            "mmaShape = [16, 8, 8]",
            // the m16n8 f32 accumulator scattered to the shared output tile
            "vector<2x2xf32>",
        ],
    );
    assert!(
        !mlir.contains("subgroup_mma") && !mlir.contains("mma_matrix"),
        "legacy WMMA ops left on the mma.sync dot_t path:\n{mlir}"
    );
}

#[test]
fn tensorcore_dot_accumulate_rides_in_fragments() {
    // o += dot(p, v) with an accumulator whose every use is fragment-
    // representable never materializes o in shared memory: the per-lane
    // mma.sync D fragments seed the MAC directly and the epilogue
    // scatters them straight to O. The NN B operand is still read
    // transposed (k-major staging).
    let mlir = emit_mlir_sync(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel pv(P: tensor<f16>[Nq, Nk], V: tensor<f16>[Nk, D], O: tensor<f32>[Nq, D]) {
            let pid = program_id(0)
            let row = pid * BR
            var o: tile<f32>[BR, D] = 0.0
            let p = P[row :+ BR, 0 :+ BC]
            let v = V[0 :+ BC, :]
            o += dot(p, v)
            O[row :+ BR, :] = o
        }",
        "sm_75",
    );
    assert_contains(
        &mlir,
        &[
            "nvgpu.ldmatrix",
            "nvgpu.mma.sync",
            // NN dot reads the k-major B staging transposed
            "transpose = true",
        ],
    );
    assert!(
        !mlir.contains("memref<64x64xf32, 3>"),
        "fragment accumulator materialized in shared memory:\n{mlir}"
    );
    assert!(
        !mlir.contains("subgroup_mma"),
        "legacy WMMA ops left on the mma.sync dot accumulate path:\n{mlir}"
    );
}
