// Elementwise chains, row reductions and the math intrinsics.

use super::*;

#[test]
fn elementwise_tile_accumulate_is_distributed() {
    let mlir = emit_mlir(
        "@autotune(T in [8, 16])
        kernel k(A: tensor<f32>[N]) {
            var acc: tile<f32>[T] = 0.0
            let a = A[0 :+ T]
            acc += a
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref<8xf32, 3>",
            "memref.get_global",
            "memref.subview",
            "gpu.thread_id",
            "gpu.block_dim",
            "scf.for",
            "arith.addf",
            // aligned operands, 8 % 4 == 0 -> vectorized accumulate
            "vector.load",
            "vector.store",
        ],
    );
}

#[test]
fn fused_slice_binary_has_no_temp_buffer() {
    let mlir = emit_mlir(
        "@autotune(BLOCK in [1024])
        @aligned(N = BLOCK)
        kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
            let base = program_id(0) * BLOCK
            c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
        }",
    );
    // c[slice] = a[slice] + b[slice] writes the target subview directly.
    assert!(
        !mlir.contains("memref.alloca"),
        "unexpected temp in:\n{mlir}"
    );
    assert!(!mlir.contains("memref.copy"), "unexpected copy in:\n{mlir}");
    assert_contains(
        &mlir,
        &[
            "gpu.thread_id",
            "gpu.block_dim",
            "scf.for",
            "arith.addf",
            // slice offsets are provably 16B-aligned (base = pid*1024),
            // so the whole add is 128-bit vectorized
            "vector.load",
            "vector.store",
            "alignment = 16",
        ],
    );
}

#[test]
fn flash_attention_lowers_softmax_builtins() {
    // The SPEC's online-softmax kernel exercises dot_t, exp, rowmax,
    // rowsum, tmax, broadcast subtract/divide, and tile-scalar scaling.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [32], BC in [32])
        @aligned(Nq = BR, Nk = BC)
        kernel flash_attention(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D],
                               V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D], scale: f32) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            var m: tile<f32>[BR, 1] = -300000000.0
            var l: tile<f32>[BR, 1] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                let v = V[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                s = s * scale
                var mnew: tile<f32>[BR, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                var p: tile<f32>[BR, BC] = exp(s - mnew)
                var corr: tile<f32>[BR, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(p)
                acc = acc * corr
                acc += dot(p, v)
                m = mnew
            }
            acc = acc / l
            O[row :+ BR, :] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @flash_attention",
            "vector.contract",    // dot / dot_t
            "ex2.approx.ftz.f32", // exp lowers to the PTX ex2 intrinsic
            "arith.cmpf ogt",     // rowmax / tmax reductions
            "arith.divf",         // the broadcast normalize divide
            "arith.subf",         // broadcast subtraction inside exp
        ],
    );
}

#[test]
fn argsel_folds_a_value_index_pair_alongside_tmax() {
    // The greedy-argmax reduction's own shape: a running (value, index) pair
    // folded against a freshly loaded chunk and its own index tile, argsel
    // carrying the index side of what tmax alone can only do for the value.
    let mlir = emit_mlir(
        "@autotune(W in [8])
        kernel argmax_step(X: tensor<f32>[M, W], IDX: tensor<f32>[M, W]) {
            var v: tile<f32>[1, W] = -300000000.0
            var i: tile<f32>[1, W] = -1.0
            let x = X[0 :+ 1, :]
            let idx = IDX[0 :+ 1, :]
            i = argsel(x, v, idx, i)
            v = tmax(x, v)
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @argmax_step",
            "arith.cmpf oge", // argsel's own comparison
            "arith.select",   // argsel picks the winning index
            "arith.cmpf ogt", // tmax's comparison, unaffected by argsel
        ],
    );
}

#[test]
fn layernorm_lowers_sqrt_to_ptx_intrinsic() {
    // A LayerNorm-shaped body: mean and variance via rowsum, an
    // inverse-stddev via sqrt, and broadcast center/scale. sqrt must lower
    // to the PTX sqrt.approx intrinsic (like exp -> ex2.approx).
    let mlir = emit_mlir(
        "@launch(128)
        @autotune(BR in [32], W in [64])
        kernel layernorm(X: tensor<f32>[N, W], Y: tensor<f32>[N, W]) {
            let row = program_id(0) * BR
            var x: tile<f32>[BR, W] = 0.0
            x += X[row :+ BR, :]
            var s: tile<f32>[BR, 1] = rowsum(x)
            var mu: tile<f32>[BR, 1] = s / 64.0
            var xc: tile<f32>[BR, W] = x - mu
            var v: tile<f32>[BR, 1] = rowsum(xc * xc)
            var sd: tile<f32>[BR, 1] = sqrt(v / 64.0 + 0.00001)
            Y[row :+ BR, :] = xc / sd
        }",
    );
    assert_contains(&mlir, &["sqrt.approx.f32", "arith.divf", "arith.subf"]);
}

#[test]
fn log_lowers_to_the_ptx_base_two_intrinsic() {
    // softplus is the reason log exists: the GatedDeltaNet decay is
    // exp(rate * log(1 + exp(x))), so the gates cannot be computed on the
    // device without it. The hardware primitive is base two, so a natural
    // log is lg2 with the change of base folded in.
    let mlir = emit_mlir(
        "@launch(256)
        @autotune(TILE in [64])
        kernel softplus(X: tensor<f32>[M, N], Y: tensor<f32>[M, N]) {
            let p = program_id(0)
            var x = X[0 :+ 1, p * TILE :+ TILE]
            Y[0 :+ 1, p * TILE :+ TILE] = log(1.0 + exp(x))
        }",
    );
    assert_contains(
        &mlir,
        &["lg2.approx.ftz.f32", "ex2.approx.ftz.f32", "arith.mulf"],
    );
}

#[test]
fn unary_minus_negates_a_tile() {
    // `-t` on a tile lowers through the scalar-broadcast path as `0 - t`,
    // so a sigmoid written the obvious way compiles.
    let mlir = emit_mlir(
        "@launch(256)
        @autotune(TILE in [64])
        kernel neg(X: tensor<f32>[M, N], Y: tensor<f32>[M, N]) {
            let p = program_id(0)
            var x = X[0 :+ 1, p * TILE :+ TILE]
            Y[0 :+ 1, p * TILE :+ TILE] = x / (1.0 + exp(-x))
        }",
    );
    assert_contains(&mlir, &["arith.subf", "arith.divf", "ex2.approx"]);
}

#[test]
fn rowreduce_cooperates_via_warp_shuffles() {
    // 128 threads over 32 rows leaves four lanes per row: each lane
    // folds a strided quarter of the columns in a register and a
    // gpu.shuffle xor butterfly combines the partials, instead of one
    // thread sweeping all 64 columns while three quarters of the CTA
    // idles.
    let mlir = emit_mlir(
        "@launch(128)
        @autotune(BR in [32], BC in [64])
        kernel rmax(A: tensor<f32>[N, BC], R: tensor<f32>[N, 1]) {
            let pid = program_id(0)
            let row = pid * BR
            var t: tile<f32>[BR, BC] = 0.0
            t += A[row :+ BR, :]
            var r: tile<f32>[BR, 1] = rowmax(t)
            R[row :+ BR, :] = r
        }",
    );
    assert_contains(&mlir, &["gpu.shuffle", "arith.cmpf ogt"]);
}

#[test]
fn rowreduce_serial_without_spare_threads() {
    // One thread per row (128 rows, 128 threads) leaves no lanes to
    // cooperate, so the reduction stays the serial per-row sweep.
    let mlir = emit_mlir(
        "@launch(128)
        @autotune(BR in [128], BC in [64])
        kernel rsum(A: tensor<f32>[N, BC], R: tensor<f32>[N, 1]) {
            let pid = program_id(0)
            let row = pid * BR
            var t: tile<f32>[BR, BC] = 0.0
            t += A[row :+ BR, :]
            var r: tile<f32>[BR, 1] = rowsum(t)
            R[row :+ BR, :] = r
        }",
    );
    assert!(
        !mlir.contains("gpu.shuffle"),
        "warp shuffles emitted with no spare lanes per row:\n{mlir}"
    );
}

/// A nested per-element chain becomes one sweep. Before this, every call in
/// `i8(i32(round(a)))` staged a shared tile of its own, so a kernel doing
/// nothing else allocated four. See codegen/elemwise.rs.
#[test]
fn elementwise_chain_fuses_into_one_sweep() {
    let mlir = emit_mlir(
        "@launch(256)
        @autotune(NB in [32])
        kernel narrow(A: tensor<f32>[RB, 32], Q: tensor<i8>[RB, 32]) {
            let r = program_id(0)
            var a = A[r * NB :+ NB, 0 :+ 32]
            Q[r * NB :+ NB, 0 :+ 32] = i8(i32(round(a)))
        }",
    );
    // Only the operand is staged; the rounding, the i32 and the i8 are all
    // register steps of the one store sweep.
    assert_eq!(
        mlir.matches("memref.global").count(),
        1,
        "the chain should stage one tile, not one per call, in:\n{mlir}"
    );
    assert!(
        !mlir.contains("xi32, 3>") && !mlir.contains("xi8, 3>"),
        "no integer tile should be staged in:\n{mlir}"
    );
    assert_contains(&mlir, &["cvt.rni", "arith.fptosi"]);
}

/// The chain must not swallow a store whose operand it cannot index safely:
/// a masked operand would be read out of bounds by a target-indexed sweep,
/// so such a store keeps the old tile-per-call path and still verifies.
#[test]
fn elementwise_chain_leaves_a_masked_operand_alone() {
    let mlir = emit_mlir(
        "@launch(256)
        kernel narrow(A: tensor<f32>[M, 24], Q: tensor<i8>[M, 24]) {
            let r = program_id(0)
            var a = A[r :+ 1, 0 :+ 24]
            Q[r :+ 1, 0 :+ 24] = i8(i32(round(a)))
        }",
    );
    assert!(module_verifies(&mlir));
}

/// A float step after a conversion away from float is a type error, not a
/// silent reinterpretation.
#[test]
fn elementwise_chain_rejects_rounding_an_integer() {
    let err = std::panic::catch_unwind(|| {
        emit_mlir(
            "@launch(256)
            @autotune(NB in [32])
            kernel narrow(A: tensor<f32>[RB, 32], O: tensor<f32>[RB, 32]) {
                let r = program_id(0)
                var a = A[r * NB :+ NB, 0 :+ 32]
                O[r * NB :+ NB, 0 :+ 32] = round(i32(a))
            }",
        )
    });
    assert!(
        err.is_err(),
        "round of an i32 chain step should not compile"
    );
}
