// The standard library: every function compiles where a kernel would call
// it, and the ones a hand-written kernel already spells out compile to that
// kernel exactly.

use super::*;

/// A kernel around `body`, which reads `x`, a `[16, 64]` tile of `X`.
fn probe(body: &str) -> String {
    format!(
        "kernel k(X: tensor<f32>[N, 64], Y: tensor<f32>[N, 64], W: tensor<f32>[1, 64], Z: tensor<f32>[N, 1],
                  C: tensor<f32>[N, 32], S: tensor<f32>[N, 32], Q: tensor<i8>[N, 64],
                  I: tensor<i32>[N, 4], V: tensor<f32>[N, 4]) {{
            let r = ctx.block.x * 16
            var x = X[r :+ 16, :]
            {body}
        }}"
    )
}

/// One call of every public library function, each in a statement.
const CALLS: &[&str] = &[
    "Y[r :+ 16, :] = cdiv(x, 2.0) + round_up(x, 2.0) * f32(cdiv(extent(X, 0), 16))",
    "Y[r :+ 16, :] = square(x) + rcp(x) + clamp(x, -2.0, 2.0) + lerp(x, Y[r :+ 16, :], 0.25)",
    "Y[r :+ 16, :] = exp2(x) + log2(x) + sign(x) + erf(x)",
    "Y[r :+ 16, :] = relu(x) + relu6(x) + leaky_relu(x, 0.01) + hardtanh(x)",
    "Y[r :+ 16, :] = elu(x, 1.0) + selu(x) + celu(x, 2.0)",
    "Y[r :+ 16, :] = sigmoid(x) + log_sigmoid(x) + hardsigmoid(x)",
    "Y[r :+ 16, :] = silu(x) + swish(x, 1.5) + hardswish(x)",
    "Y[r :+ 16, :] = gelu(x) + gelu_tanh(x) + quick_gelu(x)",
    "Y[r :+ 16, :] = softplus(x) + mish(x) + softsign(x) + squared_relu(x)",
    "Y[r :+ 16, :] = swiglu(x, X[r :+ 16, :]) + geglu(x, x) + geglu_tanh(x, x) + reglu(x, x)",
    "Z[r :+ 16, :] = rowmean(x * x)",
    "Z[r :+ 16, :] = rowmin(x)",
    "softmax(x)\n Y[r :+ 16, :] = x",
    "log_softmax(x)\n Y[r :+ 16, :] = x",
    "logsumexp(Z[r :+ 16, :], x)",
    "mask_cols(x, 40, -1.0e30)\n causal_mask(x, r, 0)\n Y[r :+ 16, :] = x",
    "var vals: tile<f32>[16, 4] = 0.0
     var idx: tile<i32>[16, 4] = 0
     topk(vals, idx, x)
     V[r :+ 16, :] = vals
     I[r :+ 16, :] = idx",
    "rms_norm(x, W[0 :+ 1, :], 0.00001)\n Y[r :+ 16, :] = x",
    "layer_norm(x, W[0 :+ 1, :], 0.5, 0.00001)\n Y[r :+ 16, :] = x",
    "l2_normalize(x, 0.000001)\n Y[r :+ 16, :] = x",
    "rope(x, C[r :+ 16, :], S[r :+ 16, :])\n Y[r :+ 16, :] = x",
    "var q: tile<i8>[16, 64]
     quantize_i8(q, Z[r :+ 16, :], x)
     Q[r :+ 16, :] = q
     dequantize_i8(Y[r :+ 16, :], q, Z[r :+ 16, :])",
    "var acc: tile<f32>[16, 16] = 0.0
     matmul(acc, X, Y, r, 0, 16)
     matmul_t(acc, X, Y, r, 16, 16)
     V[r :+ 16, :] = acc[:, 0 :+ 4]",
    "var m: tile<f32>[16, 1] = -1.0e30
     var l: tile<f32>[16, 1] = 0.0
     var acc: tile<f32>[16, 64] = 0.0
     var s: tile<f32>[16, 16] = dot_t(x, X[0 :+ 16, :])
     online_softmax(acc, m, l, s, Y[0 :+ 16, :])
     var o: tile<f32>[16, 64] = 0.0
     attention(o, x, X, Y, 0.125, 16, 16)
     Y[r :+ 16, :] = acc + o",
];

#[test]
fn every_library_function_compiles() {
    for call in CALLS {
        emit_mlir(&probe(call));
        emit_mlir_sync(&probe(call), "sm_80");
    }
}

#[test]
fn every_library_function_is_probed() {
    let missing: Vec<_> = crate::stdlib::builtin()
        .into_iter()
        .filter(|f| !f.name.starts_with('_'))
        .filter(|f| !CALLS.iter().any(|c| c.contains(&format!("{}(", f.name))))
        .map(|f| f.name)
        .collect();
    assert!(missing.is_empty(), "no call in CALLS: {missing:?}");
}

const SGEMM: &str = "@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32], T in [256, 128])
    @launch(T)
    @pipeline
    @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
    kernel sgemm(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], alpha: f32, beta: f32) {
      let pm = ctx.block.x
      let pn = ctx.block.y
      var acc: tile<f32>[TILE_M, TILE_N] = 0.0
      LOOP
      let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
      C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
    }";

#[test]
fn matmul_is_the_k_loop_written_out() {
    let written = SGEMM.replace(
        "LOOP",
        "for kt in range(0, K, TILE_K) {
           var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
           var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
           acc += dot(a, b)
         }",
    );
    let called = SGEMM.replace("LOOP", "matmul(acc, A, B, pm * TILE_M, pn * TILE_N, TILE_K)");
    assert_eq!(emit_mlir(&called), emit_mlir(&written));
    assert_eq!(emit_mlir_sync(&called, "sm_80"), emit_mlir_sync(&written, "sm_80"));
}

const FLASH: &str = "@tensorcore
    @launch(128)
    @autotune(D in [64], BR in [64], BC in [64])
    @aligned(Nq = BR, Nk = BC)
    kernel flash(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], V: tensor<f16>[Nk, D],
                 O: tensor<f16>[Nq, D], scale: f32) {
      let row = ctx.block.x * BR
      let q = Q[row :+ BR, :]
      var acc: tile<f32>[BR, D] = 0.0
      BODY
      O[row :+ BR, :] = acc
    }";

/// Flash attention's loop, with `UPDATE` for the online softmax.
const FLASH_LOOP: &str = "var m: tile<f32>[BR, 1] = -1.0e30
      var l: tile<f32>[BR, 1] = 0.0
      for kt in range(0, Nk, BC) {
        let k = K[kt :+ BC, :]
        let v = V[kt :+ BC, :]
        var s: tile<f32>[BR, BC] = dot_t(q, k)
        s = s * scale
        s = where(iota(1) < Nk - kt, s, -1.0e30)
        UPDATE
      }
      acc = acc / l";

const ONLINE_SOFTMAX: &str = "var mnew: tile<f32>[BR, 1] = rowmax(s)
        mnew = tmax(m, mnew)
        s = exp(s - mnew)
        var corr: tile<f32>[BR, 1] = exp(m - mnew)
        l = l * corr
        l += rowsum(s)
        acc = acc * corr
        acc += dot(s, v)
        m = mnew";

#[test]
fn online_softmax_and_attention_are_flash_attention_written_out() {
    let written = FLASH.replace("BODY", &FLASH_LOOP.replace("UPDATE", ONLINE_SOFTMAX));
    let update = FLASH.replace("BODY", &FLASH_LOOP.replace("UPDATE", "online_softmax(acc, m, l, s, v)"));
    let whole = FLASH.replace("BODY", "attention(acc, q, K, V, scale, BR, BC)");
    for chip in ["sm_75", "sm_80"] {
        let want = emit_mlir_sync(&written, chip);
        assert_eq!(emit_mlir_sync(&update, chip), want, "online_softmax on {chip}");
        assert_eq!(emit_mlir_sync(&whole, chip), want, "attention on {chip}");
    }
}

#[test]
fn extent_folds_a_static_axis_and_reads_a_dynamic_one() {
    let src = |e: &str| {
        format!(
            "kernel k(X: tensor<f32>[N, 64], Y: tensor<f32>[N, 64]) {{
                var x = X[0 :+ 16, :]
                var t: tile<f32>[16, 64] = x * f32({e})
                Y[0 :+ 16, :] = t
            }}"
        )
    };
    assert_eq!(emit_mlir(&src("extent(x, 1)")), emit_mlir(&src("64")));
    assert_eq!(emit_mlir(&src("extent(x * x, 0)")), emit_mlir(&src("16")));
    assert_eq!(emit_mlir(&src("extent(X, 0)")), emit_mlir(&src("N")));
    assert_eq!(emit_mlir(&src("extent(X[0 :+ 8, 0 :+ 4], 1)")), emit_mlir(&src("4")));
    let err = |e: &str| emit_err(&src(e));
    assert!(err("extent(x, 2)").contains("axis 2 of a rank-2 value"));
    assert!(err("extent(x)").contains("takes a tile or a tensor and an axis"));
}

#[test]
fn a_function_sizes_a_tile_by_its_parameters() {
    let src = |f: &str, call: &str| {
        format!(
            "{f}
            kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {{
                {call}
            }}"
        )
    };
    let sized = src(
        "fn copy_rows(Y, X, R, C) {
            var t: tile<f32>[R, C] = X[0 :+ R, 0 :+ C]
            Y[0 :+ R, 0 :+ C] = t
        }",
        "copy_rows(Y, X, 8, 32)",
    );
    let written = src("", "var t: tile<f32>[8, 32] = X[0 :+ 8, 0 :+ 32]\n Y[0 :+ 8, 0 :+ 32] = t");
    assert_eq!(emit_mlir(&sized), emit_mlir(&written));
    let err = crate::parse(&src("fn f(R) { var t: tile<f32>[R, 4] = 0.0 }", "f(2 * 4)")).unwrap_err();
    assert!(err.to_string().contains("`R` sizes a tile"), "{err}");
}

#[test]
fn a_repeated_operand_is_staged_once() {
    let src = |e: &str| {
        format!(
            "kernel k(X: tensor<f32>[N, 64], Y: tensor<f32>[N, 64]) {{
                let r = ctx.block.x * 16
                Y[r :+ 16, :] = {e}
            }}"
        )
    };
    let staged = |e: &str| emit_mlir(&src(e)).matches("memref.load %subview").count();
    assert_eq!(staged("X[r :+ 16, :] * 2.0"), 1);
    assert_eq!(staged("X[r :+ 16, :] * X[r :+ 16, :]"), 1);
    assert_eq!(staged("gelu(X[r :+ 16, :])"), 1);
}
