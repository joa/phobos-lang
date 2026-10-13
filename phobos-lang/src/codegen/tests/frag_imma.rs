// The int8 fragment accumulator: `acc += f32(dot_t(a, w)) * row * col` kept
// in the integer tensor cores' output registers across its loop.

use super::*;

/// A blocked int8 projection over 32-element blocks, with `scales` the
/// product of the two scale broadcasts as written in the update.
fn projection(m: i64, n: i64, threads: i64, aligned: &str, scales: &str) -> String {
    format!(
        "@launch({threads})
{aligned}
kernel proj(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
            W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
            C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  var acc: tile<f32>[{m}, {n}] = 0.0
  for kt in range(0, K, 32) {{
    let b = kt / 32
    let a = A[pm * {m} :+ {m}, kt :+ 32]
    let w = W[pn * {n} :+ {n}, kt :+ 32]
    let ws = WS[b :+ 1, pn * {n} :+ {n}]
    let da = AS[pm * {m} :+ {m}, b :+ 1]
    acc += f32(dot_t(a, w)) * {scales}
  }}
  C[pm * {m} :+ {m}, pn * {n} :+ {n}] = acc
}}
"
    )
}

const ALIGNED: &str = "@aligned(M = 128, N = 64, K = 32)";

#[test]
fn scaled_int8_accumulator_rides_in_registers() {
    // 128x64 over four warps: each warp owns a 32x64 patch of 8x8 tiles,
    // 32 tiles of two elements a lane, carried by the k loop.
    let src = projection(128, 64, 128, ALIGNED, "ws * da");
    for mlir in [emit_mlir(&src), emit_mlir_sync(&src, "sm_120")] {
        assert!(
            !mlir.contains("memref.global"),
            "int8 accumulator materialized in shared memory:\n{mlir}"
        );
        assert_eq!(mlir.matches("mmaShape = [8, 8, 16]").count(), 64, "{mlir}");
        assert_eq!(mlir.matches("math.fma").count(), 64, "{mlir}");
        let carried = format!("-> ({})", vec!["f32"; 64].join(", "));
        assert!(mlir.contains(&carried), "k loop does not carry 64 f32:\n{mlir}");
        // Global operands: the trailing barrier has nothing to order.
        assert!(!mlir.contains("gpu.barrier"), "{mlir}");
    }
}

#[test]
fn scales_ride_in_either_order() {
    for scales in ["ws * da", "da * ws"] {
        let mlir = emit_mlir(&projection(128, 64, 128, ALIGNED, scales));
        assert!(!mlir.contains("memref.global"), "{scales}:\n{mlir}");
        assert_eq!(mlir.matches("math.fma").count(), 64, "{scales}:\n{mlir}");
    }
}

#[test]
fn partial_operands_keep_the_shared_accumulator() {
    // Without M and N aligned the slices may be partial, and the fragments
    // are read with no bounds guard.
    let mlir = emit_mlir(&projection(128, 64, 128, "@aligned(K = 32)", "ws * da"));
    assert!(mlir.contains("memref.global"), "{mlir}");
    assert!(!mlir.contains("math.fma"), "{mlir}");
}

#[test]
fn a_wider_update_keeps_the_shared_accumulator() {
    let mlir = emit_mlir(&projection(128, 64, 128, ALIGNED, "ws * da + 1.0"));
    assert!(mlir.contains("memref.global"), "{mlir}");
}

#[test]
fn a_shared_operand_orders_its_rewrite() {
    // w is rewritten each iteration while other warps may still read the
    // last one's fragments, so the op keeps its trailing barrier.
    let src = "@launch(128)
@aligned(M = 64, N = 64, K = 32)
kernel tcs(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
           W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
           C: tensor<f32>[M, N]) {
  let pm = ctx.block.x
  let pn = ctx.block.y
  var acc: tile<f32>[64, 64] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[pm * 64 :+ 64, kt :+ 32]
    var w: tile<i8>[64, 32] = W[pn * 64 :+ 64, kt :+ 32]
    let da = AS[pm * 64 :+ 64, b :+ 1]
    let ws = WS[b :+ 1, pn * 64 :+ 64]
    acc += f32(dot_t(a, w)) * da * ws
  }
  C[pm * 64 :+ 64, pn * 64 :+ 64] = acc
}
";
    let mlir = emit_mlir(src);
    assert!(mlir.contains("math.fma"), "{mlir}");
    let yield_at = mlir.find("scf.yield").expect("the k loop yields");
    let before = mlir[..yield_at].trim_end();
    assert!(before.ends_with("gpu.barrier"), "no barrier before the next rewrite:\n{mlir}");
}
