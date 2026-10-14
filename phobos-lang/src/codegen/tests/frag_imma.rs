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
    // 32 tiles of two elements a lane, carried by the k loop. Blackwell
    // takes two row tiles over both k steps in one m16n8k32.
    let src = projection(128, 64, 128, ALIGNED, "ws * da");
    for (mlir, shape, mmas) in [
        (emit_mlir(&src), "mmaShape = [8, 8, 16]", 64),
        (emit_mlir_sync(&src, "sm_120"), "mmaShape = [16, 8, 32]", 16),
    ] {
        assert!(
            !mlir.contains("memref.global"),
            "int8 accumulator materialized in shared memory:\n{mlir}"
        );
        assert_eq!(mlir.matches(shape).count(), mmas, "{mlir}");
        assert_eq!(mlir.matches("math.fma").count(), 64, "{mlir}");
        let carried = format!("-> ({})", vec!["f32"; 64].join(", "));
        assert!(mlir.contains(&carried), "k loop does not carry 64 f32:\n{mlir}");
        // Global operands: the trailing barrier has nothing to order.
        assert!(!mlir.contains("gpu.barrier"), "{mlir}");
    }
}

#[test]
fn a_single_k_step_pairs_row_tiles_on_blackwell() {
    // A scale every 16 elements: one k step an update, so Blackwell pairs
    // the row tiles in a whole m16n8k16 rather than two half-used ones.
    let src = projection(128, 64, 128, ALIGNED, "ws * da").replace(
        "    acc += f32(dot_t(a, w)) * ws * da\n",
        "    let a0 = a[0 :+ 128, 0 :+ 16]
    let w0 = w[0 :+ 64, 0 :+ 16]
    acc += f32(dot_t(a0, w0)) * ws * da
    let a1 = a[0 :+ 128, 16 :+ 16]
    let w1 = w[0 :+ 64, 16 :+ 16]
    acc += f32(dot_t(a1, w1)) * ws * da
",
    );
    let mlir = emit_mlir_sync(&src, "sm_120");
    assert_eq!(mlir.matches("mmaShape = [16, 8, 16]").count(), 32, "{mlir}");
    assert!(!mlir.contains("mmaShape = [8, 8, 16]"), "{mlir}");
    assert_eq!(mlir.matches("math.fma").count(), 128, "{mlir}");
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

#[test]
fn past_the_warp_tile_cap_keeps_the_shared_accumulator() {
    // 256x128 over four warps is 128 tiles a warp, past the 64 a warp
    // carries without spilling.
    let mlir = emit_mlir(&projection(256, 128, 128, "@aligned(M = 256, N = 128, K = 32)", "ws * da"));
    assert!(!mlir.contains("math.fma"), "{mlir}");
}

/// A weight staged into a shared tile once per 64 elements of k, then
/// contracted 32 at a time through `updates` slices of it.
fn staged_slices(updates: usize) -> String {
    let body: String = (0..updates)
        .map(|s| {
            format!(
                "    let a{s} = A[pm * 64 :+ 64, kt + {k} :+ 32]
    let da{s} = AS[pm * 64 :+ 64, kt / 32 + {s} :+ 1]
    let w{s} = w[0 :+ 64, {k} :+ 32]
    let ws{s} = WS[kt / 32 + {s} :+ 1, pn * 64 :+ 64]
    acc += f32(dot_t(a{s}, w{s})) * ws{s} * da{s}
",
                k = 32 * s
            )
        })
        .collect();
    format!(
        "@launch(128)
@aligned(M = 64, N = 64, K = 64)
kernel staged(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
              C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  var w: tile<i8>[64, 64]
  var acc: tile<f32>[64, 64] = 0.0
  for kt in range(0, K, 64) {{
    w = W[pn * 64 :+ 64, kt :+ 64]
{body}  }}
  C[pm * 64 :+ 64, pn * 64 :+ 64] = acc
}}
"
    )
}

#[test]
fn slices_of_a_shared_operand_add_no_barriers() {
    // A view touches no memory, so the second update needs no barrier the
    // first did not: one orders the copy before the reads, one the reads
    // before the next copy.
    let one = emit_mlir(&staged_slices(1));
    let two = emit_mlir(&staged_slices(2));
    assert_eq!(two.matches("math.fma").count(), 2 * one.matches("math.fma").count(), "{two}");
    assert_eq!(
        two.matches("gpu.barrier").count(),
        one.matches("gpu.barrier").count(),
        "{two}"
    );
}

#[test]
fn a_shared_operand_loads_by_ldmatrix() {
    // Four 8-row tiles of the shared operand a warp reads are one
    // ldmatrix.x4 per 16 elements of k: nvgpu's op at 64-bit indices, nvvm's
    // on the shared address at 32, where nvgpu's does not lower.
    let src = staged_slices(1);
    assert!(emit_mlir_sync(&src, "sm_75").contains("nvgpu.ldmatrix"));
    let narrow = emit_mlir(&src);
    assert!(narrow.contains("nvvm.ldmatrix"), "{narrow}");
    assert!(narrow.contains("math.fma"), "{narrow}");
}

#[test]
fn a_column_times_a_row_adds_to_the_accumulator_in_registers() {
    // The minimum a K-quant subtracts: `acc += rs * mw`, a [M, 1] column
    // times a [1, N] row, one more fma an element, in either order.
    for outer in ["rs * mw", "mw * rs"] {
        let scales = format!(
            "ws * da
    let rs = AS[pm * 128 :+ 128, b :+ 1]
    let mw = WS[b :+ 1, pn * 64 :+ 64]
    acc += {outer}"
        );
        let mlir = emit_mlir(&projection(128, 64, 128, ALIGNED, &scales));
        assert!(!mlir.contains("memref.global"), "{outer}:\n{mlir}");
        assert_eq!(mlir.matches("math.fma").count(), 128, "{outer}:\n{mlir}");
    }
    // `-=` subtracts it, the row's value negated once.
    let scales = "ws * da
    let rs = AS[pm * 128 :+ 128, b :+ 1]
    let mw = WS[b :+ 1, pn * 64 :+ 64]
    acc -= rs * mw";
    let mlir = emit_mlir(&projection(128, 64, 128, ALIGNED, scales));
    assert!(!mlir.contains("memref.global"), "{mlir}");
    assert_eq!(mlir.matches("math.fma").count(), 128, "{mlir}");
    assert!(mlir.contains("arith.negf"), "{mlir}");
    // Two columns are no outer product: the accumulator stays a tile.
    let scales = "ws * da
    let rs = AS[pm * 128 :+ 128, b :+ 1]
    acc += rs * da";
    assert!(emit_mlir(&projection(128, 64, 128, ALIGNED, scales)).contains("memref.global"));
}

#[test]
fn a_let_in_the_loop_keeps_the_accumulator_in_registers() {
    // `kt` is bound inside the loop, so its slices' bounds are checked
    // through what it stands for.
    let src = projection(128, 64, 128, ALIGNED, "ws * da").replace(
        "  for kt in range(0, K, 32) {
    let b = kt / 32",
        "  for kq in range(0, K / 32, 1) {
    let kt = kq * 32
    let b = kt / 32",
    );
    let mlir = emit_mlir(&src);
    assert!(!mlir.contains("memref.global"), "{mlir}");
    assert_eq!(mlir.matches("mmaShape = [8, 8, 16]").count(), 64, "{mlir}");
}
