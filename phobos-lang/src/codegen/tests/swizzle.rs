// Shared tiles the scaled int8 dot reads take an XOR-swizzled layout when
// their rows are a multiple of 128 bytes; `@swizzle` insists, `@swizzle(off)`
// declines.

use super::*;

/// A projection whose weight is staged into a declared `[64, 128]` tile and
/// whose activation is a pipelined stage of the same shape. `extra` goes
/// after the contraction.
fn projection(attrs: &str, extra: &str) -> String {
    let body: String = (0..4)
        .map(|s| {
            format!(
                "    let a{s} = at[0 :+ 64, {k} :+ 32]
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
{attrs}
@aligned(M = 64, N = 64, K = 128)
kernel proj(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
            W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
            C: tensor<f32>[M, N], E: tensor<i8>[N, K]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  var w: tile<i8>[64, 128]
  var acc: tile<f32>[64, 64] = 0.0
  for kt in range(0, K, 128) {{
    var at = A[pm * 64 :+ 64, kt :+ 128]
    w = W[pn * 64 :+ 64, kt :+ 128]
    barrier()
{body}{extra}  }}
  C[pm * 64 :+ 64, pn * 64 :+ 64] = acc
}}
"
    )
}

/// The bytes of the module's shared buffer.
fn shared_bytes(mlir: &str) -> i64 {
    mlir.lines()
        .filter(|l| l.contains("memref.global") && l.contains("xi8, 3>"))
        .filter_map(|l| l.split("memref<").nth(1)?.split('x').next()?.parse::<i64>().ok())
        .sum()
}

#[test]
fn tiles_the_scaled_dot_reads_are_swizzled_unpadded() {
    let on = emit_mlir_sync(&projection("@pipeline\n@padstage", ""), "sm_75");
    let off = emit_mlir_sync(&projection("@pipeline\n@padstage\n@swizzle(off)", ""), "sm_75");
    assert!(on.matches("arith.xori").count() > off.matches("arith.xori").count(), "{on}");
    // The stage's pad goes with the swizzle: 128-byte rows, not 144.
    assert_eq!(shared_bytes(&off) - shared_bytes(&on), 64 * 16);
    assert!(on.contains("ldmatrix"), "{on}");
}

#[test]
fn swizzle_insists_or_fails_with_the_reason() {
    assert!(emit_mlir_sync(&projection("@swizzle", ""), "sm_75").contains("ldmatrix"));
    // A plain copy out of the tile would read it unpermuted.
    let e = emit_err(&projection("@swizzle(on)", "    E[pn * 64 :+ 64, kt :+ 128] = w\n"));
    assert!(e.contains("cannot swizzle tile 'w'"), "{e}");
    // Without the attribute the same kernel compiles, its `w` unswizzled.
    emit_mlir(&projection("", "    E[pn * 64 :+ 64, kt :+ 128] = w\n"));
}

#[test]
fn swizzle_with_nothing_to_swizzle_fails() {
    // 64-byte rows, whose eight-row fragment loads a swizzle cannot spread.
    let narrow = "@launch(128)
@swizzle
@aligned(M = 64, N = 64, K = 64)
kernel proj(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
            W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
            C: tensor<f32>[M, N]) {
  var w: tile<i8>[64, 64]
  var acc: tile<f32>[64, 64] = 0.0
  for kt in range(0, K, 64) {
    w = W[ctx.block.y * 64 :+ 64, kt :+ 64]
    let a = A[ctx.block.x * 64 :+ 64, kt :+ 32]
    let da = AS[ctx.block.x * 64 :+ 64, kt / 32 :+ 1]
    let ws = WS[kt / 32 :+ 1, ctx.block.y * 64 :+ 64]
    acc += f32(dot_t(a, w[0 :+ 64, 0 :+ 32])) * ws * da
  }
  C[ctx.block.x * 64 :+ 64, ctx.block.y * 64 :+ 64] = acc
}
";
    let e = emit_err(narrow);
    assert!(e.contains("@swizzle: no shared tile"), "{e}");
    let e = emit_err(&projection("@swizzle(sideways)", ""));
    assert!(e.contains("@swizzle takes on or off"), "{e}");
}
