// `generate(f)`: a tile computed where it is read. As a fragment
// accumulator's scaled weight each lane evaluates its own B fragments; any
// other read materializes it through shared memory.

use super::*;

/// A blocked int8 projection whose weight is `W` itself, generated a word at
/// a time, with `reads` after the declaration.
fn projection(reads: &str) -> String {
    format!(
        "fn wq(j, q) {{
  let v = word(W[pn * 64 + j, kt + 4 * q :+ 4], 0)
  v
}}

@launch(128)
@aligned(M = 128, N = 64, K = 32)
kernel proj(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
            W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
            C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  var acc: tile<f32>[128, 64] = 0.0
  for kt in range(0, K, 32) {{
    let b = kt / 32
    let a = A[pm * 128 :+ 128, kt :+ 32]
    let w: tile<i8>[64, 32] = generate(wq)
    let ws = WS[b :+ 1, pn * 64 :+ 64]
    let da = AS[pm * 128 :+ 128, b :+ 1]
{reads}  }}
  C[pm * 128 :+ 128, pn * 64 :+ 64] = acc
}}
"
    )
}

#[test]
fn a_generated_weight_feeds_the_fragments_with_no_shared_tile() {
    let src = projection("    acc += f32(dot_t(a, w)) * ws * da\n");
    for (mlir, shape, mmas) in [
        (emit_mlir(&src), "mmaShape = [8, 8, 16]", 64),
        (emit_with(&src, "sm_120", 64), "mmaShape = [16, 8, 32]", 16),
    ] {
        assert!(!mlir.contains("memref.global"), "{mlir}");
        assert!(!mlir.contains("gpu.barrier"), "{mlir}");
        assert_eq!(mlir.matches(shape).count(), mmas, "{mlir}");
        // Each lane loads the words of its own B fragments from `W`.
        assert!(mlir.contains("vector<4xi8>"), "{mlir}");
    }
}

#[test]
fn any_other_read_materializes_the_tile() {
    let src = projection(
        "    let w0 = w[0 :+ 64, 0 :+ 32]
    acc += f32(dot_t(a, w0)) * ws * da
",
    );
    let mlir = emit_mlir(&src);
    assert!(mlir.contains("memref.global"), "{mlir}");
    assert!(mlir.contains("gpu.barrier"), "{mlir}");
}

#[test]
fn an_accumulator_kept_in_shared_memory_materializes_its_weight() {
    // A row sum of `acc` is not a read the fragments hold, so the tile stays
    // in shared memory and the dot reads a weight written out first. At 64
    // rows the tile fits the 48 KB a block may hold.
    let src = projection("    acc += f32(dot_t(a, w)) * ws * da\n")
        .replace("M = 128", "M = 64")
        .replace("[128, 64]", "[64, 64]")
        .replace("pm * 128 :+ 128", "pm * 64 :+ 64")
        .replace("  C[pm * 64", "  var s: tile<f32>[64, 1] = rowsum(acc)\n  C[pm * 64");
    let mlir = emit_mlir(&src);
    let carried = format!("-> ({})", vec!["f32"; 32].join(", "));
    assert!(!mlir.contains(&carried), "{mlir}");
    assert!(mlir.contains("memref.global"), "{mlir}");
    assert!(mlir.contains("gpu.barrier"), "{mlir}");
    assert!(mlir.contains("memref<64x32xi8, 3>, vector<4xi8>"), "the weight is stored a word at a time:\n{mlir}");
}

#[test]
fn generate_takes_a_function_ending_in_its_value() {
    let src = projection("    acc += f32(dot_t(a, w)) * ws * da\n").replace("  v\n}", "  let u = v\n}");
    let err = crate::parse(&src).expect_err("a body ending in a `let` has no value").to_string();
    assert!(err.contains("must end in an expression"), "got: {err}");
}

#[test]
fn generated_scales_and_outer_operands_stay_in_registers() {
    // The scales computed per lane too: a row from `WS`, a column from
    // `AS`, read as the scaled dot's and the outer product's operands.
    let src = projection(
        "    let wsg: tile<f32>[1, 64] = generate(wsf)
    let dag: tile<f32>[128, 1] = generate(daf)
    acc += f32(dot_t(a, w)) * wsg * dag
    acc -= dag * wsg
",
    )
    .replace(
        "@launch(128)",
        "fn wsf(i, j) {
  WS[b, pn * 64 + j]
}

fn daf(i, j) {
  AS[pm * 128 + i, b]
}

@launch(128)",
    );
    let mlir = emit_mlir(&src);
    assert!(!mlir.contains("memref.global"), "{mlir}");
    assert!(!mlir.contains("gpu.barrier"), "{mlir}");
    assert_eq!(mlir.matches("mmaShape = [8, 8, 16]").count(), 64, "{mlir}");
}
