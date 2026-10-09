use super::*;

/// `delta_scan_t` over one head's `D` rows and `C` state columns at `threads`.
fn scan_src(threads: usize, d: usize, c: usize) -> String {
    format!(
        "@launch({threads})
@autotune(D in [{d}], C in [{c}])
@aligned(SD = D, SW = C, W = D, GW = 1)
kernel scan(Q: tensor<f32>[N, W], K: tensor<f32>[N, W], V: tensor<f32>[N, W],
            DEC: tensor<f32>[N, GW], BET: tensor<f32>[N, GW],
            S: tensor<f32>[SD, SW], O: tensor<f32>[N, W]) {{
  let h = program_id(0)
  let jn = program_id(1)
  let q = Q[:, h * D :+ D]
  let k = K[:, h * D :+ D]
  let v = V[:, h * D + jn * C :+ C]
  let o = O[:, h * D + jn * C :+ C]
  let dec = DEC[:, h :+ 1]
  let bet = BET[:, h :+ 1]
  let st = S[h * D :+ D, jn * C :+ C]
  delta_scan_t(q, k, v, dec, bet, st, o)
}}
"
    )
}

#[test]
fn the_state_stays_in_registers() {
    let mlir = emit_mlir(&scan_src(64, 128, 4));
    // Two warps of two columns: a lane holds one column of a 16-lane row
    // group, so each of the two butterflies takes four steps. The lane's rows
    // of k and q come as 16-byte loads, and nothing goes to shared memory.
    assert_eq!(mlir.matches("gpu.shuffle").count(), 8, "{mlir}");
    assert_contains(&mlir, &["vector.load", "scf.for"]);
    assert!(!mlir.contains("memref.alloc") && !mlir.contains("workgroup"), "{mlir}");
}

#[test]
fn a_warp_can_own_more_columns() {
    // Four columns a warp leave eight row groups, so three steps a
    // butterfly: more columns cost fewer shuffles.
    let mlir = emit_mlir(&scan_src(64, 128, 8));
    assert_eq!(mlir.matches("gpu.shuffle").count(), 6, "{mlir}");
}

#[test]
fn the_view_must_split_over_the_warps() {
    // Three warps cannot split eight columns, and sixteen each is too many.
    for (threads, c) in [(96, 8), (64, 32)] {
        let err = emit_err(&scan_src(threads, 128, c));
        assert!(err.contains("split evenly"), "{err}");
    }
}
