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
    // Two butterflies of five steps for each of a warp's two columns, the
    // lane's rows of k and q as 16-byte loads, and no shared memory.
    assert_eq!(mlir.matches("gpu.shuffle").count(), 20, "{mlir}");
    assert_contains(&mlir, &["vector.load", "scf.for"]);
    assert!(!mlir.contains("memref.alloc") && !mlir.contains("workgroup"), "{mlir}");
}

#[test]
fn the_view_must_match_the_warps() {
    // 64 threads are two warps, which own four columns, not eight.
    let err = emit_err(&scan_src(64, 128, 8));
    assert!(err.contains("must match"), "{err}");
}
