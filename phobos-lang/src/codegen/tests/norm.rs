use super::*;

/// `rms_norm_q_t` over a row of `blocks` Q8_0 blocks at `cta` threads,
/// with or without the normalized row written out.
fn norm_src(blocks: usize, cta: usize, with_out: bool) -> String {
    let out = match with_out {
        true => "O[0 :+ NB, 0 :+ 32], ",
        false => "",
    };
    format!(
        "@launch({cta})
@autotune(NB in [{blocks}])
@aligned(RB = NB, MB = NB, D1 = 1)
kernel norm(X: tensor<f32>[RB, 32], G: tensor<f32>[MB, 32], O: tensor<f32>[RB, 32],
            Q: tensor<i8>[RB, 32], S: tensor<f32>[RB, D1]) {{
  rms_norm_q_t(X[0 :+ NB, 0 :+ 32], G[0 :+ NB, 0 :+ 32], 0.000001, {out}Q[0 :+ NB, 0 :+ 32], S[0 :+ NB, 0 :+ 1])
}}
"
    )
}

#[test]
fn a_row_of_whole_stripes_gates_nothing() {
    // 1024 elements at 256 threads: one stripe, every piece present.
    let mlir = emit_mlir(&norm_src(32, 256, true));
    assert_contains(&mlir, &["gpu.shuffle", "vector.load", "vector.store"]);
    assert!(!mlir.contains("arith.cmpi ult"), "{mlir}");
}

#[test]
fn a_partial_last_stripe_gates_its_loads_and_stores() {
    // 2560 elements at 256 threads: two and a half stripes, so the third
    // stripe's pieces are gated on their element existing.
    let mlir = emit_mlir(&norm_src(80, 256, true));
    assert_contains(&mlir, &["gpu.shuffle", "arith.cmpi ult", "scf.if"]);
    // The gated loads yield zeros past the end. The shuffles are not gated.
    let (before, _) = mlir.split_once("gpu.shuffle").expect("a shuffle");
    assert!(before.contains("scf.if"), "{mlir}");
}

#[test]
fn the_five_operand_form_writes_no_normalized_row() {
    let with = emit_mlir(&norm_src(80, 256, true));
    let without = emit_mlir(&norm_src(80, 256, false));
    assert!(with.matches("vector.store").count() > without.matches("vector.store").count(), "{without}");
    assert_contains(&without, &["gpu.shuffle", "arith.cmpi ult"]);
}

#[test]
fn the_gated_form_reads_the_gate_and_applies_silu() {
    // One 128-element head at one warp, as a gated delta layer's norm runs.
    let src = "@launch(32)
@autotune(NB in [4])
@aligned(RB = NB, MB = NB, D1 = 1)
kernel norm(X: tensor<f32>[RB, 32], Z: tensor<f32>[RB, 32], G: tensor<f32>[MB, 32],
            O: tensor<f32>[RB, 32], Q: tensor<i8>[RB, 32], S: tensor<f32>[RB, D1]) {
  let r = program_id(0)
  rms_norm_gated_q_t(X[r * NB :+ NB, 0 :+ 32], Z[r * NB :+ NB, 0 :+ 32], G[0 :+ NB, 0 :+ 32], 0.000001,
                     O[r * NB :+ NB, 0 :+ 32], Q[r * NB :+ NB, 0 :+ 32], S[r * NB :+ NB, 0 :+ 1])
}
";
    let gated = emit_mlir(src);
    let plain = emit_mlir(&norm_src(4, 32, true));
    // One more 16-byte load per piece for the gate, and an exponential.
    assert!(gated.matches("vector.load").count() > plain.matches("vector.load").count(), "{gated}");
    assert!(gated.contains("ex2.approx") || gated.contains("math.exp"), "{gated}");
}
