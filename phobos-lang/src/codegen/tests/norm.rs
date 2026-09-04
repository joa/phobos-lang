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
    // 2560 elements at 256 threads: two stripes and a half, so the third
    // stripe's pieces are gated on their element existing.
    let mlir = emit_mlir(&norm_src(80, 256, true));
    assert_contains(&mlir, &["gpu.shuffle", "arith.cmpi ult", "scf.if"]);
    // The gated loads yield zeros past the end; the shuffles are not gated.
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
