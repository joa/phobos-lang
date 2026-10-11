// A dot fused with the update of its product.

use super::*;

/// Attention's two updates of a product. `ELEM` is the operands' type.
fn attention_updates(attrs: &str, elem: &str) -> String {
    format!(
        "{attrs}
        @launch(128)
        kernel k(Q: tensor<{elem}>[16, 64], K: tensor<{elem}>[16, 64], V: tensor<{elem}>[16, 64],
                 C: tensor<f32>[16, 1], O: tensor<f32>[16, 64]) {{
            var q = Q[0 :+ 16, 0 :+ 64]
            var k = K[0 :+ 16, 0 :+ 64]
            var v = V[0 :+ 16, 0 :+ 64]
            var c = C[0 :+ 16, 0 :+ 1]
            var acc: tile<f32>[16, 64] = 1.0
            var s: tile<f32>[16, 16] = dot_t(q, k)
            s = s * 0.125
            acc = acc * c + dot(s, v)
            O[0 :+ 16, 0 :+ 64] = acc
        }}"
    )
}

#[test]
fn the_fma_path_writes_the_update_out_with_the_product() {
    let src = attention_updates("", "f32");
    let fused = emit_mlir(&src);
    // The same update as a sweep of its own, then an accumulating dot.
    let split = emit_mlir(&src.replace("acc = acc * c + dot(s, v)", "acc = acc * c
            acc += dot(s, v)"));
    assert!(!fused.contains("subgroup_mma"), "{fused}");
    let loops = |m: &str| m.matches("scf.for").count();
    assert_eq!(loops(&fused) + 1, loops(&split), "{fused}
{split}");
}

#[test]
fn the_tensor_cores_take_the_dot_and_then_the_update_sweep() {
    let mlir = emit_mlir(&attention_updates("@tensorcore", "f16"));
    assert!(mlir.contains("gpu.subgroup_mma_compute"), "{mlir}");
}
