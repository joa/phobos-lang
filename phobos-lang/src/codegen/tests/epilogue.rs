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

/// The register GEMM over a `tm x tn` output tile, with `alpha * acc + beta * C`.
fn scaled_gemm(tm: i64, tn: i64) -> String {
    format!(
        "@autotune(TILE_M in [{tm}], TILE_N in [{tn}], TILE_K in [8])
        @pipeline
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel gemm(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], alpha: f32, beta: f32) {{
            let pm = ctx.block.x
            let pn = ctx.block.y
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {{
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }}
            let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
        }}"
    )
}

#[test]
fn a_surplus_warp_of_the_register_gemm_leaves_c_to_its_block_owner() {
    // 32x64 is 64 sub-tiles of 8x4 for 256 threads: six warps clamp onto
    // the last block, and only the owner may read and write its C.
    let stores = |mlir: &str| mlir.matches("arith.cmpi eq").count();
    let surplus = emit_mlir(&scaled_gemm(32, 64));
    assert!(module_verifies(&surplus), "{surplus}");
    let whole = emit_mlir(&scaled_gemm(64, 128));
    assert_eq!(stores(&surplus), stores(&whole) + 1, "{surplus}");
}
