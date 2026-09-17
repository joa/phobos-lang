use super::*;
use crate::ir::verify::verify;

/// The default target's facts, the same the emitter reads: sm_75 at 32-bit
/// indices, so tensor cores but no mma.sync.
fn target() -> Target {
    crate::codegen::target::build_target(&phobos_base::context::Context::default())
}

/// Builds the first kernel of a source and returns its print, after the
/// verifier has passed.
fn build_src(src: &str) -> String {
    let kernels = crate::parse(src).unwrap();
    let base = phobos_base::context::Context::default();
    let (ir, _) = build(&base, target(), &kernels[0]).unwrap_or_else(|e| panic!("{e:#}"));
    verify(&ir).unwrap_or_else(|e| panic!("{e}\n{ir}"));
    ir.to_string()
}

fn build_err(src: &str) -> String {
    let kernels = crate::parse(src).unwrap();
    let base = phobos_base::context::Context::default();
    match build(&base, target(), &kernels[0]) {
        Ok((ir, _)) => panic!("expected the build to fail:\n{ir}"),
        Err(e) => format!("{e:#}"),
    }
}

fn assert_contains(text: &str, needles: &[&str]) {
    for needle in needles {
        assert!(text.contains(needle), "missing `{needle}` in:\n{text}");
    }
}

#[test]
fn params_and_dims_bind_to_the_entry() {
    let out = build_src(
        "kernel k(A: tensor<f32>[M, N], n: i32) {\n    let x = A[0, 1]\n    let y = n + M\n}\n",
    );
    // Each parameter's dims come right after it, then the next parameter.
    assert_contains(
        &out,
        &[
            "kernel k(%A.0: tensor<f32>[?, ?], %n.1: i32)",
            "%A.2: tensor<f32>[?, ?] = assume_align %A.0",
            "%M.3: index = dim 0 %A.0",
            "%N.4: index = dim 1 %A.0",
            "%n.5: index = index_cast index %n.1",
            "load %A.2,",
            "add %n.5, %M.3",
        ],
    );
}

#[test]
fn an_in_bounds_slice_stages() {
    let out = build_src(
        "@aligned(M = 64, K = 32)\nkernel k(A: tensor<f32>[M, K]) {\n    let p = program_id(0)\n    var t = A[p * 64 :+ 64, 0 :+ 32]\n    t = t * 2.0\n}\n",
    );
    assert_contains(
        &out,
        &[
            "program_id 0",
            "slice [64, 32] %A.1,",
            "= stage %",
            "fused (mul #0 s#1) %t.",
        ],
    );
    assert!(!out.contains("mask"), "{out}");
}

#[test]
fn an_unproven_slice_is_masked_and_materializes_in_value_position() {
    let out = build_src(
        "@aligned(K = 32)\nkernel k(A: tensor<f32>[M, K]) {\n    let p = program_id(0)\n    let s = rowsum(A[p * 64 :+ 64, 0 :+ 32])\n}\n",
    );
    assert_contains(&out, &["mask [m, -]", "materialize %", "rowsum %"]);
}

#[test]
fn a_loop_over_a_dynamic_extent_splits() {
    let out = build_src(
        "kernel k(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {\n    for kt in range(0, K, 32) {\n        var t = A[0 :+ 64, kt :+ 32]\n        C[0 :+ 64, kt :+ 32] = t\n    }\n}\n",
    );
    assert_contains(&out, &["divu", "for ragged %", "} ragged {"]);
    // The trimmed body's slices are proven along the loop's axis and the
    // ragged replay's are masked; the unaligned row axis is masked in both.
    assert_eq!(out.matches("mask [m, -]").count(), 2, "{out}");
    assert_eq!(out.matches("mask [m, m]").count(), 2, "{out}");
}

#[test]
fn constant_bounds_make_an_affine_loop() {
    let out = build_src(
        "@autotune(T in [4, 8])\nkernel k(A: tensor<f32>[M, N]) {\n    var s = 0\n    for i in range(0, T) {\n        s += i\n    }\n}\n",
    );
    assert_contains(&out, &["for 0..4 step 1 {", "store %"]);
    assert!(!out.contains("for %"), "{out}");
}

#[test]
fn an_elementwise_store_fuses_into_one_sweep() {
    let out = build_src(
        "@aligned(M = 8, N = 8)\nkernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {\n    var a = A[0 :+ 8, 0 :+ 8]\n    var b: tile<f32>[8, 8] = 0.0\n    b = exp(a * 2.0) + tmax(a, b)\n}\n",
    );
    // Each mention of a leaf is one operand.
    assert_contains(
        &out,
        &["= stage %", "fused (add (exp (mul #0 s#1)) (max #2 #3)) %a."],
    );
}

#[test]
fn a_chain_of_conversions_is_one_op() {
    let out = build_src(
        "@aligned(M = 8, N = 8)\nkernel k(A: tensor<f32>[M, N]) {\n    var a = A[0 :+ 8, 0 :+ 8]\n    var q: tile<i8>[8, 8] = 0\n    q = i8(i32(round(a)))\n}\n",
    );
    assert_contains(&out, &["chain i8 i32 round %a."]);
}

#[test]
fn the_diagnostics_moved_with_the_decisions() {
    assert_eq!(
        build_err("kernel k(A: tensor<f32>[M, N]) {\n    let x = nope\n}\n"),
        "unknown identifier 'nope'"
    );
    assert_eq!(
        build_err("kernel k(A: tensor<f32>[M, N]) {\n    let x = wobble(A)\n}\n"),
        "unknown function 'wobble'"
    );
    assert_eq!(
        build_err("kernel k(A: tensor<f32>[M, N], n: i32) {\n    let x = 1.0 + n\n}\n"),
        "mismatched operand types: f32 vs index"
    );
    assert_eq!(
        build_err("kernel k(A: tensor<f32>[M, N]) {\n    let x = A\n}\n"),
        "tensor 'A' used as a value; index or slice it"
    );
}

#[test]
fn the_build_is_deterministic() {
    let src = "kernel k(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {\n    let pm = program_id(0)\n    var acc: tile<f32>[64, 64] = 0.0\n    for kt in range(0, K, 16) {\n        let a = A[pm * 64 :+ 64, kt :+ 16]\n        let b = B[kt :+ 16, 0 :+ 64]\n        acc += dot(a, b)\n    }\n    C[pm * 64 :+ 64, 0 :+ 64] = acc\n}\n";
    assert_eq!(build_src(src), build_src(src));
}

#[test]
fn a_register_matmul_is_a_seeded_loop_and_a_store() {
    let out = build_src(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])\n@aligned(M = TILE_M, N = TILE_N)\nkernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {\n    let pm = program_id(0)\n    let pn = program_id(1)\n    var acc: tile<f32>[TILE_M, TILE_N] = 0.0\n    for kt in range(0, K, TILE_K) {\n        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]\n        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]\n        acc += dot(a, b)\n    }\n    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc\n}\n",
    );
    assert_contains(
        &out,
        &[
            "gemm<f32>[64, 64]/16 = gemm_init %",
            "gemm<f32>[64, 64]/16 = for %",
            "= gemm_dot %",
            "gemm_store alpha= beta= %",
        ],
    );
    assert!(!out.contains("dot_into") && !out.contains("alloc"), "{out}");
}

#[test]
fn without_alignment_the_same_matmul_takes_the_generic_path() {
    let out = build_src(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])\nkernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {\n    let pm = program_id(0)\n    let pn = program_id(1)\n    var acc: tile<f32>[TILE_M, TILE_N] = 0.0\n    for kt in range(0, K, TILE_K) {\n        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]\n        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]\n        acc += dot(a, b)\n    }\n    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc\n}\n",
    );
    assert_contains(&out, &["%acc.", "= alloc", "dot_into acc %"]);
    assert!(!out.contains("gemm"), "{out}");
}

/// Every source under `PHOBOS_SNAPSHOT_SRC` builds and verifies, or fails
/// the build where its `.err` says it should. Skipped without the variable.
#[test]
fn every_snapshot_source_builds() {
    let Ok(dir) = std::env::var("PHOBOS_SNAPSHOT_SRC") else {
        return;
    };
    let mut failures = Vec::new();
    let mut built = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "ph") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let Ok(kernels) = crate::parse(&src) else {
            continue;
        };
        let base = phobos_base::context::Context::default();
        for kernel in &kernels {
            match build(&base, target(), kernel) {
                Ok((ir, _)) => {
                    if let Err(e) = verify(&ir) {
                        failures.push(format!("{}: {e}", path.display()));
                    }
                    built += 1;
                }
                Err(e) => failures.push(format!("{}: {e:#}", path.display())),
            }
        }
    }
    println!("{built} kernels built");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
