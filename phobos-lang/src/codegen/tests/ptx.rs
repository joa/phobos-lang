// Down to PTX: each new operation reaches one instruction, and none becomes
// a libdevice call, which nothing links.

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

const OPS: &str = "
kernel ops(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64], I: tensor<i32>[N], n: i32) {
    let i = program_id(0)
    let q = I[i]
    I[i] = max(q >> 4, abs(q - 3)) | (q << 1) ^ ~(q & 7)
    let x = X[0 :+ 16, 0 :+ 64]
    var y: tile<f32>[16, 64] = where(iota(1) < n, min(max(x, 0.0), 6.0), -1.0e30)
    y += sin(x) * cos(x) + abs(x) + floor(x) + ceil(x) + rsqrt(x)
    Y[0 :+ 16, 0 :+ 64] = y
}
";

fn ptx(src: &str, chip: &str, index_bitwidth: u32) -> String {
    let ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
        index_bitwidth,
        ..Context::default()
    };
    crate::compile_raw(&ctx, src).unwrap().code
}

#[test]
fn new_operations_lower_to_single_instructions() {
    for (chip, bits) in [("sm_75", 32), ("sm_120", 64)] {
        let code = ptx(OPS, chip, bits);
        assert!(
            !code.contains("__nv_"),
            "{chip}: a libdevice call in:\n{code}"
        );
        for want in [
            "shr.s",
            "and.b",
            "or.b",
            "xor.b",
            "shl.b",
            "max.s",
            "min.f32",
            "max.f32",
            "selp",
            "sin.approx.f32",
            "cos.approx.f32",
            "abs.f32",
            "cvt.rmi.f32.f32",
            "cvt.rpi.f32.f32",
            "rsqrt.approx.f32",
        ] {
            assert!(code.contains(want), "{chip}: no `{want}` in:\n{code}");
        }
    }
}

#[test]
fn a_qgemm_module_needs_a_wide_index() {
    let src = "kernel k(A: tensor<i8>[64, 256], C: tensor<f32>[64, 64]) {
        let c = q4k_qgemm_t(A[0 :+ 64, 0 :+ 256], A[0 :+ 64, 0 :+ 256], A[0 :+ 64, 0 :+ 256], A[0 :+ 64, 0 :+ 256])
        C[0 :+ 64, 0 :+ 64] = c
    }";
    let kernels = crate::parse(src).unwrap();
    assert!(crate::needs_wide_index(&Context::default(), &kernels));
    let plain = crate::parse("kernel k(A: tensor<f32>[N]) { let i = program_id(0) }").unwrap();
    assert!(!crate::needs_wide_index(&Context::default(), &plain));
}
