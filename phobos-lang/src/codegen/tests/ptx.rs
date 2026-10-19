// Down to PTX: each new operation reaches one instruction, and none becomes
// a libdevice call, which nothing links.

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

const OPS: &str = "
kernel ops(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64], I: tensor<i32>[N], n: i32) {
    let i = ctx.block.x
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
    let plain = crate::parse("kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }").unwrap();
    assert!(!crate::needs_wide_index(&Context::default(), &plain));
}

#[test]
fn index_bits_takes_the_tuned_width_over_the_rule() {
    let src = "@autotune(IB in [32, 64])
        @index_bits(IB)
        kernel k(A: tensor<f32>[N]) { A[ctx.block.x] = 1.0 }";
    let kernels = crate::parse(src).unwrap();
    let mut ctx = Context::default();
    assert_eq!(crate::index_bits(&ctx, &kernels).unwrap(), 32);
    ctx.shape_overrides.insert("IB".to_string(), 64);
    assert_eq!(crate::index_bits(&ctx, &kernels).unwrap(), 64);
    // Asking for 64 is a floor the attribute does not lower.
    ctx.shape_overrides.insert("IB".to_string(), 32);
    ctx.index_bitwidth = 64;
    assert_eq!(crate::index_bits(&ctx, &kernels).unwrap(), 64);
    let odd = crate::parse("@index_bits(48) kernel k(A: tensor<f32>[N]) { A[ctx.block.x] = 1.0 }").unwrap();
    let err = crate::index_bits(&Context::default(), &odd).unwrap_err();
    assert!(err.to_string().contains("32 or 64"), "{err}");
}

#[test]
fn atomics_reach_atom_instructions() {
    let src = "kernel k(L: tensor<i32>[2], F: tensor<f32>[1], OUT: tensor<i32>[N]) {
        let i = ctx.block.x
        OUT[i] = atomic_cas(L, 0, 0, i)
        OUT[0] = atomic_add(L, 1, 1)
        F[0] = atomic_add(F, 0, 0.5)
    }";
    for (chip, bits) in [("sm_75", 32), ("sm_120", 64)] {
        let code = ptx(src, chip, bits);
        assert!(!code.contains("__nv_"), "{chip}: a libdevice call in:
{code}");
        for want in [".global.cas.b32", "atom.global.add.u32", "atom.global.add.f32"] {
            assert!(code.contains(want), "{chip}: no {want} in:
{code}");
        }
    }
}

#[test]
fn unsigned_reads_zero_extend_in_ptx() {
    let src = "kernel k(B: tensor<u8>[16, 64], H: tensor<u16>[N], O: tensor<i32>[16, 64], P: tensor<f32>[N]) {
        let i = ctx.block.x
        P[i] = f32(H[i])
        var b = B[0 :+ 16, 0 :+ 64]
        var q: tile<i32>[16, 64] = (i32(b) >> 4) & 15
        O[0 :+ 16, 0 :+ 64] = q
    }";
    for (chip, bits) in [("sm_75", 32), ("sm_120", 64)] {
        let code = ptx(src, chip, bits);
        assert!(!code.contains("__nv_"), "{chip}: a libdevice call in:\n{code}");
        assert!(!code.contains(".s8") && !code.contains(".s16"), "{chip}: a signed narrow read in:\n{code}");
    }
}

#[test]
fn system_scope_reaches_ptx_widened() {
    // Asked for 32 bits, compiled at 64, since the address must be whole.
    let src = "kernel k(F: tensor<i32>[1, 1], O: tensor<i32>[1, 1]) {
        O[0, 0] = load_acquire(F, 0)
        store_release(F, 0, 0)
    }";
    let code = ptx(src, "sm_75", 32);
    for want in ["ld.acquire.sys.global.u32", "st.release.sys.global.u32", "fence.acq_rel.sys"] {
        assert!(code.contains(want), "no {want} in:\n{code}");
    }
}

// The f32 gemm stages a k-major, which cp.async could move only four bytes at
// a time, and no supported chip runs f32x2 arithmetic natively: at 64-bit
// indices on sm_120 the kernel keeps the 16-byte register staging and scalar
// FMAs the 32-bit sm_75 build has.
#[test]
fn f32_gemm_at_wide_indices_keeps_register_staging() {
    let pins = [("TILE_M", 64), ("TILE_N", 128), ("TILE_K", 16), ("U", 8), ("T", 128)];
    let ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_120")),
        index_bitwidth: 64,
        shape_overrides: pins.iter().map(|&(name, value)| (name.to_string(), value)).collect(),
        ..Context::default()
    };
    let code = crate::compile_raw(&ctx, include_str!("../../../../examples/gemm_fp32.ph")).unwrap().code;
    assert!(!code.contains("f32x2"), "packed f32 arithmetic in:\n{code}");
    assert!(!code.contains("cp.async"), "cp.async staging in:\n{code}");
    assert!(code.contains("fma.rn.f32"));
    assert!(code.contains("ld.global.v4"), "16-byte global loads in:\n{code}");
}
