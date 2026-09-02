// The raw-format matmuls: every format the device decodes in-kernel,
// against the host's dense fallback.

use super::*;

/// A raw-format super-block: full-range random bytes, with a real (small,
/// finite) `d`/`dmin` header written at the given offsets afterward.
fn random_raw_block(
    next: &mut (impl FnMut() -> f32 + ?Sized),
    block_bytes: usize,
    d_off: usize,
    dmin_off: Option<usize>,
) -> Vec<u8> {
    let mut block = vec![0u8; block_bytes];
    for byte in block.iter_mut() {
        *byte = (((next() + 1.0) * 127.5) as i32).clamp(0, 255) as u8;
    }
    let d = f32_to_f16(0.02 * (next().abs() + 0.1));
    block[d_off..d_off + 2].copy_from_slice(&d.to_le_bytes());
    if let Some(off) = dmin_off {
        let dmin = f32_to_f16(0.02 * (next().abs() + 0.1));
        block[off..off + 2].copy_from_slice(&dmin.to_le_bytes());
    }
    block
}

/// An IQ1_M super-block: full-range random bytes, then a real `d`'s four
/// nibbles written into the top nibbles of `scales[1, 3, 5, 7]`, low to
/// high (see `quant/iq1_m.rs::raw_scales`).
fn random_iq1m_block(next: &mut (impl FnMut() -> f32 + ?Sized)) -> Vec<u8> {
    let mut block = vec![0u8; 56];
    for byte in block.iter_mut() {
        *byte = (((next() + 1.0) * 127.5) as i32).clamp(0, 255) as u8;
    }
    let d = f32_to_f16(0.02 * (next().abs() + 0.1));
    for (i, off) in [49usize, 51, 53, 55].into_iter().enumerate() {
        let nibble = (d >> (4 * i)) & 0xf;
        block[off] = (block[off] & 0x0f) | ((nibble as u8) << 4);
    }
    block
}

/// `m`, `k`, `n`: small and large, single-row and batched, aligned and not.
/// Shared by every raw-kernel format's check below.
const RAW_SHAPES: [(usize, usize, usize); 10] = [
    (1, 256, 32),
    (1, 256, 64),
    (1, 512, 96),
    (1, 1024, 33),
    (2, 256, 32),
    // m > 1 at a width no format's TN divides: the masked `_dequant`
    // fallback, where the other m > 1 shapes reach `_qdecode`.
    (3, 512, 33),
    (5, 1024, 128),
    (6, 2048, 5120),
    // The only shape `project_raw_dense` hands an f16 strip: every other m
    // here is under TC_TILE_M. Without it the narrow path goes untested.
    (64, 512, 128),
    // The one shape the fused projection takes: m a multiple of IQ1S_QMMA_TM,
    // n of IQ1S_QMMA_TN, k a whole number of 256-element blocks. Without it
    // `project_raw_qmma` goes untested, and it is the prompt path.
    (128, 512, 128),
];

/// Shapes that reach the tensor cores, and so stage their weight to f16.
/// Judged like every other f16 path here; see `check_tc`.
fn raw_shape_is_tc(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(64) && n.is_multiple_of(64) && k.is_multiple_of(16)
}

/// `check_within`'s signature, named once since it appears as a parameter
/// type below and clippy would rather it not be spelled out inline.
type CheckWithin<'a> = dyn Fn(&str, f32, &[f32], &[f32]) + 'a;

/// `check_tc`'s signature, named for the same reason.
type CheckTc<'a> = dyn Fn(&str, usize, &[f32], &[f32]) + 'a;

/// [`CheckWithin`]'s signature for the spread measure, which carries its own
/// tolerance because only one comparison uses it.
type CheckSpread<'a> = dyn Fn(&str, &[f32], &[f32]) + 'a;

/// A raw-kernel format's device path against the host reference
/// (`Packed::dense`), across every shape in [`RAW_SHAPES`]. `gen_block`
/// builds one random super-block.
///
/// At one row the device has two paths and they need different oracles. The
/// float matvec is judged against the host, as everything else here is. The
/// `dp4a` one quantizes its activation, which the host reference does not, so
/// against the host it disagrees by the size of an 8-bit activation however
/// right it is -- the same reason `fuse_check` compares the fused decode path
/// against the launched one rather than against the host. So it is judged
/// against the float path it replaces, on the device, in this session:
/// `set_dp4a` runs the same projection both ways.
#[allow(clippy::too_many_arguments)]
fn check_matmul_raw(
    host: &dyn Backend,
    gpu: &dyn Backend,
    set_dp4a: &dyn Fn(bool),
    set_qmma: &dyn Fn(bool),
    check_spread: &CheckSpread,
    check_within: &CheckWithin,
    check_tc: &CheckTc,
    next: &mut dyn FnMut() -> f32,
    quant: Quant,
    name: &str,
    mut gen_block: impl FnMut(&mut dyn FnMut() -> f32) -> Vec<u8>,
) -> Result<()> {
    for (m, k, n) in RAW_SHAPES {
        let nb = k / 256;
        let mut blocks = Vec::new();
        for _ in 0..n * nb {
            blocks.extend(gen_block(next));
        }
        let packed = Packed::new(quant, blocks, k, n)?;
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.constant_raw(&format!("raw{name}{m}x{k}x{n}"), &packed)?;
            let out = b.alloc(m * n)?;
            b.matmul_raw(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        let label = format!("matmul_raw {name} [{m} x {k} x {n}]");
        set_dp4a(false);
        set_qmma(false);
        let (want, float_path) = (run(host)?, run(gpu)?);
        if raw_shape_is_tc(m, k, n) {
            check_tc(&label, k, &want, &float_path);
        } else {
            check_within(&label, 1e-3, &want, &float_path);
        }
        // Only one row reaches a `dp4a` matvec at all; anything wider takes the
        // same path either way and the comparison would be vacuous.
        if m == 1 {
            set_dp4a(true);
            let dp4a = run(gpu)?;
            set_dp4a(false);
            check_spread(&format!("{label} dp4a vs float"), &float_path, &dp4a);
        }
        // The fused projection quantizes its activation as well, so the host
        // cannot judge it either; the path it replaces can.
        if m > 1 {
            set_qmma(true);
            let fused = run(gpu)?;
            set_qmma(false);
            if fused != float_path {
                check_spread(&format!("{label} fused vs dense"), &float_path, &fused);
            }
        }
    }
    Ok(())
}

/// Every raw format, through [`check_matmul_raw`].
#[allow(clippy::too_many_arguments)]
pub(super) fn check_raw_formats(
    host: &dyn Backend,
    gpu: &dyn Backend,
    set_dp4a: &dyn Fn(bool),
    set_qmma: &dyn Fn(bool),
    check_spread: &CheckSpread,
    check_within: &CheckWithin,
    check_tc: &CheckTc,
    next: &mut dyn FnMut() -> f32,
) -> Result<()> {
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::Q2_K,
        "Q2_K",
        |n| random_raw_block(n, 84, 80, Some(82)),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::Q3_K,
        "Q3_K",
        |n| random_raw_block(n, 110, 108, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ1_S,
        "IQ1_S",
        |n| random_raw_block(n, 50, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ2_XXS,
        "IQ2_XXS",
        |n| random_raw_block(n, 66, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ1_M,
        "IQ1_M",
        |n| random_iq1m_block(n),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ2_S,
        "IQ2_S",
        |n| random_raw_block(n, 82, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ2_XS,
        "IQ2_XS",
        |n| random_raw_block(n, 74, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ3_XXS,
        "IQ3_XXS",
        |n| random_raw_block(n, 98, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ3_S,
        "IQ3_S",
        |n| random_raw_block(n, 110, 0, None),
    )?;
    check_matmul_raw(
        host,
        gpu,
        set_dp4a,
        set_qmma,
        check_spread,
        check_within,
        check_tc,
        next,
        Quant::IQ4_XS,
        "IQ4_XS",
        |n| random_raw_block(n, 136, 0, None),
    )?;
    Ok(())
}
