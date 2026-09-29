// The raw-format matmuls: every format the device decodes in-kernel,
// against the host's dense fallback.

use super::*;
use phobos_gguf::quant::quantize_row;

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
const RAW_SHAPES: [(usize, usize, usize); 11] = [
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
    // The only shape `project_raw_dense` hands an f16 strip, since every
    // other m here is under TC_TILE_M.
    (64, 512, 128),
    // The one shape the fused projection takes: m a multiple of
    // IQ1S_QMMA_TM, n of IQ1S_QMMA_TN, k whole 256-element blocks. This is
    // what covers `project_raw_qmma`, the prompt path.
    (128, 512, 128),
    // More than one row tile plus a ragged remainder. The staged kernel runs
    // padded into a scratch and copies the window out, as for any prompt
    // length the tile does not divide.
    (200, 512, 128),
];

/// Shapes that reach the tensor cores and stage their weight to f16, so they
/// are judged like other f16 paths (see `check_tc`). `Backend::matmul` sends
/// every whole 64-row band that way, so any `m` of 64 or more qualifies.
fn raw_shape_is_tc(m: usize, k: usize, n: usize) -> bool {
    m >= 64 && n.is_multiple_of(64) && k.is_multiple_of(16)
}

/// `check_within`'s signature, named so clippy accepts it as a parameter
/// type.
type CheckWithin<'a> = dyn Fn(&str, f32, &[f32], &[f32]) + 'a;

/// `check_tc`'s signature.
type CheckTc<'a> = dyn Fn(&str, usize, &[f32], &[f32]) + 'a;

/// `check_spread`'s signature. Unlike [`CheckWithin`] it takes no tolerance.
type CheckSpread<'a> = dyn Fn(&str, &[f32], &[f32]) + 'a;

/// A raw-kernel format's device path against the host reference
/// (`Packed::dense`), across every shape in [`RAW_SHAPES`]. `gen_block`
/// builds one random super-block.
///
/// At one row the `dp4a` path quantizes its activation and the host does
/// not. So `dp4a` is judged against the device float path instead, with
/// `set_dp4a` running the same projection both ways.
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
        // Only a single row reaches the `dp4a` matvec. Wider shapes take the
        // same path either way.
        if m == 1 {
            set_dp4a(true);
            let dp4a = run(gpu)?;
            set_dp4a(false);
            check_spread(&format!("{label} dp4a vs float"), &float_path, &dp4a);
        }
        // The fused projection also quantizes its activation, so it is
        // judged against the dense device path it replaces.
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

/// A K-quant format's device path against the host reference.
///
/// Both device kernels quantize the activation to Q8_0, ties to even as
/// `quantize_row` does, and neither has a float path. So the host is fed that
/// same quantized activation. Only accumulation order and f16 header
/// rounding remain, which 1e-3 covers.
#[allow(clippy::too_many_arguments)]
fn check_matmul_kquant(
    host: &dyn Backend,
    gpu: &dyn Backend,
    set_dp4a: &dyn Fn(bool),
    set_qmma: &dyn Fn(bool),
    check_within: &CheckWithin,
    next: &mut dyn FnMut() -> f32,
    quant: Quant,
    name: &str,
    mut gen_block: impl FnMut(&mut dyn FnMut() -> f32) -> Vec<u8>,
) -> Result<()> {
    set_dp4a(true);
    set_qmma(true);
    for (m, k, n) in RAW_SHAPES {
        let nb = k / 256;
        let mut blocks = Vec::new();
        for _ in 0..n * nb {
            blocks.extend(gen_block(next));
        }
        let packed = Packed::new(quant, blocks, k, n)?;
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        // The activation as the device sees it, a scale every 32.
        let mut seen = vec![0.0f32; m * k];
        let mut qs = vec![0i8; 32];
        for (row, from) in seen.chunks_mut(32).zip(a.chunks(32)) {
            let scale = quantize_row(from, &mut qs);
            for (y, &q) in row.iter_mut().zip(&qs) {
                *y = f32::from(q) * scale;
            }
        }
        let run = |b: &dyn Backend, act: &[f32]| -> Result<Vec<f32>> {
            let ab = b.upload(act)?;
            let wb = b.constant_raw(&format!("raw{name}{m}x{k}x{n}"), &packed)?;
            let out = b.alloc(m * n)?;
            b.matmul_raw(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        let (want, got) = (run(host, &seen)?, run(gpu, &a)?);
        check_within(&format!("matmul_raw {name} [{m} x {k} x {n}]"), 1e-3, &want, &got);
    }
    Ok(())
}

/// Every raw format, through [`check_matmul_raw`] or, for the formats with
/// no float device path, [`check_matmul_kquant`].
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
    // Header offsets: d at 0 and dmin at 2 for Q4_K and Q5_K, d in the last
    // two bytes of Q6_K. A PTQ1_0 registry block is two file blocks, each
    // ending in its d. Any byte is a valid five trits.
    for (quant, name, block_bytes, d_off, dmin_off) in [
        (Quant::Q4_K, "Q4_K", 144usize, 0usize, Some(2usize)),
        (Quant::Q5_K, "Q5_K", 176, 0, Some(2)),
        (Quant::Q6_K, "Q6_K", 210, 208, None),
        (Quant::PTQ1_0, "PTQ1_0", 56, 26, Some(54)),
    ] {
        check_matmul_kquant(host, gpu, set_dp4a, set_qmma, check_within, next, quant, name, |n| {
            random_raw_block(n, block_bytes, d_off, dmin_off)
        })?;
    }
    Ok(())
}

/// The Hadamard transform ahead of a folded weight, at the widths the
/// ternary 27B folds and with its delta net's head regrouping.
pub(super) fn check_hadamard(
    host: &dyn Backend,
    gpu: &dyn Backend,
    check_within: &CheckWithin,
    next: &mut dyn FnMut() -> f32,
) -> Result<()> {
    let regroup = HeadPerm { head_dim: 128, groups: 16, repeat: 3 };
    for (rows, width, perm) in [
        (1usize, 5120usize, None),
        (3, 5120, None),
        (1, 17408, None),
        (1, 6144, Some(regroup)),
        (4, 6144, Some(regroup)),
    ] {
        let x: Vec<f32> = (0..rows * width).map(|_| next()).collect();
        let signs: Vec<f32> = (0..width).map(|_| if next() < 0.0 { -1.0 } else { 1.0 }).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, sb) = (b.upload(&x)?, b.upload(&signs)?);
            let out = b.alloc(rows * width)?;
            b.hadamard(xb, rows, width, sb, perm, out)?;
            read_vec(b, out, rows * width)
        };
        let label = format!("hadamard [{rows} x {width}]{}", if perm.is_some() { " regrouped" } else { "" });
        check_within(&label, 1e-4, &run(host)?, &run(gpu)?);
    }

    // The quantizing forms, with the int8 copy read back through a Q8_0
    // projection. The normalizing form also returns its plain row.
    let (n, eps) = (64usize, 1e-6f32);
    for (rows, width, perm, norm) in [
        (1usize, 5120usize, None, true),
        (3, 5120, None, true),
        (1, 17408, None, false),
        (4, 6144, Some(regroup), false),
    ] {
        let x: Vec<f32> = (0..rows * width).map(|_| next() * 4.0).collect();
        let signs: Vec<f32> = (0..width).map(|_| if next() < 0.0 { -1.0 } else { 1.0 }).collect();
        let gain: Vec<f32> = (0..width).map(|_| next() + 1.5).collect();
        let qs: Vec<i8> = (0..width * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(width / 32) * n).map(|_| next().abs() + 0.01).collect();
        let packed = pack_q8_0(&qs, &scales, width, n)?;
        let run = |b: &dyn Backend| -> Result<[Vec<f32>; 3]> {
            let (xb, sb, gb) = (b.upload(&x)?, b.upload(&signs)?, b.upload(&gain)?);
            let (normed, out) = (b.alloc(rows * width)?, b.alloc(rows * width)?);
            let act = match norm {
                true => b.rms_norm_hadamard_q(xb, rows, width, gb, eps, sb, normed, out)?,
                false => b.hadamard_q(xb, rows, width, sb, perm, out)?,
            };
            let wb = b.constant_quant(&format!("hadamard_q{width}"), &packed)?;
            let proj = b.alloc(rows * n)?;
            b.matmul_quant_act(act, rows, width, wb, n, proj)?;
            let normed = if norm { read_vec(b, normed, rows * width)? } else { Vec::new() };
            Ok([normed, read_vec(b, out, rows * width)?, read_vec(b, proj, rows * n)?])
        };
        let ([want_n, want_o, want_p], [got_n, got_o, got_p]) = (run(host)?, run(gpu)?);
        let label = format!("{} [{rows} x {width}]", if norm { "rms_norm_hadamard_q" } else { "hadamard_q" });
        check_within(&format!("{label} normed"), 1e-4, &want_n, &got_n);
        check_within(&label, 1e-4, &want_o, &got_o);
        check_within(&format!("{label} act x {n}"), 1e-3, &want_p, &got_p);
    }
    Ok(())
}
