// The device backend against the host reference, op by op:
//
//   cargo run --release -p phobos-gguf --features cuda --example backend_check
//
// The host implementation defines the semantics, so anything the device does
// differently by more than f32 reduction noise is a bug.

use anyhow::Result;
use phobos_gguf::backend::{
    Attn, Backend, Buf, DeltaMix, HBuf, HPlane, HeadPerm, HostBackend, Plane, Rope, read_vec,
};

use phobos_base::half::f32_to_f16;
use phobos_gguf::backend::device;
use phobos_gguf::quant::{Packed, Quant, pack_q8_0};


mod raw;

use raw::{check_hadamard, check_raw_formats};

fn main() -> Result<()> {
    let gpu = device::DeviceBackend::new(&phobos_gguf::Quant::ALL)?;
    let host = HostBackend::new();
    // Cells so that every checker closure below can update them.
    let worst = std::cell::Cell::new(0.0f32);
    let failures = std::cell::Cell::new(0u32);

    // Switches for the device paths that quantize their activation, which the
    // host cannot judge. See `check_matmul_raw`.
    let set_dp4a = |on: bool| gpu.set_iq_dp4a(on);
    let set_qmma = |on: bool| gpu.set_raw_qmma(on);

    /// The largest gap as a fraction of the spread of `want`.
    ///
    /// Random weights make matvec outputs cancel toward zero, so a
    /// per-element relative error would mislead.
    fn spread_error(want: &[f32], got: &[f32]) -> f32 {
        let (lo, hi) = want
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &w| (l.min(w), h.max(w)));
        let gap = want
            .iter()
            .zip(got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        gap / (hi - lo).max(f32::MIN_POSITIVE)
    }

    // Looser than the float path's 1e-3, since it bounds a known
    // approximation. A layout mistake would be orders of magnitude worse.
    let check_spread = |name: &str, want: &[f32], got: &[f32]| {
        let error = spread_error(want, got);
        worst.set(worst.get().max(error));
        let ok = error < 1e-2 && want.len() == got.len();
        if !ok {
            failures.set(failures.get() + 1);
        }
        println!(
            "{} {name:<34} spread {error:>10.3e}",
            if ok { "ok  " } else { "FAIL" }
        );
    };

    let check_within = |name: &str, tolerance: f32, want: &[f32], got: &[f32]| {
        let error = want
            .iter()
            .zip(got)
            .map(|(w, g)| (w - g).abs() / w.abs().max(1.0))
            .fold(0.0f32, f32::max);
        worst.set(worst.get().max(error));
        // f32 reductions in a different order will not match bit for bit; a
        // layout mistake would be orders of magnitude worse than this.
        let ok = error < tolerance && want.len() == got.len();
        if !ok {
            failures.set(failures.get() + 1);
        }
        println!(
            "{} {name:<34} rel err {error:>10.3e}",
            if ok { "ok  " } else { "FAIL" }
        );
        if !ok {
            let first = want
                .iter()
                .zip(got)
                .position(|(w, g)| (w - g).abs() / w.abs().max(1.0) > tolerance);
            println!("      first bad index {first:?} of {}", want.len());
        }
    };
    // The default tolerance for ops without their own.
    let check = |name: &str, want: &[f32], got: &[f32]| check_within(name, 1e-4, want, got);

    // Tensor-core matmul rounds inputs to fp16, so the error is relative to
    // the dot's RMS magnitude, sqrt(K)/3. Same rule as `verify_matmul` in
    // phobos-kbench/src/gemm.rs.
    let check_tc = |name: &str, k: usize, want: &[f32], got: &[f32]| {
        let floor = (k as f32).sqrt() / 3.0;
        let error = want
            .iter()
            .zip(got)
            .map(|(w, g)| (w - g).abs() / w.abs().max(floor))
            .fold(0.0f32, f32::max);
        worst.set(worst.get().max(error));
        let ok = error < 1e-2 && want.len() == got.len();
        if !ok {
            failures.set(failures.get() + 1);
        }
        println!(
            "{} {name:<34} rel err {error:>10.3e}",
            if ok { "ok  " } else { "FAIL" }
        );
    };

    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    // Sizes the model actually uses, plus a ragged one.
    let (rows, width) = (5usize, 1024usize);
    let x: Vec<f32> = (0..rows * width).map(|_| next()).collect();
    let gain: Vec<f32> = (0..width).map(|_| next().abs() + 0.5).collect();

    let on_gpu = gpu.upload(&x)?;
    check("upload+read", &x, &read_vec(&gpu, on_gpu, x.len())?);

    let eps = 1e-6f32;
    let run_rms = |b: &dyn Backend| -> Result<Vec<f32>> {
        let xb = b.upload(&x)?;
        let gb = b.upload(&gain)?;
        let out = b.alloc(rows * width)?;
        b.rms_norm(xb, rows, width, gb, eps, out)?;
        read_vec(b, out, rows * width)
    };
    let (rms_want, rms_got) = (run_rms(&host)?, run_rms(&gpu)?);
    check("rms_norm [5 x 1024]", &rms_want, &rms_got);
    let bad_rows: Vec<usize> = (0..rows)
        .filter(|&r| {
            rms_want[r * width..(r + 1) * width]
                .iter()
                .zip(&rms_got[r * width..(r + 1) * width])
                .any(|(w, g)| (w - g).abs() / w.abs().max(1.0) > 1e-4)
        })
        .collect();
    println!("     rms_norm rows disagreeing: {bad_rows:?}");

    // `rms_norm_q`: the normalized rows, and the int8 copy read back through
    // a Q8_0 projection. At this width and the 27B's.
    for width_q in [1024usize, 5120] {
        let x: Vec<f32> = (0..rows * width_q).map(|_| next() * 4.0).collect();
        let gain: Vec<f32> = (0..width_q).map(|_| next() + 1.5).collect();
        let n = 64;
        let qs: Vec<i8> = (0..width_q * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(width_q / 32) * n).map(|_| next().abs() + 0.01).collect();
        let packed = pack_q8_0(&qs, &scales, width_q, n)?;
        let run = |b: &dyn Backend| -> Result<(Vec<f32>, Vec<f32>)> {
            let xb = b.upload(&x)?;
            let gb = b.upload(&gain)?;
            let out = b.alloc(rows * width_q)?;
            let act = b.rms_norm_q(xb, rows, width_q, gb, eps, out)?;
            let wb = b.constant_quant(&format!("rmsq{width_q}"), &packed)?;
            let proj = b.alloc(rows * n)?;
            b.matmul_quant_act(act, rows, width_q, wb, n, proj)?;
            Ok((read_vec(b, out, rows * width_q)?, read_vec(b, proj, rows * n)?))
        };
        let ((want_o, want_p), (got_o, got_p)) = (run(&host)?, run(&gpu)?);
        check(&format!("rms_norm_q [{rows} x {width_q}]"), &want_o, &got_o);
        check_within(&format!("rms_norm_q act [{rows} x {width_q} x {n}]"), 1e-3, &want_p, &got_p);
    }

    let y: Vec<f32> = (0..rows * width).map(|_| next()).collect();
    let run_add = |b: &dyn Backend| -> Result<Vec<f32>> {
        let acc = b.upload(&x)?;
        let add = b.upload(&y)?;
        b.add_into(acc, add)?;
        read_vec(b, acc, x.len())
    };
    check("add_into [3072]", &run_add(&host)?, &run_add(&gpu)?);

    // A projection bias down every row: GLM-4's key and value width, its
    // query width, and a width that leaves a partial tile.
    for (bias_rows, bias_width) in [(1, 256), (7, 4096), (3, 1000)] {
        let xs: Vec<f32> = (0..bias_rows * bias_width).map(|_| next()).collect();
        let bias: Vec<f32> = (0..bias_width).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let xb = b.upload(&xs)?;
            b.add_rows(xb, bias_rows, bias_width, b.upload(&bias)?)?;
            read_vec(b, xb, xs.len())
        };
        check(&format!("add_rows [{bias_rows} x {bias_width}]"), &run(&host)?, &run(&gpu)?);
    }

    // At the FFN width and at a length that is not a tile multiple. A nonzero
    // `at` reads the second half of a shared buffer, as the fused
    // gate-and-up projection does.
    for (len, at) in [(rows * 3584, 0), (1000, 0), (1024, 1024)] {
        let g: Vec<f32> = (0..len + at).map(|_| next()).collect();
        let u: Vec<f32> = (0..len + at).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let gb = b.upload(&g)?;
            let ub = b.upload(&u)?;
            let out = b.alloc(len)?;
            b.swiglu(gb, 0, ub, at, out, len)?;
            read_vec(b, out, len)
        };
        check(&format!("swiglu [{len} @ {at}]"), &run(&host)?, &run(&gpu)?);
    }

    // The prompt-pass form: gate and up interleave in the fused projection's
    // output, so both are strided. At 3584 one program holds a whole row;
    // 4608 overruns a kernel's static shared memory.
    for ffn in [3584usize, 4608] {
        let both: Vec<f32> = (0..rows * 2 * ffn).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let src = b.upload(&both)?;
            let out = b.alloc(rows * ffn)?;
            let half = |offset| Plane {
                buf: src,
                offset,
                pitch: 2 * ffn,
            };
            b.swiglu_planes(half(0), half(ffn), out, rows, ffn)?;
            read_vec(b, out, rows * ffn)
        };
        check(
            &format!("swiglu_planes [{rows} x {ffn}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    let run_copy = |b: &dyn Backend| -> Result<Vec<f32>> {
        let src = b.upload(&x)?;
        let dst = b.alloc(x.len())?;
        b.copy(src, width, dst, 0, width)?;
        read_vec(b, dst, width)
    };
    check("copy [1024 @ offset]", &run_copy(&host)?, &run_copy(&gpu)?);

    // The LM head reads the last row of a [rows, d] buffer.
    let run_last = |b: &dyn Backend| -> Result<Vec<f32>> {
        let src = b.upload(&x)?;
        let dst = b.alloc(width)?;
        b.copy(src, (rows - 1) * width, dst, 0, width)?;
        read_vec(b, dst, width)
    };
    check("copy [last row of 5]", &run_last(&host)?, &run_last(&gpu)?);

    // Every (m, k, n) a prefill and a decode step produce.
    for (m, k, n) in [
        (1usize, 1024usize, 2048usize),
        (1, 1024, 16),
        (1, 1024, 4096),
        (1, 1024, 6144),
        (1, 1024, 512),
        (1, 1024, 3584),
        (1, 3584, 1024),
        (1, 2048, 1024),
        (5, 1024, 2048),
        (5, 1024, 16),
        (5, 1024, 4096),
        (5, 1024, 6144),
        (5, 1024, 512),
        (5, 1024, 3584),
        (5, 3584, 1024),
        (5, 2048, 1024),
        // Ragged N and ragged M, which leave the last tile partially outside
        // the tensor in each direction.
        (5, 1024, 40),
        (5, 1024, 2049),
        (5, 1024, 33),
        (33, 1024, 64),
        (33, 1024, 40),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.upload(&w)?;
            let out = b.alloc(m * n)?;
            b.matmul(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        check(
            &format!("matmul [{m} x {k} x {n}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // matmul at shapes that reach the tensor-core band (m >= TC_TILE_M, n a
    // whole number of TC_TILE_N); none of the shapes above do.
    for (m, k, n) in [
        (128usize, 1024usize, 2048usize), // whole TC bands, no remainder
        (100, 1024, 2048),                // one TC band, 36-row remainder
        (65, 1024, 3584),                 // one TC band, 1-row remainder
        (128, 1024, 2049),                // n not TC_TILE_N-wide: TC skipped entirely
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.upload(&w)?;
            let out = b.alloc(m * n)?;
            b.matmul(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        check_tc(
            &format!("matmul_tc [{m} x {k} x {n}]"),
            k,
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The delta rule, checked on its state as well as its output. A slightly
    // wrong decay shows in the state before it shows in the output.
    for (rows, heads, head_dim) in [
        (1usize, 16usize, 128usize),
        (7, 16, 128),
        (64, 4, 32),
        // Prompt depths, where f32 accumulation order starts to separate the
        // two backends.
        (48, 16, 128),
        (47, 16, 128),
        (128, 32, 128),
        (512, 32, 128),
    ] {
        let n = rows * heads * head_dim;
        let vecs: Vec<Vec<f32>> = (0..3).map(|_| (0..n).map(|_| next()).collect()).collect();
        // Decays live in (0, 1) and betas in (0, 1), as the gates produce them.
        let decay: Vec<f32> = (0..rows * heads).map(|_| next().abs().min(1.0)).collect();
        let beta: Vec<f32> = (0..rows * heads).map(|_| next().abs().min(1.0)).collect();
        let plane = heads * head_dim * head_dim;
        let state0: Vec<f32> = (0..plane).map(|_| next()).collect();

        let mut staged = Vec::new();
        for part in &vecs {
            staged.extend_from_slice(part);
        }
        staged.extend_from_slice(&decay);
        staged.extend_from_slice(&beta);
        let run = |b: &dyn Backend| -> Result<(Vec<f32>, Vec<f32>)> {
            let packed = b.upload(&staged)?;
            let state = b.upload(&state0)?;
            let out = b.alloc(n)?;
            b.delta_rule(packed, rows, heads, head_dim, state, out)?;
            Ok((read_vec(b, out, n)?, read_vec(b, state, plane)?))
        };
        let (host_out, host_state) = run(&host)?;
        let (gpu_out, gpu_state) = run(&gpu)?;
        // Accumulation order differs from the host's over many rows, so this
        // needs a looser tolerance than the default.
        check_within(
            &format!("delta_rule [{rows} x {heads} x {head_dim}]"),
            3e-4,
            &host_out,
            &gpu_out,
        );
        check_within(
            &format!("delta_rule state [{rows} x {heads} x {head_dim}]"),
            3e-4,
            &host_state,
            &gpu_state,
        );
    }

    // The Q8_0 matmul, which decode runs.
    for (m, k, n) in [
        (1usize, 1024usize, 2048usize),
        (1, 1024, 16),
        (1, 1024, 6144),
        (1, 1024, 3584),
        (1, 3584, 1024),
        (1, 2048, 1024),
        (2, 1024, 512),
        // From 8 rows the tensor-core kernel takes over. These cover the
        // seam: an exact tile, one row short of two, one row past, and an n
        // that does not tile.
        (8, 1024, 512),
        (15, 1024, 512),
        (17, 1024, 512),
        (128, 1024, 2048),
        // A width a 48-wide tile divides but no 64-wide one: the delta net's
        // alpha/beta projections, which run split-K. k is 1024 rather than
        // the model's 5120 to stay inside the 1e-3 bound.
        (128, 1024, 48),
        (8, 1024, 40),
        (33, 1024, 33),
        // From 64 rows the fused tensor-core kernel takes over and passes
        // what it cannot tile to the two kernels below it. These cover that
        // seam: an exact tile, one short, and 101 = 64 + 32 + 5, which gives
        // all three kernels rows. The last two have an n the 64-wide column
        // tile does not divide, so none of their rows may reach the fused
        // kernel.
        (64, 1024, 512),
        (63, 1024, 512),
        (101, 1024, 512),
        (128, 1024, 2049),
        (101, 1024, 40),
    ] {
        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(k / 32) * n).map(|_| next().abs() + 0.01).collect();
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let packed = pack_q8_0(&qs, &scales, k, n)?;
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.constant_quant(&format!("q{m}x{k}x{n}"), &packed)?;
            let out = b.alloc(m * n)?;
            b.matmul_quant(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        // Looser than the dense ops. Quantized sums reach a few thousand, and
        // the kernel sums per 32-element block where the reference uses one
        // flat accumulator. `q8diag` shows the kernel is the more accurate.
        check_within(
            &format!("matmul_quant [{m} x {k} x {n}]"),
            1e-3,
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // Q5_0 through `q50_qdot_t` and `q50_qmma_t`, at GLM-4's down projection
    // (k = 13696, no multiple of 256). 200 = 128 + 64 + 8 rows gives the deep
    // tile, the shallow one and the matvec rows each a share.
    for (m, k, n, add) in [(1, 13696, 4096, false), (1, 13696, 4096, true), (200, 13696, 4096, false), (64, 1024, 48, false)] {
        let blocks: Vec<u8> = (0..k / 32 * n)
            .flat_map(|_| {
                let d = phobos_base::half::f32_to_f16(next().abs() * 0.01 + 0.001).to_le_bytes();
                let rest: Vec<u8> = (0..20).map(|_| (next().abs() * 256.0) as u8).collect();
                d.into_iter().chain(rest)
            })
            .collect();
        let packed = Packed::new(Quant::Q5_0, blocks, k, n)?;
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let base: Vec<f32> = (0..m * n).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.constant_quant(&format!("q50_{m}x{k}x{n}"), &packed)?;
            let out = b.upload(&base)?;
            let act = b.quantize_act(ab, m, k)?;
            match add {
                true => b.matmul_quant_add(act, m, k, wb, n, out)?,
                false => b.matmul_quant_act(act, m, k, wb, n, out)?,
            }
            read_vec(b, out, m * n)
        };
        let what = if add { "matmul_quant_add" } else { "matmul_quant" };
        check_within(&format!("{what} Q5_0 [{m} x {k} x {n}]"), 1e-3, &run(&host)?, &run(&gpu)?);
    }

    // `matmul_raw` for every raw-kernel format: random weight bytes through
    // the device's in-kernel decode and the host's dense fallback
    // (`Packed::dense`). `d`/`dmin` are small positive floats, not random
    // bits, so they are never f16 Inf or NaN.
    check_raw_formats(
        &host,
        &gpu,
        &set_dp4a,
        &set_qmma,
        &check_spread,
        &check_within,
        &check_tc,
        &mut next,
    )?;

    // The convolution feeding the delta rule, in both fused layouts. Watch the
    // one-row shapes: there the carried positions are most of the input, so a
    // padding off-by-one still gives a plausible number.
    for (rows, heads, kv_heads, head_dim, interleaved, normalize) in [
        (1usize, 16usize, 16usize, 128usize, false, true),
        (7, 16, 16, 128, false, true),
        (5, 4, 4, 32, true, true),
        (3, 4, 4, 32, false, false),
        // A prompt, where a program carries eight positions at once rather than
        // one, and a length that only reaches half of that.
        (16, 16, 16, 128, false, true),
        (12, 4, 4, 32, true, true),
        (8, 4, 4, 32, false, false),
        // Grouped-query deltanet: 48 value heads sharing 16 key/query heads
        // (UD-IQ1_M's shape), plus a smaller group.
        (1, 48, 16, 128, false, true),
        (9, 48, 16, 128, false, true),
        (1, 6, 2, 32, false, false),
        (5, 6, 2, 32, false, true),
    ] {
        let kv_width = kv_heads * head_dim;
        let (planes, head_stride) = if interleaved {
            ([0, head_dim, 2 * head_dim], 3 * head_dim)
        } else {
            ([0, kv_width, 2 * kv_width], head_dim)
        };
        let mix = DeltaMix {
            rows,
            heads,
            head_dim,
            kv_heads,
            kernel: 4,
            planes,
            head_stride,
            normalize,
            query_scale: (head_dim as f32).sqrt().recip(),
        };
        let history: Vec<f32> = (0..mix.history_len()).map(|_| next()).collect();
        let taps: Vec<f32> = (0..mix.kernel * mix.channels()).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let stream = b.upload(&history)?;
            let weights = b.upload(&taps)?;
            let packed = b.alloc(mix.packed_len())?;
            b.delta_conv(stream, weights, mix, packed)?;
            read_vec(b, packed, 3 * mix.span())
        };
        let layout = if interleaved { "interleaved" } else { "planar" };
        check(
            &format!("delta_conv [{rows} x {heads} (kv {kv_heads}) x {head_dim}] {layout}"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The gates. The decay input reaches far enough that a softplus written
    // the obvious way overflows.
    for (rows, heads) in [(1usize, 16usize), (9, 16), (64, 4)] {
        let mix = DeltaMix {
            rows,
            heads,
            head_dim: 32,
            kv_heads: heads,
            kernel: 4,
            planes: [0, 0, 0],
            head_stride: 32,
            normalize: false,
            query_scale: 1.0,
        };
        let mut alpha: Vec<f32> = (0..mix.gates()).map(|_| next() * 8.0).collect();
        alpha[0] = 120.0;
        let beta: Vec<f32> = (0..mix.gates()).map(|_| next() * 8.0).collect();
        let rate: Vec<f32> = (0..heads).map(|_| -next().abs()).collect();
        let bias: Vec<f32> = (0..heads).map(|_| next()).collect();
        let zeros = vec![0.0f32; mix.packed_len()];
        // Both gates come from one stacked projection, so they arrive as
        // windows of one buffer. The decay sits second so its offset is
        // nonzero.
        let stacked: Vec<f32> = beta.iter().chain(&alpha).copied().collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let both = b.upload(&stacked)?;
            let (r, d) = (b.upload(&rate)?, b.upload(&bias)?);
            let packed = b.upload(&zeros)?;
            b.delta_gates(both, mix.gates(), both, 0, r, d, mix, packed)?;
            let all = read_vec(b, packed, mix.packed_len())?;
            Ok(all[3 * mix.span()..].to_vec())
        };
        check(
            &format!("delta_gates [{rows} x {heads}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // Attention. The scan covers whole key tiles, then the rest one key at a
    // time, so totals on and off a tile multiple take different paths.
    for (rows, start_pos, n_head, n_kv, head_dim) in [
        (1usize, 0usize, 16usize, 8usize, 128usize),
        (1, 63, 16, 8, 128),
        (1, 64, 16, 8, 128),
        (1, 100, 16, 8, 128),
        (29, 0, 16, 8, 128),
        (5, 31, 16, 2, 64),
        (7, 0, 4, 4, 32),
        // The model's own shape, and the one the blocked kernel runs on: a
        // prompt whose length is a whole number of query blocks, and one that
        // leaves a partial block at the end.
        (64, 0, 8, 4, 256),
        (29, 0, 8, 4, 256),
        (1, 40, 8, 4, 256),
        // Decode deep enough that the key axis is split several ways, with
        // one piece holding a partial key tile that the merge pass must
        // rescale. The last is shorter than the split count, so most pieces
        // see no keys at all.
        (1, 300, 8, 4, 256),
        (1, 511, 8, 4, 256),
        (1, 3, 8, 4, 256),
        // Prompts whose rows, cache and head dimension all tile by 64, as the
        // matmul path needs: into an empty cache, onto a cache a whole number
        // of tiles deep, and exactly one tile. The 96-row case tiles by 32
        // but not 64 and must not take that path.
        (128, 0, 8, 4, 256),
        (128, 64, 8, 4, 256),
        (64, 0, 8, 4, 256),
        (96, 0, 8, 4, 256),
        // A llama block's shape at eight query heads per key head, decoding and
        // continuing a prompt, against a cache several hundred positions deep.
        // Nothing above reaches both a wide group and a deep cache at once.
        (1, 300, 16, 2, 128),
        (1, 512, 16, 2, 128),
        (1, 600, 16, 2, 128),
        (8, 512, 16, 2, 128),
        (88, 512, 16, 2, 128),
        // The 4B's attention block, sixteen query heads over four key heads:
        // a pp512 prompt, and a continuation onto a cache two tiles deep.
        (512, 0, 16, 4, 256),
        (192, 128, 16, 4, 256),
        // Ragged prompts: the tensor cores take the whole tiles and the
        // blocked kernel the rest, continuing from where the tiles end.
        (300, 0, 16, 4, 256),
        (65, 64, 8, 4, 256),
    ] {
        let spec = Attn {
            rows,
            start_pos,
            n_head,
            n_kv,
            head_dim,
        };
        let cached = spec.total() * spec.kv_width();
        let q: Vec<f32> = (0..rows * n_head * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..cached).map(|_| next()).collect();
        let v: Vec<f32> = (0..cached).map(|_| next()).collect();
        // Fill the caches as a block does, through f32 and the narrowing
        // store. Both backends then round the same way, and only the
        // arithmetic is compared.
        let cache = |b: &dyn Backend, data: &[f32]| -> Result<HBuf> {
            let width = spec.kv_width();
            let dense = b.upload(data)?;
            let into = b.alloc_h(data.len())?;
            b.store_2d(
                Plane {
                    buf: dense,
                    offset: 0,
                    pitch: width,
                },
                HPlane {
                    buf: into,
                    offset: 0,
                    pitch: width,
                },
                spec.total(),
                width,
            )?;
            b.release(dense);
            Ok(into)
        };
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (qb, kb, vb) = (b.upload(&q)?, cache(b, &k)?, cache(b, &v)?);
            let out = b.alloc(q.len())?;
            b.attention(qb, kb, vb, spec, out)?;
            read_vec(b, out, q.len())
        };
        // A prompt of at least one tile onto a cache and with a head dimension
        // that tile by 64 takes the tensor-core path for its whole tiles,
        // which rounds the queries, keys and probabilities to f16.
        let tensor_core = rows >= 64 && start_pos.is_multiple_of(64) && head_dim.is_multiple_of(64);
        check_within(
            &format!("attention [{rows} @ {start_pos} x {n_head}/{n_kv} x {head_dim}]"),
            if tensor_core { 3e-3 } else { 1e-4 },
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    for (rows, heads, head_dim, rope_dim, start_pos) in [
        (1usize, 16usize, 128usize, 128usize, 40usize),
        (9, 4, 64, 32, 0),
    ] {
        let table: Vec<f32> = (0..(start_pos + rows) * rope_dim).map(|_| next()).collect();
        let x: Vec<f32> = (0..rows * heads * head_dim).map(|_| next()).collect();
        let spec = Rope {
            heads,
            head_dim,
            rope_dim,
            start_pos,
        };
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, tb) = (b.upload(&x)?, b.upload(&table)?);
            b.rope(xb, rows, tb, spec)?;
            read_vec(b, xb, x.len())
        };
        check(
            &format!("rope [{rows} x {heads} x {head_dim}/{rope_dim}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // A strided window of a fused QKV buffer, offset past other heads in the
    // same row. `llama.rs::attention` passes this for K, and for Q past one
    // row. The first shape has rope_dim == head_dim, as minicpm does. The
    // second has a passthrough range, as Qwen does; Qwen's forward pass never
    // calls this op, so this is the only check of that path.
    for (rows, heads, head_dim, rope_dim, stride_heads, slot, start_pos) in [
        (5usize, 2usize, 32usize, 32usize, 8usize, 4usize, 10usize),
        (9, 4, 64, 32, 8, 4, 0),
    ] {
        let table: Vec<f32> = (0..(start_pos + rows) * rope_dim).map(|_| next()).collect();
        let src: Vec<f32> = (0..rows * stride_heads * head_dim)
            .map(|_| next())
            .collect();
        let spec = Rope {
            heads,
            head_dim,
            rope_dim,
            start_pos,
        };
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (sb, tb) = (b.upload(&src)?, b.upload(&table)?);
            let dest = b.alloc(rows * heads * head_dim)?;
            b.rope_gather(
                Plane {
                    buf: sb,
                    offset: slot * head_dim,
                    pitch: stride_heads * head_dim,
                },
                rows,
                tb,
                spec,
                dest,
            )?;
            read_vec(b, dest, rows * heads * head_dim)
        };
        check(
            &format!(
                "rope_gather [{rows} x {heads} x {head_dim}/{rope_dim} @ slot {slot} of {stride_heads}]"
            ),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    for (rows, width, pitch, offset) in
        [(64usize, 128usize, 256usize, 128usize), (5, 2048, 4096, 0)]
    {
        let src: Vec<f32> = (0..rows * pitch).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let sb = b.upload(&src)?;
            let db = b.alloc(rows * width)?;
            b.copy_2d(
                Plane {
                    buf: sb,
                    offset,
                    pitch,
                },
                Plane {
                    buf: db,
                    offset: 0,
                    pitch: width,
                },
                rows,
                width,
            )?;
            read_vec(b, db, rows * width)
        };
        check(
            &format!("copy_2d [{rows} x {width} of {pitch}]"),
            &run(&host)?,
            &run(&gpu)?,
        );

        // The same block into an f16 cache. Checked on its own, since a
        // rounding bug would look like noise inside attention. The bar is
        // equality, which `f32::MIN_POSITIVE` expresses to this checker.
        let store = |b: &dyn Backend| -> Result<Vec<f32>> {
            let sb = b.upload(&src)?;
            let db = b.alloc_h(rows * width)?;
            b.store_2d(
                Plane {
                    buf: sb,
                    offset,
                    pitch,
                },
                HPlane {
                    buf: db,
                    offset: 0,
                    pitch: width,
                },
                rows,
                width,
            )?;
            let mut out = vec![0.0; rows * width];
            b.read_h(db, &mut out)?;
            Ok(out)
        };
        check_within(
            &format!("store_2d [{rows} x {width} of {pitch}]"),
            f32::MIN_POSITIVE,
            &store(&host)?,
            &store(&gpu)?,
        );
    }

    {
        let n = 3000;
        let x: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let g: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, gb) = (b.upload(&x)?, b.upload(&g)?);
            b.gate_into(xb, gb)?;
            read_vec(b, xb, n)
        };
        check("gate_into", &run(&host)?, &run(&gpu)?);
    }

    // New checks go last. They draw from the same random stream, so one
    // inserted earlier would change every later check's data.
    {
        // A delta net's gate projections, one program unsplit.
        let (k, n) = (5120, 48);
        let a: Vec<f32> = (0..k).map(|_| next()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (ab, wb) = (b.upload(&a)?, b.upload(&w)?);
            let out = b.alloc(n)?;
            b.matmul(ab, 1, k, wb, n, out)?;
            read_vec(b, out, n)
        };
        check(&format!("matmul [1 x {k} x {n}]"), &run(&host)?, &run(&gpu)?);
    }
    check_hadamard(&host, &gpu, &check_within, &mut next)?;

    // The narrow dense projection against a weight held `[n, k]`: a decode
    // row, a prompt of whole tiles, one with a tail, and a width the prompt
    // tile does not divide.
    for (m, k, n) in [(1usize, 5120usize, 48usize), (128, 5120, 48), (37, 5120, 96), (19, 1024, 20), (512, 5120, 96)] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (ab, wb) = (b.upload(&a)?, b.upload(&w)?);
            let out = b.alloc(m * n)?;
            b.matmul_rows(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        check(&format!("matmul_rows [{m} x {k} x {n}]"), &run(&host)?, &run(&gpu)?);
    }

    // The plain norm at the widths a model runs it: attention's per-head
    // norms and the model width.
    for (rows, width) in [(96usize, 256usize), (3, 5120), (1, 5120)] {
        let x: Vec<f32> = (0..rows * width).map(|_| next() * 3.0).collect();
        let gain: Vec<f32> = (0..width).map(|_| next() + 1.5).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, gb) = (b.upload(&x)?, b.upload(&gain)?);
            let out = b.alloc(rows * width)?;
            b.rms_norm(xb, rows, width, gb, eps, out)?;
            read_vec(b, out, rows * width)
        };
        check(&format!("rms_norm [{rows} x {width}]"), &run(&host)?, &run(&gpu)?);
    }

    let _ = Buf(0);
    println!("\nworst relative error {:.3e}", worst.get());
    if failures.get() > 0 {
        anyhow::bail!(
            "{} device ops disagree with the host reference",
            failures.get()
        );
    }
    Ok(())
}
