// The fp32 and fp16 GEMM benchmarks, each with its own verifier.

use crate::harness::*;
use crate::*;

/// Spot-checks a sample of output elements against an f64 reference,
/// alpha * (A*B)[i,j] + beta * c_in[i,j]. A full check would cost O(M*N*K).
///
/// The f32 kernel checks at 1e-3 relative tolerance. The tensor-core kernel
/// also rounds inputs to fp16, so its error is normalized by
/// max(|want|, sqrt(K)/3), the dot's RMS magnitude, at 1e-2 tolerance.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
pub(crate) fn verify_matmul(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    c_in: &[f32],
    alpha: f64,
    beta: f64,
    M: usize,
    N: usize,
    K: usize,
    fp16: bool,
) -> anyhow::Result<()> {
    let (tol, floor) = if fp16 {
        (1e-2, (K as f64).sqrt() / 3.0)
    } else {
        (1e-3, f64::MIN_POSITIVE)
    };
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, j) = match s {
            0 => (0, 0),
            1 => (M - 1, N - 1),
            _ => (sample(M), sample(N)),
        };
        let (want, got) = if fp16 {
            // Reference sees the same fp16-rounded inputs the tensor cores do.
            let mut dot = 0.0f64;
            for k in 0..K {
                dot += fp16_round(a[i * K + k]) as f64 * fp16_round(b[k * N + j]) as f64;
            }
            (
                alpha * dot + beta * c_in[i * N + j] as f64,
                c[i * N + j] as f64,
            )
        } else {
            let mut dot = 0.0f64;
            for k in 0..K {
                dot += a[i * K + k] as f64 * b[k * N + j] as f64;
            }
            (
                alpha * dot + beta * c_in[i * N + j] as f64,
                c[i * N + j] as f64,
            )
        };
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "c[{i},{j}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // gemm dims plus the result sink
pub(crate) fn bench_gemm_fp32(
    stream: &cust::stream::Stream,
    code: &'static str,
    name: &str,
    label: &str,
    fp16: bool,
    alpha: f32,
    beta: f32,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(code)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let M: i32 = 4096;
    let N: i32 = 4096;
    let K: i32 = 4096;
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);

    let mut rng = SmallRng::seed_from_u64(42);
    let a: Vec<f32> = (0..(M * K)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let b: Vec<f32> = (0..(K * N)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();

    let c_in = vec![0.0f32; (M * N) as usize];
    let mut c = c_in.clone();

    let a_dev = a.as_slice().as_dbuf()?;
    let b_dev = b.as_slice().as_dbuf()?;
    let c_dev = c.as_slice().as_dbuf()?;

    let (a_ptr, b_ptr, c_ptr) = (
        a_dev.as_device_ptr(),
        b_dev.as_device_ptr(),
        c_dev.as_device_ptr(),
    );

    // Must match the kernel's @launch thread count (.maxntid); more is a hard error.
    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // The default @tensorcore path (mma.sync) compiles at 64-bit index, so
    // the host's memref descriptors must be 64-bit too.
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code,
        grid_for: |cfg: &[autotune::Setting]| {
            let tile_m = autotune::cfg_value(cfg, "TILE_M")? as u32;
            let tile_n = autotune::cfg_value(cfg, "TILE_N")? as u32;
            anyhow::ensure!(
                (M as u32).is_multiple_of(tile_m) && (N as u32).is_multiple_of(tile_n),
                "tile does not divide the problem size"
            );
            Ok(autotune::Grid(M as u32 / tile_m, N as u32 / tile_n))
        },

        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            zero_device_async(c_ptr.as_raw(), (M * N) as usize * 4, stream)?;
            let func = module.get_function("gemm")?;
            launch_gemm(
                &func,
                (grid.0, grid.1),
                block,
                stream,
                a_ptr.as_raw(),
                b_ptr.as_raw(),
                c_ptr.as_raw(),
                M,
                N,
                K,
                alpha,
                beta,
                wide,
            )
        },
        verify: || {
            c_dev.copy_to(&mut c)?;
            verify_matmul(
                &c,
                &a,
                &b,
                &c_in,
                alpha as f64,
                beta as f64,
                M as usize,
                N as usize,
                K as usize,
                fp16,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench(label, || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS sgemm ", || {
        zero_device_async(c_ptr.as_raw(), (M * N) as usize * 4, stream)?;
        blas.matmul(
            M,
            N,
            K,
            a_dev.as_device_ptr().as_raw(),
            b_dev.as_device_ptr().as_raw(),
            c_ptr.as_raw(),
            alpha,
            beta,
        )?;
        stream.synchronize()?;
        Ok(())
    })?;
    c_dev.copy_to(&mut c)?;

    verify_matmul(
        &c,
        &a,
        &b,
        &c_in,
        alpha as f64,
        beta as f64,
        M as usize,
        N as usize,
        K as usize,
        false,
    )?;

    phinfo!("check: {} elements, both correct", M * N);
    let gflop = (2.0 * m64 as f64 * n64 as f64 * k64 as f64 + m64 as f64 * n64 as f64) / 1e9;
    phinfo!(
        "phobos: {:.1} GFLOP/s, cuBLAS: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x / {:.2}%",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64(),
        100.0f64 * cublas_avg.as_secs_f64() / phobos_avg.as_secs_f64()
    );

    // The tensor-core path rounds inputs to fp16, so phobos is measured
    // against the fp16 f32-acc peak. The cuBLAS baseline is f32 sgemm.
    let phobos_prec = if fp16 {
        Precision::F16TcF32
    } else {
        Precision::F32
    };
    results.push(
        name,
        "phobos",
        phobos_prec,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        name,
        "cuBLAS",
        Precision::F32,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}

/// Reference for the fp16-accumulate GEMM (examples/gemm_fp16.ph). Rounds
/// inputs to fp16 and accumulates in fp16 like the kernel's WMMA fragment,
/// then scales in f32 and rounds back to fp16.
///
/// fp16 accumulation is imprecise, so error is normalized by
/// max(|want|, sqrt(K)/3), the dot's RMS magnitude, at a loose 1.5e-1
/// tolerance.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_matmul_fp16acc(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    c_in: &[f32],
    alpha: f32,
    beta: f32,
    M: usize,
    N: usize,
    K: usize,
) -> anyhow::Result<()> {
    let (tol, floor) = (1.5e-1f64, (K as f64).sqrt() / 3.0);
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, j) = match s {
            0 => (0, 0),
            1 => (M - 1, N - 1),
            _ => (sample(M), sample(N)),
        };
        let mut acc = 0.0f32;
        for k in 0..K {
            let af = fp16_round(a[i * K + k]);
            let bf = fp16_round(b[k * N + j]);
            acc = fp16_round(acc + af * bf);
        }
        let want = fp16_round(alpha * acc + beta * c_in[i * N + j]) as f64;
        let got = c[i * N + j] as f64;
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "c[{i},{j}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

/// Half-precision GEMM benchmark (examples/gemm_fp16.ph): fp16 A, B, C and
/// fp16 accumulation on the tensor cores. Device data is fp16, uploaded as
/// u16 bit patterns. Compared against cublasHgemm, which also takes fp16
/// operands and accumulates in fp16.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // gemm dims plus the result sink
pub(crate) fn bench_gemm_fp16(
    stream: &cust::stream::Stream,
    code: &'static str,
    name: &str,
    label: &str,
    alpha: f32,
    beta: f32,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(code)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let M: i32 = 4096;
    let N: i32 = 4096;
    let K: i32 = 4096;
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);

    let mut rng = SmallRng::seed_from_u64(42);
    let a: Vec<f32> = (0..(M * K)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let b: Vec<f32> = (0..(K * N)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let c_in = vec![0.0f32; (M * N) as usize];

    // fp16 device tensors, uploaded as u16 bit patterns.
    let a_dev = to_fp16_bits(&a).as_slice().as_dbuf()?;
    let b_dev = to_fp16_bits(&b).as_slice().as_dbuf()?;
    let c_dev = vec![0u16; (M * N) as usize].as_slice().as_dbuf()?;
    let mut c_bits = vec![0u16; (M * N) as usize];

    let (a_ptr, b_ptr, c_ptr) = (
        a_dev.as_device_ptr(),
        b_dev.as_device_ptr(),
        c_dev.as_device_ptr(),
    );

    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore (mma.sync) compiles at 64-bit index, which widens the
    // memref descriptor fields to i64. Host metadata must match, or the
    // kernel reads garbage past the first pointer. The WMMA opt-out,
    // @tensorcore(wmma), stays 32-bit.
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code,
        grid_for: |cfg: &[autotune::Setting]| {
            let tile_m = autotune::cfg_value(cfg, "TILE_M")? as u32;
            let tile_n = autotune::cfg_value(cfg, "TILE_N")? as u32;
            anyhow::ensure!(
                (M as u32).is_multiple_of(tile_m) && (N as u32).is_multiple_of(tile_n),
                "tile does not divide the problem size"
            );
            Ok(autotune::Grid(M as u32 / tile_m, N as u32 / tile_n))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            // Zero C with a byte memset, 2 bytes per element.
            zero_device_async(c_ptr.as_raw(), (M * N) as usize * 2, stream)?;
            let func = module.get_function("gemm")?;
            launch_gemm(
                &func,
                (grid.0, grid.1),
                block,
                stream,
                a_ptr.as_raw(),
                b_ptr.as_raw(),
                c_ptr.as_raw(),
                M,
                N,
                K,
                alpha,
                beta,
                wide,
            )
        },
        verify: || {
            c_dev.copy_to(&mut c_bits)?;
            let c = from_fp16_bits(&c_bits);
            verify_matmul_fp16acc(
                &c, &a, &b, &c_in, alpha, beta, M as usize, N as usize, K as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench(label, || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS hgemm ", || {
        // Zero C, 2 bytes per element.
        zero_device_async(c_ptr.as_raw(), (M * N) as usize * 2, stream)?;
        blas.matmul_fp16(
            M,
            N,
            K,
            a_dev.as_device_ptr().as_raw(),
            b_dev.as_device_ptr().as_raw(),
            c_ptr.as_raw(),
            alpha,
            beta,
        )?;
        stream.synchronize()?;
        Ok(())
    })?;
    c_dev.copy_to(&mut c_bits)?;
    let c = from_fp16_bits(&c_bits);
    verify_matmul_fp16acc(
        &c, &a, &b, &c_in, alpha, beta, M as usize, N as usize, K as usize,
    )?;

    phinfo!("check: {} elements, both correct", M * N);
    let gflop = (2.0 * m64 as f64 * n64 as f64 * k64 as f64 + m64 as f64 * n64 as f64) / 1e9;
    phinfo!(
        "phobos gemm_fp16: {:.1} GFLOP/s, cuBLAS hgemm: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x / {:.2}%",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64(),
        100.0f64 * cublas_avg.as_secs_f64() / phobos_avg.as_secs_f64()
    );

    // Both sides are fp16 on the tensor cores.
    results.push(
        name,
        "phobos",
        Precision::F16Tc,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        name,
        "cuBLAS",
        Precision::F16Tc,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}
