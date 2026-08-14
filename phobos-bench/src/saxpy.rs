// The saxpy benchmark, the one that is purely bandwidth-bound.

use crate::harness::*;
use crate::*;

/// SAXPY reference: out = alpha * x + y. The kernel lowers to a separate
/// mulf then addf in f32 (no fused multiply-add), so an exact f32 match is
/// expected.
pub(crate) fn verify_saxpy(out: &[f32], x: &[f32], y: &[f32], alpha: f32) {
    for i in 0..out.len() {
        let want = alpha * x[i] + y[i];
        assert_eq!(
            out[i], want,
            "expected {}, got {} at index {}",
            want, out[i], i
        );
    }
}

pub(crate) fn bench_saxpy(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_SAXPY)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let n: i32 = 1 << 25;
    let alpha: f32 = 2.0;

    let mut rng = SmallRng::seed_from_u64(42);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let mut out = vec![0.0f32; n as usize];

    let x_dev = x.as_slice().as_dbuf()?;
    let y_dev = y.as_slice().as_dbuf()?;
    let out_dev = out.as_slice().as_dbuf()?;

    let (x_ptr, y_ptr, out_ptr) = (
        x_dev.as_device_ptr(),
        y_dev.as_device_ptr(),
        out_dev.as_device_ptr(),
    );

    let block: u32 = 1024;

    let mut tuner = autotune::Autotuner {
        code: CODE_SAXPY,
        grid_for: |cfg: &[autotune::Setting]| {
            let block_size = autotune::cfg_value(cfg, "BLOCK")? as u32;
            Ok(autotune::Grid((n as u32).div_ceil(block_size), 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            let func = module.get_function("saxpy")?;
            let grid_x = grid.0;
            // Kernel params: x, y, out (each a tensor<f32>[N] => 5-scalar
            // memref descriptor), then the alpha scalar.
            unsafe {
                launch!(func<<<grid_x, block, 0, stream>>>(
                    x_ptr, x_ptr, 0i32, n, 1i32,
                    y_ptr, y_ptr, 0i32, n, 1i32,
                    out_ptr, out_ptr, 0i32, n, 1i32,
                    alpha
                ))?;
            }
            stream.synchronize()?;
            Ok(())
        },
        verify: || {
            out_dev.copy_to(&mut out)?;
            verify_saxpy(&out, &x, &y, alpha);
            Ok(())
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;

    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos saxpy", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    // cuBLAS saxpy is in-place (y := alpha*x + y), so it runs on a dedicated
    // buffer. The timed loop lets it accumulate (irrelevant to the runtime);
    // correctness is checked separately on a freshly reset buffer.
    let mut yb = y.clone();
    let mut yb_dev = y.as_slice().as_dbuf()?;
    let yb_ptr = yb_dev.as_device_ptr();

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS saxpy", || {
        blas.saxpy(n, alpha, x_dev.as_device_ptr().as_raw(), yb_ptr.as_raw())?;
        stream.synchronize()?;
        Ok(())
    })?;
    yb_dev.copy_from(y.as_slice())?;
    blas.saxpy(n, alpha, x_dev.as_device_ptr().as_raw(), yb_ptr.as_raw())?;
    stream.synchronize()?;
    yb_dev.copy_to(&mut yb)?;
    verify_saxpy(&yb, &x, &y, alpha);

    phinfo!("check: {} elements, both correct", n);
    // SAXPY is one fused multiply-add per element: 2 flops. It is memory-bound,
    // so this sits far below the f32 FLOP peak, but it is reported like any
    // other bench.
    let gflop = 2.0 * n as f64 / 1e9;
    phinfo!(
        "phobos saxpy: {:.1} GFLOP/s, cuBLAS saxpy: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64()
    );

    results.push(
        "saxpy_fp32",
        "phobos",
        Precision::F32,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        "saxpy_fp32",
        "cuBLAS",
        Precision::F32,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}
