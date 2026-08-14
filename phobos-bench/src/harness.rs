// Timing a kernel and launching it: what every benchmark below shares.

use crate::*;

/// Times `launch` and returns the fastest of N runs with the spread beside it.
/// Run-to-run variance is external noise (clock ramp, scheduling, driver work)
/// that only ever slows a sample, so the minimum is the truest measure. This is
/// also the autotuner's ranking metric, so the winner it picks is the one
/// reported here.
pub(crate) fn bench(
    name: &str,
    mut launch: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<(Duration, Duration)> {
    const N: u32 = 100;

    for _ in 0..10 {
        launch()?;
    }

    let (mut min, mut max) = (f64::INFINITY, 0f64);
    for _ in 0..N {
        let now = Instant::now();
        launch()?;
        let dt = now.elapsed().as_secs_f64();
        min = min.min(dt);
        max = max.max(dt);
    }
    let (d, s) = (
        Duration::from_secs_f64(min),
        Duration::from_secs_f64(max - min),
    );
    phinfo!("{name}: {d:.2?} (spread {s:.2?})");
    Ok((d, s))
}

/// Zeroes `bytes` at `ptr`. A byte memset to 0 is +0.0 for both f32 and fp16.
pub(crate) fn zero_device_async(
    ptr: cust::sys::CUdeviceptr,
    bytes: usize,
    stream: &cust::stream::Stream,
) -> anyhow::Result<()> {
    let r = unsafe { cust::sys::cuMemsetD8Async(ptr, 0, bytes, stream.as_inner()) };
    anyhow::ensure!(
        r == cust::sys::CUresult::CUDA_SUCCESS,
        "cuMemsetD8Async failed: {:?}",
        r
    );
    Ok(())
}

/// Launches a matmul kernel over C[M,N] = A[M,K] @ B[K,N], dispatching on `wide`
/// so the memref descriptor's offset/size/stride fields match the kernel's index
/// width: i64 for the default @tensorcore mma.sync path, i32 otherwise. Get that
/// wrong and every field after the first pointer shifts. Pointers are raw device
/// addresses either way, so this is element-type agnostic.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gemm(
    func: &cust::function::Function,
    grid: (u32, u32),
    block: u32,
    stream: &cust::stream::Stream,
    a: cust::sys::CUdeviceptr,
    b: cust::sys::CUdeviceptr,
    c: cust::sys::CUdeviceptr,
    M: i32,
    N: i32,
    K: i32,
    alpha: f32,
    beta: f32,
    wide: bool,
) -> anyhow::Result<()> {
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);
    unsafe {
        if wide {
            launch!(func<<<grid, block, 0, stream>>>(
                a, a, 0i64, m64, k64, k64, 1i64,
                b, b, 0i64, k64, n64, n64, 1i64,
                c, c, 0i64, m64, n64, n64, 1i64,
                alpha,
                beta
            ))?;
        } else {
            launch!(func<<<grid, block, 0, stream>>>(
                a, a, 0i32, M, K, K, 1i32,
                b, b, 0i32, K, N, N, 1i32,
                c, c, 0i32, M, N, N, 1i32,
                alpha,
                beta
            ))?;
        }
    }
    stream.synchronize()?;
    Ok(())
}

/// Launches a flash_attention kernel over Q/K/V/O (each [rows, D]),
/// dispatching on wide exactly as [`launch_gemm`] does. Pointers are raw
/// device addresses, so this serves both the f32 and fp16 benches.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_flash(
    func: &cust::function::Function,
    grid_x: u32,
    block: u32,
    stream: &cust::stream::Stream,
    q: cust::sys::CUdeviceptr,
    k: cust::sys::CUdeviceptr,
    v: cust::sys::CUdeviceptr,
    o: cust::sys::CUdeviceptr,
    Nq: i32,
    Nk: i32,
    D: i32,
    scale: f32,
    wide: bool,
) -> anyhow::Result<()> {
    let (nq64, nk64, d64) = (Nq as i64, Nk as i64, D as i64);
    unsafe {
        if wide {
            launch!(func<<<grid_x, block, 0, stream>>>(
                q, q, 0i64, nq64, d64, d64, 1i64,
                k, k, 0i64, nk64, d64, d64, 1i64,
                v, v, 0i64, nk64, d64, d64, 1i64,
                o, o, 0i64, nq64, d64, d64, 1i64,
                scale
            ))?;
        } else {
            launch!(func<<<grid_x, block, 0, stream>>>(
                q, q, 0i32, Nq, D, D, 1i32,
                k, k, 0i32, Nk, D, D, 1i32,
                v, v, 0i32, Nk, D, D, 1i32,
                o, o, 0i32, Nq, D, D, 1i32,
                scale
            ))?;
        }
    }
    stream.synchronize()?;
    Ok(())
}

pub(crate) fn fp16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// fp32 -> fp16 bit pattern, for uploading host data to an fp16 device tensor.
pub(crate) fn to_fp16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

/// Reads back an fp16 device tensor (u16 bit patterns) as fp32.
pub(crate) fn from_fp16_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}
