mod common;

use std::time::{Duration, Instant};

use anyhow::Result;
use cust::prelude::*;
use phobos_base::context::Context;
use phobos_cluster::storage;
use phobos_sched::server::DispatchConfig;

use common::{check_matmul, matmul_job, matmul_src, seed_matmul, start_cluster, temp_uris};

/// The matmul of [`matmul_src`] without `@cluster`, for the bare device
/// launch. The tile dims are pinned through `shape_overrides`.
const DEVICE_SRC: &str = r#"
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
    let pm = ctx.block.x
    let pn = ctx.block.y
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;

/// Mean wall time per call over `iters` runs, after one warmup.
fn timed(iters: u32, mut f: impl FnMut() -> Result<()>) -> Result<Duration> {
    f()?;
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(start.elapsed() / iters)
}

/// Spot-check C = A @ B on a spread of entries against an f64 reference.
fn verify(c: &[f32], a: &[f32], b: &[f32], n: usize) -> Result<()> {
    check_matmul(c, a, b, n, &[(0, 0), (1, n / 2 + 1), (n - 1, n - 1), (n / 2, 7)])
}

/// Compile the matmul with the cluster leaf's device tile (32/32/4) and launch
/// it over the whole MxN grid, with all of K in the kernel's loop. Verifies the
/// result and returns the mean execution time.
fn bare_launch(stream: &Stream, n: usize, a: &[f32], b: &[f32], iters: u32) -> Result<Duration> {
    const DEV: i32 = 32; // device tile (matches the leaf's @autotune default)
    let ctx = Context {
        shape_overrides: [("TILE_M", 32i64), ("TILE_N", 32), ("TILE_K", 4)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        ..Default::default()
    };
    let ptx = phobos_lang::compile(&ctx, DEVICE_SRC)?;
    let kernel = phobos_lang::parse(DEVICE_SRC)?.remove(0);
    let block = kernel.cta_threads().map_err(anyhow::Error::msg)? as u32;

    let module = Module::from_ptx(ptx.as_str(), &[])?;
    let func = module.get_function("matmul")?;

    let a_dev = a.as_dbuf()?;
    let b_dev = b.as_dbuf()?;
    let c_dev = vec![0.0f32; n * n].as_slice().as_dbuf()?;
    let (ap, bp, cp) = (
        a_dev.as_device_ptr(),
        b_dev.as_device_ptr(),
        c_dev.as_device_ptr(),
    );
    let ni = n as i32;
    let grid = (ni as u32 / DEV as u32, ni as u32 / DEV as u32);

    // Each tensor is a memref<?x?xf32>: (alloc, aligned, offset, sizes[2],
    // strides[2]). The kernel overwrites C, so it needs no memset.
    let launch = || -> Result<()> {
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                ap, ap, 0i32, ni, ni, ni, 1i32,
                bp, bp, 0i32, ni, ni, ni, 1i32,
                cp, cp, 0i32, ni, ni, ni, 1i32
            ))?;
        }
        stream.synchronize()?;
        Ok(())
    };

    let dt = timed(iters, launch)?;
    let mut c = vec![0.0f32; n * n];
    c_dev.copy_to(&mut c)?;
    verify(&c, a, b, n)?;
    Ok(dt)
}

fn gflops(n: usize, dt: Duration) -> f64 {
    (2.0 * n as f64 * n as f64 * n as f64) / 1e9 / dt.as_secs_f64()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let sizes = [2048usize, 4096];
    let arena = 1usize << 30; // 1 GiB per node
    let iters = 5;

    // Main-thread CUDA context for the bare launches. The pod engine makes its
    // own on its thread, and both attach the device's primary context.
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Scheduler and one node over localhost gRPC, shared across all runs.
    let (sched, _) = start_cluster(1, arena).await?;

    let uri = temp_uris("phobos_bench_cluster")?;

    println!("phobos single-GPU cluster benchmark");
    if cfg!(debug_assertions) {
        println!("note: debug build; rerun with --release for representative numbers");
    }
    println!(
        "{:>6}  {:>16}  {:>11}  {:>11}  {:>9}",
        "size", "stage", "time", "GFLOP/s", "vs bare"
    );

    for &n in &sizes {
        let (a, b) = seed_matmul(&uri, n)?;

        let bare = bare_launch(&stream, n, &a, &b, iters)?;
        println!(
            "{n:>6}  {:>16}  {:>11}  {:>11.1}  {:>8}",
            "bare launch",
            format!("{bare:.2?}"),
            gflops(n, bare),
            "1.00x"
        );

        for (label, super_lo) in [("cluster 1x1x1", n), ("cluster 2x2x2", n / 2)] {
            let source = matmul_src(super_lo, 16384);
            let job = || matmul_job(&source, n, &uri);
            let cfg = || DispatchConfig {
                nodes: 1,
                ..Default::default()
            };
            // Warm the pod's PTX cache, then time, then verify the last run.
            sched.dispatch(job(), cfg()).await?;
            let start = Instant::now();
            for _ in 0..iters {
                sched.dispatch(job(), cfg()).await?;
            }
            let dt = start.elapsed() / iters;

            let c = storage::read_tensor_f32(&uri("C.bin"), n * n)?;
            verify(&c, &a, &b, n)?;

            println!(
                "{n:>6}  {:>16}  {:>11}  {:>11.1}  {:>7.1}x",
                label,
                format!("{dt:.2?}"),
                gflops(n, dt),
                dt.as_secs_f64() / bare.as_secs_f64()
            );
        }
    }

    let _ = &_ctx; // keep the main-thread context alive to here
    Ok(())
}
