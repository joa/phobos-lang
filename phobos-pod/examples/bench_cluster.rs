use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use cust::prelude::*;
use phobos_base::context::Context;
use phobos_cluster::proto::{self, TensorInput};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, DataType};
use phobos_sched::server::{DispatchConfig, Scheduler, make_job};

/// The cluster matmul (mirrors examples/matmul_cluster_fp32.ph). The @cluster
/// lower bound is filled per run so default_supers picks the supertile we
/// want; the device-tile @autotune defaults (32/32/4) are shared with the
/// leaf the scheduler compiles.
fn matmul_src(super_lo: usize) -> String {
    format!(
        r#"
@cluster(TILE_M in [{super_lo}, 16384], TILE_N in [{super_lo}, 16384], TILE_K in [{super_lo}, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {{
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {{
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }}
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}}"#
    )
}

/// The same matmul without @cluster, for the bare device launch (device
/// codegen need not know the cluster attribute; the tile dims are pinned via
/// shape_overrides).
const DEVICE_SRC: &str = r#"
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;

fn randoms(n: usize, seed: u64) -> Vec<f32> {
    phobos_base::rng::Lcg::new(seed).unit_f32s(n)
}

fn tensor(name: &str, n: usize, mode: AccessMode, uri: String) -> TensorInput {
    TensorInput {
        name: name.to_string(),
        data_type: proto::data_type_to_i32(DataType::F32),
        shape: vec![n as u64, n as u64],
        mode: proto::am_to_i32(mode),
        uri,
    }
}

/// Mean wall time per call over iters runs, after one warmup.
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
    for (r, col) in [(0usize, 0usize), (1, n / 2 + 1), (n - 1, n - 1), (n / 2, 7)] {
        let want: f64 = (0..n)
            .map(|k| a[r * n + k] as f64 * b[k * n + col] as f64)
            .sum();
        let got = c[r * n + col] as f64;
        ensure!(
            (got - want).abs() / want.abs().max(1.0) < 1e-3,
            "C[{r},{col}] = {got}, CPU says {want}"
        );
    }
    Ok(())
}

/// Compile the matmul to PTX with the device tile pinned (the same 32/32/4 the
/// cluster leaf uses) and launch it once over the whole MxN grid, full K in the
/// kernel's loop. Returns mean execution time; verifies the result.
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
    // strides[2]); C = acc overwrites, so no memset needed.
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

    // Main-thread CUDA context for the bare launches. The pod engine inits its
    // own on its thread; both attach the same device primary context.
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Scheduler + one node over localhost gRPC (shared across all runs).
    let sched = Scheduler::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let sched_addr = listener.local_addr()?.to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let server = sched.clone().into_server();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_incoming(incoming)
            .await
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    {
        let sa = sched_addr.clone();
        tokio::spawn(async move {
            if let Err(e) = phobos_pod::serve(0, sa, "127.0.0.1:0".into(), None, arena).await {
                eprintln!("node 0: {e:#}");
            }
        });
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    let dir = std::env::temp_dir().join("phobos_bench_cluster");
    std::fs::create_dir_all(&dir)?;
    let uri = |f: &str| format!("file://{}", dir.join(f).display());

    println!("phobos single-GPU cluster benchmark");
    if cfg!(debug_assertions) {
        println!("note: debug build; rerun with --release for representative numbers");
    }
    println!(
        "{:>6}  {:>16}  {:>11}  {:>11}  {:>9}",
        "size", "stage", "time", "GFLOP/s", "vs bare"
    );

    for &n in &sizes {
        let a = randoms(n * n, 1);
        let b = randoms(n * n, 2);
        storage::write_tensor_f32(&uri("A.bin"), &a)?;
        storage::write_tensor_f32(&uri("B.bin"), &b)?;
        storage::write_tensor_f32(&uri("C.bin"), &vec![0.0; n * n])?;

        // 1. bare single full-K launch.
        let bare = bare_launch(&stream, n, &a, &b, iters)?;
        println!(
            "{n:>6}  {:>16}  {:>11}  {:>11.1}  {:>8}",
            "bare launch",
            format!("{bare:.2?}"),
            gflops(n, bare),
            "1.00x"
        );

        // 2 & 3. cluster job latency at one supertile (1x1x1) then tiled (2x2x2).
        for (label, super_lo) in [("cluster 1x1x1", n), ("cluster 2x2x2", n / 2)] {
            let source = matmul_src(super_lo);
            let job = || {
                make_job(
                    &source,
                    &[("M", n as i64), ("N", n as i64), ("K", n as i64)],
                    vec![
                        tensor("A", n, AccessMode::Read, uri("A.bin")),
                        tensor("B", n, AccessMode::Read, uri("B.bin")),
                        tensor("C", n, AccessMode::Write, uri("C.bin")),
                    ],
                )
            };
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
