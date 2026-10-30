// What the pod examples share: a scheduler with pods attached in-process over
// localhost gRPC, file:// tensors in a temp directory, and the cluster matmul
// with its CPU reference.

// Each example uses only some of these.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, ensure};
use phobos_cluster::proto::{self, Job, TensorInput};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, DataType};
use phobos_sched::server::{Scheduler, make_job};

/// The cluster matmul, its `@cluster` dims searched from `super_lo` to
/// `super_hi`. `default_supers` takes the lower bound, and each output
/// streams its K/SUPER k-steps rather than holding them all resident. The
/// `@autotune` defaults (32/32/4) are the device tile the scheduler compiles
/// the leaf at.
pub fn matmul_src(super_lo: usize, super_hi: usize) -> String {
    format!(
        r#"
@cluster(TILE_M in [{super_lo}, {super_hi}], TILE_N in [{super_lo}, {super_hi}], TILE_K in [{super_lo}, {super_hi}])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {{
    let pm = ctx.block.x
    let pn = ctx.block.y
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

/// The matmul's dims at M = N = K = `n`, as the planner takes them.
pub fn matmul_dims(n: usize) -> HashMap<String, i64> {
    ["M", "N", "K"]
        .into_iter()
        .map(|d| (d.to_string(), n as i64))
        .collect()
}

/// The job computing C = A @ B, each `n` x `n`, from `source` over the
/// tensors at `uri("A.bin")`, `uri("B.bin")` and `uri("C.bin")`.
pub fn matmul_job(source: &str, n: usize, uri: impl Fn(&str) -> String) -> Job {
    make_job(
        source,
        &[("M", n as i64), ("N", n as i64), ("K", n as i64)],
        vec![
            f32_tensor("A", &[n, n], AccessMode::Read, uri("A.bin")),
            f32_tensor("B", &[n, n], AccessMode::Read, uri("B.bin")),
            f32_tensor("C", &[n, n], AccessMode::Write, uri("C.bin")),
        ],
    )
}

/// Writes a random A and B (seeds 1 and 2) and a zeroed C, each `n` x `n`,
/// where [`matmul_job`] reads them, and returns A and B for the reference.
pub fn seed_matmul(uri: impl Fn(&str) -> String, n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let a = randoms(n * n, 1);
    let b = randoms(n * n, 2);
    storage::write_tensor_f32(&uri("A.bin"), &a)?;
    storage::write_tensor_f32(&uri("B.bin"), &b)?;
    storage::write_tensor_f32(&uri("C.bin"), &vec![0.0; n * n])?;
    Ok((a, b))
}

/// Checks C = A @ B, each `n` x `n`, at `cells` against an f64 reference.
pub fn check_matmul(c: &[f32], a: &[f32], b: &[f32], n: usize, cells: &[(usize, usize)]) -> Result<()> {
    for &(r, col) in cells {
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

pub fn randoms(n: usize, seed: u64) -> Vec<f32> {
    phobos_base::rng::Lcg::new(seed).unit_f32s(n)
}

pub fn f32_tensor(name: &str, shape: &[usize], mode: AccessMode, uri: String) -> TensorInput {
    TensorInput {
        name: name.to_string(),
        data_type: proto::data_type_to_i32(DataType::F32),
        shape: shape.iter().map(|&d| d as u64).collect(),
        mode: proto::am_to_i32(mode),
        uri,
    }
}

/// Creates `dir` under the temp directory and returns the file:// URI of a
/// file in it.
pub fn temp_uris(dir: &str) -> Result<impl Fn(&str) -> String> {
    let dir = std::env::temp_dir().join(dir);
    std::fs::create_dir_all(&dir)?;
    Ok(move |file: &str| format!("file://{}", dir.join(file).display()))
}

/// A scheduler serving on localhost, with `nodes` pods attached in-process,
/// each with an arena of `arena_bytes`. Returns the scheduler and its
/// address. Every port is OS-assigned, since Windows reserves many fixed
/// ranges, and the pods register the address they bound.
pub async fn start_cluster(nodes: u16, arena_bytes: usize) -> Result<(Scheduler, String)> {
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
    for id in 0..nodes {
        let sa = sched_addr.clone();
        tokio::spawn(async move {
            if let Err(e) = phobos_pod::serve(id, sa, "127.0.0.1:0".into(), None, arena_bytes).await {
                eprintln!("node {id}: {e:#}");
            }
        });
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    Ok((sched, sched_addr))
}
