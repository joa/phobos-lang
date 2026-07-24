use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Result, ensure};
use phobos_cluster::proto::{self, TensorInput};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, DataType};
use phobos_pod::server::{GET_KERNEL_CALLS, SERVED_BYTES};
use phobos_sched::server::{DispatchConfig, Scheduler, make_job};

const MATMUL: &str = r#"
@cluster(TILE_M in [512, 16384], TILE_N in [512, 16384], TILE_K in [512, 16384])
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

#[tokio::main(flavor = "multi_thread", worker_threads = 6)]
async fn main() -> Result<()> {
    const N: usize = 1024;
    let arena = 256 << 20;

    // Seed file:// tensors in a temp dir.
    let dir = std::env::temp_dir().join("phobos_m1");
    std::fs::create_dir_all(&dir)?;
    let uri = |f: &str| format!("file://{}", dir.join(f).display());
    let a = randoms(N * N, 1);
    let b = randoms(N * N, 2);
    storage::write_tensor_f32(&uri("A.bin"), &a)?;
    storage::write_tensor_f32(&uri("B.bin"), &b)?;
    storage::write_tensor_f32(&uri("C.bin"), &vec![0.0; N * N])?;

    // The analytic peer minimum for owner-computes (for the bytes assertion).
    let kernel = phobos_lang::parse(MATMUL)?.remove(0);
    let program = phobos_cluster::compile(&kernel)?;
    let supers = phobos_sched::default_supers(&program);
    let dims = [("M", N as i64), ("N", N as i64), ("K", N as i64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    // This exercises the peer data plane, so force the home-LOAD/peer-FETCH
    // ingest policy (the default DirectLoad would have every node LOAD its own
    // inputs and move zero bytes between peers).
    let pl = phobos_sched::plan_with(
        &program,
        &dims,
        &supers,
        2,
        phobos_sched::IngestPolicy::HomeLoadPeerFetch,
    )?;
    let want_bytes = pl.fetch_bytes;

    // Scheduler gRPC server on an OS-assigned port (Windows reserves many
    // fixed ranges); nodes also listen on :0 and register their bound addr.
    let sched = Scheduler::new();
    let sched_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let sched_addr = sched_listener.local_addr()?.to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(sched_listener);
    let server = sched.clone().into_server();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_incoming(incoming)
            .await
    });
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Two nodes.
    for id in 0..2u16 {
        let sa = sched_addr.clone();
        tokio::spawn(async move {
            if let Err(e) = phobos_pod::serve(id, sa, "127.0.0.1:0".into(), None, arena).await {
                eprintln!("node {id}: {e:#}");
            }
        });
    }

    SERVED_BYTES.store(0, Ordering::SeqCst);
    GET_KERNEL_CALLS.store(0, Ordering::SeqCst);

    // Dispatch: withhold the step leaf (kernel 0) from node 1 so it must peer
    // GetKernel-fill from node 0.
    let job = make_job(
        MATMUL,
        &[("M", N as i64), ("N", N as i64), ("K", N as i64)],
        vec![
            tensor("A", N, AccessMode::Read, uri("A.bin")),
            tensor("B", N, AccessMode::Read, uri("B.bin")),
            tensor("C", N, AccessMode::Write, uri("C.bin")),
        ],
    );
    let cfg = DispatchConfig {
        nodes: 2,
        withhold: vec![(1, 0)],
        ingest: phobos_sched::IngestPolicy::HomeLoadPeerFetch,
        ..Default::default()
    };
    sched.dispatch(job, cfg).await?;

    // Verify C against CPU ground truth on a spread of entries.
    let c = storage::read_tensor_f32(&uri("C.bin"), N * N)?;
    for (r, col) in [
        (0usize, 0usize),
        (1, 513),
        (511, 512),
        (512, 511),
        (1023, 1023),
    ] {
        let want: f64 = (0..N)
            .map(|k| a[r * N + k] as f64 * b[k * N + col] as f64)
            .sum();
        let got = c[r * N + col] as f64;
        ensure!(
            (got - want).abs() / want.abs().max(1.0) < 1e-3,
            "C[{r},{col}] = {got}, CPU says {want}"
        );
    }

    let served = SERVED_BYTES.load(Ordering::SeqCst);
    ensure!(
        served == want_bytes,
        "peer bytes served {served} != analytic owner-computes minimum {want_bytes}"
    );
    let gk = GET_KERNEL_CALLS.load(Ordering::SeqCst);
    ensure!(gk >= 1, "GetKernel was never exercised (got {gk})");

    println!(
        "cluster_grpc OK: 2 nodes, {} MiB peer transfer = analytic minimum, {gk} GetKernel fill(s)",
        served >> 20
    );
    Ok(())
}
