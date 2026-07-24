use std::time::Duration;

use anyhow::{Result, bail, ensure};
use phobos_cluster::proto::scheduler_client::SchedulerClient;
use phobos_cluster::proto::{self, Job, TensorInput};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, DataType};
use phobos_sched::server::{Scheduler, make_job};

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

const ADD: &str = r#"
@cluster(BLOCK in [512, 16384])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
    let base = program_id(0) * BLOCK
    c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;

fn randoms(n: usize, seed: u64) -> Vec<f32> {
    phobos_base::rng::Lcg::new(seed).unit_f32s(n)
}

async fn submit(client: &mut SchedulerClient<tonic::transport::Channel>, job: Job) -> Result<()> {
    let resp = client.submit(job).await?;
    let mut events = resp.into_inner();
    while let Some(ev) = events.message().await? {
        match ev.kind {
            Some(proto::job_event::Kind::Done(_)) => return Ok(()),
            Some(proto::job_event::Kind::Progress(p)) => bail!("job failed: {p}"),
            None => {}
        }
    }
    bail!("submit stream ended without a Done event")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("phobos_m1_loopback");
    std::fs::create_dir_all(&dir)?;
    let uri = |f: &str| format!("file://{}", dir.join(f).display());

    // OS-assigned ports (Windows reserves many fixed ranges).
    let sched_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let sched_addr = sched_listener.local_addr()?.to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(sched_listener);
    let server = Scheduler::new().into_server();
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
            if let Err(e) = phobos_pod::serve(0, sa, "127.0.0.1:0".into(), None, 256 << 20).await {
                eprintln!("node 0: {e:#}");
            }
        });
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut client = SchedulerClient::connect(format!("http://{sched_addr}")).await?;

    let t2 = |name: &str, n: usize, mode: AccessMode, uri: String| TensorInput {
        name: name.to_string(),
        data_type: proto::data_type_to_i32(DataType::F32),
        shape: vec![n as u64, n as u64],
        mode: proto::am_to_i32(mode),
        uri,
    };

    // --- matmul (1 node, no FETCH) ---
    const N: usize = 1024;
    let a = randoms(N * N, 1);
    let b = randoms(N * N, 2);
    storage::write_tensor_f32(&uri("A.bin"), &a)?;
    storage::write_tensor_f32(&uri("B.bin"), &b)?;
    storage::write_tensor_f32(&uri("C.bin"), &vec![0.0; N * N])?;
    submit(
        &mut client,
        make_job(
            MATMUL,
            &[("M", N as i64), ("N", N as i64), ("K", N as i64)],
            vec![
                t2("A", N, AccessMode::Read, uri("A.bin")),
                t2("B", N, AccessMode::Read, uri("B.bin")),
                t2("C", N, AccessMode::Write, uri("C.bin")),
            ],
        ),
    )
    .await?;
    let c = storage::read_tensor_f32(&uri("C.bin"), N * N)?;
    for (r, col) in [(0usize, 0usize), (1, 513), (511, 512), (1023, 1023)] {
        let want: f64 = (0..N)
            .map(|k| a[r * N + k] as f64 * b[k * N + col] as f64)
            .sum();
        let got = c[r * N + col] as f64;
        ensure!(
            (got - want).abs() / want.abs().max(1.0) < 1e-3,
            "C[{r},{col}] = {got}, CPU says {want}"
        );
    }
    println!("matmul loopback: 2x2x2 supertile grid OK");

    // --- elementwise add (bit-exact) ---
    const M: usize = 1024;
    let x = randoms(M, 3);
    let y = randoms(M, 4);
    storage::write_tensor_f32(&uri("a.bin"), &x)?;
    storage::write_tensor_f32(&uri("b.bin"), &y)?;
    storage::write_tensor_f32(&uri("c.bin"), &vec![0.0; M])?;
    let t1 = |name: &str, mode: AccessMode, uri: String| TensorInput {
        name: name.to_string(),
        data_type: proto::data_type_to_i32(DataType::F32),
        shape: vec![M as u64],
        mode: proto::am_to_i32(mode),
        uri,
    };
    submit(
        &mut client,
        make_job(
            ADD,
            &[("N", M as i64)],
            vec![
                t1("a", AccessMode::Read, uri("a.bin")),
                t1("b", AccessMode::Read, uri("b.bin")),
                t1("c", AccessMode::Write, uri("c.bin")),
            ],
        ),
    )
    .await?;
    let cc = storage::read_tensor_f32(&uri("c.bin"), M)?;
    for i in 0..M {
        let want = x[i] + y[i];
        ensure!(
            cc[i].to_bits() == want.to_bits(),
            "c[{i}] = {} != {want} (must be bit-exact)",
            cc[i]
        );
    }
    println!("add loopback: bit-exact OK");
    println!("loopback PASS");
    Ok(())
}
