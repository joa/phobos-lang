mod common;

use std::sync::atomic::Ordering;

use anyhow::{Result, ensure};
use phobos_cluster::storage;
use phobos_pod::server::{GET_KERNEL_CALLS, SERVED_BYTES};
use phobos_sched::server::DispatchConfig;

use common::{check_matmul, matmul_dims, matmul_job, matmul_src, seed_matmul, start_cluster, temp_uris};

#[tokio::main(flavor = "multi_thread", worker_threads = 6)]
async fn main() -> Result<()> {
    const N: usize = 1024;
    let arena = 256 << 20;

    // Seed file:// tensors in a temp dir.
    let uri = temp_uris("phobos_m1")?;
    let (a, b) = seed_matmul(&uri, N)?;

    // The analytic minimum of peer bytes under owner-computes, for the assertion below.
    let source = matmul_src(512, 16384);
    let kernel = phobos_lang::parse(&source)?.remove(0);
    let program = phobos_cluster::compile(&kernel)?;
    let supers = phobos_sched::default_supers(&program);
    let dims = matmul_dims(N);
    // Use the home-LOAD, peer-FETCH ingest policy to exercise the peer data
    // plane. The default DirectLoad moves no bytes between peers.
    let pl = phobos_sched::plan_with(
        &program,
        &dims,
        &supers,
        2,
        phobos_sched::IngestPolicy::HomeLoadPeerFetch,
    )?;
    let want_bytes = pl.fetch_bytes;

    let (sched, _) = start_cluster(2, arena).await?;

    SERVED_BYTES.store(0, Ordering::SeqCst);
    GET_KERNEL_CALLS.store(0, Ordering::SeqCst);

    // Withhold the step leaf (kernel 0) from node 1, so it has to GetKernel it from node 0.
    let cfg = DispatchConfig {
        nodes: 2,
        withhold: vec![(1, 0)],
        ingest: phobos_sched::IngestPolicy::HomeLoadPeerFetch,
        ..Default::default()
    };
    sched.dispatch(matmul_job(&source, N, &uri), cfg).await?;

    // Verify C against CPU ground truth on a spread of entries.
    let c = storage::read_tensor_f32(&uri("C.bin"), N * N)?;
    check_matmul(&c, &a, &b, N, &[(0, 0), (1, 513), (511, 512), (512, 511), (1023, 1023)])?;

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
