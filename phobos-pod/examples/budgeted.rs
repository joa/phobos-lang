mod common;

use anyhow::{Result, ensure};
use phobos_cluster::storage;
use phobos_sched::autotune::ClusterFingerprint;
use phobos_sched::server::DispatchConfig;

use common::{check_matmul, matmul_dims, matmul_job, matmul_src, seed_matmul, start_cluster, temp_uris};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    const N: usize = 1024;
    let arena = 256 << 20;

    let uri = temp_uris("phobos_m2_budgeted")?;
    let (a, b) = seed_matmul(&uri, N)?;

    // Pick SUPER_* for one node, then plan with a budget of about three
    // supertiles, so the node's program spans several segments.
    let source = matmul_src(512, 16384);
    let kernel = phobos_lang::parse(&source)?.remove(0);
    let program = phobos_cluster::compile(&kernel)?;
    let dims = matmul_dims(N);
    let fp = ClusterFingerprint {
        nodes: 1,
        vram_bytes: arena as u64,
        ..Default::default()
    };
    let supers = phobos_sched::autotune::best(&program, &dims, fp)?;
    let chosen: i64 = supers["TILE_M"];
    let tile_bytes = (chosen as u64) * (chosen as u64) * 4;
    let budget = 3 * tile_bytes;
    let pl = phobos_sched::plan_budgeted(&program, &dims, &supers, 1, budget)?;
    let nsegs = pl.node_segments[0].len();
    ensure!(nsegs > 1, "budget should force >1 segment, got {nsegs}");
    println!(
        "autotuner picked SUPER={chosen}; budget {} MiB -> {nsegs} segments (peak {} MiB)",
        budget >> 20,
        pl.peak_resident >> 20
    );

    // Scheduler and one node over localhost gRPC.
    let (sched, _) = start_cluster(1, arena).await?;

    let cfg = DispatchConfig {
        nodes: 1,
        budget_bytes: Some(budget),
        autotune: Some(fp),
        ..Default::default()
    };
    sched.dispatch(matmul_job(&source, N, &uri), cfg).await?;

    let c = storage::read_tensor_f32(&uri("C.bin"), N * N)?;
    check_matmul(&c, &a, &b, N, &[(0, 0), (1, 513), (511, 512), (512, 511), (1023, 1023)])?;
    println!("budgeted OK: {nsegs}-segment plan executed epsilon-correct");
    Ok(())
}
