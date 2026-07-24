use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use phobos_cluster::isa::Region;
use phobos_cluster::proto::{self, TensorInput};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, DataType};
use phobos_sched::server::{DispatchConfig, Scheduler, make_job};
use phobos_sched::{IngestPolicy, default_supers, plan_budgeted_with};

/// Cluster matmul. The @cluster lower bounds are what default_supers picks, so
/// they set the supertile shape directly; SUPER_K = SUPER means K/SUPER k-steps
/// per output supertile (operands stream rather than all co-residing).
fn matmul_src(super_dim: usize) -> String {
    format!(
        r#"
@cluster(TILE_M in [{super_dim}, 65536], TILE_N in [{super_dim}, 65536], TILE_K in [{super_dim}, 65536])
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

fn fmt_bytes(b: u64) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let (mut v, mut u) = (b as f64, 0);
    while v >= 1024.0 && u < U.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.1} {}", U[u])
}

/// Stream n f32 in [-1, 1] to a fresh file:// tensor in chunks, never
/// holding more than one chunk in RAM. Returns the wall time.
fn seed_tensor(uri: &str, n: usize, seed: u64) -> Result<Duration> {
    const CHUNK: usize = 1 << 22; // 16 MiB of f32
    let start = Instant::now();
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(storage::file_path(uri)?)?;
    let mut w = BufWriter::with_capacity(1 << 22, file);
    let mut buf = vec![0f32; CHUNK];
    let mut lcg = phobos_base::rng::Lcg::new(seed);
    let mut left = n;
    while left > 0 {
        let m = left.min(CHUNK);
        for x in buf[..m].iter_mut() {
            *x = lcg.next_unit_f32();
        }
        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, m * 4) };
        w.write_all(bytes)?;
        left -= m;
    }
    w.flush()?;
    Ok(start.elapsed())
}

/// Pre-size a write-only tensor on disk without writing its bytes (the pods
/// STORE into regions of it; DirectLoad never reads a write-only tensor).
fn alloc_tensor(uri: &str, n: usize) -> Result<()> {
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(storage::file_path(uri)?)?;
    f.set_len((n * 4) as u64)?;
    Ok(())
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

/// Recompute C[r, c] on the CPU in f64 by reading just row r of A and
/// column c of B from disk, and compare to the stored distributed value.
/// Returns the relative error.
fn sample_rel_err(ua: &str, ub: &str, uc: &str, n: usize, r: usize, c: usize) -> Result<f64> {
    let shape = [n as u64, n as u64];
    let row = storage::load_f32(
        ua,
        &shape,
        &Region {
            offset: vec![r as u64, 0],
            shape: vec![1, n as u64],
        },
    )?;
    let col = storage::load_f32(
        ub,
        &shape,
        &Region {
            offset: vec![0, c as u64],
            shape: vec![n as u64, 1],
        },
    )?;
    let want: f64 = (0..n).map(|k| row[k] as f64 * col[k] as f64).sum();
    let got = storage::load_f32(
        uc,
        &shape,
        &Region {
            offset: vec![r as u64, c as u64],
            shape: vec![1, 1],
        },
    )?[0] as f64;
    Ok((got - want).abs() / want.abs().max(1.0))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let n = arg(1, 16384);
    let super_dim = arg(2, 8192);
    let nodes = arg(3, 4) as u16;

    ensure!(
        n % super_dim == 0,
        "N ({n}) must be a multiple of SUPER ({super_dim})"
    );
    ensure!(
        super_dim % 32 == 0,
        "SUPER ({super_dim}) must be a multiple of 32"
    );
    let grid = n / super_dim;
    ensure!(
        grid * grid >= nodes as usize,
        "output grid {grid}x{grid} = {} supertiles < {nodes} nodes",
        grid * grid
    );

    let footprint = 3 * (n as u64) * (n as u64) * 4;
    println!(
        "SGEMM N={n} (M=N=K), super={super_dim} -> {grid}x{grid} grid = {} supertiles, {nodes} nodes",
        grid * grid
    );

    // Plan first (CPU-only, instant): peak_resident is the per-node resident
    // high-water (operands stream k-step by k-step and free, so it's a few
    // supertiles, not the whole problem). The node engine honors it via arena
    // backpressure -- it allocates in topological order and parks ALLOCs that
    // would overflow until a FREE makes room -- so arena = peak + headroom runs
    // within bound.
    let source = matmul_src(super_dim);
    let program = phobos_cluster::compile(&phobos_lang::parse(&source)?.remove(0))?;
    let supers = default_supers(&program);
    let dims = [("M", n as i64), ("N", n as i64), ("K", n as i64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let pl = plan_budgeted_with(
        &program,
        &dims,
        &supers,
        nodes,
        u64::MAX,
        IngestPolicy::DirectLoad,
        &Default::default(),
    )?;
    let peak = pl.peak_resident;
    let arena = (peak + peak / 5) as usize; // +20% headroom for prefetch/alignment
    let budget = arena as u64;

    // Each in-process node keeps its own arena on the shared GPU, so the card
    // must hold the sum. Simulating N nodes on one GPU replicates operands,
    // so it costs more VRAM than the problem; the cluster's real win needs
    // separate VRAM per physical node.
    let aggregate = arena as u64 * nodes as u64;
    println!(
        "on disk: {} of f32 (A+B+C)\nplan: {} instrs, peak resident {}/node -> arena {}/node",
        fmt_bytes(footprint),
        pl.total_instrs(),
        fmt_bytes(peak),
        fmt_bytes(arena as u64),
    );
    println!(
        "the shared GPU must hold {} x {} = {} at once (operands replicate per node)",
        nodes,
        fmt_bytes(arena as u64),
        fmt_bytes(aggregate),
    );

    // Seed inputs (the slow part), pre-size the output.
    let dir = std::env::temp_dir().join("phobos_cluster_correctness");
    std::fs::create_dir_all(&dir)?;
    let uri = |f: &str| format!("file://{}", dir.join(f).display());
    let (ua, ub, uc) = (uri("A.bin"), uri("B.bin"), uri("C.bin"));
    let ta = seed_tensor(&ua, n * n, 1)?;
    let tb = seed_tensor(&ub, n * n, 2)?;
    alloc_tensor(&uc, n * n)?;
    println!("seeded A in {ta:.1?}, B in {tb:.1?}");

    // Scheduler + nodes in-process pods over localhost gRPC.
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
            if let Err(e) = phobos_pod::serve(id, sa, "127.0.0.1:0".into(), None, arena).await {
                eprintln!("node {id}: {e:#}");
            }
        });
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    let job = make_job(
        &source,
        &[("M", n as i64), ("N", n as i64), ("K", n as i64)],
        vec![
            tensor("A", n, AccessMode::Read, ua.clone()),
            tensor("B", n, AccessMode::Read, ub.clone()),
            tensor("C", n, AccessMode::Write, uc.clone()),
        ],
    );
    let cfg = DispatchConfig {
        nodes,
        budget_bytes: Some(budget),
        ..Default::default()
    };
    let run = Instant::now();
    sched.dispatch(job, cfg).await?;
    println!("dispatched + ran in {:.1?}", run.elapsed());

    // Sample-verify: corners, the diagonal, and a few scattered interior cells,
    // each recomputed in f64 from disk-read row/col.
    let mut samples = vec![
        (0, 0),
        (n - 1, n - 1),
        (0, n - 1),
        (n - 1, 0),
        (n / 2, n / 2),
    ];
    let mut s = 0x9e3779b97f4a7c15u64;
    for _ in 0..7 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        samples.push(((s >> 33) as usize % n, (s >> 1) as usize % n));
    }

    let mut max_rel = 0.0f64;
    for &(r, c) in &samples {
        let rel = sample_rel_err(&ua, &ub, &uc, n, r, c)
            .with_context(|| format!("verifying C[{r},{c}]"))?;
        max_rel = max_rel.max(rel);
        ensure!(rel < 1e-3, "C[{r},{c}] mismatch: rel err {rel:.2e}");
    }

    println!(
        "check: {} sampled entries, distributed vs CPU reference match (max rel err {max_rel:.1e})",
        samples.len()
    );
    Ok(())
}
