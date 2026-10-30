mod common;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use phobos_base::cli::format_size;
use phobos_cluster::isa::Region;
use phobos_cluster::storage;
use phobos_sched::server::DispatchConfig;
use phobos_sched::{IngestPolicy, default_supers, plan_budgeted_with};

use common::{matmul_dims, matmul_job, matmul_src, start_cluster, temp_uris};

/// Write `n` f32 in [-1, 1] to a fresh file:// tensor, one chunk in RAM at a
/// time. Returns the wall time.
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

/// Pre-size a write-only tensor on disk without writing its bytes. The pods
/// STORE into regions of it, and DirectLoad never reads a write-only tensor.
fn alloc_tensor(uri: &str, n: usize) -> Result<()> {
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(storage::file_path(uri)?)?;
    f.set_len((n * 4) as u64)?;
    Ok(())
}

/// Recomputes C[r, c] on the CPU in f64, from row r of A and column c of B
/// read off disk, and returns the relative error against the stored value.
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

    // Plan first, on the CPU. peak_resident is the per-node high-water mark,
    // since operands are loaded and freed one k-step at a time. The engine
    // parks an ALLOC that would overflow until a FREE makes room, so an arena
    // of peak plus headroom is enough.
    let source = matmul_src(super_dim, 65536);
    let program = phobos_cluster::compile(&phobos_lang::parse(&source)?.remove(0))?;
    let supers = default_supers(&program);
    let dims = matmul_dims(n);
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

    // Each in-process node has its own arena on the shared GPU, so the card
    // must hold their sum. That is more VRAM than a real cluster needs per card.
    let aggregate = arena as u64 * nodes as u64;
    println!(
        "on disk: {} of f32 (A+B+C)\nplan: {} instrs, peak resident {}/node -> arena {}/node",
        format_size(footprint),
        pl.total_instrs(),
        format_size(peak),
        format_size(arena as u64),
    );
    println!(
        "the shared GPU must hold {} x {} = {} at once (operands replicate per node)",
        nodes,
        format_size(arena as u64),
        format_size(aggregate),
    );

    // Seed the inputs, the slow part, and pre-size the output.
    let uri = temp_uris("phobos_cluster_correctness")?;
    let (ua, ub, uc) = (uri("A.bin"), uri("B.bin"), uri("C.bin"));
    let ta = seed_tensor(&ua, n * n, 1)?;
    let tb = seed_tensor(&ub, n * n, 2)?;
    alloc_tensor(&uc, n * n)?;
    println!("seeded A in {ta:.1?}, B in {tb:.1?}");

    // Scheduler and in-process pods over localhost gRPC.
    let (sched, _) = start_cluster(nodes, arena).await?;

    let job = matmul_job(&source, n, &uri);
    let cfg = DispatchConfig {
        nodes,
        budget_bytes: Some(budget),
        ..Default::default()
    };
    let run = Instant::now();
    sched.dispatch(job, cfg).await?;
    println!("dispatched + ran in {:.1?}", run.elapsed());

    // Spot-check corners, the diagonal and a few interior cells, each
    // recomputed in f64 from the row and column on disk.
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
