use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::tile::ScalarValue;
use phobos_sched::autotune::{ClusterFingerprint, RankedConfig, autotune};
use phobos_sched::{IngestPolicy, Plan, default_supers, plan_budgeted_with};

const MATMUL: &str = r#"
@cluster(TILE_M in [2048, 16384], TILE_N in [2048, 16384], TILE_K in [2048, 16384])
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
@cluster(BLOCK in [1048576, 16777216])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
    let base = program_id(0) * BLOCK
    c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;

const FLASH: &str = r#"
@cluster(BR in [1024, 4096])
@autotune(D in [64], BR in [32, 128], BC in [32, 128])
kernel attn(Q: tensor<f32>[Nq, D],
            K: tensor<f32>[Nk, D],
            V: tensor<f32>[Nk, D],
            O: tensor<f32>[Nq, D],
            scale: f32) {
    let pid = program_id(0)
    let row = pid * BR
    let q = Q[row :+ BR, :]
    var acc: tile<f32>[BR, D] = 0.0
    var l: tile<f32>[BR, 1] = 0.0
    for kt in range(0, Nk, BC) {
        let k = K[kt :+ BC, :]
        let v = V[kt :+ BC, :]
        var s: tile<f32>[BR, BC] = dot_t(q, k)
        s = s * scale
        var p: tile<f32>[BR, BC] = exp(s)
        l += rowsum(p)
        acc += dot(p, v)
    }
    acc = acc / l
    O[row :+ BR, :] = acc
}"#;

/// One thing to benchmark: a compiled cluster program plus a size -> dims
/// rule and any scalar bindings the kernel needs.
struct Workload {
    name: &'static str,
    /// Human-readable shape, e.g. "M=N=K".
    shape: &'static str,
    program: ClusterProgram,
    dims_for: fn(i64) -> HashMap<String, i64>,
    scalars: HashMap<String, ScalarValue>,
    /// Problem sizes for the throughput sweep.
    sizes: &'static [i64],
}

fn compile(src: &str) -> Result<ClusterProgram> {
    let kernel = phobos_lang::parse(src)?
        .into_iter()
        .next()
        .context("source has no kernel")?;
    phobos_cluster::compile(&kernel)
}

fn dims3(size: i64) -> HashMap<String, i64> {
    [("M", size), ("N", size), ("K", size)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

fn dims_add(size: i64) -> HashMap<String, i64> {
    [("N".to_string(), size)].into_iter().collect()
}

fn dims_flash(size: i64) -> HashMap<String, i64> {
    [("Nq", size), ("Nk", size), ("D", 64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// Mean wall time per call of f over iters runs, after two warmups. f is
/// also the work whose result the caller wants from the last run.
fn timed<T>(iters: u32, mut f: impl FnMut() -> Result<T>) -> Result<(Duration, T)> {
    f()?;
    let mut last = f()?;
    let start = Instant::now();
    for _ in 0..iters {
        last = f()?;
    }
    Ok((start.elapsed() / iters, last))
}

fn fmt_bytes(b: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.1} {}", UNITS[u])
}

/// Bytes the busiest node pulls from peers under the lowered plan (the comm
/// term that races compute for the makespan). Mirrors the scheduler's own
/// busiest_comm_sec, but on a plan lowered with peer FETCHes.
fn busiest_fetch_bytes(p: &ClusterProgram, pl: &Plan) -> u64 {
    pl.fetches
        .iter()
        .map(|fetches| {
            fetches
                .iter()
                .map(|(tile, _)| {
                    let t = tile.tensor() as usize;
                    pl.super_shapes[t].iter().product::<u64>()
                        * p.tensors[t].data_type.bytes() as u64
                })
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0)
}

/// The autotuner's chosen supertile config, as a compact K=V K=V string in a
/// stable order.
fn fmt_supers(rc: &RankedConfig) -> String {
    let mut kv: Vec<_> = rc.supers.iter().collect();
    kv.sort_by(|a, b| a.0.cmp(b.0));
    kv.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Section 1: time the full planning pipeline as problem size and node count
/// grow, and report the instruction-emission rate.
fn throughput(w: &Workload, node_counts: &[u16]) -> Result<()> {
    println!("== {} planner throughput ({}) ==", w.name, w.shape);
    println!(
        "{:>10}  {:>6}  {:>9}  {:>11}  {:>10}",
        "size", "nodes", "instrs", "plan/iter", "instr/s"
    );
    let supers = default_supers(&w.program);
    for &size in w.sizes {
        let dims = (w.dims_for)(size);
        for &nodes in node_counts {
            // Skip configs the default supertile can't divide into >= nodes
            // output supertiles; the planner would just error.
            let plan = plan_budgeted_with(
                &w.program,
                &dims,
                &supers,
                nodes,
                u64::MAX,
                IngestPolicy::DirectLoad,
                &w.scalars,
            );
            let Ok(probe) = plan else { continue };
            let instrs = probe.total_instrs();
            // More iterations for cheap (small) plans, fewer for the big ones.
            let iters = if instrs < 10_000 { 200 } else { 20 };
            let (dt, _) = timed(iters, || {
                plan_budgeted_with(
                    &w.program,
                    &dims,
                    &supers,
                    nodes,
                    u64::MAX,
                    IngestPolicy::DirectLoad,
                    &w.scalars,
                )
            })?;
            let per_sec = instrs as f64 / dt.as_secs_f64();
            println!(
                "{size:>10}  {nodes:>6}  {instrs:>9}  {:>11}  {:>9.2}M",
                format!("{dt:.2?}"),
                per_sec / 1e6,
            );
        }
    }
    println!();
    Ok(())
}

/// Section 2: hold the problem fixed and let the autotuner pick the supertile
/// granularity for each node count, reporting the cost model's predictions and
/// the strong-scaling speedup they imply.
fn scaling(w: &Workload, size: i64, node_counts: &[u16], fp: ClusterFingerprint) -> Result<()> {
    println!("== {} analytic scaling ({} = {size}) ==", w.name, w.shape);
    println!(
        "{:>6}  {:>22}  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>5}",
        "nodes",
        "super",
        "instrs",
        "peak/node",
        "netvol",
        "compute",
        "comm",
        "makespan",
        "speedup",
        "eff"
    );
    let dims = (w.dims_for)(size);
    let mut base_makespan = None;
    for &nodes in node_counts {
        let fp = ClusterFingerprint { nodes, ..fp };
        let ranked = match autotune(&w.program, &dims, fp) {
            Ok(r) => r,
            Err(_) => {
                println!("{nodes:>6}  {:>22}", "(infeasible)");
                continue;
            }
        };
        let rc = &ranked[0];

        // The autotuner ranks under DirectLoad: on shared storage every node
        // reads its own operands and zero bytes cross the network, so its
        // makespan is purely compute-bound. To show the communication the
        // owner-computes placement would move when operands live on a peer's
        // disk, re-plan the winner under HomeLoadPeerFetch and fold the busiest
        // node's fetch time into the makespan ourselves (compute is identical
        // under either ingest, so rc.compute_sec carries over).
        let plan = plan_budgeted_with(
            &w.program,
            &dims,
            &rc.supers,
            nodes,
            u64::MAX,
            IngestPolicy::HomeLoadPeerFetch,
            &w.scalars,
        )?;
        let instrs = plan.total_instrs();
        let comm_sec = busiest_fetch_bytes(&w.program, &plan) as f64 / fp.link_bytes_per_sec;
        let makespan = rc.compute_sec.max(comm_sec);

        let base = *base_makespan.get_or_insert(makespan);
        let speedup = base / makespan;
        let eff = 100.0 * speedup / nodes as f64;
        let dur = |s: f64| format!("{:.3?}", Duration::from_secs_f64(s));
        println!(
            "{nodes:>6}  {:>22}  {instrs:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>7.2}x  {:>4.0}%",
            fmt_supers(rc),
            fmt_bytes(rc.peak_resident),
            fmt_bytes(plan.fetch_bytes),
            dur(rc.compute_sec),
            dur(comm_sec),
            dur(makespan),
            speedup,
            eff,
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let fp = ClusterFingerprint {
        nodes: 1,
        vram_bytes: 40 << 30,      // 40 GiB
        link_bytes_per_sec: 10e9,  // 10 GB/s
        leaf_flops_per_sec: 10e12, // 10 TFLOP/s
    };

    let workloads = [
        Workload {
            name: "sgemm_f32",
            shape: "M=N=K",
            program: compile(MATMUL)?,
            dims_for: dims3,
            scalars: HashMap::new(),
            sizes: &[8192, 16384, 32768],
        },
        Workload {
            name: "add_f32",
            shape: "N",
            program: compile(ADD)?,
            dims_for: dims_add,
            scalars: HashMap::new(),
            sizes: &[16 << 20, 64 << 20, 256 << 20],
        },
        Workload {
            name: "flash_f32",
            shape: "Nq=Nk, D=64",
            program: compile(FLASH)?,
            dims_for: dims_flash,
            scalars: [("scale".to_string(), ScalarValue::F32(0.125))]
                .into_iter()
                .collect(),
            sizes: &[4096, 8192, 16384],
        },
    ];

    println!("phobos cluster scheduler benchmark (analytic, CPU-only)");
    println!(
        "fingerprint: link {:.1} GB/s, leaf {:.1} TFLOP/s, vram {}",
        fp.link_bytes_per_sec / 1e9,
        fp.leaf_flops_per_sec / 1e12,
        fmt_bytes(fp.vram_bytes),
    );
    if cfg!(debug_assertions) {
        println!("note: debug build; rerun with --release for representative plan/iter rates");
    }
    println!();

    let throughput_nodes = [1u16, 4, 16];
    let scaling_nodes = [1u16, 2, 4, 8, 16];

    for w in &workloads {
        throughput(w, &throughput_nodes)?;
    }

    // Strong scaling: fix the middle problem size of each workload.
    for w in &workloads {
        let size = w.sizes[1];
        scaling(w, size, &scaling_nodes, fp)?;
    }

    // The same sgemm over a 10x slower link: communication stops hiding behind
    // compute past a point, so the cost model's makespan goes comm-bound and
    // strong-scaling efficiency falls off. This is the arithmetic-intensity
    // wall made visible.
    let slow = ClusterFingerprint {
        link_bytes_per_sec: 1e9,
        ..fp
    };
    println!(
        "--- sgemm again, link throttled to {:.1} GB/s ---",
        slow.link_bytes_per_sec / 1e9
    );
    scaling(&workloads[0], workloads[0].sizes[1], &scaling_nodes, slow)?;

    Ok(())
}
