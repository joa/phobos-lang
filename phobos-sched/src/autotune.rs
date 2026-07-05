use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::isa::Op;
use phobos_cluster::tile::{AccessMode, DataType, ScalarValue};

use crate::{IngestPolicy, Plan, plan_budgeted_with};

#[derive(Clone, Copy, Debug)]
pub struct ClusterFingerprint {
    pub nodes: u16,
    pub vram_bytes: u64, // per node
    pub link_bytes_per_sec: f64,
    pub leaf_flops_per_sec: f64,
}

impl Default for ClusterFingerprint {
    fn default() -> ClusterFingerprint {
        ClusterFingerprint {
            nodes: 1,
            vram_bytes: 16 << 30,      // 16 GiB
            link_bytes_per_sec: 10e9,  // ca 10GB/s
            leaf_flops_per_sec: 10e12, // 10 TFLOP/s
        }
    }
}

#[derive(Clone, Debug)]
pub struct RankedConfig {
    pub supers: HashMap<String, i64>,
    pub makespan_sec: f64,  // lower is better
    pub compute_sec: f64,   // compute of busiest node
    pub comm_sec: f64,      // communication of busiest node
    pub peak_resident: u64, // high watermark
    pub fetch_bytes: u64,   // total fetch bytes
}

pub fn autotune(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    fp: ClusterFingerprint,
) -> Result<Vec<RankedConfig>> {
    if fp.nodes == 0 {
        bail!("cluster has no nodes");
    }

    let mut ranked = Vec::new();
    let mut last_err = None;

    for supers in configs(p) {
        match evaluate(p, dims, &supers, fp) {
            Ok(Some(rc)) => ranked.push(rc),
            Ok(None) => {}
            Err(e) => last_err = Some(e),
        }
    }

    if ranked.is_empty() {
        match last_err {
            Some(e) => bail!("no feasible cluster config: last error: {e:#}"),
            None => bail!("no feasible cluster config (all pruned by VRAM/parallelism)"),
        }
    }

    // order of priority:
    // 1. shotrest makespan wins
    // 2. smallest working set
    // 3. larger supertiles -> fewer launches or better FLOP/byte
    ranked.sort_by(|a, b| {
        a.makespan_sec
            .total_cmp(&b.makespan_sec)
            .then(a.peak_resident.cmp(&b.peak_resident))
            .then(b.fetch_bytes.cmp(&a.fetch_bytes))
    });

    Ok(ranked)
}

pub fn best(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    fp: ClusterFingerprint,
) -> Result<HashMap<String, i64>> {
    Ok(autotune(p, dims, fp)?.swap_remove(0).supers)
}
fn zero_scalar(d: DataType) -> ScalarValue {
    match d {
        DataType::F32 | DataType::F16 => ScalarValue::F32(0.0),
        DataType::F64 => ScalarValue::F64(0.0),
        DataType::I32 | DataType::I8 => ScalarValue::I32(0),
        DataType::I64 => ScalarValue::I64(0),
        DataType::Bool => ScalarValue::Bool(false),
    }
}

fn evaluate(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    fp: ClusterFingerprint,
) -> Result<Option<RankedConfig>> {
    let scalars: HashMap<String, ScalarValue> = p
        .scalars
        .iter()
        .map(|s| (s.name.clone(), zero_scalar(s.data_type)))
        .collect();

    let pl = plan_budgeted_with(
        p,
        dims,
        supers,
        fp.nodes,
        u64::MAX,
        IngestPolicy::default(),
        &scalars,
    )?;

    let out = p
        .tensors
        .iter()
        .position(|t| matches!(t.mode, AccessMode::Write | AccessMode::RMW))
        .unwrap_or(p.tensors.len() - 1);

    let parallelism: u64 = pl.super_grids[out].iter().product();
    if parallelism < fp.nodes as u64 {
        return Ok(None);
    }

    if pl.peak_resident > fp.vram_bytes {
        return Ok(None);
    }

    let compute_s = busiest_compute_sec(p, &pl, fp);
    let comm_s = busiest_comm_sec(p, &pl, fp);
    let makespan_s = compute_s.max(comm_s);

    Ok(Some(RankedConfig {
        supers: supers.clone(),
        makespan_sec: makespan_s,
        compute_sec: compute_s,
        comm_sec: comm_s,
        peak_resident: pl.peak_resident,
        fetch_bytes: pl.fetch_bytes,
    }))
}

fn busiest_compute_sec(p: &ClusterProgram, pl: &Plan, fp: ClusterFingerprint) -> f64 {
    let mut max_flops = 0u64;

    for node in 0..pl.node_segments.len() {
        let mut flops = 0u64;

        for i in pl.node_instrs(node) {
            if let Op::Compute { args, .. } = &i.op {
                let mut axes: HashMap<&str, u64> = HashMap::new();

                for (tile, _) in args {
                    let t = tile.tensor() as usize;
                    for (sym, &val) in p.tensors[t].super_syms.iter().zip(&pl.super_shapes[t]) {
                        axes.insert(sym.as_str(), val);
                    }
                }

                flops += 2 * axes.values().product::<u64>();
            }
        }

        max_flops = max_flops.max(flops);
    }

    max_flops as f64 / fp.leaf_flops_per_sec
}

fn busiest_comm_sec(p: &ClusterProgram, pl: &Plan, fp: ClusterFingerprint) -> f64 {
    let mut max_bytes = 0u64;

    for fetches in &pl.fetches {
        let mut bytes = 0u64;

        for (tile, _) in fetches {
            let t = tile.tensor() as usize;
            bytes +=
                pl.super_shapes[t].iter().product::<u64>() * p.tensors[t].data_type.bytes() as u64;
        }

        max_bytes = max_bytes.max(bytes);
    }

    max_bytes as f64 / fp.link_bytes_per_sec
}

fn configs(p: &ClusterProgram) -> Vec<HashMap<String, i64>> {
    p.super_dims.iter().fold(vec![HashMap::new()], |acc, d| {
        acc.iter()
            .flat_map(|base| {
                d.choices.iter().map(|&c| {
                    let mut m = base.clone();
                    m.insert(d.name.clone(), c);
                    m
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use phobos_cluster::ir::ClusterProgram;

    use super::{ClusterFingerprint, autotune, best, configs};

    const MATMUL: &str = r#"
@cluster(TILE_M in [2048, 8192], TILE_N in [2048, 8192], TILE_K in [2048, 8192])
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

    fn program() -> ClusterProgram {
        let kernel = phobos_lang::parse(MATMUL).unwrap().remove(0);
        phobos_cluster::compile(&kernel).unwrap()
    }

    fn dims(v: i64) -> HashMap<String, i64> {
        [("M", v), ("N", v), ("K", v)]
            .into_iter()
            .map(|(k, x)| (k.to_string(), x))
            .collect()
    }

    #[test]
    fn cartesian_product_covers_search_space() {
        let p = program();
        // [2048, 8192] doubling -> {2048, 4096, 8192}; 3 dims -> 27 configs.
        assert_eq!(configs(&p).len(), 27);
    }

    #[test]
    fn ranks_and_prunes() {
        let p = program();
        let fp = ClusterFingerprint {
            nodes: 4,
            vram_bytes: 8 << 30,
            link_bytes_per_sec: 10e9,
            leaf_flops_per_sec: 10e12,
        };
        // 16384^3: divisible by all the {2048,4096,8192} choices.
        let ranked = autotune(&p, &dims(16384), fp).unwrap();
        assert!(!ranked.is_empty());
        // sorted best-first
        for w in ranked.windows(2) {
            assert!(w[0].makespan_sec <= w[1].makespan_sec);
        }
        // every survivor has enough output supertiles for 4 nodes and fits VRAM
        for rc in &ranked {
            assert!(rc.peak_resident <= fp.vram_bytes);
        }
    }

    #[test]
    fn parallelism_prune_drops_under_node_grids() {
        let p = program();
        // M=N=K=8192 with min super 2048 -> grids up to 4x4; with super 8192 ->
        // 1x1 (one output supertile). On 16 nodes the coarse configs are pruned
        // for under-parallelism but fine-grained ones survive.
        let fp = ClusterFingerprint {
            nodes: 16,
            vram_bytes: 32 << 30,
            link_bytes_per_sec: 10e9,
            leaf_flops_per_sec: 10e12,
        };
        let ranked = autotune(&p, &dims(8192), fp).unwrap();
        assert!(!ranked.is_empty());
        // the winner must have at least 16 output supertiles, i.e. SUPER_M and
        // SUPER_N small enough that (8192/SM)*(8192/SN) >= 16.
        let w = &ranked[0];
        let sm = w.supers["TILE_M"];
        let sn = w.supers["TILE_N"];
        assert!(
            (8192 / sm) * (8192 / sn) >= 16,
            "winner under-parallel: {w:?}"
        );
    }

    #[test]
    fn tiny_vram_prunes_everything() {
        let p = program();
        let fp = ClusterFingerprint {
            nodes: 1,
            vram_bytes: 1 << 20, // 1 MiB; no supertile fits
            ..Default::default()
        };
        let err = autotune(&p, &dims(8192), fp).unwrap_err().to_string();
        assert!(err.contains("no feasible cluster config"), "got: {err}");
    }

    #[test]
    fn best_returns_top_config() {
        let p = program();
        let fp = ClusterFingerprint {
            nodes: 4,
            ..Default::default()
        };
        let b = best(&p, &dims(16384), fp).unwrap();
        assert!(b.contains_key("TILE_M"));
        assert!(b.contains_key("TILE_N"));
        assert!(b.contains_key("TILE_K"));
    }

    #[test]
    fn autotunes_a_scalar_kernel() {
        // A kernel with a scalar param must tune without a job binding: the
        // trial plan uses placeholder scalar values (the config is independent
        // of them).
        const FLASH: &str = r#"
@cluster(BR in [1024, 4096])
@autotune(D in [64], BR in [32, 128], BC in [32, 128])
kernel attn(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D],
            V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D], scale: f32) {
    let pid = program_id(0)
    let row = pid * BR
    let q = Q[row :+ BR, :]
    var acc: tile<f32>[BR, D] = 0.0
    for kt in range(0, Nk, BC) {
        let k = K[kt :+ BC, :]
        let v = V[kt :+ BC, :]
        var s: tile<f32>[BR, BC] = dot_t(q, k)
        s = s * scale
        acc += dot(s, v)
    }
    O[row :+ BR, :] = acc
}"#;
        let p = phobos_cluster::compile(&phobos_lang::parse(FLASH).unwrap().remove(0)).unwrap();
        let dims: HashMap<String, i64> = [("Nq", 8192), ("Nk", 4096), ("D", 64)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let fp = ClusterFingerprint {
            nodes: 2,
            ..Default::default()
        };
        let b = best(&p, &dims, fp).unwrap();
        assert!(b.contains_key("BR"), "tuned a BR supertile: {b:?}");
    }
}
