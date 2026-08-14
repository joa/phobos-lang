use std::collections::{HashMap, HashSet};

use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::isa::{Instr, Op};
use phobos_cluster::tile::TileId;

use super::{
    IngestPolicy, Plan, default_supers, plan, plan_budgeted, plan_budgeted_with, plan_with,
    recover_plan, validate,
};

const MATMUL: &str = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
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

fn matmul_program() -> ClusterProgram {
    let kernel = phobos_lang::parse(MATMUL).unwrap().remove(0);
    phobos_cluster::compile(&kernel).unwrap()
}

fn dims(v: i64) -> HashMap<String, i64> {
    [("M", v), ("N", v), ("K", v)]
        .into_iter()
        .map(|(k, x)| (k.to_string(), x))
        .collect()
}

/// Count ops across a node's whole (possibly multi-segment) program.
fn count(pl: &Plan, node: usize, f: impl Fn(&Op) -> bool) -> usize {
    pl.node_instrs(node).filter(|i| f(&i.op)).count()
}

fn node_ops(pl: &Plan, node: usize) -> Vec<&Instr> {
    pl.node_instrs(node).collect()
}

#[test]
fn matmul_2x2x2_single_node() {
    let p = matmul_program();
    let supers = default_supers(&p); // 4096 each
    let pl = plan(&p, &dims(8192), &supers, 1).unwrap();
    validate(&pl).unwrap();

    // a single node consumes every supertile itself: no peer transfer
    assert!(pl.fetches.iter().all(|f| f.is_empty()));
    assert_eq!(pl.fetch_bytes, 0);

    // unbudgeted -> one segment per node
    assert_eq!(pl.node_segments.len(), 1);
    assert_eq!(pl.node_segments[0].len(), 1);

    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Alloc { .. })), 12);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Load { .. })), 8);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Compute { .. })), 12);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Store { .. })), 4);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Free { .. })), 12);
    assert_eq!(node_ops(&pl, 0).len(), 48);

    // the C(0,0) chain: init, then k-steps each depending on the previous
    let instrs = node_ops(&pl, 0);
    let c00 = TileId::new(2, 0, 0);
    let chain: Vec<&Instr> = instrs
        .iter()
        .copied()
        .filter(
            |i| matches!(&i.op, Op::Compute { args, .. } if args.iter().any(|(t, _)| *t == c00)),
        )
        .collect();
    assert_eq!(chain.len(), 3, "init + 2 k-steps");
    let (Op::Compute { kernel: k0, .. }, Op::Compute { kernel: k1, .. }) =
        (&chain[0].op, &chain[1].op)
    else {
        unreachable!()
    };
    assert_eq!(*k0, 1, "chain starts with the init leaf");
    assert_eq!(*k1, 0, "k-steps run the step leaf");
    assert!(
        chain[1].deps.contains(&chain[0].iid),
        "step 0 waits on init"
    );
    assert!(
        chain[2].deps.contains(&chain[1].iid),
        "step 1 waits on step 0"
    );

    // launch dims: 4096 supertile over 32x32 device tiles, flat CTA
    let Op::Compute { grid, cta, .. } = &chain[1].op else {
        unreachable!()
    };
    assert_eq!(*grid, (128, 128, 1));
    assert_eq!(*cta, (256, 1, 1));

    // input lifetime: A(0,1) is consumed by exactly the two k=1 steps
    let a01 = TileId::new(0, 0, 1);
    let free = instrs
        .iter()
        .find(|i| matches!(&i.op, Op::Free { tile, .. } if *tile == a01))
        .expect("A(0,1) is freed");
    assert_eq!(free.deps.len(), 2, "freed after its two consumers");

    // output lifetime: STORE after last write, FREE after STORE
    let store = instrs
        .iter()
        .find(|i| matches!(&i.op, Op::Store { tile, .. } if *tile == c00))
        .expect("C(0,0) is stored");
    assert_eq!(store.deps, vec![chain[2].iid]);
    let cfree = instrs
        .iter()
        .find(|i| matches!(&i.op, Op::Free { tile, .. } if *tile == c00))
        .unwrap();
    assert_eq!(cfree.deps, vec![store.iid]);
}

#[test]
fn launch_attr_sets_compute_cta() {
    // @launch overrides the default CTA on every leaf's COMPUTE.
    let src = MATMUL.replace("@cluster", "@launch(128)\n@cluster");
    let kernel = phobos_lang::parse(&src).unwrap().remove(0);
    let p = phobos_cluster::compile(&kernel).unwrap();
    let supers = default_supers(&p);
    let pl = plan(&p, &dims(8192), &supers, 1).unwrap();
    for instr in node_ops(&pl, 0) {
        if let Op::Compute { cta, .. } = &instr.op {
            assert_eq!(*cta, (128, 1, 1), "every leaf launches with @launch CTA");
        }
    }
}

#[test]
fn matmul_two_nodes_owner_computes() {
    let p = matmul_program();
    let supers = default_supers(&p);
    // The FETCH/serve behavior is the HomeLoadPeerFetch ingest policy.
    let pl = plan_with(&p, &dims(8192), &supers, 2, IngestPolicy::HomeLoadPeerFetch).unwrap();
    validate(&pl).unwrap();

    assert_eq!(pl.node_segments.len(), 2);
    for n in 0..2 {
        // 2 owned C supertiles x (init + 2 steps)
        assert_eq!(count(&pl, n, |o| matches!(o, Op::Compute { .. })), 6);
        assert_eq!(count(&pl, n, |o| matches!(o, Op::Store { .. })), 2);

        // every compute writes a C supertile this node owns (block-cyclic)
        for i in node_ops(&pl, n) {
            if let Op::Compute { args, .. } = &i.op {
                let (out, _) = args.iter().find(|(t, _)| t.tensor() == 2).unwrap();
                assert_eq!(out.coord() % 2, n as u64);
            }
        }
    }

    // each input supertile is LOADed from storage exactly once cluster-wide
    // (4 A + 4 B = 8); the home node owns the LOAD, peers FETCH.
    let total_loads: usize = (0..2)
        .map(|n| count(&pl, n, |o| matches!(o, Op::Load { .. })))
        .sum();
    assert_eq!(total_loads, 8);

    // owner-computes by C linear coord (lin = i*2+j, node = lin%2): both
    // nodes consume all 4 A supertiles, so A homes to node 0 and node 1
    // FETCHes all 4; B splits by column with no fetch. Hence node 0 LOADs
    // 4 A + 2 B = 6 and node 1 LOADs 2 B; node 1 issues 4 FETCHes, node 0
    // none; all from node 0.
    let loads = |n: usize| count(&pl, n, |o| matches!(o, Op::Load { .. }));
    assert_eq!(loads(0), 6);
    assert_eq!(loads(1), 2);
    assert!(pl.fetches[0].is_empty());
    assert_eq!(pl.fetches[1].len(), 4);
    assert!(pl.fetches[1].iter().all(|(_, from)| *from == 0));
    let fetch_count = |n: usize| count(&pl, n, |o| matches!(o, Op::Fetch { .. }));
    assert_eq!(fetch_count(0), 0);
    assert_eq!(fetch_count(1), 4);

    // node 0 serves each of its 4 A supertiles once (to node 1); every
    // other FREE expects zero serves.
    let a_serves: Vec<u32> = node_ops(&pl, 0)
        .iter()
        .filter_map(|i| match &i.op {
            Op::Free {
                tile,
                expected_serves,
            } if tile.tensor() == 0 => Some(*expected_serves),
            _ => None,
        })
        .collect();
    assert_eq!(a_serves, vec![1, 1, 1, 1]);
    for n in 0..2 {
        for i in node_ops(&pl, n) {
            if let Op::Free {
                tile,
                expected_serves,
            } = &i.op
                && tile.tensor() != 0
            {
                assert_eq!(*expected_serves, 0);
            }
        }
    }

    // analytic minimum: 4 fetched A supertiles of 4096x4096 f32
    assert_eq!(pl.fetch_bytes, 4 * 4096 * 4096 * 4);
}

#[test]
fn matmul_two_nodes_direct_load() {
    // The default policy: every node LOADs the inputs it consumes straight
    // from storage, with no peer FETCH and no serve counts. Each node owns 2 C
    // supertiles and consumes all 4 A (both rows) + its own 2 B (one
    // column) = 6 distinct input supertiles.
    let p = matmul_program();
    let supers = default_supers(&p);
    let pl = plan(&p, &dims(8192), &supers, 2).unwrap();
    validate(&pl).unwrap();

    assert_eq!(pl.fetch_bytes, 0);
    for n in 0..2 {
        assert_eq!(count(&pl, n, |o| matches!(o, Op::Fetch { .. })), 0);
        assert_eq!(count(&pl, n, |o| matches!(o, Op::Load { .. })), 6);
        assert!(pl.fetches[n].is_empty());
        // no input is served to a peer
        for i in node_ops(&pl, n) {
            if let Op::Free {
                expected_serves, ..
            } = &i.op
            {
                assert_eq!(*expected_serves, 0);
            }
        }
    }
    // Under HomeLoadPeerFetch the 4 A-supertiles cross the network once;
    // DirectLoad instead re-reads them: node1 LOADs its 4 A directly, so
    // the cluster does 12 LOADs (6 each) and 0 peer bytes.
    let total_loads: usize = (0..2)
        .map(|n| count(&pl, n, |o| matches!(o, Op::Load { .. })))
        .sum();
    assert_eq!(total_loads, 12);
}

#[test]
fn add_elementwise_plan() {
    let src = r#"
@cluster(BLOCK in [1048576, 16777216])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
let base = program_id(0) * BLOCK
c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;
    let kernel = phobos_lang::parse(src).unwrap().remove(0);
    let p = phobos_cluster::compile(&kernel).unwrap();
    let supers = default_supers(&p);
    let d: HashMap<String, i64> = [("N".to_string(), 2 * 1048576)].into_iter().collect();
    let pl = plan(&p, &d, &supers, 1).unwrap();
    validate(&pl).unwrap();

    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Alloc { .. })), 6);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Load { .. })), 4);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Compute { .. })), 2);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Store { .. })), 2);
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Free { .. })), 6);
}

#[test]
fn rejects_indivisible_shapes() {
    let p = matmul_program();
    let supers = default_supers(&p);
    let err = plan(&p, &dims(10000), &supers, 1).unwrap_err().to_string();
    assert!(err.contains("not a multiple"), "got: {err}");
}

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

#[test]
fn flash_single_leaf_plan() {
    use phobos_cluster::isa::ScalarArg;
    use phobos_cluster::tile::ScalarValue;

    let kernel = phobos_lang::parse(FLASH).unwrap().remove(0);
    let p = phobos_cluster::compile(&kernel).unwrap();
    let supers = default_supers(&p); // BR = 1024
    let dims: HashMap<String, i64> = [("Nq", 4096), ("Nk", 2048), ("D", 64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let scalars: HashMap<String, ScalarValue> = [("scale".to_string(), ScalarValue::F32(0.125))]
        .into_iter()
        .collect();

    let pl = plan_budgeted_with(
        &p,
        &dims,
        &supers,
        1,
        u64::MAX,
        IngestPolicy::default(),
        &scalars,
    )
    .unwrap();
    validate(&pl).unwrap();

    // grid over query blocks: Nq / BR = 4096 / 1024 = 4 leaf computes
    assert_eq!(count(&pl, 0, |o| matches!(o, Op::Compute { .. })), 4);

    // K and V are read whole: their supertile spans the full Nk x D
    assert_eq!(pl.super_shapes[1], vec![2048, 64]); // K
    assert_eq!(pl.super_shapes[2], vec![2048, 64]); // V
    // Q and O are tiled to a BR-row supertile (grid extent 4 along Nq)
    assert_eq!(pl.super_shapes[0], vec![1024, 64]); // Q
    assert_eq!(pl.super_grids[0], vec![4, 1]);
    assert_eq!(pl.super_grids[1], vec![1, 1]); // K: one supertile

    // every compute carries the bound scalar at param position 4
    let want = vec![ScalarArg {
        pos: 4,
        value: ScalarValue::F32(0.125),
    }];
    for i in node_ops(&pl, 0) {
        if let Op::Compute { scalars, .. } = &i.op {
            assert_eq!(scalars, &want);
        }
    }
}

#[test]
fn flash_leaf_lowers_to_ptx() {
    // The single leaf is the whole flash kernel; it must lower all the way
    // to PTX through the device pipeline (GPU-free: LLVM/NVPTX codegen),
    // the same path dispatch runs. Exercises the scalar param plus
    // exp/dot_t/softmax codegen end to end.
    let kernel = phobos_lang::parse(FLASH).unwrap().remove(0);
    let p = phobos_cluster::compile(&kernel).unwrap();
    assert_eq!(p.leaves.len(), 1);
    let base = phobos_base::context::Context::default();
    let ptx = phobos_mlir::gen_ptx(&base, |b, c, m| {
        phobos_lang::codegen::emit(b, std::slice::from_ref(&p.leaves[0].kernel), c, m).map(|_| ())
    })
    .unwrap();
    assert!(
        ptx.contains(".visible .entry attn"),
        "missing PTX entry:\n{ptx}"
    );
}

#[test]
fn flash_unbound_scalar_errors() {
    let kernel = phobos_lang::parse(FLASH).unwrap().remove(0);
    let p = phobos_cluster::compile(&kernel).unwrap();
    let supers = default_supers(&p);
    let dims: HashMap<String, i64> = [("Nq", 4096), ("Nk", 2048), ("D", 64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    // no scalar binding for scale
    let err = plan(&p, &dims, &supers, 1).unwrap_err().to_string();
    assert!(
        err.contains("scalar parameter 'scale' is unbound"),
        "got: {err}"
    );
}

#[test]
fn budget_splits_into_segments() {
    // 2x2x2 grid on one node. Each supertile is 4096x4096 f32 = 64 MiB.
    // The steady-state working set for one C chain is acc + a + b = 3
    // tiles = 192 MiB. A budget below the whole program but at least one
    // tile forces multiple segments; each segment's incremental footprint
    // must stay within budget, and the plan must still validate.
    let p = matmul_program();
    let supers = default_supers(&p);
    let tile = 4096u64 * 4096 * 4;
    let budget = 3 * tile; // room for one C chain's live set at a time
    let pl = plan_budgeted(&p, &dims(8192), &supers, 1, budget).unwrap();
    validate(&pl).unwrap();

    assert_eq!(pl.node_segments.len(), 1);
    let segs = &pl.node_segments[0];
    assert!(segs.len() > 1, "budget should force more than one segment");

    // total instruction count is unchanged by segmentation
    assert_eq!(node_ops(&pl, 0).len(), 48);

    // every segment respects the incremental budget
    for m in &pl.segment_mem[0] {
        assert!(
            m.incremental <= budget,
            "segment incremental {} exceeds budget {budget}",
            m.incremental
        );
    }

    // ids are unique and contiguous within the node
    let ids: Vec<u64> = segs.iter().map(|s| s.id).collect();
    assert_eq!(ids, (0..segs.len() as u64).collect::<Vec<_>>());
}

#[test]
fn tight_budget_across_chains_no_overflow() {
    // Regression: a node owning multiple output chains frees the first
    // chain's operands (resident drops below the segment's starting floor)
    // before allocating the next chain; the incremental calc must saturate
    // rather than underflow the u64. A budget of ~2 supertiles makes the
    // floor climb high enough to expose it. Exercised at 1 and 3 nodes.
    let p = matmul_program();
    let supers = default_supers(&p);
    let tile = 4096u64 * 4096 * 4;
    let budget = 2 * tile;
    for nodes in [1u16, 3] {
        let pl = plan_budgeted(&p, &dims(8192), &supers, nodes, budget).unwrap();
        validate(&pl).unwrap();
        for node in &pl.segment_mem {
            for m in node {
                assert!(
                    m.incremental <= budget,
                    "incremental {} > {budget}",
                    m.incremental
                );
            }
        }
    }
}

#[test]
fn budget_too_small_for_one_tile_errors() {
    let p = matmul_program();
    let supers = default_supers(&p);
    let err = plan_budgeted(&p, &dims(8192), &supers, 1, 1024)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds the memory budget"), "got: {err}");
}

#[test]
fn peak_resident_is_reported() {
    let p = matmul_program();
    let supers = default_supers(&p);
    let pl = plan(&p, &dims(8192), &supers, 1).unwrap();
    // at least one C chain's live set (acc + a + b = 3 x 64 MiB) is resident
    let tile = 4096u64 * 4096 * 4;
    assert!(pl.peak_resident >= 3 * tile);
}

// --- lineage re-execution ---

/// The output supertiles a plan recomputes, as sorted (tensor, lin).
fn recovered_outputs(pl: &Plan) -> Vec<(usize, u64)> {
    let mut v: Vec<(usize, u64)> = pl.stores.iter().map(|&(_, _, t, lin)| (t, lin)).collect();
    v.sort();
    v
}

#[test]
fn plan_exposes_stores_and_owners() {
    // The initial plan records every output STORE and each output's owner,
    // the data the dispatcher needs to drive recovery.
    let p = matmul_program();
    let supers = default_supers(&p);
    let pl = plan(&p, &dims(8192), &supers, 2).unwrap();
    // 4 C supertiles (2x2 grid), each STOREd once and owned by lin % 2
    assert_eq!(pl.stores.len(), 4);
    assert_eq!(pl.output_owner.len(), 4);
    for (&(t, lin), &owner) in &pl.output_owner {
        assert_eq!(t, 2, "only C is an output");
        assert_eq!(owner, (lin % 2) as u16);
    }
    // a STORE runs on the node that owns the tile it writes
    for &(_, node, t, lin) in &pl.stores {
        assert_eq!(pl.output_owner[&(t, lin)], node);
    }
}

#[test]
fn recover_reassigns_lost_chains_to_survivor() {
    // 2x2x2 matmul on 2 nodes; node 1 dies with nothing durable. Node 1
    // owned C(lin=1) and C(lin=3); both chains re-run on node 0 (the only
    // survivor), re-LOADing their inputs from durable storage.
    let p = matmul_program();
    let supers = default_supers(&p);
    let base = plan(&p, &dims(8192), &supers, 2).unwrap();

    let rec = recover_plan(
        &p,
        &dims(8192),
        &supers,
        2,
        &[1],
        &HashSet::new(),
        u64::MAX,
        IngestPolicy::default(),
        1,
        base.max_iid(),
        &HashMap::new(),
    )
    .unwrap();
    validate(&rec).unwrap();

    // exactly the two lost C supertiles, recomputed
    assert_eq!(recovered_outputs(&rec), vec![(2, 1), (2, 3)]);
    // 2 chains x (init + 2 k-steps)
    assert_eq!(count(&rec, 0, |o| matches!(o, Op::Compute { .. })), 6);
    assert_eq!(count(&rec, 0, |o| matches!(o, Op::Store { .. })), 2);
    // inputs come back from storage, not a peer; the dead node held none
    assert_eq!(rec.fetch_bytes, 0);
    assert_eq!(count(&rec, 0, |o| matches!(o, Op::Fetch { .. })), 0);
    assert!(count(&rec, 0, |o| matches!(o, Op::Load { .. })) > 0);

    // all recovery work lands on the survivor; the dead node gets nothing
    assert_eq!(node_ops(&rec, 1).len(), 0);
    assert!(!node_ops(&rec, 0).is_empty());
    // every recomputed chain was originally node 1's
    for key in recovered_outputs(&rec) {
        assert_eq!(base.output_owner[&key], 1);
    }
}

#[test]
fn recover_reissues_at_fresh_ids_and_versions() {
    // Reissued instructions must not alias iids still live in the survivor's
    // table, and reissued tiles must carry a bumped version so they can't
    // collide with version-0 tiles the survivor may still hold.
    let p = matmul_program();
    let supers = default_supers(&p);
    let base = plan(&p, &dims(8192), &supers, 2).unwrap();
    let iid_base = base.max_iid();

    let rec = recover_plan(
        &p,
        &dims(8192),
        &supers,
        2,
        &[1],
        &HashSet::new(),
        u64::MAX,
        IngestPolicy::default(),
        7,
        iid_base,
        &HashMap::new(),
    )
    .unwrap();
    validate(&rec).unwrap();

    for i in node_ops(&rec, 0) {
        assert!(i.iid > iid_base, "iid {} not above base {iid_base}", i.iid);
        if let Op::Alloc { tile, .. } = &i.op {
            assert_eq!(tile.version(), 7, "reissued tile not re-versioned");
        }
    }
}

#[test]
fn recover_skips_already_stored_outputs() {
    // An output that reached storage before the crash is durable; never
    // recomputed. Mark C(lin=1) durable; only C(lin=3) comes back.
    let p = matmul_program();
    let supers = default_supers(&p);
    let base = plan(&p, &dims(8192), &supers, 2).unwrap();
    let durable: HashSet<(usize, u64)> = [(2usize, 1u64)].into_iter().collect();

    let rec = recover_plan(
        &p,
        &dims(8192),
        &supers,
        2,
        &[1],
        &durable,
        u64::MAX,
        IngestPolicy::default(),
        1,
        base.max_iid(),
        &HashMap::new(),
    )
    .unwrap();
    validate(&rec).unwrap();
    assert_eq!(recovered_outputs(&rec), vec![(2, 3)]);
    assert_eq!(count(&rec, 0, |o| matches!(o, Op::Compute { .. })), 3);
}

#[test]
fn recover_balances_across_multiple_survivors() {
    // 4x4 grid on 4 nodes; node 2 dies. Its 4 outputs (lin % 4 == 2:
    // 2,6,10,14) redistribute over the survivors [0,1,3] block-cyclically:
    // none land back on the dead node, and all four are recomputed.
    let p = matmul_program();
    let supers = default_supers(&p); // 4096 each -> 4x4 grid at 16384
    let base = plan(&p, &dims(16384), &supers, 4).unwrap();

    let rec = recover_plan(
        &p,
        &dims(16384),
        &supers,
        4,
        &[2],
        &HashSet::new(),
        u64::MAX,
        IngestPolicy::default(),
        1,
        base.max_iid(),
        &HashMap::new(),
    )
    .unwrap();
    validate(&rec).unwrap();

    assert_eq!(
        recovered_outputs(&rec),
        vec![(2, 2), (2, 6), (2, 10), (2, 14)]
    );
    assert_eq!(node_ops(&rec, 2).len(), 0, "the dead node gets no work");
    let survivors: HashSet<u16> = [0, 1, 3].into_iter().collect();
    for &(_, node, t, lin) in &rec.stores {
        assert!(
            survivors.contains(&node),
            "STORE on dead/unknown node {node}"
        );
        assert_eq!(
            base.output_owner[&(t, lin)],
            2,
            "recovered a chain node 2 didn't own"
        );
    }
    let busy = (0..4).filter(|&n| !node_ops(&rec, n).is_empty()).count();
    assert!(
        busy >= 2,
        "recovery should spread across survivors, got {busy}"
    );
}

#[test]
fn recover_handles_simultaneous_failures() {
    // 4x4 grid; nodes 1 and 3 both fail. Their outputs (lin % 4 in {1,3})
    // all come back on the survivors [0, 2], none on a dead node.
    let p = matmul_program();
    let supers = default_supers(&p);
    let base = plan(&p, &dims(16384), &supers, 4).unwrap();

    let rec = recover_plan(
        &p,
        &dims(16384),
        &supers,
        4,
        &[1, 3],
        &HashSet::new(),
        u64::MAX,
        IngestPolicy::default(),
        1,
        base.max_iid(),
        &HashMap::new(),
    )
    .unwrap();
    validate(&rec).unwrap();

    let got = recovered_outputs(&rec);
    let want: Vec<(usize, u64)> = (0..16u64)
        .filter(|l| l % 4 == 1 || l % 4 == 3)
        .map(|l| (2, l))
        .collect();
    assert_eq!(got, want);
    assert_eq!(node_ops(&rec, 1).len(), 0);
    assert_eq!(node_ops(&rec, 3).len(), 0);
    for &(_, node, ..) in &rec.stores {
        assert!(
            node == 0 || node == 2,
            "recovery placed work on dead node {node}"
        );
    }
}

#[test]
fn recover_all_dead_errors() {
    let p = matmul_program();
    let supers = default_supers(&p);
    let err = recover_plan(
        &p,
        &dims(8192),
        &supers,
        1,
        &[0],
        &HashSet::new(),
        u64::MAX,
        IngestPolicy::default(),
        1,
        0,
        &HashMap::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("every node has failed"), "got: {err}");
}
