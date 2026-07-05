pub mod autotune;
pub mod dot;
pub mod job;
pub mod server;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Context, Result, bail};
use phobos_cluster::ir::{ClusterProgram, ClusterStmt, Coord};
use phobos_cluster::isa::{Instr, InstrId, Op, Region, ScalarArg, Segment, StorageRef};
use phobos_cluster::tile::{AccessMode, DataType, NodeId, ScalarValue, TileId};
use phobos_lang::ast::{AttrArg, Dim};

/// Default launch args
pub const CTA: (u32, u32, u32) = (phobos_lang::ast::DEFAULT_CTA_THREADS as u32, 1, 1);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IngestPolicy {
    /// LOAD the supertile directly via its URI.
    #[default]
    DirectLoad,

    /// Lowest-id consumer LOADs; every other node FETCHes from it.
    HomeLoadPeerFetch,
}

/// Per-segment memory accounting, parallel to [`Plan::node_segments`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegMem {
    pub peak: u64, // high-watermark for this segment

    pub incremental: u64, // bytes NEWLY allocated within the segment
}

#[derive(Debug)]
pub struct Plan {
    /// The ordered list of segments to dispatch; indexed by node id
    pub node_segments: Vec<Vec<Segment>>,

    /// Memory accounting parallel to node_segments.
    pub segment_mem: Vec<Vec<SegMem>>,

    /// Supertile shape per tensor.
    pub super_shapes: Vec<Vec<u64>>,

    /// Supertile-grid extents per tensor.
    pub super_grids: Vec<Vec<u64>>,

    /// Tile homes for FETCH instructions.
    pub fetches: Vec<Vec<(TileId, NodeId)>>,

    /// Total bytes moved by FETCHes.
    pub fetch_bytes: u64,

    /// Highest resident bytes (must fit in VRAM).
    pub peak_resident: u64,

    /// Expected STOREs as (iid, node, tensor, linear coord).
    pub stores: Vec<(InstrId, NodeId, usize, u64)>,

    /// Where each supertile (tensor, linear coord) was placed.
    pub output_owner: HashMap<(usize, u64), NodeId>,
}

impl Plan {
    /// Flattened instructions in dispatch order for a node.
    pub fn node_instrs(&self, node: usize) -> impl Iterator<Item = &Instr> {
        self.node_segments[node]
            .iter()
            .flat_map(|s| s.instructions.iter())
    }

    pub fn total_instrs(&self) -> u64 {
        self.node_segments
            .iter()
            .flat_map(|segs| segs.iter())
            .map(|s| s.instructions.len() as u64)
            .sum()
    }

    pub fn max_iid(&self) -> InstrId {
        self.node_segments
            .iter()
            .flat_map(|segs| segs.iter())
            .flat_map(|s| s.instructions.iter())
            .map(|i| i.iid)
            .max()
            .unwrap_or(0)
    }
}

pub fn default_supers(p: &ClusterProgram) -> HashMap<String, i64> {
    p.super_dims
        .iter()
        .map(|d| (d.name.clone(), d.choices[0]))
        .collect()
}

pub fn plan(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    nodes: u16,
) -> Result<Plan> {
    plan_budgeted_with(
        p,
        dims,
        supers,
        nodes,
        u64::MAX,
        IngestPolicy::default(),
        &HashMap::new(),
    )
}

pub fn plan_with(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    nodes: u16,
    policy: IngestPolicy,
) -> Result<Plan> {
    plan_budgeted_with(p, dims, supers, nodes, u64::MAX, policy, &HashMap::new())
}

/// Plan by partitioning each node's program into segments where the
/// incremental working set stays within budget bytes.
/// u64::MAX is one segment per node.
pub fn plan_budgeted(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    nodes: u16,
    budget: u64,
) -> Result<Plan> {
    plan_budgeted_with(
        p,
        dims,
        supers,
        nodes,
        budget,
        IngestPolicy::default(),
        &HashMap::new(),
    )
}

pub fn plan_budgeted_with(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    nodes: u16,
    budget: u64,
    policy: IngestPolicy,
    scalars: &HashMap<String, ScalarValue>,
) -> Result<Plan> {
    if nodes == 0 {
        bail!("cluster has no nodes");
    }
    let inst = instantiate(p, dims, supers)?;
    let cfg = LowerCfg {
        live: (0..nodes).collect(),
        width: nodes as usize,
        restrict: None,
        version: 0,
        iid_base: 0,
        budget,
        policy,
    };
    lower(p, &inst, &cfg, scalars)
}

/// Re-plan the work lost when dead fails, for redispatch onto the survivors.
///
/// Lineage recovery for owner-computes: every output supertile dead owned but
/// had not yet STOREd is recomputed from scratch on a surviving node. Its
/// operands are durable inputs (re-LOADed from storage), and the chain is
/// self-contained, so the lost subgraph is just those chains placed over the
/// survivor set. Deterministic TileIds keep the rest of the DAG valid; the
/// reissued tiles carry a bumped version so they can't collide with tiles still
/// resident on the survivor that adopts them, and iids start at iid_base (use
/// the original plan's [`Plan::max_iid`] + 1, advanced across successive
/// failures) so they never alias instructions still in a survivor's table.
///
/// width is the original cluster size (max node id + 1); dead is every node
/// that has failed so far (so survivors exclude them all, and chains a prior
/// recovery placed on a now-dead node are re-recovered); durable is the set of
/// output supertiles (tensor, linear coord) the dispatcher should not
/// recompute: those already STOREd, plus those an outstanding recovery is
/// already redoing. Returns an empty plan (no segments) when nothing is lost.
#[allow(clippy::too_many_arguments)]
pub fn recover_plan(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    width: u16,
    dead: &[NodeId],
    durable: &HashSet<(usize, u64)>,
    budget: u64,
    policy: IngestPolicy,
    version: u16,
    iid_base: InstrId,
    scalars: &HashMap<String, ScalarValue>,
) -> Result<Plan> {
    let dead_set: HashSet<NodeId> = dead.iter().copied().collect();
    let survivors: Vec<NodeId> = (0..width).filter(|n| !dead_set.contains(n)).collect();
    if survivors.is_empty() {
        bail!("every node has failed; cannot recover");
    }

    let inst = instantiate(p, dims, supers)?;

    // lost are the tiles that were owned by dead nodes
    let mut lost: HashSet<(usize, u64)> = HashSet::new();

    for c in &inst.computes {
        let key = inst.output_key(c)?;
        let owner = (key.1 % width as u64) as NodeId;
        if dead_set.contains(&owner) && !durable.contains(&key) {
            lost.insert(key);
        }
    }

    let cfg = LowerCfg {
        live: survivors,
        width: width as usize,
        restrict: Some(lost),
        version,
        iid_base,
        budget,
        policy,
    };

    lower(p, &inst, &cfg, scalars)
}

struct LowerCfg {
    /// Live nodes in ascending id order.
    /// Output lin maps to live[lin % live.len()]
    live: Vec<NodeId>,
    /// Max node id + 1
    width: usize,
    /// Restrict to the (tensor, lin) set when given; otherwise emit every chain.
    restrict: Option<HashSet<(usize, u64)>>,
    version: u16,
    /// Exclusive lower bound -> the first iid is iid_base + 1.
    iid_base: InstrId,
    budget: u64,
    policy: IngestPolicy,
}

struct Instantiated {
    computes: Vec<CompInst>,
    super_shapes: Vec<Vec<u64>>,
    super_grids: Vec<Vec<u64>>,
    leaf_grids: Vec<(u32, u32, u32)>,
    /// Per-leaf CTA shape from @launch/default.
    leaf_ctas: Vec<(u32, u32, u32)>,
}

impl Instantiated {
    fn lin(&self, tensor: usize, coords: &[u64]) -> u64 {
        let mut l = 0;

        for (c, g) in coords.iter().zip(&self.super_grids[tensor]) {
            l = l * g + c;
        }

        l
    }

    /// (tensor, lin) of a compute's (single) written supertile.
    fn output_key(&self, c: &CompInst) -> Result<(usize, u64)> {
        let mut outs = c
            .args
            .iter()
            .filter(|(_, _, m)| matches!(m, AccessMode::Write | AccessMode::RMW));
        let out = outs.next().context("compute writes no supertile")?;

        if outs.next().is_some() {
            bail!("compute writes more than one supertile (unsupported)");
        }

        Ok((out.0, self.lin(out.0, &out.1)))
    }
}

fn advance(point: &mut [u64], ext: &[u64]) -> bool {
    for (p, &e) in point.iter_mut().zip(ext).rev() {
        *p += 1;
        if *p < e {
            return true;
        }
        *p = 0;
    }
    false
}

fn instantiate(
    p: &ClusterProgram,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
) -> Result<Instantiated> {
    if p.tensors.len() >= 1 << 12 {
        bail!("too many tensors for TileId encoding");
    }

    let dim_val = |d: &Dim| -> Result<u64> {
        match d {
            Dim::Sym(s) => dims
                .get(s)
                .copied()
                .with_context(|| format!("problem dim '{s}' is unbound"))
                .and_then(|v| {
                    if v <= 0 {
                        bail!("problem dim '{s}' must be positive, got {v}")
                    } else {
                        Ok(v as u64)
                    }
                }),
            Dim::Int(n) => Ok(*n as u64),
        }
    };

    let super_val = |s: &str| -> Result<u64> {
        match supers.get(s).or_else(|| dims.get(s)).copied() {
            Some(v) if v > 0 => Ok(v as u64),
            Some(v) => bail!("supertile dim '{s}' must be positive, got {v}"),
            None => bail!("supertile dim '{s}' is unbound"),
        }
    };

    // supertile grid axes
    let mut grid_ext = Vec::new();
    for ax in &p.grid {
        let d = dim_val(&ax.dim)?;
        let s = super_val(&ax.super_sym)?;
        if d % s != 0 {
            bail!(
                "dim {:?} = {d} is not a multiple of supertile dim {} = {s}",
                ax.dim,
                ax.super_sym
            );
        }
        grid_ext.push(d / s);
    }

    // per-tensor supertile shapes and grids
    let mut super_shapes = Vec::new();
    let mut super_grids = Vec::new();

    for t in &p.tensors {
        let mut shape = Vec::new();
        let mut grid = Vec::new();

        for (dim, sym) in t.dims.iter().zip(&t.super_syms) {
            let d = dim_val(dim)?;
            let s = super_val(sym)?;
            if d % s != 0 {
                bail!(
                    "tensor '{}' dim {:?} = {d} is not a multiple of supertile dim {sym} = {s}",
                    t.name,
                    dim
                );
            }
            shape.push(s);
            grid.push(d / s);
        }

        if grid.iter().product::<u64>() >= 1 << 36 {
            bail!(
                "tensor '{}' has too many supertiles for TileId encoding",
                t.name
            );
        }

        super_shapes.push(shape);
        super_grids.push(grid);
    }

    // leaf launch grids: same (dim, sym) formula scaled down.
    // the leaf's runtime dims are the supertile shape divided by the @autotune default for now
    let mut leaf_grids = Vec::new();
    let mut leaf_ctas = Vec::new();

    for leaf in &p.leaves {
        let cta_threads = leaf.kernel.cta_threads().map_err(|e| anyhow::anyhow!(e))?;

        leaf_ctas.push((cta_threads as u32, 1, 1));

        let devs: HashMap<&str, i64> = leaf
            .kernel
            .attrs
            .iter()
            .filter(|a| a.name == "autotune")
            .flat_map(|a| a.args.iter())
            .filter_map(|arg| match arg {
                AttrArg::Search { name, choices } => Some((name.as_str(), choices[0])),
                _ => None,
            })
            .collect();

        let mut g = [1u32; 3];

        for (i, ax) in p.grid.iter().enumerate() {
            if i >= 3 {
                bail!("more than 3 grid axes");
            }

            let sup = super_val(&ax.super_sym)?;
            let dev = devs.get(ax.super_sym.as_str()).copied().with_context(|| {
                format!(
                    "leaf '{}' has no @autotune default for '{}'",
                    leaf.kernel.name, ax.super_sym
                )
            })?;

            if dev <= 0 || sup % dev as u64 != 0 {
                bail!(
                    "supertile dim {} = {sup} is not a multiple of leaf '{}' device tile {dev}",
                    ax.super_sym,
                    leaf.kernel.name
                );
            }

            g[i] = (sup / dev as u64) as u32;
        }

        leaf_grids.push((g[0], g[1], g[2]));
    }

    // unroll: one body execution per supertile-grid point
    let mut computes = Vec::new();
    let mut gpt = vec![0u64; grid_ext.len()];

    loop {
        unroll(
            &p.body,
            &gpt,
            &mut HashMap::new(),
            dims,
            supers,
            &mut computes,
        )?;

        if !advance(&mut gpt, &grid_ext) {
            break;
        }
    }

    Ok(Instantiated {
        computes,
        super_shapes,
        super_grids,
        leaf_grids,
        leaf_ctas,
    })
}

struct CompInst {
    leaf: usize,
    args: Vec<(usize, Vec<u64>, AccessMode)>,
    scalars: Vec<usize>,
}

fn unroll(
    stmts: &[ClusterStmt],
    gpt: &[u64],
    loops: &mut HashMap<String, u64>,
    dims: &HashMap<String, i64>,
    supers: &HashMap<String, i64>,
    out: &mut Vec<CompInst>,
) -> Result<()> {
    for stmt in stmts {
        match stmt {
            ClusterStmt::Compute {
                leaf,
                args,
                scalars,
            } => {
                let mut cluster_args = Vec::new();

                for (r, mode) in args {
                    let coords = r
                        .coords
                        .iter()
                        .map(|c| match c {
                            Coord::Grid(i) => gpt[*i],
                            Coord::Loop(v) => loops[v],
                            Coord::Full => 0,
                        })
                        .collect();
                    cluster_args.push((r.tensor, coords, *mode));
                }

                out.push(CompInst {
                    leaf: *leaf,
                    args: cluster_args,
                    scalars: scalars.clone(),
                });
            }

            ClusterStmt::Loop {
                var,
                dim,
                super_sym,
                body,
            } => {
                let d = match dim {
                    Dim::Sym(s) => *dims
                        .get(s)
                        .with_context(|| format!("loop dim '{s}' is unbound"))?
                        as u64,
                    Dim::Int(n) => *n as u64,
                };

                let s = *supers
                    .get(super_sym)
                    .with_context(|| format!("supertile dim '{super_sym}' is unbound"))?
                    as u64;

                if d % s != 0 {
                    bail!("loop bound {d} is not a multiple of supertile dim {super_sym} = {s}");
                }

                for i in 0..d / s {
                    loops.insert(var.clone(), i);
                    unroll(body, gpt, loops, dims, supers, out)?;
                }

                loops.remove(var);
            }
        }
    }

    Ok(())
}

struct TileSlot {
    tile: TileId,
    alloc: InstrId,
    last_write: Option<InstrId>,
    consumers: Vec<InstrId>,
}

#[allow(clippy::map_entry)]
fn lower(
    p: &ClusterProgram,
    inst: &Instantiated,
    cfg: &LowerCfg,
    scalars: &HashMap<String, ScalarValue>,
) -> Result<Plan> {
    let resolve_scalars = |idxs: &[usize]| -> Result<Vec<ScalarArg>> {
        idxs.iter()
            .map(|&si| {
                let decl = &p.scalars[si];
                let value = *scalars
                    .get(&decl.name)
                    .with_context(|| format!("scalar parameter '{}' is unbound", decl.name))?;

                if value.data_type() != decl.data_type {
                    bail!(
                        "scalar '{}' bound as {:?} but declared {:?}",
                        decl.name,
                        value.data_type(),
                        decl.data_type
                    );
                }

                Ok(ScalarArg {
                    pos: decl.param_pos as u32,
                    value,
                })
            })
            .collect()
    };

    let computes = &inst.computes;
    let super_shapes = &inst.super_shapes;
    let leaf_grids = &inst.leaf_grids;
    let leaf_ctas = &inst.leaf_ctas;
    let lin = |tensor: usize, coords: &[u64]| -> u64 { inst.lin(tensor, coords) };
    let tile_bytes = |tensor: usize| -> u64 {
        super_shapes[tensor].iter().product::<u64>() * p.tensors[tensor].data_type.bytes() as u64
    };

    let live = &cfg.live;
    let mut placement = vec![0usize; computes.len()];
    let mut active: Vec<usize> = Vec::new();
    let mut output_owner: HashMap<(usize, u64), NodeId> = HashMap::new();

    for (ci, c) in computes.iter().enumerate() {
        let key = inst.output_key(c)?;

        placement[ci] = live[(key.1 % live.len() as u64) as usize] as usize;

        if cfg.restrict.as_ref().is_none_or(|r| r.contains(&key)) {
            active.push(ci);
            output_owner.insert(key, placement[ci] as NodeId);
        }
    }

    let mut consumers_of: HashMap<(usize, u64), BTreeSet<usize>> = HashMap::new();
    for &ci in &active {
        let node = placement[ci];
        for (t, coords, mode) in &computes[ci].args {
            if matches!(p.tensors[*t].mode, AccessMode::Read) && matches!(mode, AccessMode::Read) {
                consumers_of
                    .entry((*t, lin(*t, coords)))
                    .or_default()
                    .insert(node);
            }
        }
    }

    // home of an input supertile = lowest-id consuming node (it LOADs from storage);
    // every other consuming node FETCHes from it. The home serves the tile once per remote consuming node.
    //
    // note: this depends on the fetch mode; could also be nodes just loading from storage
    let home = |key: &(usize, u64)| -> usize { *consumers_of[key].iter().next().unwrap() };
    let remote_serves = |key: &(usize, u64)| -> u32 { (consumers_of[key].len() - 1) as u32 };

    let mut node_computes: Vec<Vec<usize>> = vec![Vec::new(); cfg.width];
    for &ci in &active {
        node_computes[placement[ci]].push(ci);
    }

    let last_use: Vec<HashMap<(usize, u64), usize>> = node_computes
        .iter()
        .map(|cis| {
            let mut m = HashMap::new();
            for (pos, &ci) in cis.iter().enumerate() {
                for (t, coords, _) in &computes[ci].args {
                    m.insert((*t, lin(*t, coords)), pos);
                }
            }
            m
        })
        .collect();

    let mut iid: InstrId = cfg.iid_base;
    let mut next = || {
        iid += 1;
        iid
    };

    let mut node_lists: Vec<Vec<Instr>> = vec![Vec::new(); cfg.width];
    let mut fetches: Vec<Vec<(TileId, NodeId)>> = vec![Vec::new(); cfg.width];
    let mut fetch_bytes: u64 = 0;
    let mut stores: Vec<(InstrId, NodeId, usize, u64)> = Vec::new();

    for node in 0..cfg.width {
        // (tensor, coord) -> slot; BTreeMap for deterministic output
        let mut slots: BTreeMap<(usize, u64), TileSlot> = BTreeMap::new();
        let instrs = &mut node_lists[node];

        for (pos, &ci) in node_computes[node].iter().enumerate() {
            let c = &computes[ci];

            // materialize args
            let mut deps = Vec::new();
            let mut iargs = Vec::new();

            for (t, coords, mode) in &c.args {
                let key = (*t, lin(*t, coords));
                if !slots.contains_key(&key) {
                    let tile = TileId::new(*t as u16, cfg.version, key.1);
                    let alloc = next();
                    instrs.push(Instr {
                        iid: alloc,
                        deps: vec![],
                        op: Op::Alloc {
                            tile,
                            shape: super_shapes[*t].clone(),
                            data_type: p.tensors[*t].data_type,
                        },
                    });

                    let last_write = match p.tensors[*t].mode {
                        AccessMode::RMW => {
                            let load = next();
                            instrs.push(Instr {
                                iid: load,
                                deps: vec![alloc],
                                op: Op::Load {
                                    tile,
                                    src: storage_ref(*t, coords, super_shapes),
                                },
                            });
                            Some(load)
                        }
                        AccessMode::Read => {
                            let ingest = next();
                            let op = match cfg.policy {
                                IngestPolicy::HomeLoadPeerFetch if node != home(&key) => {
                                    // must fetch
                                    let from = home(&key) as NodeId;
                                    fetches[node].push((tile, from));
                                    fetch_bytes += tile_bytes(*t);
                                    Op::Fetch { tile, from }
                                }
                                _ => Op::Load {
                                    // direct load
                                    tile,
                                    src: storage_ref(*t, coords, super_shapes),
                                },
                            };
                            instrs.push(Instr {
                                iid: ingest,
                                deps: vec![alloc],
                                op,
                            });
                            Some(ingest)
                        }
                        AccessMode::Write => None,
                    };

                    slots.insert(
                        key,
                        TileSlot {
                            tile,
                            alloc,
                            last_write,
                            consumers: vec![],
                        },
                    );
                }
                let slot = &slots[&key];
                match mode {
                    AccessMode::Read | AccessMode::RMW => match slot.last_write {
                        Some(w) => deps.push(w),
                        None => bail!(
                            "supertile of '{}' is read before it is produced",
                            p.tensors[*t].name
                        ),
                    },
                    AccessMode::Write => deps.push(slot.alloc),
                }
                iargs.push((slot.tile, *mode));
            }
            deps.sort_unstable();
            deps.dedup();

            let ciid = next();
            instrs.push(Instr {
                iid: ciid,
                deps,
                op: Op::Compute {
                    kernel: c.leaf as u32,
                    args: iargs,
                    scalars: resolve_scalars(&c.scalars)?,
                    grid: leaf_grids[c.leaf],
                    cta: leaf_ctas[c.leaf],
                },
            });

            for (t, coords, mode) in &c.args {
                let slot = slots.get_mut(&(*t, lin(*t, coords))).unwrap();
                match mode {
                    AccessMode::Write | AccessMode::RMW => slot.last_write = Some(ciid),
                    AccessMode::Read => slot.consumers.push(ciid),
                }
            }

            // inline lifetime end: STORE/FREE each tile whose last use is now
            // last_use is exact!
            for (t, coords, _) in &c.args {
                let key = (*t, lin(*t, coords));
                if last_use[node][&key] != pos {
                    continue;
                }
                let slot = slots.get(&key).unwrap();
                let tile = slot.tile;
                if matches!(p.tensors[*t].mode, AccessMode::Write | AccessMode::RMW) {
                    let last = slot.last_write.with_context(|| {
                        format!(
                            "output supertile of '{}' never produced",
                            p.tensors[*t].name
                        )
                    })?;
                    let store = next();
                    stores.push((store, node as NodeId, *t, key.1));
                    instrs.push(Instr {
                        iid: store,
                        deps: vec![last],
                        op: Op::Store {
                            tile,
                            dst: storage_ref(*t, coords, super_shapes),
                        },
                    });
                    instrs.push(Instr {
                        iid: next(),
                        deps: vec![store],
                        op: Op::Free {
                            tile,
                            expected_serves: 0,
                        },
                    });
                } else {
                    let expected_serves = match cfg.policy {
                        IngestPolicy::HomeLoadPeerFetch if node == home(&key) => {
                            remote_serves(&key)
                        }
                        _ => 0,
                    };
                    instrs.push(Instr {
                        iid: next(),
                        deps: slot.consumers.clone(),
                        op: Op::Free {
                            tile,
                            expected_serves,
                        },
                    });
                }
            }
        }
    }

    // partition each node's flat list into memory-budgeted segments
    let mut node_segments = Vec::with_capacity(cfg.width);
    let mut segment_mem = Vec::with_capacity(cfg.width);
    let mut seg_id: u64 = 0;
    let mut peak_resident = 0u64;

    for instrs in node_lists {
        let (segs, mems, node_peak) = segment(instrs, cfg.budget, &mut seg_id, p.tensors.len())?;
        peak_resident = peak_resident.max(node_peak);
        node_segments.push(segs);
        segment_mem.push(mems);
    }

    Ok(Plan {
        node_segments,
        segment_mem,
        super_shapes: inst.super_shapes.clone(),
        super_grids: inst.super_grids.clone(),
        fetches,
        fetch_bytes,
        peak_resident,
        stores,
        output_owner,
    })
}

/// Partition one node's topologically ordered instruction list into segments
/// whose incremental working set (bytes allocated since the segment began
/// that are still live at its peak) stays within budget. Returns the
/// segments, their [`SegMem`], and the node's absolute resident high-water.
///
/// Boundaries fall before an ALLOC that would push the current segment's
/// incremental footprint over budget; a single tile larger than the budget is
/// a hard error (the autotuner's feasibility prune should have rejected the
/// config first). Cutting never reorders; deps that now cross a boundary are
/// ordinary same-node InstrId deps and stay valid (see [`validate`]).
fn segment(
    instrs: Vec<Instr>,
    budget: u64,
    seg_id: &mut u64,
    ntensors: usize,
) -> Result<(Vec<Segment>, Vec<SegMem>, u64)> {
    // tile -> bytes, learned from ALLOC ops (FREE carries no size).
    let mut bytes_of: HashMap<TileId, u64> = HashMap::new();
    let data_type_bytes = |d: DataType| d.bytes() as u64;
    for i in &instrs {
        if let Op::Alloc {
            tile,
            shape,
            data_type,
        } = &i.op
        {
            bytes_of.insert(
                *tile,
                shape.iter().product::<u64>() * data_type_bytes(*data_type),
            );
        }
    }
    let _ = ntensors; // (reserved: per-tensor budgeting could key on this)

    let mut segs = Vec::new();
    let mut mems = Vec::new();
    let mut cur: Vec<Instr> = Vec::new();

    let mut resident: u64 = 0; // absolute, across the whole node program
    let mut seg_start: u64 = 0; // resident when the current segment began
    let mut seg_peak: u64 = 0; // absolute high-water within the current segment
    let mut node_peak: u64 = 0;

    let flush = |cur: &mut Vec<Instr>,
                 segs: &mut Vec<Segment>,
                 mems: &mut Vec<SegMem>,
                 seg_peak: u64,
                 seg_start: u64,
                 seg_id: &mut u64| {
        if cur.is_empty() {
            return;
        }
        segs.push(Segment {
            id: *seg_id,
            instructions: std::mem::take(cur),
        });
        mems.push(SegMem {
            peak: seg_peak,
            incremental: seg_peak.saturating_sub(seg_start),
        });
        *seg_id += 1;
    };

    for instr in instrs {
        if let Op::Alloc { tile, .. } = &instr.op {
            let b = bytes_of[tile];
            if b > budget {
                bail!("supertile of {b} bytes exceeds the memory budget of {budget} bytes",);
            }
            // Would this allocation push the segment's incremental working
            // set over budget? Inline FREEs can drop resident below the
            // segment's starting floor, so this must saturate; being below the
            // floor means negative incremental, never a reason to cut (and a
            // plain - would underflow the u64, panicking).
            if !cur.is_empty() && (resident + b).saturating_sub(seg_start) > budget {
                flush(&mut cur, &mut segs, &mut mems, seg_peak, seg_start, seg_id);
                seg_start = resident;
                seg_peak = resident;
            }
            resident += b;
        } else if let Op::Free { tile, .. } = &instr.op {
            resident = resident.saturating_sub(bytes_of.get(tile).copied().unwrap_or(0));
        }
        seg_peak = seg_peak.max(resident);
        node_peak = node_peak.max(resident);
        cur.push(instr);
    }
    flush(&mut cur, &mut segs, &mut mems, seg_peak, seg_start, seg_id);

    Ok((segs, mems, node_peak))
}

fn storage_ref(tensor: usize, coords: &[u64], super_shapes: &[Vec<u64>]) -> StorageRef {
    StorageRef::Tensor {
        tensor: tensor as u32,
        region: Region {
            offset: coords
                .iter()
                .zip(&super_shapes[tensor])
                .map(|(c, s)| c * s)
                .collect(),
            shape: super_shapes[tensor].clone(),
        },
    }
}

pub fn validate(plan: &Plan) -> Result<()> {
    for (node, segs) in plan.node_segments.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for seg in segs {
            for instr in &seg.instructions {
                for d in &instr.deps {
                    if !seen.contains(d) {
                        bail!(
                            "instr {} on node {node} depends on {} which is not an \
                             earlier instruction in the node's dispatch order",
                            instr.iid,
                            d
                        );
                    }
                }
                seen.insert(instr.iid);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
            .filter(|i| {
                matches!(&i.op, Op::Compute { args, .. } if args.iter().any(|(t, _)| *t == c00))
            })
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
        let scalars: HashMap<String, ScalarValue> =
            [("scale".to_string(), ScalarValue::F32(0.125))]
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
            phobos_lang::codegen::emit(b, std::slice::from_ref(&p.leaves[0].kernel), c, m)
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
}
