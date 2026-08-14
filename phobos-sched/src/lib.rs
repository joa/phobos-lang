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

/// Re-plan the work lost when `dead` fails, for redispatch onto the survivors.
///
/// Lineage recovery for owner-computes: every output supertile a dead node owned
/// but had not yet STOREd is recomputed from scratch on a survivor. Its operands
/// are durable inputs re-LOADed from storage, so each chain is self-contained
/// and the lost subgraph is just those chains placed over the survivor set.
/// Deterministic TileIds keep the rest of the DAG valid; reissued tiles carry a
/// bumped version so they cannot collide with tiles still resident on the
/// survivor that adopts them, and iids start at `iid_base` so they never alias
/// instructions still in a survivor's table.
///
/// `width` is the original cluster size, `dead` every node that has failed so
/// far, and `durable` the output supertiles the dispatcher must not recompute:
/// those already STOREd, plus those an outstanding recovery is redoing. Returns
/// an empty plan when nothing is lost.
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
mod tests;
