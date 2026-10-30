// Plans a cluster program onto nodes. The private submodules are its phases:
// instantiate, lower, then segment. [`server`] dispatches the plan.

pub mod autotune;
pub mod dot;
mod instantiate;
pub mod job;
mod lower;
mod segment;
pub mod server;

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::isa::{Instr, InstrId, Segment};
use phobos_cluster::tile::{NodeId, ScalarValue, TileId};

use instantiate::instantiate;
use lower::lower;

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

    pub incremental: u64, // bytes allocated within the segment
}

#[derive(Debug)]
pub struct Plan {
    /// Segments to dispatch, in order, indexed by node id.
    pub node_segments: Vec<Vec<Segment>>,

    /// Memory accounting parallel to node_segments.
    pub segment_mem: Vec<Vec<SegMem>>,

    /// Supertile shape per tensor.
    pub super_shapes: Vec<Vec<u64>>,

    /// Supertile-grid extents per tensor.
    pub super_grids: Vec<Vec<u64>>,

    /// Per node, each FETCHed tile and the node it comes from.
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

/// Plans with each node's program split into segments whose incremental
/// working set stays within `budget` bytes. `u64::MAX` gives one segment per
/// node.
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

/// Re-plans the work lost when `dead` fails, for redispatch onto survivors.
///
/// `width` is the original cluster size, dead nodes included. Output
/// ownership is keyed by `lin % width`, so it must match the initial plan.
///
/// Every output a dead node owned and had not STOREd is recomputed from
/// inputs re-LOADed from storage. `durable` lists outputs to skip, either
/// already STOREd or claimed by another recovery. Reissued tiles carry the
/// bumped `version` so they cannot collide with resident tiles, and iids
/// start above `iid_base` so they cannot alias earlier instructions.
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

    // Outputs owned by a dead node and not yet durable.
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
    /// Live nodes in ascending id order. Output `lin` goes to
    /// `live[lin % live.len()]`.
    live: Vec<NodeId>,
    /// Max node id + 1.
    width: usize,
    /// When set, emit only the chains for these (tensor, lin) outputs.
    restrict: Option<HashSet<(usize, u64)>>,
    version: u16,
    /// Exclusive lower bound: the first iid is `iid_base + 1`.
    iid_base: InstrId,
    budget: u64,
    policy: IngestPolicy,
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
