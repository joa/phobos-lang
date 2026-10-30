// Lowering: places each compute on the node that owns its output and emits
// every node's instruction list, which segment.rs then cuts to the budget.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, bail};
use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::isa::{Instr, InstrId, Op, Region, ScalarArg, StorageRef};
use phobos_cluster::tile::{AccessMode, NodeId, ScalarValue, TileId};

use crate::instantiate::Instantiated;
use crate::segment::segment;
use crate::{IngestPolicy, LowerCfg, Plan};

struct TileSlot {
    tile: TileId,
    alloc: InstrId,
    last_write: Option<InstrId>,
    consumers: Vec<InstrId>,
}

#[allow(clippy::map_entry)]
pub(super) fn lower(
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

    // An input supertile's home is its lowest-id consumer, which LOADs it from
    // storage. Every other consumer FETCHes it from the home, once each.
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
        // Keyed by (tensor, coord). A BTreeMap keeps the output deterministic.
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
                                    let from = home(&key) as NodeId;
                                    fetches[node].push((tile, from));
                                    fetch_bytes += tile_bytes(*t);
                                    Op::Fetch { tile, from }
                                }
                                _ => Op::Load {
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

            // STORE and FREE each tile whose last use is this compute.
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
        let (segs, mems, node_peak) = segment(instrs, cfg.budget, &mut seg_id)?;
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
