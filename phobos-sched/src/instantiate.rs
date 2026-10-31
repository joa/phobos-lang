// Instantiation: a cluster program at concrete dims and supertile sizes,
// unrolled into one compute per supertile-grid point and loop iteration.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use phobos_cluster::ir::{ClusterProgram, ClusterStmt, Coord};
use phobos_cluster::tile::AccessMode;
use phobos_lang::ast::{AttrArg, Dim};

pub(super) struct Instantiated {
    pub(super) computes: Vec<CompInst>,
    pub(super) super_shapes: Vec<Vec<u64>>,
    pub(super) super_grids: Vec<Vec<u64>>,
    pub(super) leaf_grids: Vec<(u32, u32, u32)>,
    /// Per-leaf CTA shape from @launch/default.
    pub(super) leaf_ctas: Vec<(u32, u32, u32)>,
}

impl Instantiated {
    pub(super) fn lin(&self, tensor: usize, coords: &[u64]) -> u64 {
        let mut l = 0;

        for (c, g) in coords.iter().zip(&self.super_grids[tensor]) {
            l = l * g + c;
        }

        l
    }

    /// (tensor, lin) of a compute's (single) written supertile.
    pub(super) fn output_key(&self, c: &CompInst) -> Result<(usize, u64)> {
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

pub(super) fn instantiate(
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

    // leaf launch grids: supertile shape divided by each leaf's @autotune default.
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

pub(super) struct CompInst {
    pub(super) leaf: usize,
    pub(super) args: Vec<(usize, Vec<u64>, AccessMode)>,
    pub(super) scalars: Vec<usize>,
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
