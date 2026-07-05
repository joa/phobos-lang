use std::collections::HashMap;
use std::fmt::Write as _;

use phobos_cluster::ir::ClusterProgram;
use phobos_cluster::isa::{InstrId, Op};
use phobos_cluster::tile::{AccessMode, TileId};

use crate::Plan;

pub fn plan_dot(program: &ClusterProgram, plan: &Plan) -> String {
    let mut producer: HashMap<(usize, TileId), InstrId> = HashMap::new();

    for (node, segs) in plan.node_segments.iter().enumerate() {
        for instr in segs.iter().flat_map(|s| s.instructions.iter()) {
            match &instr.op {
                Op::Load { tile, .. } => {
                    producer.insert((node, *tile), instr.iid);
                }
                Op::Compute { args, .. } => {
                    for (t, m) in args {
                        if matches!(m, AccessMode::Write | AccessMode::RMW) {
                            producer.insert((node, *t), instr.iid);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut s = String::new();
    writeln!(s, "digraph \"{}\" {{", program.name).unwrap();
    writeln!(s, "  rankdir=TB;").unwrap();
    writeln!(s, "  labelloc=t;").unwrap();
    writeln!(
        s,
        "  label=\"plan {} ({} node(s))\";",
        program.name,
        plan.node_segments.len()
    )
    .unwrap();
    writeln!(
        s,
        "  node [shape=box, style=filled, fontname=\"monospace\", fontsize=10];"
    )
    .unwrap();
    writeln!(s, "  edge [fontname=\"monospace\", fontsize=9];").unwrap();

    let mut fetch_edges: Vec<(InstrId, InstrId, String)> = Vec::new();

    for (node, segs) in plan.node_segments.iter().enumerate() {
        let segn = segs.len();
        writeln!(s, "  subgraph cluster_n{node} {{").unwrap();
        writeln!(s, "    style=rounded; color=gray40;").unwrap();
        writeln!(s, "    label=\"node {node} ({segn} segment(s))\";").unwrap();
        for instr in segs.iter().flat_map(|s| s.instructions.iter()) {
            let (label, fill) = op_node(&instr.op, program);
            writeln!(
                s,
                "    i{} [label=\"#{} {}\", fillcolor={fill}];",
                instr.iid, instr.iid, label
            )
            .unwrap();
            if let Op::Fetch { tile, from } = &instr.op
                && let Some(&p) = producer.get(&(*from as usize, *tile))
            {
                fetch_edges.push((p, instr.iid, tile_label(program, *tile)));
            }
        }
        writeln!(s, "  }}").unwrap();
    }

    for instr in plan
        .node_segments
        .iter()
        .flat_map(|segs| segs.iter())
        .flat_map(|seg| seg.instructions.iter())
    {
        for d in &instr.deps {
            writeln!(s, "  i{d} -> i{};", instr.iid).unwrap();
        }
    }

    for (from, to, label) in fetch_edges {
        writeln!(
            s,
            "  i{from} -> i{to} [style=dashed, color=red, constraint=false, label=\"{label}\"];"
        )
        .unwrap();
    }

    writeln!(s, "}}").unwrap();
    s
}

fn op_node(op: &Op, program: &ClusterProgram) -> (String, &'static str) {
    match op {
        Op::Alloc { tile, .. } => (format!("ALLOC {}", tile_label(program, *tile)), "gray90"),
        Op::Load { tile, .. } => (format!("LOAD {}", tile_label(program, *tile)), "lightblue"),
        Op::Fetch { tile, from } => (
            format!("FETCH {} <-n{from}", tile_label(program, *tile)),
            "lightcyan",
        ),
        Op::Compute {
            kernel,
            args,
            scalars,
            ..
        } => {
            let out = args
                .iter()
                .find(|(_, m)| matches!(m, AccessMode::Write | AccessMode::RMW))
                .map(|(t, _)| tile_label(program, *t))
                .unwrap_or_default();
            let sc = if scalars.is_empty() {
                String::new()
            } else {
                format!(" +{}sc", scalars.len())
            };
            (format!("COMPUTE k{kernel}{sc}\\n-> {out}"), "palegreen")
        }
        Op::Store { tile, .. } => (
            format!("STORE {}", tile_label(program, *tile)),
            "navajowhite",
        ),
        Op::Free { tile, .. } => (format!("FREE {}", tile_label(program, *tile)), "mistyrose"),
    }
}

fn tile_label(program: &ClusterProgram, tile: TileId) -> String {
    let name = program
        .tensors
        .get(tile.tensor() as usize)
        .map(|t| t.name.as_str())
        .unwrap_or("?");
    if tile.version() == 0 {
        format!("{name}#{}", tile.coord())
    } else {
        format!("{name}#{}v{}", tile.coord(), tile.version())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::plan_dot;
    use crate::{IngestPolicy, default_supers, plan, plan_with};

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

    #[test]
    fn renders_instruction_dag() {
        let kernel = phobos_lang::parse(MATMUL).unwrap().remove(0);
        let program = phobos_cluster::compile(&kernel).unwrap();
        let supers = default_supers(&program);
        let dims: HashMap<String, i64> = [("M", 8192), ("N", 8192), ("K", 8192)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let pl = plan(&program, &dims, &supers, 2).unwrap();
        let dot = plan_dot(&program, &pl);

        assert!(dot.starts_with("digraph \"matmul\""));
        // one subgraph per node
        assert!(dot.contains("subgraph cluster_n0"));
        assert!(dot.contains("subgraph cluster_n1"));
        // the real op vocabulary shows up, tile-labelled with tensor names
        assert!(dot.contains("COMPUTE k"));
        assert!(dot.contains("LOAD A#"));
        assert!(dot.contains("STORE C#"));
        // dependency edges are present (the whole point)
        assert!(dot.contains(" -> i"), "no dependency edges in:\n{dot}");
        // DirectLoad: no peer FETCH, so no dashed cross-node edges
        assert!(!dot.contains("style=dashed"));
    }

    #[test]
    fn renders_cross_node_fetch_edges() {
        // Under HomeLoadPeerFetch the home LOADs an input and peers FETCH it;
        // the DAG should carry a FETCH node and a dashed producer->fetch edge.
        let kernel = phobos_lang::parse(MATMUL).unwrap().remove(0);
        let program = phobos_cluster::compile(&kernel).unwrap();
        let supers = default_supers(&program);
        let dims: HashMap<String, i64> = [("M", 8192), ("N", 8192), ("K", 8192)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let pl = plan_with(&program, &dims, &supers, 2, IngestPolicy::HomeLoadPeerFetch).unwrap();
        let dot = plan_dot(&program, &pl);
        assert!(dot.contains("FETCH A#"), "no FETCH node in:\n{dot}");
        assert!(
            dot.contains("style=dashed"),
            "no cross-node fetch edge in:\n{dot}"
        );
    }
}
