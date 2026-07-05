use std::collections::HashMap;
use std::fmt::Write as _;

use anyhow::{Context, Result, anyhow};
use phobos_cluster::ir::{ClusterProgram, ClusterStmt, Coord, SuperTile};
use phobos_cluster::tile::AccessMode;
use phobos_lang::ast::Dim;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: dag_dot <file.ph> [out.dot]");
        std::process::exit(2);
    };
    let out = args.next();

    let src = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
    let kernels = phobos_lang::parse(&src)?;
    let kernel = kernels
        .first()
        .ok_or_else(|| anyhow!("{path} has no kernel"))?;
    let prog = phobos_cluster::compile(kernel)?;

    let dot = Dot::render(&prog);
    match out {
        Some(p) => {
            std::fs::write(&p, dot).with_context(|| format!("writing {p}"))?;
            eprintln!("wrote {p}");
        }
        None => print!("{dot}"),
    }
    Ok(())
}

struct Dot<'a> {
    prog: &'a ClusterProgram,
    s: String,
    compute: usize,
    loops: usize,
    writer: HashMap<(usize, String), String>,
}

impl<'a> Dot<'a> {
    fn render(prog: &'a ClusterProgram) -> String {
        let mut d = Dot {
            prog,
            s: String::new(),
            compute: 0,
            loops: 0,
            writer: HashMap::new(),
        };

        let grid = prog
            .grid
            .iter()
            .map(|ax| format!("p{}={}/{}", ax.pid, dim_str(&ax.dim), ax.super_sym))
            .collect::<Vec<_>>()
            .join(", ");

        writeln!(d.s, "digraph \"{}\" {{", prog.name).unwrap();
        writeln!(d.s, "  rankdir=LR;").unwrap();
        writeln!(d.s, "  labelloc=t;").unwrap();
        writeln!(d.s, "  label=\"cluster {}\\ngrid: {grid}\";", prog.name).unwrap();
        writeln!(d.s, "  node [fontname=\"monospace\", fontsize=10];").unwrap();
        writeln!(d.s, "  edge [fontname=\"monospace\", fontsize=9];").unwrap();

        for (i, t) in prog.tensors.iter().enumerate() {
            let fill = match t.mode {
                AccessMode::Read => "lightblue",
                AccessMode::Write => "lightcoral",
                AccessMode::RMW => "khaki",
            };
            writeln!(
                d.s,
                "  t{i} [shape=box, style=\"rounded,filled\", fillcolor={fill}, label=\"{} : {}\"];",
                t.name,
                mode_str(t.mode),
            )
            .unwrap();
        }

        d.walk(&prog.body, 0);
        d.sink_outputs();

        writeln!(d.s, "}}").unwrap();
        d.s
    }

    fn walk(&mut self, body: &[ClusterStmt], depth: usize) {
        for stmt in body {
            match stmt {
                ClusterStmt::Compute { leaf, args, .. } => self.compute(*leaf, args, depth),
                ClusterStmt::Loop {
                    var,
                    dim,
                    super_sym,
                    body,
                } => {
                    let id = self.loops;
                    self.loops += 1;
                    writeln!(self.s, "  subgraph cluster_l{id} {{").unwrap();
                    writeln!(self.s, "    style=dashed; color=gray50;").unwrap();
                    writeln!(
                        self.s,
                        "    label=\"for {var} in {}/{super_sym}\";",
                        dim_str(dim)
                    )
                    .unwrap();
                    self.walk(body, depth + 1);
                    writeln!(self.s, "  }}").unwrap();
                }
            }
        }
    }

    fn compute(&mut self, leaf: usize, args: &[(SuperTile, AccessMode)], depth: usize) {
        let node = format!("c{}", self.compute);
        self.compute += 1;
        let name = &self.prog.leaves[leaf].kernel.name;
        writeln!(
            self.s,
            "  {node} [shape=box, style=filled, fillcolor=white, label=\"{name}\"];"
        )
        .unwrap();

        for (r, mode) in args {
            let key = (r.tensor, coords_str(r));
            let label = self.ref_label(r);
            if matches!(mode, AccessMode::Read | AccessMode::RMW) {
                let src = self
                    .writer
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| format!("t{}", r.tensor));
                writeln!(self.s, "  {src} -> {node} [label=\"{label}\"];").unwrap();
                if *mode == AccessMode::RMW && depth > 0 {
                    writeln!(
                        self.s,
                        "  {node} -> {node} [label=\"carry {label}\", style=dashed, color=gray50];"
                    )
                    .unwrap();
                }
            }
            if matches!(mode, AccessMode::Write | AccessMode::RMW) {
                self.writer.insert(key, node.clone());
            }
        }
    }

    fn sink_outputs(&mut self) {
        let mut edges: Vec<(usize, String, String)> = self
            .writer
            .iter()
            .filter(|((t, _), _)| {
                matches!(
                    self.prog.tensors[*t].mode,
                    AccessMode::Write | AccessMode::RMW
                )
            })
            .map(|((t, coords), node)| (*t, coords.clone(), node.clone()))
            .collect();
        edges.sort();
        for (t, coords, node) in edges {
            let name = &self.prog.tensors[t].name;
            writeln!(self.s, "  {node} -> t{t} [label=\"{name}({coords})\"];").unwrap();
        }
    }

    fn ref_label(&self, r: &SuperTile) -> String {
        format!("{}({})", self.prog.tensors[r.tensor].name, coords_str(r))
    }
}

fn coords_str(r: &SuperTile) -> String {
    r.coords
        .iter()
        .map(|c| match c {
            Coord::Grid(pid) => format!("p{pid}"),
            Coord::Loop(v) => v.clone(),
            Coord::Full => ":".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn dim_str(d: &Dim) -> String {
    match d {
        Dim::Sym(s) => s.clone(),
        Dim::Int(n) => n.to_string(),
    }
}

fn mode_str(m: AccessMode) -> &'static str {
    match m {
        AccessMode::Read => "read",
        AccessMode::Write => "write",
        AccessMode::RMW => "rmw",
    }
}
