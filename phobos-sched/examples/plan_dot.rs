use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use phobos_cluster::proto;
use phobos_cluster::tile::ScalarValue;
use phobos_sched::dot::plan_dot;
use phobos_sched::job::parse_job;
use phobos_sched::{IngestPolicy, default_supers, plan_budgeted_with};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: plan_dot <job.txt> [--nodes N] [--ingest direct|home-fetch] [out.dot]";
    let mut positional = Vec::new();
    let mut nodes: u16 = 2;
    let mut ingest = IngestPolicy::DirectLoad;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--nodes" => {
                nodes = args
                    .get(i + 1)
                    .context("--nodes expects a value")?
                    .parse()
                    .context("--nodes must be a u16")?;
                i += 2;
            }
            "--ingest" => {
                ingest = match args.get(i + 1).map(String::as_str) {
                    Some("direct") => IngestPolicy::DirectLoad,
                    Some("home-fetch") => IngestPolicy::HomeLoadPeerFetch,
                    other => bail!("--ingest expects direct|home-fetch, got {other:?}"),
                };
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("{usage}");
                return Ok(());
            }
            _ => {
                positional.push(args[i].clone());
                i += 1;
            }
        }
    }
    let Some(job_path) = positional.first() else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let out = positional.get(1).cloned();

    let job = parse_job(job_path)?;
    let kernel = phobos_lang::parse(&job.source)?
        .into_iter()
        .next()
        .context("source has no kernel")?;
    let program = phobos_cluster::compile(&kernel)?;

    let dims: HashMap<String, i64> = job
        .dimensions
        .iter()
        .map(|d| (d.name.clone(), d.value))
        .collect();
    let scalars: HashMap<String, ScalarValue> = job
        .scalars
        .iter()
        .map(|s| {
            let v = proto::scalar_value_from_proto(
                s.value.as_ref().context("scalar binding missing value")?,
            )?;
            Ok((s.name.clone(), v))
        })
        .collect::<Result<_>>()?;
    let supers = default_supers(&program);

    if nodes == 0 {
        bail!("--nodes must be at least 1");
    }
    let plan = plan_budgeted_with(&program, &dims, &supers, nodes, u64::MAX, ingest, &scalars)?;

    let dot = plan_dot(&program, &plan);
    match out {
        Some(p) => {
            std::fs::write(&p, dot).with_context(|| format!("writing {p}"))?;
            eprintln!("wrote {p}");
        }
        None => print!("{dot}"),
    }
    Ok(())
}
