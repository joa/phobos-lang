use anyhow::{Context, Result, bail};
use phobos_base::phinfo;
use phobos_sched::IngestPolicy;
use phobos_sched::autotune::ClusterFingerprint;
use phobos_sched::job::parse_job;
use phobos_sched::server::{DispatchConfig, Scheduler};

const USAGE: &str = "\
usage: phobos-sched --listen <host:port> --nodes <n> --job <file> [--budget <bytes>]
                    [--autotune [--vram <bytes>] [--link-bw <bytes/s>] [--leaf-flops <flop/s>]]
  --listen      Attach/Submit bind address (host:port)
  --nodes       number of nodes to wait for before dispatching
  --job         path to the job-file (see the binary docs)
  --budget      per-node memory budget in bytes (enables segmentation)
  --ingest      input ingest policy: 'direct' (every node LOADs from the URI;
                default, optimal for shared storage) or 'home-fetch' (one node
                LOADs, peers FETCH it; for inputs on a single node's local disk)
  --autotune    pick SUPER_* via the cluster autotuner instead of @cluster defaults
  the cluster autotuner's fingerprint (only used with --autotune; defaults shown):
  --vram        per-node VRAM the feasibility prune fits the working set into
                (bytes, default 16 GiB); set this to match the nodes' --arena
  --link-bw     measured peer link bandwidth (bytes/s, default 10e9)
  --leaf-flops  device-leaf throughput (FLOP/s, default 10e12)";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }

    let listen = arg(&args, "--listen")?.unwrap_or_else(|| "0.0.0.0:7000".to_string());
    let nodes: u16 = arg(&args, "--nodes")?
        .context("missing --nodes")?
        .parse()
        .context("--nodes must be a u16")?;
    let job_path = arg(&args, "--job")?.context("missing --job")?;
    let budget_bytes = match arg(&args, "--budget")? {
        Some(v) => Some(v.parse::<u64>().context("--budget must be a byte count")?),
        None => None,
    };
    let autotune = args.iter().any(|a| a == "--autotune");
    let ingest = match arg(&args, "--ingest")?.as_deref() {
        None | Some("direct") => IngestPolicy::DirectLoad,
        Some("home-fetch") => IngestPolicy::HomeLoadPeerFetch,
        Some(other) => bail!("unknown --ingest '{other}' (direct|home-fetch)"),
    };

    let mut fp = ClusterFingerprint::default();
    if let Some(v) = arg(&args, "--vram")? {
        fp.vram_bytes = v.parse().context("--vram must be a byte count")?;
    }
    if let Some(v) = arg(&args, "--link-bw")? {
        fp.link_bytes_per_sec = v.parse().context("--link-bw must be a number")?;
    }
    if let Some(v) = arg(&args, "--leaf-flops")? {
        fp.leaf_flops_per_sec = v.parse().context("--leaf-flops must be a number")?;
    }

    let job = parse_job(&job_path)?;

    let sched = Scheduler::new();
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .with_context(|| format!("binding scheduler to {listen}"))?;
    let bound = listener.local_addr()?;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let server = sched.clone().into_server();
    tokio::spawn(async move {
        if let Err(e) = tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_incoming(incoming)
            .await
        {
            eprintln!("scheduler server error: {e:#}");
        }
    });

    eprintln!("phobos-sched: listening on {bound}, waiting for {nodes} node(s)");
    let cfg = DispatchConfig {
        nodes,
        budget_bytes,
        autotune: autotune.then_some(fp),
        ingest,
        ..Default::default()
    };
    let outputs = sched.dispatch(job, cfg).await?;
    phinfo!("job done; outputs:");
    for uri in &outputs {
        phinfo!("  {uri}");
    }
    Ok(())
}

fn arg(args: &[String], flag: &str) -> Result<Option<String>> {
    match args.iter().position(|a| a == flag) {
        Some(i) => match args.get(i + 1) {
            Some(v) => Ok(Some(v.clone())),
            None => bail!("{flag} expects a value\n{USAGE}"),
        },
        None => Ok(None),
    }
}
