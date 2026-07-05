use anyhow::{Context, Result, bail};

const USAGE: &str = "\
usage: phobos-pod --id <node-id> --sched <host:port> [--listen <host:port>] [--advertise <host:port>] [--arena <bytes>]
  --id        this node's id (0-based, must be unique in the cluster)
  --sched     the scheduler's Attach endpoint (host:port)
  --listen    this node's TileServer bind address (default 0.0.0.0:0)
  --advertise the address peers FETCH from (default: the bound address, with a
              wildcard bind IP rewritten to loopback). Set this to the node's
              routable IP for a multi-host cluster bound to 0.0.0.0.
  --arena     device arena bytes (default 512 MiB); accepts a plain integer";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }

    let id: u16 = arg(&args, "--id")?
        .context("missing --id")?
        .parse()
        .context("--id must be a u16")?;
    let sched = arg(&args, "--sched")?.context("missing --sched")?;
    let listen = arg(&args, "--listen")?.unwrap_or_else(|| "0.0.0.0:0".to_string());
    let advertise = arg(&args, "--advertise")?;
    let arena = match arg(&args, "--arena")? {
        Some(v) => v.parse::<usize>().context("--arena must be a byte count")?,
        None => phobos_pod::DEFAULT_ARENA_BYTES,
    };

    eprintln!(
        "phobos-pod {id}: arena {} MiB, listen {listen}{}, scheduler {sched}",
        arena >> 20,
        advertise
            .as_deref()
            .map(|a| format!(", advertise {a}"))
            .unwrap_or_default(),
    );
    phobos_pod::serve(id, sched, listen, advertise, arena).await
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
