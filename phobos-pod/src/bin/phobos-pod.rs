use anyhow::Result;
use phobos_base::cli::Args;

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
    let args = Args::from_env();
    if args.wants_help() {
        println!("{USAGE}");
        return Ok(());
    }

    let id: u16 = args.parse_required("--id")?;
    let sched = args.required("--sched")?.to_string();
    let listen = args.value("--listen")?.unwrap_or("0.0.0.0:0").to_string();
    let advertise = args.value("--advertise")?.map(str::to_string);
    let arena = args
        .parse("--arena")?
        .unwrap_or(phobos_pod::DEFAULT_ARENA_BYTES);

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
