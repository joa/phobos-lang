use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{Context, Result, bail};
use phobos_base::cli::Args;
use phobos_base::phinfo;
use phobos_base::rng::Lcg;
use phobos_base::shape;
use phobos_cluster::isa::Region;
use phobos_cluster::storage;

const USAGE: &str = "\
usage:
  phobos-tensor init --uri <file://...> --shape <RxC|N> [--fill <mode>] [--value <f>] [--seed <s>]
  phobos-tensor peek --uri <file://...> --shape <RxC|N>

init fills:
  zero            all zeros (default; use for output tensors to pre-allocate)
  random          reproducible LCG in [-1, 1) (matches the example harnesses; --seed, default 1)
  const           every element = --value (default 0)
  iota            0, 1, 2, ... row-major (debugging)

shapes are row-major `RxC` (rank-2) or `N` (rank-1), as in the --job file.";

const CHUNK: usize = 1 << 20;

fn main() -> Result<()> {
    let args = Args::from_env();
    if args.is_empty() || args.wants_help() {
        println!("{USAGE}");
        return Ok(());
    }
    match args.subcommand() {
        Some(("init", rest)) => init(&rest),
        Some(("peek", rest)) => peek(&rest),
        Some((other, _)) => bail!("unknown subcommand '{other}'\n{USAGE}"),
        None => unreachable!("non-empty args always carry a subcommand"),
    }
}

fn init(args: &Args) -> Result<()> {
    let uri = args.required("--uri")?;
    let shape = shape::parse(args.required("--shape")?)?;
    let fill = args.value("--fill")?.unwrap_or("zero");
    let value = args.parse::<f32>("--value")?.unwrap_or(0.0);
    let seed = args.parse::<u64>("--seed")?.unwrap_or(1);

    let total: u64 = shape.iter().product();
    let path = storage::file_path(uri)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut w = BufWriter::new(File::create(&path).with_context(|| format!("creating {path:?}"))?);

    let mut lcg = Lcg::new(seed);
    let mut idx: u64 = 0;
    let mut remaining = total;
    let mut buf = vec![0f32; CHUNK.min(total.max(1) as usize)];
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        for slot in buf[..n].iter_mut() {
            *slot = match fill {
                "zero" => 0.0,
                "const" => value,
                "iota" => idx as f32,
                "random" => lcg.next_unit_f32(),
                other => bail!("unknown --fill '{other}'\n{USAGE}"),
            };
            idx += 1;
        }
        w.write_all(&storage::f32_to_le_bytes(&buf[..n]))?;
        remaining -= n as u64;
    }
    w.flush()?;
    phinfo!(
        "wrote {} ({} elements, {} bytes, fill={fill}) to {}",
        uri,
        total,
        total * 4,
        path.display()
    );
    Ok(())
}

fn peek(args: &Args) -> Result<()> {
    let uri = args.required("--uri")?;
    let shape = shape::parse(args.required("--shape")?)?;

    let coords: Vec<Vec<u64>> = match shape.as_slice() {
        [n] => {
            let n = *n;
            [0, n / 2, n.saturating_sub(1)]
                .iter()
                .map(|&i| vec![i])
                .collect()
        }
        [r, c] => {
            let (r, c) = (*r, *c);
            [
                (0, 0),
                (0, c - 1),
                (r / 2, c / 2),
                (r - 1, 0),
                (r - 1, c - 1),
            ]
            .iter()
            .map(|&(i, j)| vec![i, j])
            .collect()
        }
        _ => bail!("peek supports rank <= 2"),
    };

    println!("{uri} {shape:?}:");
    for at in coords {
        let region = Region {
            offset: at.clone(),
            shape: vec![1; at.len()],
        };
        let v = storage::load_f32(uri, &shape, &region)?;
        println!(
            "  [{}] = {}",
            at.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            v[0]
        );
    }
    Ok(())
}

