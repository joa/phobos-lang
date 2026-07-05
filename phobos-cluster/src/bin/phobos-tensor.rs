use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{Context, Result, bail};
use phobos_base::phinfo;
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    match args[0].as_str() {
        "init" => init(&args[1..]),
        "peek" => peek(&args[1..]),
        other => bail!("unknown subcommand '{other}'\n{USAGE}"),
    }
}

fn init(args: &[String]) -> Result<()> {
    let uri = arg(args, "--uri")?.context("missing --uri")?;
    let shape = parse_shape(&arg(args, "--shape")?.context("missing --shape")?)?;
    let fill = arg(args, "--fill")?.unwrap_or_else(|| "zero".to_string());
    let value: f32 = match arg(args, "--value")? {
        Some(v) => v.parse().context("--value must be a float")?,
        None => 0.0,
    };
    let seed: u64 = match arg(args, "--seed")? {
        Some(v) => v.parse().context("--seed must be an integer")?,
        None => 1,
    };

    let total: u64 = shape.iter().product();
    let path = storage::file_path(&uri)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut w = BufWriter::new(File::create(&path).with_context(|| format!("creating {path:?}"))?);

    let mut lcg = seed;
    let mut idx: u64 = 0;
    let mut remaining = total;
    let mut buf = vec![0f32; CHUNK.min(total.max(1) as usize)];
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        for slot in buf[..n].iter_mut() {
            *slot = match fill.as_str() {
                "zero" => 0.0,
                "const" => value,
                "iota" => idx as f32,
                "random" => {
                    lcg = lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((lcg >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                }
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

fn peek(args: &[String]) -> Result<()> {
    let uri = arg(args, "--uri")?.context("missing --uri")?;
    let shape = parse_shape(&arg(args, "--shape")?.context("missing --shape")?)?;

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
        let v = storage::load_f32(&uri, &shape, &region)?;
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

fn parse_shape(s: &str) -> Result<Vec<u64>> {
    s.split('x')
        .map(|d| {
            d.trim()
                .parse::<u64>()
                .with_context(|| format!("bad shape extent '{d}'"))
        })
        .collect()
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
