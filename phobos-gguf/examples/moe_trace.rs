// Record what a mixture-of-experts model's routers choose, token by token,
// over a set of prompts and their greedy continuations:
//
//   cargo run --release -p phobos-gguf --example moe_trace -- \
//       MODEL.gguf -n 64 -o trace.jsonl "Explain how a hash map works." ...
//   cargo run --release -p phobos-gguf --example moe_trace -- \
//       MODEL.gguf -n 64 -o trace.jsonl --prompts prompts.txt
//
// One JSON line a position: the prompt it belongs to, whether it was part of
// the prompt or generated, the token, every block's chosen experts, and
// every block's one-block lookahead (see `RouteTrace`). `scripts/moe_sim.py`
// replays the file against cache policies. Runs on whatever backend the
// build has; on the host reference a 35B token is a second or two, so a
// trace of a few thousand tokens is an hour or so.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use phobos_gguf::backend::Backend;
use phobos_gguf::qwen35::RouteTrace;
use phobos_gguf::{Bpe, Decoder, Gguf};

fn make_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(phobos_gguf::backend::device::DeviceBackend::new()?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(phobos_gguf::backend::HostBackend::new()))
    }
}

fn main() -> Result<()> {
    let (mut path, mut out, mut prompts) = (None::<PathBuf>, None::<PathBuf>, Vec::new());
    let mut generate = 64usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-n" => generate = args.next().unwrap_or_default().parse()?,
            "-o" => out = Some(args.next().unwrap_or_default().into()),
            "--prompts" => {
                let file = args.next().unwrap_or_default();
                let text = std::fs::read_to_string(&file).with_context(|| format!("read {file}"))?;
                prompts.extend(text.lines().filter(|l| !l.trim().is_empty()).map(|l| l.replace("\\n", "\n")));
            }
            other if path.is_none() => path = Some(other.into()),
            other => prompts.push(other.to_string()),
        }
    }
    let (Some(path), Some(out)) = (path, out) else {
        bail!("usage: moe_trace [-n GENERATE] -o TRACE.jsonl [--prompts FILE] MODEL.gguf PROMPT...");
    };
    if prompts.is_empty() {
        bail!("no prompts given");
    }

    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let backend = make_backend()?;
    let backend = backend.as_ref();
    let mut sink = BufWriter::new(std::fs::File::create(&out).with_context(|| format!("create {}", out.display()))?);

    let started = Instant::now();
    let mut positions = 0usize;
    for (p, prompt) in prompts.iter().enumerate() {
        let ids = bpe.encode(prompt)?;
        let mut state = model.new_state();
        let (logits, trace) = model.forward_traced(&mut state, &ids, backend)?;
        for (t, &id) in ids.iter().enumerate() {
            write_row(&mut sink, p, "prompt", t, id, &trace, ids.len(), t)?;
        }
        positions += ids.len();
        let mut next = argmax(&logits);
        let mut pos = ids.len();
        for _ in 0..generate {
            if bpe.is_eog(next) {
                break;
            }
            let (logits, trace) = model.forward_traced(&mut state, &[next], backend)?;
            write_row(&mut sink, p, "decode", pos, next, &trace, 1, 0)?;
            positions += 1;
            pos += 1;
            next = argmax(&logits);
        }
        state.release(backend);
        sink.flush()?;
        eprintln!(
            "prompt {p}: {} prompt + {} generated positions, {positions} in all, {:.1} s/position so far",
            ids.len(),
            pos - ids.len(),
            started.elapsed().as_secs_f64() / positions as f64
        );
    }
    Ok(())
}

/// One position of a trace: `row` of a pass over `rows` positions.
#[allow(clippy::too_many_arguments)]
fn write_row(
    sink: &mut impl Write,
    prompt: usize,
    phase: &str,
    pos: usize,
    token: u32,
    trace: &RouteTrace,
    rows: usize,
    row: usize,
) -> Result<()> {
    let n = trace.n_used;
    let blocks = trace.routes.len() / (rows * n);
    let pick = |flat: &[u32], b: usize| -> String {
        let at = (b * rows + row) * n;
        let ids: Vec<String> = flat[at..at + n].iter().map(u32::to_string).collect();
        format!("[{}]", ids.join(","))
    };
    let routes: Vec<String> = (0..blocks).map(|b| pick(&trace.routes, b)).collect();
    let lookahead: Vec<String> = (0..blocks).map(|b| pick(&trace.lookahead, b)).collect();
    writeln!(
        sink,
        "{{\"prompt\":{prompt},\"phase\":\"{phase}\",\"pos\":{pos},\"token\":{token},\"routes\":[{}],\"lookahead\":[{}]}}",
        routes.join(","),
        lookahead.join(",")
    )?;
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
