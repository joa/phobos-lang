// Prompt passes through a `GgufModel` session, as the generation loop feeds
// them, with the prompt batch however the environment sets it:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example batch_tune_check -- MODEL.gguf [-p 2048,8192] [-r 2] [-n 32]
//
// Each prompt runs into a fresh session, its prefill timed, then `-n` greedy
// steps timed as decode. Run it once with `PHOBOS_PROMPT_BATCH` pinned and
// once tuned, in alternation, to compare the two; with `PHOBOS_LOG=debug` the
// tuner prints its steps. Token ids are pseudorandom, a different sequence
// per prompt, as phobos-bench feeds them.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use phobos_gguf::runtime::GgufModel;
use phobos_inference::Model;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut path, mut lengths, mut reps, mut steps) = (None, vec![2048, 8192], 2, 32);
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-p" => lengths = value()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "-r" => reps = value()?.parse()?,
            "-n" => steps = value()?.parse()?,
            _ if path.is_none() => path = Some(arg),
            _ => bail!("unexpected argument {arg:?}"),
        }
    }
    let Some(path) = path else {
        bail!("usage: batch_tune_check MODEL.gguf [-p LEN,...] [-r REPS] [-n STEPS]");
    };
    let pinned = std::env::var("PHOBOS_PROMPT_BATCH").ok();

    let model = GgufModel::load(path.as_ref())?;
    model.warm_up()?;
    let vocab = model.info().vocab_size;
    let arm = pinned.map_or("tuned".to_string(), |rows| format!("fixed {rows}"));
    println!("arm\tlen\trep\tprefill_s\tpp_tps\ttg_tps");
    let (mut prefill_secs, mut decode_secs) = (0.0, 0.0);
    let mut stream = 0;
    for rep in 0..reps {
        for &len in &lengths {
            stream += 1;
            let tokens = synthetic_tokens(len + steps, vocab, stream);
            let mut session = model.session()?;
            let batch = session.prompt_batch().unwrap_or(len);
            let started = Instant::now();
            for chunk in tokens[..len].chunks(batch) {
                session.extend(chunk)?;
            }
            let prefill = started.elapsed().as_secs_f64();
            let started = Instant::now();
            for &token in &tokens[len..] {
                session.extend_greedy(&[token])?;
            }
            let decode = started.elapsed().as_secs_f64();
            prefill_secs += prefill;
            decode_secs += decode;
            println!(
                "{arm}\t{len}\t{rep}\t{prefill:.3}\t{:.1}\t{:.1}",
                len as f64 / prefill,
                steps as f64 / decode
            );
        }
    }
    let prompt_tokens: usize = lengths.iter().sum::<usize>() * reps;
    println!(
        "total\t{arm}\tprefill {prefill_secs:.2} s ({:.1} t/s)\tdecode {:.1} t/s",
        prompt_tokens as f64 / prefill_secs,
        (steps * lengths.len() * reps) as f64 / decode_secs
    );
    Ok(())
}

/// A deterministic spread of valid token ids, a different one per `stream`.
fn synthetic_tokens(count: usize, vocab: usize, stream: u64) -> Vec<i64> {
    let mut seed = 0x2545_f491_4f6c_dd1du64 ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % vocab as u64) as i64
        })
        .collect()
}
