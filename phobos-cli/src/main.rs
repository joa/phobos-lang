// Run text inference on the Phobos runtimes:
//
//   phobos-cli --gguf MODEL.gguf "The color of the sky is"
//   phobos-cli --onnx MODEL_DIR "The color of the sky is"
//
// Encodes the prompt, runs it through either a GGUF model or an ONNX export,
// and streams the continuation. With no prompt it drops into a REPL that keeps
// the loaded model warm.
//

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use phobos_gguf::GgufModel;
use phobos_inference::Model;
use phobos_inference::generate::{self, Flow};
use phobos_inference::sampling::{Rng, SampleConfig, Sequence};
use phobos_inference::server;
use phobos_onnx::OnnxModel;

struct Args {
    gguf: Option<PathBuf>,
    onnx: Option<PathBuf>,
    num_tokens: usize,
    show: usize,
    sample: SampleConfig,
    seed: u64,
    listen: Option<String>,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut gguf: Option<PathBuf> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut num_tokens = 200usize;
    let mut show = 5usize;
    let mut sample = SampleConfig::greedy();
    let mut seed = 0u64;
    let mut listen: Option<String> = None;
    let mut words: Vec<String> = Vec::new();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--gguf" => gguf = Some(next("--gguf")?.into()),
            "--onnx" => onnx = Some(next("--onnx")?.into()),
            "-n" | "--num" => num_tokens = next("-n")?.parse().context("-n count")?,
            "--show" => show = next("--show")?.parse().context("--show count")?,
            "-t" | "--temp" => sample.temperature = next("--temp")?.parse().context("--temp")?,
            "-k" | "--top-k" => sample.top_k = next("--top-k")?.parse().context("--top-k")?,
            "-p" | "--top-p" => sample.top_p = next("--top-p")?.parse().context("--top-p")?,
            "--min-p" => sample.min_p = next("--min-p")?.parse().context("--min-p")?,
            "--presence-penalty" => {
                sample.presence_penalty = next("--presence-penalty")?
                    .parse()
                    .context("--presence-penalty")?
            }
            "--repetition-penalty" => {
                sample.repetition_penalty = next("--repetition-penalty")?
                    .parse()
                    .context("--repetition-penalty")?
            }
            "--seed" => seed = next("--seed")?.parse().context("--seed")?,
            "--listen" => listen = Some(next("--listen")?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => words.push(other.to_string()),
        }
    }

    let prompt = (!words.is_empty()).then(|| words.join(" "));
    Ok(Args {
        gguf,
        onnx,
        num_tokens: num_tokens.max(1),
        show,
        sample,
        seed,
        listen,
        prompt,
    })
}

fn load(args: &Args) -> Result<Box<dyn Model>> {
    match (&args.gguf, &args.onnx) {
        (Some(path), None) => Ok(Box::new(GgufModel::load(path)?)),
        (None, Some(dir)) => Ok(Box::new(OnnxModel::load(dir)?)),
        (None, None) => bail!("no model: pass --gguf FILE or --onnx DIR"),
        (Some(_), Some(_)) => bail!("--gguf and --onnx are alternatives, not both"),
    }
}

fn backend_name(args: &Args) -> &'static str {
    match args.gguf {
        Some(_) => phobos_gguf::runtime::backend_name(),
        None => phobos_onnx::runtime::backend_name(),
    }
}

fn print_usage() {
    eprintln!(
        "\
usage: phobos-cli (--gguf FILE | --onnx DIR) [OPTIONS] [PROMPT]

With a PROMPT: print a continuation. Without one: start a REPL.

One of --gguf or --onnx is required; there is no default model.

OPTIONS:
      --gguf FILE     run a GGUF model, dispatching on the architecture the
                      file declares, with the tokenizer the file carries
      --onnx DIR      run an ONNX export from DIR. A decoder.onnx beside a
                      decoder_with_past.onnx is the KV-cached engine; a
                      model.onnx is the full-recompute one
  -n, --num N         generate up to N tokens (default: 200; stops early on an
                      end-of-turn token or the model's context limit)
  -t, --temp T        sampling temperature; 0 = greedy/argmax (default: 0)
  -k, --top-k K       sample only from the K highest-logit tokens (default: 0 = off)
  -p, --top-p P       nucleus sampling threshold (default: 1.0 = off)
      --min-p P       keep tokens at least P as likely as the best one
                      (default: 0.0 = off)
      --presence-penalty P
                      subtract P from the logit of every token generated so far,
                      the prompt excluded (default: 0.0 = off)
      --repetition-penalty P
                      scale the logit of every token already in the sequence,
                      prompt included, towards zero by P (default: 1.0 = off)
      --seed S        PRNG seed for sampling (default: 0)
      --show K        show the top-K candidates for the first token (default: 5)
      --listen ADDR   run an OpenAI compatible HTTP server on ADDR (e.g.
                      127.0.0.1:8080). The sampling options above become what a
                      request falls back to for every field it does not send.
  -h, --help          print this message"
    );
}

fn main() -> Result<()> {
    let args = parse_args()?;

    eprint!("loading model on {}... ", backend_name(&args));
    io::stderr().flush().ok();
    let model = load(&args)?;
    let info = model.info();
    eprintln!("ready ({} engine, {} vocab).", info.label, info.vocab_size);

    if let Some(addr) = args.listen.clone() {
        let defaults = server::Defaults {
            sample: args.sample,
            seed: args.seed,
            max_tokens: args.num_tokens,
        };
        return server::serve(addr, model, defaults);
    }

    match &args.prompt {
        Some(prompt) => oneshot(model.as_ref(), prompt, &args, &mut Rng::new(args.seed)),
        None => repl(model.as_ref(), &args),
    }
}

fn oneshot(model: &dyn Model, prompt: &str, args: &Args, rng: &mut Rng) -> Result<()> {
    let tokenizer = model.tokenizer();
    let ids = tokenizer.encode(prompt)?;
    if ids.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    println!("prompt: {prompt:?} ({} tokens)", ids.len());

    let mut session = model.session()?;
    let logits = generate::prefill(session.as_mut(), &ids)?;

    if args.show > 0 {
        println!("\ntop {} next tokens:", args.show);
        for (id, prob) in top_candidates(&logits, args.show) {
            println!("  {:>8.2}%  {:?}", prob * 100.0, tokenizer.decode(&[id]));
        }
    }

    println!("\n--- continuation ---");
    print!("{prompt}");
    io::stdout().flush().ok();

    let config = generate::Config {
        sample: args.sample,
        max_tokens: args.num_tokens,
    };
    let mut sequence = Sequence::new(ids);
    let mut sink = |text: &str| {
        print!("{text}");
        io::stdout().flush().ok();
        Flow::Continue
    };
    let outcome = generate::continue_from(
        model,
        session.as_mut(),
        &mut sequence,
        &logits,
        &config,
        rng,
        &mut sink,
    )?;
    println!("\n[stopped: {}]", outcome.stop);
    Ok(())
}

fn repl(model: &dyn Model, args: &Args) -> Result<()> {
    println!("REPL: type a prompt and press enter ('exit' or Ctrl-D to quit).");
    let mut rng = Rng::new(args.seed);
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if matches!(prompt, "exit" | "quit") {
            break;
        }
        if let Err(e) = oneshot(model, prompt, args, &mut rng) {
            eprintln!("error: {e:#}");
        }
        println!();
    }
    Ok(())
}

fn top_candidates(logits: &[f32], k: usize) -> Vec<(i64, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.into_iter()
        .take(k)
        .map(|i| (i as i64, exps[i] / sum))
        .collect()
}
