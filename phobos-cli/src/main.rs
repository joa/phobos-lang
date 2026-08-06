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

use anyhow::{Result, bail};

use phobos_base::cli;
use phobos_gguf::GgufModel;
use phobos_inference::Model;
use phobos_inference::generate::{self, Flow};
use phobos_inference::sampling::{Rng, SampleConfig, Sequence};
use phobos_inference::server;
use phobos_onnx::OnnxModel;

/// Every flag that takes a value. Whatever is left over is the prompt.
const VALUED: &[&str] = &[
    "--gguf",
    "--onnx",
    "--tokenizer",
    "-n",
    "--num",
    "--show",
    "-t",
    "--temp",
    "-k",
    "--top-k",
    "-p",
    "--top-p",
    "--min-p",
    "--presence-penalty",
    "--repetition-penalty",
    "--seed",
    "--listen",
];

/// Which model to run, and where its tokenizer comes from.
enum Source {
    /// A GGUF file, which carries its own vocabulary.
    Gguf(PathBuf),
    /// An ONNX export, plus the directory holding the tokenizer it was
    /// exported against. Defaults to the export's own directory.
    Onnx { dir: PathBuf, tokenizer: PathBuf },
}

struct Args {
    source: Source,
    num_tokens: usize,
    show: usize,
    sample: SampleConfig,
    seed: u64,
    listen: Option<String>,
    prompt: Option<String>,
}

fn parse_args(args: &cli::Args) -> Result<Args> {
    let words = args.positional(VALUED)?;
    Ok(Args {
        source: source(args)?,
        num_tokens: args
            .parse_of::<usize>(&["-n", "--num"])?
            .unwrap_or(200)
            .max(1),
        show: args.parse("--show")?.unwrap_or(5),
        sample: sampling(args)?,
        seed: args.parse("--seed")?.unwrap_or(0),
        listen: args.value("--listen")?.map(str::to_string),
        prompt: (!words.is_empty()).then(|| words.join(" ")),
    })
}

/// Greedy, with each field a flag overrides.
fn sampling(args: &cli::Args) -> Result<SampleConfig> {
    let base = SampleConfig::greedy();
    Ok(SampleConfig {
        temperature: args
            .parse_of(&["-t", "--temp"])?
            .unwrap_or(base.temperature),
        top_k: args.parse_of(&["-k", "--top-k"])?.unwrap_or(base.top_k),
        top_p: args.parse_of(&["-p", "--top-p"])?.unwrap_or(base.top_p),
        min_p: args.parse("--min-p")?.unwrap_or(base.min_p),
        presence_penalty: args
            .parse("--presence-penalty")?
            .unwrap_or(base.presence_penalty),
        repetition_penalty: args
            .parse("--repetition-penalty")?
            .unwrap_or(base.repetition_penalty),
    })
}

fn source(args: &cli::Args) -> Result<Source> {
    match (args.value("--gguf")?, args.value("--onnx")?) {
        (Some(path), None) => Ok(Source::Gguf(path.into())),
        (None, Some(dir)) => Ok(Source::Onnx {
            // An ONNX export carries no vocabulary, so the tokenizer is a pair
            // of files that by default sit beside the model.
            tokenizer: args.value("--tokenizer")?.unwrap_or(dir).into(),
            dir: dir.into(),
        }),
        (None, None) => bail!("no model: pass --gguf FILE or --onnx DIR"),
        (Some(_), Some(_)) => bail!("--gguf and --onnx are alternatives, not both"),
    }
}

impl Source {
    fn load(&self) -> Result<Box<dyn Model>> {
        match self {
            Source::Gguf(path) => Ok(Box::new(GgufModel::load(path)?)),
            Source::Onnx { dir, tokenizer } => {
                Ok(Box::new(OnnxModel::load_with_tokenizer(dir, tokenizer)?))
            }
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Source::Gguf(_) => phobos_gguf::runtime::backend_name(),
            Source::Onnx { .. } => phobos_onnx::runtime::backend_name(),
        }
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
      --tokenizer DIR where to read the tokenizer an ONNX export was made
                      against: a vocab.json beside a merges.txt, or an
                      encoder.json beside a vocab.bpe. Defaults to --onnx DIR.
                      An ONNX file carries no vocabulary of its own
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
    let raw = cli::Args::from_env();
    if raw.wants_help() {
        print_usage();
        return Ok(());
    }
    let args = parse_args(&raw)?;

    eprint!("loading model on {}... ", args.source.backend_name());
    io::stderr().flush().ok();
    let model = args.source.load()?;
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
