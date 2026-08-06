// Run text inference on the Phobos runtimes:
//
//   phobos-inference "The color of the sky is"
//   phobos-inference --gguf MODEL.gguf "The color of the sky is"
//
// Encodes the prompt, runs it through either the ONNX engines or a GGUF model,
// and streams the continuation. With no prompt it drops into a REPL that keeps
// the loaded model warm.

#[cfg(feature = "cuda")]
mod device;
mod gguf;
mod sampling;
mod server;
mod session;
mod tokenizer;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use gguf::{GgufRuntime, GgufState};
use sampling::{Rng, SampleConfig, Sequence, choose};
use session::{Engine, GenState};
use tokenizer::Tokenizer;

const DEFAULT_MODEL: &str = "models/gpt2-lm-head-10/GPT-2-LM-HEAD";
const DEFAULT_KV_DIR: &str = "models/gpt2-kv";

/// GPT-2 has 1024 learned position embeddings and cannot exceed them.
const GPT2_CONTEXT: usize = 1024;

struct Args {
    model_dir: PathBuf,
    kv_dir: PathBuf,
    gguf: Option<PathBuf>,
    num_tokens: usize,
    show: usize,
    sample: SampleConfig,
    seed: u64,
    listen: Option<String>,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut model_dir = PathBuf::from(DEFAULT_MODEL);
    let mut kv_dir = PathBuf::from(DEFAULT_KV_DIR);
    let mut gguf: Option<PathBuf> = None;
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
            "-m" | "--model" => model_dir = next("--model")?.into(),
            "--kv" => kv_dir = next("--kv")?.into(),
            "--gguf" => gguf = Some(next("--gguf")?.into()),
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
        model_dir,
        kv_dir,
        gguf,
        num_tokens: num_tokens.max(1),
        show,
        sample,
        seed,
        listen,
        prompt,
    })
}

enum Runtime {
    Onnx {
        engine: Box<Engine>,
        tok: Box<Tokenizer>,
    },
    Gguf(Box<GgufRuntime>),
}

enum RunState {
    Onnx(GenState),
    Gguf(GgufState),
}

impl Runtime {
    fn load(args: &Args) -> Result<Runtime> {
        match &args.gguf {
            Some(path) => Ok(Runtime::Gguf(Box::new(GgufRuntime::load(path)?))),
            None => Ok(Runtime::Onnx {
                engine: Box::new(Engine::load(&args.kv_dir, &args.model_dir)?),
                tok: Box::new(Tokenizer::gpt2()?),
            }),
        }
    }

    fn label(&self) -> String {
        match self {
            Runtime::Onnx { engine, .. } => format!("ONNX, {}", engine.label()),
            Runtime::Gguf(rt) => format!("GGUF, {}", rt.label()),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Runtime::Onnx { engine, .. } => engine.vocab_size(),
            Runtime::Gguf(rt) => rt.vocab_size(),
        }
    }

    fn context_limit(&self) -> usize {
        match self {
            Runtime::Onnx { .. } => GPT2_CONTEXT,
            Runtime::Gguf(rt) => rt.context_limit(),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        match self {
            Runtime::Onnx { tok, .. } => tok.encode(text),
            Runtime::Gguf(rt) => rt.encode(text),
        }
    }

    fn decode(&self, ids: &[i64]) -> String {
        match self {
            Runtime::Onnx { tok, .. } => tok.decode(ids),
            Runtime::Gguf(rt) => rt.decode(ids),
        }
    }

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        match self {
            Runtime::Onnx { tok, .. } => tok.decode_bytes(ids),
            Runtime::Gguf(rt) => rt.decode_bytes(ids),
        }
    }

    fn is_eog(&self, id: i64) -> bool {
        match self {
            Runtime::Onnx { tok, .. } => tok.eos() == Some(id),
            Runtime::Gguf(rt) => rt.is_eog(id),
        }
    }

    pub fn chat_template(&self) -> Option<&str> {
        match self {
            Runtime::Onnx { .. } => None,
            Runtime::Gguf(rt) => rt.chat_template(),
        }
    }

    pub fn bos_text(&self) -> Option<&str> {
        match self {
            Runtime::Onnx { .. } => None,
            Runtime::Gguf(rt) => rt.bos_text(),
        }
    }

    fn start(&self, ids: &[i64]) -> Result<(Vec<f32>, RunState)> {
        match self {
            Runtime::Onnx { engine, .. } => {
                let (logits, state) = engine.start(ids)?;
                Ok((logits, RunState::Onnx(state)))
            }
            Runtime::Gguf(rt) => {
                let (logits, state) = rt.start(ids)?;
                Ok((logits, RunState::Gguf(state)))
            }
        }
    }

    fn advance(&self, state: &mut RunState, token: i64) -> Result<Vec<f32>> {
        match (self, state) {
            (Runtime::Onnx { engine, .. }, RunState::Onnx(s)) => engine.advance(s, token),
            (Runtime::Gguf(rt), RunState::Gguf(s)) => rt.advance(s, token),
            _ => bail!("runtime and generation state are mismatched"),
        }
    }

    /// Returns a finished generation's device allocations.
    /// 
    /// Dropping the state does not.
    fn finish(&self, state: RunState) {
        if let (Runtime::Gguf(rt), RunState::Gguf(s)) = (self, state) {
            rt.finish(s);
        }
    }
}

impl RunState {
    fn len(&self) -> usize {
        match self {
            RunState::Onnx(s) => s.len(),
            RunState::Gguf(s) => s.len(),
        }
    }
}

fn print_usage() {
    eprintln!(
        "\
usage: phobos-inference [OPTIONS] [PROMPT]

With a PROMPT: print a continuation. Without one: start a REPL.

OPTIONS:
      --gguf FILE     run a GGUF model, dispatching on the architecture the
                      file declares, with the tokenizer the file carries.
                      Without it the exported GPT-2 ONNX engines load.
      --kv DIR        KV-cache models dir (decoder + with-past;
                      default: {DEFAULT_KV_DIR}). Used when present, else --model.
  -m, --model DIR     full-recompute LM-head model (default: {DEFAULT_MODEL})
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

    let backend = match args.gguf {
        Some(_) => gguf::backend_name(),
        None => session::backend_name(),
    };
    eprint!("loading model on {backend}... ");
    io::stderr().flush().ok();
    let runtime = Runtime::load(&args)?;
    eprintln!(
        "ready ({} engine, {} vocab).",
        runtime.label(),
        runtime.vocab_size()
    );

    if let Some(addr) = args.listen.clone() {
        let defaults = server::Defaults {
            sample: args.sample,
            seed: args.seed,
            max_tokens: args.num_tokens,
        };
        return server::serve(addr, runtime, defaults);
    }

    match &args.prompt {
        Some(prompt) => oneshot(&runtime, prompt, &args, &mut Rng::new(args.seed)),
        None => repl(&runtime, &args),
    }
}

fn oneshot(runtime: &Runtime, prompt: &str, args: &Args, rng: &mut Rng) -> Result<()> {
    let ids = runtime.encode(prompt)?;
    if ids.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    println!("prompt: {prompt:?} ({} tokens)", ids.len());

    let (logits, mut state) = runtime.start(&ids)?;
    if args.show > 0 {
        println!("\ntop {} next tokens:", args.show);
        for (id, prob) in top_candidates(&logits, args.show) {
            println!("  {:>8.2}%  {:?}", prob * 100.0, runtime.decode(&[id]));
        }
    }

    println!("\n--- continuation ---");
    print!("{prompt}");
    let mut sequence = Sequence::new(ids);
    let first = choose(&logits, &args.sample, sequence.history(), rng);
    let stop = stream_generate(
        runtime,
        &mut state,
        first,
        args.num_tokens,
        &args.sample,
        &mut sequence,
        rng,
    )?;
    println!("\n[stopped: {stop}]");
    runtime.finish(state);
    Ok(())
}

enum Stop {
    Eos,
    Context(usize),
    Limit,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Eos => write!(f, "end-of-sequence token"),
            Stop::Context(limit) => write!(f, "{limit}-token context limit"),
            Stop::Limit => write!(f, "token limit"),
        }
    }
}

/// Extend a generation in place, printing decoded text as it is produced.
/// `first` is the next token the prompt pass already chose.
fn stream_generate(
    runtime: &Runtime,
    state: &mut RunState,
    first: i64,
    num_tokens: usize,
    sample: &SampleConfig,
    sequence: &mut Sequence,
    rng: &mut Rng,
) -> Result<Stop> {
    let context_limit = runtime.context_limit();
    // Bytes short of a complete UTF-8 char; a token can split one.
    let mut pending: Vec<u8> = Vec::new();
    let mut next = first;
    let mut produced = 0usize;

    let stop = loop {
        if runtime.is_eog(next) {
            break Stop::Eos;
        }
        if state.len() >= context_limit {
            break Stop::Context(context_limit);
        }
        emit(runtime, &[next], &mut pending);
        sequence.push(next);
        next = choose(
            &runtime.advance(state, next)?,
            sample,
            sequence.history(),
            rng,
        );
        produced += 1;
        if produced >= num_tokens {
            break Stop::Limit;
        }
    };
    // Trailing incomplete bytes, rendered lossy.
    if !pending.is_empty() {
        print!("{}", String::from_utf8_lossy(&pending));
        io::stdout().flush().ok();
    }
    Ok(stop)
}

/// Append a token's bytes to `pending` and print every complete UTF-8 char.
fn emit(runtime: &Runtime, ids: &[i64], pending: &mut Vec<u8>) {
    pending.extend(runtime.decode_bytes(ids));
    let valid = match std::str::from_utf8(pending) {
        Ok(s) => s.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid > 0 {
        // The prefix is valid UTF-8 by construction.
        print!("{}", std::str::from_utf8(&pending[..valid]).unwrap());
        io::stdout().flush().ok();
        pending.drain(..valid);
    }
}

/// Read prompts until EOF or `exit`, reusing the warm model each turn.
fn repl(runtime: &Runtime, args: &Args) -> Result<()> {
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
        if let Err(e) = oneshot(runtime, prompt, args, &mut rng) {
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
