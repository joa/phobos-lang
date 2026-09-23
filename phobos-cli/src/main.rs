// Run text inference on the Phobos runtimes:
//
//   phobos-cli --gguf MODEL.gguf "The color of the sky is"
//   phobos-cli --onnx MODEL_DIR "The color of the sky is"
//
// Encodes the prompt, runs it through either a GGUF model or an ONNX export,
// and streams the continuation. With no prompt it drops into a REPL that keeps
// the loaded model warm. With --listen it serves instead, and puts a dashboard
// on the terminal unless told not to.
//

mod tui;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use phobos_base::cli;
use phobos_base::progress::{self, Event, Step};
use phobos_gguf::GgufModel;
use phobos_inference::Model;
use phobos_inference::generate::{self, Flow};
use phobos_inference::sampling::{Rng, SampleConfig, Sequence};
use phobos_inference::server;
use phobos_inference::telemetry::Meter;
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
    "--expert-cache",
];

/// Lines of the dashboard's log to reprint if serving ends badly. The tail is
/// where the failure is; the rest is the session that led up to it.
const REPLAY_LINES: usize = 32;

/// Every flag that is a switch rather than a value. See [`cli::Args::positional_with`].
const SWITCHES: &[&str] = &["--no-tui", "--tui", "--no-prefix-cache"];

/// Which model to run, and where its tokenizer comes from.
enum Source {
    /// A GGUF file, which carries its own vocabulary, and how much of the
    /// device to give a streamed model's expert cache.
    Gguf(PathBuf, phobos_gguf::runtime::LoadOptions),
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
    /// Whether serving should put a dashboard on the terminal.
    tui: bool,
    /// Whether serving should keep a finished request's session.
    prefix_cache: bool,
}

fn parse_args(args: &cli::Args) -> Result<Args> {
    let words = args.positional_with(VALUED, SWITCHES)?;
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
        // On when there is a terminal to draw on, and off when there is not,
        // which is what a server being logged to a file wants. `--tui` says
        // to draw anyway, for a terminal this fails to recognize.
        tui: !args.has("--no-tui") && (args.has("--tui") || tui::unavailable().is_none()),
        prefix_cache: !args.has("--no-prefix-cache"),
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
        (Some(path), None) => {
            let expert_cache_bytes = args
                .value("--expert-cache")?
                .map(phobos_base::cli::parse_size)
                .transpose()
                .context("--expert-cache")?;
            Ok(Source::Gguf(path.into(), phobos_gguf::runtime::LoadOptions { expert_cache_bytes }))
        }
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
            Source::Gguf(path, options) => Ok(Box::new(GgufModel::load_with(path, *options)?)),
            Source::Onnx { dir, tokenizer } => {
                Ok(Box::new(OnnxModel::load_with_tokenizer(dir, tokenizer)?))
            }
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Source::Gguf(..) => phobos_gguf::runtime::backend_name(),
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
      --expert-cache SIZE
                      device memory for a streamed model's expert cache,
                      2g, 1500m or bytes; default what the resident weights
                      leave, less a reserve. Ignored without experts
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
      --no-tui        serve without the dashboard, printing the runtime's lines
                      instead. Implied when stdout is not a terminal, and the
                      reason is printed whenever that happens
      --tui           draw the dashboard even where the terminal was not
                      recognized as one
      --no-prefix-cache
                      start every request from an empty cache. By default a
                      finished request's session is kept and the next request
                      reuses as much of it as the two prompts agree on, which
                      is most of a chat. Turning it off costs that and gains
                      the cache's buffers back for a pass to use as scratch,
                      which is the difference on a card that only just fits
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
    let serving = args.listen.clone();

    // Only a server draws, and only it needs the dashboard up before the
    // model: a cold start compiles every kernel from source and takes
    // minutes, which without something watching looks like a hung process.
    let drawing = match (&serving, args.tui) {
        (Some(_), true) => Some(tui::start()?),
        _ => None,
    };
    if serving.is_some() && !args.tui && !raw.has("--no-tui") {
        no_dashboard(&tui::unavailable().unwrap_or_else(|| "unknown".to_string()));
    } else if args.tui
        && let Some(name) = tui::raw_report()
    {
        eprintln!("note: {name} is set, so its report will overwrite part of the dashboard.");
    }
    if drawing.is_none() {
        // Nothing is drawing, so the lines go to stderr as they happen.
        progress::set_sink(report_line);
        eprint!("loading model on {}... ", args.source.backend_name());
        io::stderr().flush().ok();
    }

    let model = args.source.load()?;
    let info = model.info();
    match drawing.as_ref() {
        Some(drawing) => drawing.loaded(info),
        None => eprintln!("ready ({} engine, {} vocab).", info.label, info.vocab_size),
    }

    if let Some(addr) = serving {
        let defaults = server::Defaults {
            sample: args.sample,
            seed: args.seed,
            max_tokens: args.num_tokens,
            prefix_cache: args.prefix_cache,
        };
        return serve(addr, model, defaults, drawing);
    }

    match &args.prompt {
        Some(prompt) => oneshot(model.as_ref(), prompt, &args, &mut Rng::new(args.seed)),
        None => repl(model.as_ref(), &args),
    }
}

/// One line for each kernel that had to be built, for a run with no dashboard.
///
/// Only the ones built: a warm start finds every kernel in the on-disk cache
/// and is over in seconds, and forty lines saying so are forty lines of
/// nothing. A cold start is the case worth narrating.
fn report_line(event: Event<'_>) {
    static BUILT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static STARTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    use std::sync::atomic::Ordering::Relaxed;

    match event {
        // Said as each one begins, because a batch begins all at once and a
        // kernel that takes ten minutes is otherwise never named until it is
        // over. The name that matters during the wait is the one that has
        // not come back.
        Event::Started { item, .. } => {
            // The first line interrupts the unfinished "loading model on ...",
            // which a warm start would have completed with "ready".
            if STARTED.fetch_add(1, Relaxed) == 0 {
                eprintln!();
            }
            eprintln!("lowering kernel {item}");
        }
        Event::Finished(step) => {
            if step.cached {
                return;
            }
            let built = BUILT.fetch_add(1, Relaxed) + 1;
            eprintln!("{}", progress_line(&step, built));
        }
    }
}

/// A finished kernel, as a line. Said in the past tense: it is reported when
/// the lowering comes back, and `lowering ...` already said when it began.
///
/// A batch knows how many it holds and can say how far along it is. A kernel
/// compiled on its own is a batch of one, where a percentage would always
/// read 100 and the only honest figure is how many have been built so far.
fn progress_line(step: &Step<'_>, built: usize) -> String {
    if step.total > 1 {
        let done = step.done as f64 / step.total as f64 * 100.0;
        format!(
            "compiled kernel {} - {}/{} - {done:.2}% complete",
            step.item, step.done, step.total
        )
    } else {
        format!("compiled kernel {} - {built} built so far", step.item)
    }
}

/// Say why there is no dashboard, loudly enough to be found again.
///
/// A screen of build output usually sits above this, and the alternative to
/// being hard to miss is a user who asked for a dashboard watching a server
/// print lines and wondering which part went wrong.
fn no_dashboard(why: &str) {
    let rule = "=".repeat(72);
    eprintln!("{rule}");
    eprintln!("  NO DASHBOARD");
    eprintln!("  {why}.");
    eprintln!("  Remove the cause, or pass --tui to draw one regardless.");
    eprintln!("  --no-tui says you meant this and silences the notice.");
    eprintln!("{rule}");
}

/// Serve, with the dashboard drawing beside the engine if it was asked for.
///
/// The engine stays on this thread because that is where the model was loaded
/// and where its device context lives; the dashboard is what moves. They meet
/// only at the [`Meter`], and either one ending clears its running flag, so
/// the other stops too.
fn serve(
    addr: String,
    model: Box<dyn Model>,
    defaults: server::Defaults,
    drawing: Option<tui::Dashboard>,
) -> Result<()> {
    let Some(drawing) = drawing else {
        // Nothing will draw the meter, so its lines go where the server's
        // own used to.
        let meter = Arc::new(Meter::new());
        meter.echo_to_stderr();
        return server::serve(addr, model, defaults, meter);
    };

    let meter = drawing.meter();
    let result = server::serve(addr, model, defaults, meter.clone());
    // Whichever of the two finished first, the other is told to stop.
    meter.stop();
    let outcome = drawing.join().and(result);

    // On the way out after a failure, print what the dashboard was showing
    // when it happened. The terminal is back by now, and the ring is the only
    // copy; a clean quit needs none of it, since the user was watching.
    if outcome.is_err() {
        for line in meter.snapshot().log.iter().rev().take(REPLAY_LINES) {
            eprintln!("[{:>7.3}s] {}", line.at.as_secs_f64(), line.text);
        }
    }
    outcome
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

    let config = generate::Config::new(args.sample, args.num_tokens);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn step(item: &'static str, done: usize, total: usize) -> Step<'static> {
        Step {
            stage: "kernels",
            item,
            done,
            total,
            cached: false,
            took: std::time::Duration::from_millis(1200),
            source: "kernel x() { }",
            ptx: ".visible .entry x",
        }
    }

    #[test]
    fn a_batch_reports_its_position_and_a_percentage() {
        assert_eq!(
            progress_line(&step("iq2s_matvec", 12, 37), 12),
            "compiled kernel iq2s_matvec - 12/37 - 32.43% complete"
        );
        assert_eq!(
            progress_line(&step("q8_qmma", 37, 37), 37),
            "compiled kernel q8_qmma - 37/37 - 100.00% complete"
        );
    }

    #[test]
    fn a_kernel_on_its_own_reports_the_running_count() {
        // A batch of one is every kernel asked for outside a batch, where
        // 1/1 and 100% would be true and useless.
        assert_eq!(
            progress_line(&step("argmax_finish", 1, 1), 4),
            "compiled kernel argmax_finish - 4 built so far"
        );
    }
}
