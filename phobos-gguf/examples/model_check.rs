// One forward pass on both backends, logits compared:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example model_check -- MODEL.gguf [-p PROMPT] [--single]
//
// The individual ops agreeing does not prove the sequence does: buffer
// reuse, aliasing and ordering only show up once a whole block runs.
// `--single` truncates the prompt to one token, past the tiled matmul.

use anyhow::{Result, bail};
use phobos_gguf::backend::HostBackend;
use phobos_gguf::{Bpe, Decoder, Gguf};

use phobos_gguf::backend::device;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: model_check MODEL.gguf [-p PROMPT] [--single]");
    };
    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let args: Vec<String> = std::env::args().collect();
    // `-p` sets the prompt. How much of the bound below quantization alone
    // uses varies by prompt, and a prompt that uses most of it judges
    // nothing else.
    let prompt = match args.iter().position(|a| a == "-p") {
        Some(at) => args.get(at + 1).cloned().unwrap_or_default(),
        None => "The capital of France is".to_string(),
    };
    let mut tokens = bpe.encode(&prompt)?;
    if args.iter().any(|a| a == "--single") {
        tokens.truncate(1);
    }

    let host = HostBackend::new();
    let gpu = device::DeviceBackend::new(&phobos_gguf::Quant::ALL)?;

    // The prompt pass takes the tiled path, then two decode steps take the
    // single-row one.
    let mut host_state = model.new_state();
    let mut gpu_state = model.new_state();
    let mut step = 0;
    let mut feed: Vec<u32> = tokens.clone();

    loop {
        let want = model.forward(&mut host_state, &feed, &host)?;
        let got = model.forward(&mut gpu_state, &feed, &gpu)?;

        // Measured against the logit spread, not per element. Reordered f32
        // arithmetic will not match digit for digit, and a near-zero logit
        // inflates a per-element relative error. What must hold is the
        // distribution's shape and the chosen token.
        let spread = want.iter().fold(f32::MIN, |a, &b| a.max(b))
            - want.iter().fold(f32::MAX, |a, &b| a.min(b));
        let worst = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        let error = worst / spread;
        let host_top = argmax(&want);
        let gpu_top = argmax(&got);
        // A flipped top token counts only when the host was decisive. Two top
        // logits closer than the drift can come out either way.
        let decisive = top_margin(&want) > 2.0 * worst;
        println!(
            "step {step}: spread err {error:>10.3e}   host {:?}  gpu {:?}{}",
            bpe.decode(&[host_top]),
            bpe.decode(&[gpu_top]),
            if decisive { "" } else { "  (tied)" }
        );
        // 2e-2 sits above where accumulated rounding over two dozen blocks
        // lands, so only a real defect crosses it.
        if error > 2e-2 || (decisive && host_top != gpu_top) {
            println!("  host[..8] {:?}", &want[..8]);
            println!("  gpu [..8] {:?}", &got[..8]);
            bail!("backends disagree at step {step}");
        }

        step += 1;
        if step > 2 {
            break;
        }
        feed = vec![host_top];
    }

    println!("\nbackends agree");
    Ok(())
}

/// Gap between the best and second-best logit.
fn top_margin(logits: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::MIN, f32::MIN);
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    best - second
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
