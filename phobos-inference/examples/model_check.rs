// One forward pass on both backends, logits compared:
//
//   cargo run --release -p phobos-inference --features cuda \
//       --example model_check -- MODEL.gguf [--single]
//
// The individual ops agreeing does not prove the sequence does: buffer reuse,
// aliasing and stream ordering only show up once a whole block runs.
//
// `--single` truncates the prompt to one token, which never reaches the tiled
// matmul, so a disagreement surviving it is not the tiled path's.

use anyhow::{Result, bail};
use phobos_gguf::compute::HostBackend;
use phobos_gguf::{Bpe, Decoder, Gguf};

#[path = "../src/device.rs"]
mod device;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: model_check MODEL.gguf [--single]");
    };
    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let mut tokens = bpe.encode("The capital of France is")?;
    // A single-token prompt never reaches the tiled matmul, so if this agrees
    // and the multi-token one does not, the tiled path is the difference.
    if std::env::args().any(|a| a == "--single") {
        tokens.truncate(1);
    }

    let host = HostBackend::new();
    let gpu = device::DeviceBackend::new()?;

    // Prompt pass, then two decode steps: the first exercises the tiled path,
    // the rest the single-row one.
    let mut host_state = model.new_state();
    let mut gpu_state = model.new_state();
    let mut step = 0;
    let mut feed: Vec<u32> = tokens.clone();

    loop {
        let want = model.forward(&mut host_state, &feed, &host)?;
        let got = model.forward(&mut gpu_state, &feed, &gpu)?;

        // Against the logit spread, not element by element. Two dozen blocks of
        // f32 arithmetic in a different order do not agree to a fixed number of
        // digits per element, and a logit near zero would make a per-element
        // relative error report a difference that changes nothing. What has to
        // hold is that the distribution is the same shape and picks the same
        // token; a kernel writing outside its output moves it far more than
        // this bound, which is how the tiled matmul's overrun showed up.
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
        // A flipped top token only means something when the host was decided:
        // if the two leading logits sit closer together than the drift, either
        // one can come out on top and the flip carries no information.
        let decisive = top_margin(&want) > 2.0 * worst;
        println!(
            "step {step}: spread err {error:>10.3e}   host {:?}  gpu {:?}{}",
            bpe.decode(&[host_top]),
            bpe.decode(&[gpu_top]),
            if decisive { "" } else { "  (tied)" }
        );
        // About 1% of spread is where two dozen blocks of accumulated rounding
        // land. A kernel writing outside its output moves it far more than
        // this, which is how the tiled matmul's overrun showed up here.
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
