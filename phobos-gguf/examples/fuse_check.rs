// The fused path against the launched one, on the device, with no host in it:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example fuse_check -- MODEL.gguf [-n STEPS] [--single]
//
// Both paths run on one backend over one upload of weights, so fusion is the
// only difference. A pass of more than one row must match exactly, since no
// stage fuses past one row. A decode step may differ by accumulation order,
// up to `APART_MAX`.

use anyhow::{Result, bail};
use phobos_gguf::{Bpe, Decoder, Gguf};

use phobos_gguf::backend::device;

/// The largest allowed gap, as a fraction of the logit spread. Well above a
/// correct fusion's accumulation drift, and well below what a broken one
/// produces.
///
/// Token flips alone cannot catch a defect. With enough drift every flip
/// counts as tied, so a gross defect would report only ties.
const APART_MAX: f32 = 1e-1;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        bail!("usage: fuse_check MODEL.gguf [-n STEPS] [--single]");
    };
    let steps = match args.iter().position(|a| a == "-n") {
        Some(at) => args
            .get(at + 1)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("-n wants a step count")),
        None => 32,
    };

    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let mut tokens = bpe.encode("The capital of France is")?;
    if args.iter().any(|a| a == "--single") {
        tokens.truncate(1);
    }

    // One backend, one set of weights, two states. The recorded pass is keyed
    // by its launch list, so alternating configurations rebuilds the graph
    // every pass. That only costs time.
    let mut gpu = device::DeviceBackend::new(&phobos_gguf::Quant::ALL)?;
    let mut plain_state = model.new_state();
    let mut fused_state = model.new_state();

    let mut feed = tokens.clone();
    let mut worst = 0.0f32;
    let mut worst_at = 0;
    let mut total = 0.0f32;
    let mut decodes = 0;
    let mut flips = 0;
    let mut multirow = false;
    for step in 0..=steps {
        gpu.set_fused(false);
        let want = model.forward(&mut plain_state, &feed, &gpu)?;
        gpu.set_fused(true);
        let got = model.forward(&mut fused_state, &feed, &gpu)?;

        // The same measure `model_check` and `batch_check` report, so the
        // numbers compare.
        let spread = want.iter().fold(f32::MIN, |a, &b| a.max(b))
            - want.iter().fold(f32::MAX, |a, &b| a.min(b));
        // Checked before the distance. `f32::max` drops NaN, so a NaN logit
        // would fold to zero distance and look like perfect agreement.
        if let Some(at) = got.iter().position(|v| !v.is_finite()) {
            bail!(
                "the fused path put {} in logit {at} at step {step}",
                got[at]
            );
        }
        let apart = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max)
            / spread;
        let (plain_top, fused_top) = (argmax(&want), argmax(&got));
        // A flip counts only when the launched path was decisive. Two top
        // logits closer than the drift can come out either way.
        let decisive = top_margin(&want) > 2.0 * apart * spread;
        if plain_top != fused_top {
            flips += 1;
        }
        if apart > worst {
            (worst, worst_at) = (apart, step);
        }
        let rows = feed.len();
        // Average over one-row passes only, the ones that fuse. A one-token
        // prompt counts.
        if rows == 1 {
            total += apart;
            decodes += 1;
        }

        println!(
            "step {step:>3}  rows {rows:>4}  apart {apart:>10.3e}  launched {:?}  fused {:?}{}",
            bpe.decode(&[plain_top]),
            bpe.decode(&[fused_top]),
            match (plain_top == fused_top, decisive) {
                (true, _) => "",
                (false, true) => "  FLIPPED",
                (false, false) => "  flipped (tied)",
            }
        );
        // Rows, not the step: a one-token prompt is a decode-shaped pass and
        // does fuse.
        multirow |= rows > 1;
        if rows > 1 && apart != 0.0 {
            bail!("no stage fuses past one row, so a pass of {rows} must agree exactly");
        }
        if apart > APART_MAX {
            bail!(
                "the paths are {apart:.3e} apart at step {step}, which is a defect and not a fold"
            );
        }
        if plain_top != fused_top && decisive {
            bail!("the fused path picks a different token at step {step}");
        }

        // Both paths are fed the launched path's token, so the contexts stay
        // identical and every step compares like with like.
        feed = vec![plain_top];
    }

    println!(
        "\n{}over {decodes} decode steps the paths are at most {worst:.3e} of the logit\n\
         spread apart (step {worst_at}), {:.3e} on average, with {flips} top-token flips, none of\n\
         them decided",
        match multirow {
            true => "the prompt pass agrees exactly, and ",
            false => "",
        },
        total / decodes as f32
    );
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
