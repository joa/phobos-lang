// The device-side argmax reduction (Backend::argmax) against host argmax
// over the same device logits, plus a synthetic edge case the model's own
// vocab shape never exercises:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example argmax_check -- MODEL.gguf [-p PROMPT]
//
// Two backends' logits (host and device) can legitimately disagree by a
// rounding difference too small to flip the winning token; comparing device
// argmax against a host argmax computed over a *different* backend's logits
// would conflate that with an argmax bug. So this reads the full logits back
// from the device (`Session::extend`'s own path) and compares the device
// kernel's answer only against the host reduction of that same vector --
// isolating the one thing this beam changed.

use anyhow::{Result, bail};
use phobos_gguf::backend::device;
use phobos_gguf::backend::Backend;
use phobos_gguf::{Bpe, Decoder, Gguf};
use phobos_inference::sampling::argmax;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: argmax_check MODEL.gguf [-p PROMPT]");
    };
    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let args: Vec<String> = std::env::args().collect();
    let prompt = match args.iter().position(|a| a == "-p") {
        Some(at) => args.get(at + 1).cloned().unwrap_or_default(),
        None => "The capital of France is".to_string(),
    };
    let tokens = bpe.encode(&prompt)?;

    // Two independent sessions of the same model, fed the identical token
    // sequence: one reads back the full logits every step (Decoder::forward,
    // what Session::extend runs), the other takes the fast path
    // (Decoder::forward_greedy). Both run on their own DeviceBackend and
    // State so neither's scratch reuse can leak into the other's numbers.
    let full_backend = device::DeviceBackend::new()?;
    let fast_backend = device::DeviceBackend::new()?;
    let mut full_state = model.new_state();
    let mut fast_state = model.new_state();

    let mut mismatches = 0usize;
    let mut feed: Vec<u32> = tokens.clone();
    for step in 0..32 {
        let logits = model.forward(&mut full_state, &feed, &full_backend)?;
        let fast_id = model.forward_greedy(&mut fast_state, &feed, &fast_backend)?;
        let host_id = argmax(&logits);

        let agrees = fast_id == host_id;
        if !agrees {
            // A real tie: two logits equal to float precision, which the
            // spatial halving tree does not resolve the same way `argmax`'s
            // last-of-equal-maxima rule does (see argsel's own doc comment).
            // Anything else is a bug.
            let top = logits.iter().cloned().fold(f32::MIN, f32::max);
            let tied = logits[fast_id as usize] == top && logits[host_id as usize] == top;
            if !tied {
                println!(
                    "step {step}: MISMATCH host {:?} (id {host_id}) vs device-argmax {:?} (id {fast_id}), \
                     not a tie (host logit {}, device logit {})",
                    bpe.decode(&[host_id as u32]),
                    bpe.decode(&[fast_id as u32]),
                    logits[host_id as usize],
                    logits[fast_id as usize],
                );
                bail!("argmax disagreement at step {step} that is not a float tie");
            }
            mismatches += 1;
            println!(
                "step {step}: tie, host {:?} device {:?} (both logit {top})",
                bpe.decode(&[host_id as u32]),
                bpe.decode(&[fast_id as u32])
            );
        } else {
            println!(
                "step {step}: agree, {:?} (id {host_id})",
                bpe.decode(&[host_id as u32])
            );
        }

        feed = vec![host_id as u32];
    }

    println!(
        "\n{} of 32 decode steps agree device-argmax with host argmax over the same device logits \
         ({mismatches} float tie{})",
        32 - mismatches,
        if mismatches == 1 { "" } else { "s" }
    );

    // A synthetic edge case no real model logit vector exercises: every
    // element negative, so a masked tail's zero fill (this kernel's
    // argmax_chunk_width never produces one, but the sentinel logic is worth
    // pinning directly) would silently win if the reduction's identity were
    // wrong. Also covers the winner sitting at index 0 and at the last index.
    println!("\nsynthetic edge cases:");
    let cases: [(&str, Vec<f32>); 3] = [
        ("all negative, winner in the middle", {
            let mut v = vec![-5.0f32; 130_560];
            v[70_000] = -0.5;
            v
        }),
        ("winner at index 0", {
            let mut v = vec![-1.0f32; 4096];
            v[0] = 10.0;
            v
        }),
        ("winner at the last index", {
            let mut v = vec![-1.0f32; 4096];
            let n = v.len();
            v[n - 1] = 10.0;
            v
        }),
    ];
    for (name, data) in cases {
        let buf = full_backend.upload(&data)?;
        let got = full_backend.argmax(buf, data.len())?;
        full_backend.release(buf);
        let want = argmax(&data);
        println!(
            "  {name}: device {got} host {want} {}",
            if got == want { "ok" } else { "MISMATCH" }
        );
        if got != want {
            bail!("synthetic case '{name}' disagrees: device {got} host {want}");
        }
    }

    full_state.release(&full_backend);
    fast_state.release(&fast_backend);
    println!("\nargmax agrees");
    Ok(())
}
