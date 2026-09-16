// Feed a prompt as one batch and as one token at a time, and compare:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example batch_check -- [--gpu] MODEL.gguf
//
// Compares a backend against itself, since host-against-device also picks up
// rounding differences far larger than a batching bug. The measure is the
// largest gap as a fraction of the logit spread, which a batching mistake
// moves and rounding does not, plus the token each picks.

use anyhow::{Result, bail};
use phobos_gguf::Decoder;
use phobos_gguf::backend::{Backend, HostBackend};
use phobos_gguf::{Bpe, Gguf};

use phobos_gguf::backend::device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--gpu` skips the host reference and compares the device's batched
    // pass against its own single-token one.
    let device_only = args.iter().any(|a| a == "--gpu");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        bail!("usage: batch_check [--gpu] MODEL.gguf");
    };
    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    // Two lengths on purpose: the short one is under the tensor-core row
    // tile and batches on the matvec; the long one crosses it into a
    // remainder, the only place the two kernels must agree on a seam.
    let prompts = [
        "The capital of France is",
        "The capital of France is Paris, and the capital of Germany is Berlin, \
         and the capital of Italy is Rome, and the capital of Spain is",
    ];

    let mut failed = false;
    let host = HostBackend::new();
    #[cfg(feature = "cuda")]
    let gpu = device::DeviceBackend::new()?;

    for prompt in prompts {
        let tokens = bpe.encode(prompt)?;
        if !device_only {
            failed |= !compare(&model, &host, "host", &tokens)?;
        }

        #[cfg(feature = "cuda")]
        {
            failed |= !compare(&model, &gpu, "gpu", &tokens)?;
        }
    }

    // A prompt longer than one pass arrives as several batches, so later ones
    // run against a cache that is already deep -- the only shape with both
    // rows > 1 and start_pos > 0. Checked against one wide pass rather than
    // one token at a time, which would take minutes; the sizes bracket the
    // tiles attention picks a kernel by. Device only: the host has no
    // shape-dependent paths.
    #[cfg(feature = "cuda")]
    {
        let long: Vec<u32> = (0..600).map(|i| tokens_of(i, model.vocab())).collect();
        // Largest first: a lazily sized table grows once per process, on
        // whichever size runs first, and 512 is what the runtime batches at.
        for batch in [512usize, 100, 64] {
            failed |= !compare_batches(&model, &gpu, "gpu", &long, batch)?;
        }
    }

    if failed {
        bail!("a backend does not batch consistently");
    }
    println!("\nbatched and sequential agree");
    Ok(())
}

/// A deterministic spread of valid token ids (xorshift64*).
fn tokens_of(index: usize, vocab: usize) -> u32 {
    let mut seed = 0x2545_f491_4f6c_dd1du64 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    (seed % vocab as u64) as u32
}

/// One pass over the whole prompt against the same prompt in `batch`-sized
/// passes, which is what a prompt longer than one pass takes.
fn compare_batches(
    model: &Decoder,
    backend: &dyn Backend,
    name: &str,
    tokens: &[u32],
    batch: usize,
) -> Result<bool> {
    let split = |batch: usize| -> Result<Vec<f32>> {
        let mut state = model.new_state();
        let mut logits = Vec::new();
        for chunk in tokens.chunks(batch) {
            logits = model.forward(&mut state, chunk, backend)?;
        }
        state.release(backend);
        Ok(logits)
    };
    // The split order runs first, deliberately: anything grown lazily (the
    // rotary table above all) grows on whichever order runs first. A wide
    // pass grows it once at the front, with nothing released to collide
    // with, which hides a fault a split pass would hit mid-growth instead.
    let label = format!("{} tokens in batches of {batch}", tokens.len());
    let split_first = split(batch)?;
    report(name, &label, &split(tokens.len())?, &split_first)
}

/// Returns whether the two orders agreed.
fn compare(model: &Decoder, backend: &dyn Backend, name: &str, tokens: &[u32]) -> Result<bool> {
    let mut batched_state = model.new_state();
    let batched = model.forward(&mut batched_state, tokens, backend)?;

    let mut serial_state = model.new_state();
    let mut serial = Vec::new();
    for &token in tokens {
        serial = model.forward(&mut serial_state, &[token], backend)?;
    }
    report(name, &format!("{} tokens", tokens.len()), &serial, &batched)
}

/// Compare two runs that should have produced the same logits.
fn report(name: &str, what: &str, reference: &[f32], got: &[f32]) -> Result<bool> {
    let (serial, batched) = (reference, got);

    let spread = serial.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - serial.iter().copied().fold(f32::INFINITY, f32::min);
    let error = batched
        .iter()
        .zip(serial)
        .map(|(b, s)| (b - s).abs())
        .fold(0.0f32, f32::max)
        / spread.max(1.0);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 { (i, x) } else { best }
            })
            .0
    };
    let agreed = argmax(batched) == argmax(serial);
    println!(
        "{name:>5}: spread err {error:>10.3e}   token {}   over {what}",
        if agreed { "agrees" } else { "DIFFERS" }
    );
    // The device accumulates the two orders differently (a single row splits
    // its contraction across the grid, eight or more go to the tensor
    // cores), while the host stays exact. A batching mistake misplaces whole
    // rows and moves the logits by a large fraction of their spread, so a
    // loose tolerance still catches it.
    let ok = error <= 5e-2 && agreed;
    if !ok {
        println!("  reference [..6] {:?}", &serial[..6]);
        println!("  compared  [..6] {:?}", &batched[..6]);
    }
    Ok(ok)
}
