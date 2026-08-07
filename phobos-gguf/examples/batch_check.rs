// Feed a prompt as one batch and as one token at a time, and compare:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example batch_check -- MODEL.gguf
//
// Both orders run the same sum on the same weights, so a backend that batches
// correctly returns the same final logits either way. Comparing a backend
// against itself is what isolates batching: host against device also picks up
// every rounding difference between them, which after two dozen blocks is far
// larger than the thing being looked for.
//
// The same sum, but no longer in the same order on the device: a single row
// splits its contraction across the grid to fill the machine while eight rows
// or more go to the integer tensor cores, and the two accumulate differently.
// That is a legitimate 1e-4 or so per projection, and two dozen blocks of it
// compounds past what an exact comparison tolerates. So the measure is the
// largest gap as a fraction of the logit spread, which a batching mistake moves
// by a large fraction and a rounding difference does not, plus the token each
// order picks.

use anyhow::{Result, bail};
use phobos_gguf::Decoder;
use phobos_gguf::backend::{Backend, HostBackend};
use phobos_gguf::{Bpe, Gguf};

use phobos_gguf::backend::device;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: batch_check MODEL.gguf");
    };
    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    // Two lengths on purpose. The short prompt is under the row tile of the
    // quantized tensor-core projection, so it batches on the matvec; the long
    // one crosses it and leaves a remainder, which is the only place the two
    // kernels have to agree on a seam.
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
        failed |= !compare(&model, &host, "host", &tokens)?;

        #[cfg(feature = "cuda")]
        {
            failed |= !compare(&model, &gpu, "gpu", &tokens)?;
        }
    }

    // A prompt longer than one pass arrives as several batches, so the second
    // and later ones run with a cache that is already deep. That is the only
    // shape with both rows > 1 and start_pos > 0: a single-batch prefill has the
    // rows without the offset and a decode step has the offset without the rows.
    //
    // Against one wide pass rather than against the sequential order, because
    // the reference here is 600 positions and stepping it one at a time on the
    // host is minutes. Both orders batch, so this isolates the seam and nothing
    // else. The sizes bracket the tiles attention picks a kernel by: a whole
    // multiple of them, a ragged one, and one that leaves a ragged remainder.
    // The device only, because this is a device question: the host backend has
    // no shape-dependent paths, so every batch size runs the same loop there,
    // and 600 positions through it is minutes.
    #[cfg(feature = "cuda")]
    {
        let long: Vec<u32> = (0..600).map(|i| tokens_of(i, model.vocab())).collect();
        // Largest first: the growth a lazily sized table does is only fresh
        // once per process, so the first size through here is the one that
        // exercises it, and 512 is what the runtime actually batches at.
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
    // The split order runs first, and that is not arbitrary. Anything a model
    // grows lazily, the rotary table above all, is grown by whichever order
    // runs first and is then already big enough for the second. Running the one
    // wide pass first hides every fault in the growth, because a single pass
    // grows once, at the front, with nothing yet released to collide with. The
    // split order grows in the middle of a later pass, which is the case that
    // was actually broken.
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
    // The device measures about 1.1e-2 on the long prompt from accumulation
    // order alone, and the host stays exact. A batching mistake misplaces whole
    // rows, which moves the logits by a large fraction of their spread rather
    // than a hundredth of it, so this leaves room for the one and not the other.
    let ok = error <= 5e-2 && agreed;
    if !ok {
        println!("  reference [..6] {:?}", &serial[..6]);
        println!("  compared  [..6] {:?}", &batched[..6]);
    }
    Ok(ok)
}
