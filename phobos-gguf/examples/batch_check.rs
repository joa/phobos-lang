// Feed a prompt as one batch and as one token at a time, and compare:
//
//   cargo run --release -p phobos-gguf --features cuda \
//       --example batch_check -- [--gpu] MODEL.gguf
//
// A backend is compared against itself, since host-against-device rounding
// differences are far larger than a batching bug. The measure is the largest
// gap as a fraction of the logit spread, plus the token each run picks.

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
    // The short prompt is under the tensor-core row tile and batches on the
    // matvec. The long one crosses the tile into a remainder, the seam where
    // the two kernels must agree.
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

    // A long prompt arrives as several batches, and later batches run
    // against a deep cache: rows > 1 with start_pos > 0. Checked against one
    // wide pass, since token by token would take minutes. The batch sizes
    // bracket the tiles attention picks its kernel by. Device only, since the
    // host has no shape-dependent paths.
    #[cfg(feature = "cuda")]
    {
        let long: Vec<u32> = (0..600).map(|i| tokens_of(i, model.vocab())).collect();
        // Largest first. 512 is the runtime's batch size, and a lazily sized
        // table grows once per process, on whichever size runs first.
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
    // The split run must go first. Lazily grown state, mainly the rotary
    // table, grows during whichever run comes first. A wide pass would grow
    // it once up front and hide a fault the split run hits mid-growth.
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
    // The device accumulates the two orders differently: one row splits its
    // contraction across the grid, eight or more rows use the tensor cores.
    // A batching bug misplaces whole rows and moves the logits by a large
    // fraction of their spread, so a loose tolerance still catches it.
    let ok = error <= 5e-2 && agreed;
    if !ok {
        println!("  reference [..6] {:?}", &serial[..6]);
        println!("  compared  [..6] {:?}", &batched[..6]);
    }
    Ok(ok)
}
