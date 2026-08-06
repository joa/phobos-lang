// Dump the header of a GGUF file:
//
//   cargo run -p phobos-gguf --example inspect -- MODEL.gguf
//
// Prints metadata, the tensor directory grouped by shape, and a quantization
// histogram. --tensors lists every tensor instead of a summary, --dump NAME
// dequantizes one and prints its leading values.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_gguf::Gguf;

fn main() -> Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut list_tensors = false;
    let mut dump: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tensors" => list_tensors = true,
            "--dump" => dump = Some(args.next().unwrap_or_default()),
            other => path = Some(other.into()),
        }
    }
    let Some(path) = path else {
        bail!("usage: inspect [--tensors] [--dump NAME] MODEL.gguf");
    };

    if let Some(name) = dump {
        return dump_tensor(&path, &name);
    }

    let gguf = Gguf::open(&path)?;
    println!("{} (GGUF v{})", path.display(), gguf.version());
    println!(
        "{} tensors, {} metadata keys",
        gguf.tensors().len(),
        gguf.metadata().len()
    );
    println!("{:.2}B parameters\n", gguf.parameter_count() as f64 / 1e9);

    println!("=== metadata ===");
    for (key, value) in gguf.metadata().iter() {
        println!("  {key:<48} {}", value.preview());
    }

    println!("\n=== tensors ===");
    let mut by_type: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for info in gguf.tensors() {
        let entry = by_type.entry(info.ggml_type.name()).or_default();
        entry.0 += 1;
        entry.1 += info.numel();
        if list_tensors {
            println!(
                "  {:<44} {:<24} {}",
                info.name,
                format!("{:?}", info.row_major_dims()),
                info.ggml_type.name()
            );
        }
    }
    for (name, (count, numel)) in &by_type {
        println!("  {name:<8} {count:>5} tensors  {:>12} elements", numel);
    }

    if let Ok(vocab) = gguf.vocab() {
        println!("\n=== tokenizer ===");
        println!("  model      {}", vocab.model);
        println!("  pre        {}", vocab.pre.as_deref().unwrap_or("(none)"));
        println!("  tokens     {}", vocab.len());
        println!("  merges     {}", vocab.merges.len());
        println!("  specials   {}", vocab.special_tokens().count());
        println!("  bos/eos    {:?} / {:?}", vocab.bos, vocab.eos);
    }

    Ok(())
}

/// Dequantize one tensor and report its leading values and range, for checking
/// the block decoders against a reference.
fn dump_tensor(path: &std::path::Path, name: &str) -> Result<()> {
    let gguf = Gguf::open(path)?;
    let info = gguf
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("no tensor '{name}'"))?;
    let values = gguf.dequantize(name)?;

    println!(
        "{name}: {:?} {} ({} elements)",
        info.row_major_dims(),
        info.ggml_type.name(),
        values.len()
    );
    let head: Vec<String> = values.iter().take(12).map(|v| format!("{v:.8}")).collect();
    println!("  first 12   [{}]", head.join(", "));

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    let sum_squares: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    println!("  min/max    {min:.8} / {max:.8}");
    println!("  mean       {:.8}", sum / values.len() as f64);
    println!(
        "  rms        {:.8}",
        (sum_squares / values.len() as f64).sqrt()
    );
    Ok(())
}
