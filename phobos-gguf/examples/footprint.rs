// What a GGUF model will occupy on the backend, and whether the card has room.
//
// The same estimate `GgufModel::load` refuses on, printed rather than enforced,
// so a model that is close to the line can be looked at before it is run. Under
// `--features cuda` it also reports what the device has free.

use std::env;
use std::path::Path;

use anyhow::{Context, Result};
use phobos_gguf::{Decoder, Gguf};

const DEFAULT_MODEL: &str = "models/Qwen3.5-0.8B-Q8_0.gguf";

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn main() -> Result<()> {
    let path = env::args().nth(1).unwrap_or(DEFAULT_MODEL.to_string());
    let path = Path::new(&path);
    let file_bytes = std::fs::metadata(path)
        .with_context(|| format!("stat '{}'", path.display()))?
        .len() as usize;

    let gguf = Gguf::open(path)?;
    let decoder = Decoder::load(&gguf)?;
    let context_length = decoder.context_length();
    let footprint = decoder.footprint(context_length);

    println!("{}", path.display());
    println!("  architecture     {}", decoder.architecture());
    println!("  file             {:.2} GiB", gib(file_bytes));
    println!("  weights          {:.2} GiB", gib(footprint.weight_bytes));
    println!("    of which f32   {:.2} GiB", gib(footprint.dense_bytes));
    println!(
        "  kv cache         {} KiB per token, {:.2} GiB at the trained context of {context_length}",
        footprint.kv_bytes_per_token / (1 << 10),
        gib(footprint.kv_bytes_per_token * context_length),
    );

    #[cfg(feature = "cuda")]
    {
        use phobos_gguf::backend::Backend;
        let backend = phobos_gguf::backend::DeviceBackend::new()?;
        match backend.device_memory() {
            Some((free_bytes, total_bytes)) => println!(
                "  device           {:.2} GiB free of {:.2} GiB",
                gib(free_bytes),
                gib(total_bytes)
            ),
            None => println!("  device           unavailable"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("  device           not built (needs --features cuda)");

    Ok(())
}
