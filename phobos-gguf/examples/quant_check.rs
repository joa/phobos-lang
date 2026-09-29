// A quantized file against a wider one of the same model, tensor by tensor:
//
//   cargo run --release -p phobos-gguf --example quant_check -- WIDE.gguf NARROW.gguf
//
// Each narrow tensor should be the wide one plus quantization error. A
// relative error of a few percent means the format decodes correctly; one
// near 1.0 means the decoder reads the wrong bits. Make the pair with:
//
//   llama-quantize --allow-requantize WIDE.gguf NARROW.gguf Q4_K_M
use std::path::Path;

use anyhow::{Result, bail};
use phobos_gguf::Gguf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(wide_path), Some(narrow_path)) = (args.next(), args.next()) else {
        bail!("usage: quant_check WIDE.gguf NARROW.gguf");
    };
    let wide = Gguf::open(Path::new(&wide_path))?;
    let narrow = Gguf::open(Path::new(&narrow_path))?;

    // Worst error per format, not per tensor, since the question is whether
    // each format's decoder works.
    let mut worst: Vec<(&'static str, f64, String)> = Vec::new();
    let mut checked = 0usize;

    for info in narrow.tensors() {
        let Some(reference) = wide.tensor(&info.name) else {
            continue;
        };
        if !info.ggml_type.is_dequantizable() || !reference.ggml_type.is_dequantizable() {
            continue;
        }
        let want = wide.dequantize(&info.name)?;
        let got = narrow.dequantize(&info.name)?;
        if want.len() != got.len() {
            bail!(
                "'{}' has {} elements against {}",
                info.name,
                got.len(),
                want.len()
            );
        }

        // Relative RMS error. It is the usual measure, and it does not blow
        // up on the near-zero weights that make up most of a tensor.
        let (mut error, mut scale) = (0.0f64, 0.0f64);
        for (&w, &g) in want.iter().zip(&got) {
            error += (f64::from(w) - f64::from(g)).powi(2);
            scale += f64::from(w).powi(2);
        }
        let rel = (error / scale.max(f64::MIN_POSITIVE)).sqrt();

        let name = info.ggml_type.name();
        match worst.iter_mut().find(|(f, _, _)| *f == name) {
            Some(entry) if rel > entry.1 => *entry = (name, rel, info.name.clone()),
            Some(_) => {}
            None => worst.push((name, rel, info.name.clone())),
        }
        checked += 1;
    }

    if checked == 0 {
        bail!("the two files share no dequantizable tensor");
    }
    worst.sort_by(|a, b| a.0.cmp(b.0));

    let mut failures = 0;
    for (format, rel, tensor) in &worst {
        // A four-bit format costs a few percent. A wrong decoder correlates
        // with nothing and lands near 1.0. The 10% bar sits well between.
        let verdict = if *rel < 0.10 { "ok  " } else { "WRONG" };
        failures += u32::from(*rel >= 0.10);
        println!("{verdict} {format:<6} worst rel rms {rel:9.3e}  {tensor}");
    }
    println!("\n{checked} tensors, {} formats", worst.len());
    if failures > 0 {
        bail!("{failures} format(s) disagree with the reference file");
    }
    Ok(())
}
