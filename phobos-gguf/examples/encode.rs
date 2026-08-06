// Encode text with a GGUF file's own tokenizer:
//
//   cargo run -p phobos-gguf --example encode -- MODEL.gguf "some text"
//
// A JSON array of strings instead of one argument prints a JSON array of token
// id arrays, for diffing against llama.cpp's /tokenize endpoint.

use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_gguf::{Bpe, Gguf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(text)) = (args.next().map(PathBuf::from), args.next()) else {
        bail!("usage: encode MODEL.gguf TEXT");
    };

    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;

    // A leading bracket means a JSON array of prompts rather than one prompt.
    let batch = text.trim_start().starts_with('[');
    let texts: Vec<String> = if batch {
        serde_json::from_str(&text)?
    } else {
        vec![text]
    };

    let encoded: Vec<Vec<u32>> = texts.iter().map(|t| bpe.encode(t)).collect::<Result<_>>()?;
    if batch {
        println!("{}", serde_json::to_string(&encoded)?);
        return Ok(());
    }
    for id in &encoded[0] {
        println!("{id:>7}  {:?}", bpe.decode(&[*id]));
    }
    Ok(())
}
