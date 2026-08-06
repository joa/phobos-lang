//! Byte-level BPE for ONNX exports.
//!
//! An ONNX file carries no vocabulary, so the tokenizer is a pair of files
//! sitting beside the model and loaded at run time.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use phobos_inference::bpe::{ByteBpe, END_OF_TURN, PreTokenizer, Specials};

/// The file-name pairs a byte-level BPE tokenizer ships under: the Hugging Face
/// naming first, then the original OpenAI GPT-2 one. Both hold the same two
/// things, a JSON token-to-id map and a ranked merge list.
const NAMINGS: &[(&str, &str)] = &[("vocab.json", "merges.txt"), ("encoder.json", "vocab.bpe")];

/// A byte-level BPE tokenizer read from a vocabulary and a merge list.
pub struct BpeTokenizer {
    bpe: ByteBpe,
    encoder: HashMap<String, i64>,
    decoder: HashMap<i64, String>,
    specials: Specials<i64>,
    /// The turn-ending tokens this vocabulary happens to carry, ascending.
    eog: Vec<i64>,
}

impl BpeTokenizer {
    pub fn load(dir: &Path) -> Result<BpeTokenizer> {
        let (vocab_path, merges_path) = Self::locate(dir)?;
        let vocab = std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("read {}", vocab_path.display()))?;
        let merges = std::fs::read_to_string(&merges_path)
            .with_context(|| format!("read {}", merges_path.display()))?;
        BpeTokenizer::new(&vocab, &merges)
            .with_context(|| format!("build the tokenizer in {}", dir.display()))
    }

    fn locate(dir: &Path) -> Result<(PathBuf, PathBuf)> {
        for (vocab, merges) in NAMINGS {
            let (vocab, merges) = (dir.join(vocab), dir.join(merges));
            if vocab.is_file() && merges.is_file() {
                return Ok((vocab, merges));
            }
        }
        let wanted: Vec<String> = NAMINGS
            .iter()
            .map(|(v, m)| format!("{v} and {m}"))
            .collect();
        bail!(
            "{} holds no tokenizer: expected {}. An ONNX export carries no \
             vocabulary, so the files have to sit beside the model or be \
             pointed at explicitly",
            dir.display(),
            wanted.join(", or ")
        )
    }

    /// Build from the contents of the two files: a JSON token-to-id object and
    /// a ranked merge list.
    pub fn new(vocab_json: &str, merges: &str) -> Result<BpeTokenizer> {
        let encoder: HashMap<String, i64> = serde_json::from_str(vocab_json)
            .context("parse the vocabulary as a token-to-id map")?;
        if encoder.is_empty() {
            bail!("the vocabulary is empty");
        }
        let merges = ByteBpe::parse_merges(merges)?;
        if merges.is_empty() {
            bail!("the merge list is empty");
        }

        let decoder = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        // A byte-level vocabulary marks nothing as special, so the turn-enders
        // it happens to contain are all we can go on.
        let markers: Vec<(String, i64)> = END_OF_TURN
            .iter()
            .filter_map(|&m| encoder.get(m).map(|&id| (m.to_string(), id)))
            .collect();
        let mut eog: Vec<i64> = markers.iter().map(|&(_, id)| id).collect();
        eog.sort_unstable();

        Ok(BpeTokenizer {
            // Nothing in the files names a pre-tokenizer; every export we have
            // seen wants GPT-2's.
            bpe: ByteBpe::new(PreTokenizer::Gpt2, merges)?,
            encoder,
            decoder,
            specials: Specials::new(markers),
            eog,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        self.bpe.encode(text, &self.specials, |symbol| {
            self.encoder.get(symbol).copied()
        })
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Decode ids to their raw byte stream. A token can hold a partial UTF-8
    /// sequence, so a streaming caller must buffer these and emit only complete
    /// characters.
    pub fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        self.bpe.decode_bytes(
            ids.iter()
                .filter_map(|id| self.decoder.get(id).map(String::as_str)),
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Every turn-ending token, ascending.
    pub fn eog(&self) -> &[i64] {
        &self.eog
    }
}

impl phobos_inference::Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        BpeTokenizer::encode(self, text)
    }

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        BpeTokenizer::decode_bytes(self, ids)
    }

    fn is_eog(&self, id: i64) -> bool {
        self.eog.binary_search(&id).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPT-2 tokenizer beside the zoo export, when it is there. The model
    /// directories are not in the repository, so a checkout without one skips
    /// the tests that need a real vocabulary instead of failing them. A pair
    /// that is present but unreadable still fails, loudly.
    ///
    /// See `models/README.md` for how the files get there.
    fn gpt2() -> Option<BpeTokenizer> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/GPT2");
        if BpeTokenizer::locate(&dir).is_err() {
            eprintln!("skipping: {} holds no tokenizer", dir.display());
            return None;
        }
        Some(BpeTokenizer::load(&dir).expect("load the GPT-2 fixture"))
    }

    #[test]
    fn matches_reference_gpt2_encoding() {
        let Some(tok) = gpt2() else { return };
        // The ids GPT2Tokenizer produces for the model's export prompt.
        let ids = tok
            .encode("Here is some text to encode Hello World")
            .unwrap();
        assert_eq!(ids, vec![4342, 318, 617, 2420, 284, 37773, 18435, 2159]);
    }

    #[test]
    fn round_trips_text() {
        let Some(tok) = gpt2() else { return };
        for text in [
            "The color of the sky is",
            "  leading spaces\tand tabs",
            "unicode: cafe\u{301}",
        ] {
            let ids = tok.encode(text).unwrap();
            assert_eq!(tok.decode(&ids), text);
        }
    }

    #[test]
    fn leading_space_is_significant() {
        let Some(tok) = gpt2() else { return };
        assert_ne!(tok.encode("hello").unwrap(), tok.encode(" hello").unwrap());
    }

    #[test]
    fn finds_the_end_of_text_marker() {
        let Some(tok) = gpt2() else { return };
        let eot = tok.encode("<|endoftext|>").unwrap();
        assert_eq!(eot.len(), 1, "the marker matches literally, not by merging");
        assert_eq!(tok.eog(), eot.as_slice());
    }

    #[test]
    fn names_the_files_it_wanted() {
        let err = match BpeTokenizer::load(Path::new("does-not-exist")) {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("a missing directory holds no tokenizer"),
        };
        assert!(err.contains("vocab.json and merges.txt"), "got: {err}");
        assert!(err.contains("encoder.json and vocab.bpe"), "got: {err}");
    }
}
