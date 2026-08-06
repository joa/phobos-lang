use std::collections::HashMap;

use anyhow::{Context, Result};
use fancy_regex::Regex;

const ENCODER_JSON: &str = include_str!("../assets/gpt2-encoder.json");
const VOCAB_BPE: &str = include_str!("../assets/gpt2-vocab.bpe");

const PAT: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// The GPT-2 byte-level BPE tokenizer.
pub struct Tokenizer {
    encoder: HashMap<String, i64>,
    decoder: HashMap<i64, String>,
    /// Adjacent symbol pair to merge rank, lowest first.
    bpe_ranks: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pat: Regex,
}

impl Tokenizer {
    pub fn gpt2() -> Result<Tokenizer> {
        let encoder: HashMap<String, i64> =
            serde_json::from_str(ENCODER_JSON).context("parse encoder.json")?;
        let decoder = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut bpe_ranks = HashMap::new();
        for (rank, line) in VOCAB_BPE.lines().enumerate().skip(1) {
            // The first line is a `#version` header, and blank lines end it.
            if line.is_empty() {
                continue;
            }
            let (a, b) = line.split_once(' ').context("malformed merge line")?;
            bpe_ranks.insert((a.to_string(), b.to_string()), rank as u32 - 1);
        }

        let (byte_to_char, char_to_byte) = byte_char_maps();

        Ok(Tokenizer {
            encoder,
            decoder,
            bpe_ranks,
            byte_to_char,
            char_to_byte,
            pat: Regex::new(PAT).context("compile GPT-2 regex")?,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let mut ids = Vec::new();
        for piece in self.pat.find_iter(text) {
            let piece = piece.context("regex match")?.as_str();
            let mapped: String = piece
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();
            for token in self.bpe(&mapped) {
                let id = *self
                    .encoder
                    .get(&token)
                    .with_context(|| format!("token '{token}' not in vocabulary"))?;
                ids.push(id);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Decode ids to their raw byte stream. A token can hold a partial UTF-8
    /// sequence, so a streaming caller must buffer these and emit only complete
    /// characters.
    pub fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        ids.iter()
            .filter_map(|id| self.decoder.get(id))
            .flat_map(|tok| tok.chars())
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect()
    }

    /// The `<|endoftext|>` id, GPT-2's end-of-sequence marker.
    pub fn eos(&self) -> Option<i64> {
        self.encoder.get("<|endoftext|>").copied()
    }

    /// Merge one pre-token, a string of mapped chars, into BPE symbols.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() < 2 {
            return word;
        }
        loop {
            let best = word
                .windows(2)
                .filter_map(|w| {
                    let pair = (w[0].clone(), w[1].clone());
                    self.bpe_ranks.get(&pair).map(|&r| (r, pair))
                })
                .min_by_key(|(r, _)| *r);
            let Some((_, (first, second))) = best else {
                break;
            };

            // Every occurrence of that pair, in one left-to-right pass.
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() < 2 {
                break;
            }
        }
        word
    }
}

/// The reversible byte-to-unicode table: printable bytes map to themselves,
/// the rest to a contiguous run above 0xFF, so every byte is a visible char.
fn byte_char_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut printable: Vec<u8> = Vec::new();
    printable.extend(b'!'..=b'~');
    printable.extend(0xA1u8..=0xAC);
    printable.extend(0xAEu8..=0xFF);

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::new();
    let mut extra = 0u32;
    for b in 0u16..256 {
        let ch = if printable.contains(&(b as u8)) {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_gpt2_encoding() {
        let tok = Tokenizer::gpt2().unwrap();
        // The ids GPT2Tokenizer produces for the model's export prompt.
        let ids = tok
            .encode("Here is some text to encode Hello World")
            .unwrap();
        assert_eq!(ids, vec![4342, 318, 617, 2420, 284, 37773, 18435, 2159]);
    }

    #[test]
    fn round_trips_text() {
        let tok = Tokenizer::gpt2().unwrap();
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
        let tok = Tokenizer::gpt2().unwrap();
        assert_ne!(tok.encode("hello").unwrap(), tok.encode(" hello").unwrap());
    }
}
