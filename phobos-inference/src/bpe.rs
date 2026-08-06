//! Byte-level BPE, the part that is the same whatever supplied the vocabulary.
//!
//! A GGUF file carries its tokens and merges as metadata and an ONNX export
//! carries none at all, so the two front ends build their vocabularies from
//! different places. What happens after that is identical: split the text on a
//! pre-tokenizer pattern, map each byte to a visible character, and merge
//! adjacent pairs by rank. That much lives here.

use std::collections::HashMap;

use anyhow::{Context, Result};
use fancy_regex::Regex;

/// The GPT-2 pre-split: contractions, letter runs, digit runs, symbol runs, and
/// whitespace, each optionally led by a single space.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// The Qwen2-family pre-split: GPT-2 with case-insensitive contractions, digits
/// split one at a time, and runs of newlines kept together.
const QWEN_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The Llama-3 pre-split used by the `llama-bpe` vocabularies: the Qwen pattern
/// with digits taken up to three at a time.
const LLAMA3_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The markers that end an assistant turn, by name.
///
/// A vocabulary that declares an end-of-sequence token names one, which on some
/// models is not the one the chat template closes turns with: MiniCPM5 declares
/// `</s>` but ends every turn with `<|im_end|>`. An ONNX export declares
/// nothing at all. So we simply brute-force this.
pub const END_OF_TURN: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end|>",
    "<|return|>",
    "<end_of_turn>",
    "</s>",
];

/// Tokens matched literally rather than merged, control markers among them.
pub struct Specials<T>(Vec<(String, T)>);

impl<T: Copy> Specials<T> {
    /// Longest first, so `<|im_start|>` wins over any shorter marker sharing
    /// its prefix.
    pub fn new(entries: impl IntoIterator<Item = (String, T)>) -> Specials<T> {
        let mut entries: Vec<(String, T)> = entries.into_iter().collect();
        entries.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));
        Specials(entries)
    }

    pub fn none() -> Specials<T> {
        Specials(Vec::new())
    }

    /// The earliest special in `text`, as (byte offset, length, id).
    fn next_in(&self, text: &str) -> Option<(usize, usize, T)> {
        self.0
            .iter()
            .filter_map(|(marker, id)| text.find(marker.as_str()).map(|at| (at, marker.len(), *id)))
            .min_by_key(|&(at, len, _)| (at, usize::MAX - len)) // longest wins
    }
}

/// Which pre-tokenizer regex a vocabulary wants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreTokenizer {
    #[default]
    Gpt2,
    Qwen,
    Llama3,
}

impl PreTokenizer {
    /// Resolve a name such as GGUF's `tokenizer.ggml.pre`. Unknown names fall
    /// back to GPT-2, as llama.cpp does, so an unrecognized model still encodes.
    pub fn from_name(name: Option<&str>) -> PreTokenizer {
        match name {
            Some(n) if n.starts_with("qwen") => PreTokenizer::Qwen,
            Some("llama3" | "llama-v3" | "llama-bpe") => PreTokenizer::Llama3,
            _ => PreTokenizer::Gpt2,
        }
    }

    fn pattern(self) -> &'static str {
        match self {
            PreTokenizer::Gpt2 => GPT2_PATTERN,
            PreTokenizer::Qwen => QWEN_PATTERN,
            PreTokenizer::Llama3 => LLAMA3_PATTERN,
        }
    }

    /// Whether a piece found in the vocabulary whole skips the merge loop.
    fn ignores_merges(self) -> bool {
        matches!(self, PreTokenizer::Llama3)
    }
}

/// A pre-tokenizer, the byte-to-character mapping, and the merge ranks.
///
/// It holds no vocabulary: a caller supplies the symbol-to-id lookup, which is
/// where the two front ends differ.
pub struct ByteBpe {
    ranks: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pattern: Regex,
    ignore_merges: bool,
}

impl ByteBpe {
    pub fn new(
        pre: PreTokenizer,
        merges: impl IntoIterator<Item = (String, String)>,
    ) -> Result<ByteBpe> {
        let (byte_to_char, char_to_byte) = byte_char_maps();
        Ok(ByteBpe {
            ranks: merges
                .into_iter()
                .enumerate()
                .map(|(rank, pair)| (pair, rank as u32))
                .collect(),
            byte_to_char,
            char_to_byte,
            pattern: Regex::new(pre.pattern()).context("compile pre-tokenizer pattern")?,
            ignore_merges: pre.ignores_merges(),
        })
    }

    /// The merge list of a `merges.txt` or `vocab.bpe` file, best rank first.
    ///
    /// One `a b` pair a line, led by a `#version` header that carries no pair.
    pub fn parse_merges(text: &str) -> Result<Vec<(String, String)>> {
        text.lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let (a, b) = line
                    .split_once(' ')
                    .with_context(|| format!("merge line {line:?} is not a pair"))?;
                Ok((a.to_string(), b.to_string()))
            })
            .collect()
    }

    /// The visible character byte `b` maps to.
    ///
    /// A byte-level vocabulary's single-character tokens are exactly these 256.
    pub fn byte_char(&self, b: u8) -> char {
        self.byte_to_char[b as usize]
    }

    /// Encode `text`, matching `specials` literally and BPE-merging the rest.
    ///
    /// `lookup` resolves a symbol against the caller's vocabulary, which is the
    /// one part a front end supplies itself.
    pub fn encode<T: Copy>(
        &self,
        text: &str,
        specials: &Specials<T>,
        lookup: impl Fn(&str) -> Option<T>,
    ) -> Result<Vec<T>> {
        let mut ids = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            match specials.next_in(rest) {
                Some((at, len, id)) => {
                    self.merge_into(&rest[..at], &lookup, &mut ids)?;
                    ids.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.merge_into(rest, &lookup, &mut ids)?;
                    break;
                }
            }
        }

        Ok(ids)
    }

    /// Split `text` and merge each piece, appending an id per symbol.
    fn merge_into<T>(
        &self,
        text: &str,
        lookup: impl Fn(&str) -> Option<T>,
        out: &mut Vec<T>,
    ) -> Result<()> {
        for piece in self.pattern.find_iter(text) {
            let piece = piece.context("pre-tokenizer match")?.as_str();
            let mapped: String = piece
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();

            if self.ignore_merges
                && let Some(id) = lookup(&mapped)
            {
                out.push(id);
                continue;
            }

            for symbol in self.merge(&mapped) {
                let id = lookup(&symbol)
                    .with_context(|| format!("token {symbol:?} is not in the vocabulary"))?;
                out.push(id);
            }
        }
        Ok(())
    }

    /// The raw byte stream behind a run of vocabulary tokens.
    ///
    /// A token can end mid-character, so a streaming caller must buffer these
    /// and emit complete UTF-8 only.
    pub fn decode_bytes<'a>(&self, tokens: impl IntoIterator<Item = &'a str>) -> Vec<u8> {
        tokens
            .into_iter()
            .flat_map(str::chars)
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect()
    }

    /// Merge one pre-token, a string of mapped characters, into BPE symbols,
    /// lowest-ranked adjacent pair first.
    fn merge(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|w| {
                    let pair = (w[0].clone(), w[1].clone());
                    self.ranks.get(&pair).map(|&rank| (rank, pair))
                })
                .min_by_key(|(rank, _)| *rank);

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
        }

        word
    }
}

/// The reversible byte-to-unicode table: printable bytes map to themselves, the
/// rest to a contiguous run above 0xFF, so every byte is a visible character.
fn byte_char_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut printable = [false; 256];
    for b in (b'!'..=b'~').chain(0xA1u8..=0xAC).chain(0xAEu8..=0xFF) {
        printable[b as usize] = true;
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::new();
    let mut extra = 0u32;
    for b in 0usize..256 {
        let ch = if printable[b] {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };

        byte_to_char[b] = ch;
        char_to_byte.insert(ch, b as u8);
    }

    (byte_to_char, char_to_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mapped byte as its own token, plus two merged ones.
    fn toy(pre: PreTokenizer) -> (ByteBpe, HashMap<String, u32>) {
        let (byte_to_char, _) = byte_char_maps();
        let mut vocab: HashMap<String, u32> = (0..256)
            .map(|b| (byte_to_char[b].to_string(), b as u32))
            .collect();
        vocab.insert("hi".into(), 256);
        vocab.insert("\u{0120}th".into(), 257); // a leading space plus "th"
        vocab.insert("12".into(), 258);

        let merges = vec![
            ("h".to_string(), "i".to_string()),
            ("\u{0120}".to_string(), "th".to_string()),
            ("1".to_string(), "2".to_string()),
        ];
        (ByteBpe::new(pre, merges).unwrap(), vocab)
    }

    fn encode(bpe: &ByteBpe, vocab: &HashMap<String, u32>, text: &str) -> Vec<u32> {
        bpe.encode(text, &Specials::none(), |s| vocab.get(s).copied())
            .unwrap()
    }

    #[test]
    fn merges_by_rank() {
        let (bpe, vocab) = toy(PreTokenizer::Gpt2);
        assert_eq!(encode(&bpe, &vocab, "hi"), vec![256]);
        // "h" and "i" alone stay two byte tokens.
        assert_eq!(encode(&bpe, &vocab, "h i").len(), 3);
    }

    #[test]
    fn round_trips_through_the_byte_map() {
        let (bpe, vocab) = toy(PreTokenizer::Gpt2);
        let by_id: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        for text in [
            "hi there",
            "  spaces\tand\ttabs",
            "caf\u{e9} \u{4e2d}\u{6587}",
        ] {
            let ids = encode(&bpe, &vocab, text);
            let tokens: Vec<&str> = ids.iter().map(|id| by_id[id].as_str()).collect();
            assert_eq!(
                String::from_utf8(bpe.decode_bytes(tokens)).unwrap(),
                text,
                "round trip of {text:?}"
            );
        }
    }

    #[test]
    fn qwen_splits_digits_singly() {
        // GPT-2 takes the run whole, so the 1+2 merge applies and "123" is
        // ["12", "3"]. Qwen hands the merge loop one digit at a time.
        let (bpe, vocab) = toy(PreTokenizer::Gpt2);
        assert_eq!(encode(&bpe, &vocab, "123"), vec![258, b'3' as u32]);
        let (bpe, vocab) = toy(PreTokenizer::Qwen);
        assert_eq!(encode(&bpe, &vocab, "123").len(), 3);
    }

    #[test]
    fn reports_a_symbol_the_vocabulary_lacks() {
        let (bpe, _) = toy(PreTokenizer::Gpt2);
        let err = bpe
            .encode("x", &Specials::none(), |_| None::<u32>)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not in the vocabulary"), "got: {err}");
    }

    #[test]
    fn matches_a_special_literally_and_longest_first() {
        let (bpe, mut vocab) = toy(PreTokenizer::Gpt2);
        // The merges never build either marker, so only a literal match finds
        // them, and the longer one has to win where both could start.
        vocab.insert("<|end|>".into(), 300);
        vocab.insert("<|end_of_text|>".into(), 301);
        let specials = Specials::new([
            ("<|end|>".to_string(), 300),
            ("<|end_of_text|>".into(), 301),
        ]);

        let encode = |text| {
            bpe.encode(text, &specials, |s| vocab.get(s).copied())
                .unwrap()
        };
        assert_eq!(encode("hi<|end_of_text|>"), vec![256, 301]);
        assert_eq!(encode("hi<|end|>"), vec![256, 300]);
    }

    #[test]
    fn parses_a_merge_file_and_skips_its_header() {
        let merges = ByteBpe::parse_merges("#version: 0.2\na b\nc d\n\n").unwrap();
        assert_eq!(
            merges,
            vec![
                ("a".to_string(), "b".to_string()),
                ("c".to_string(), "d".to_string())
            ]
        );
    }

    #[test]
    fn rejects_a_merge_line_that_is_not_a_pair() {
        assert!(ByteBpe::parse_merges("#version: 0.2\nlonely\n").is_err());
    }
}
