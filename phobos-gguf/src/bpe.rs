use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use fancy_regex::Regex;

use crate::vocab::Vocab;

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
/// `tokenizer.ggml.eos_token_id` names one token, which on some models is not
/// the one the chat template closes turns with.
///
/// I've seen MiniCPM5 declares `</s>` but ends every turn with `<|im_end|>`
/// so we simply brute-force this.
const END_OF_TURN: &[&str] = &[
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

/// Which pre-tokenizer regex a vocabulary wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    Gpt2,
    Qwen,
    Llama3,
}

impl PreTokenizer {
    /// Resolve `tokenizer.ggml.pre`. Unknown names fall back to GPT-2, as
    /// llama.cpp does, so an unrecognized model still encodes.
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

    fn ignores_merges(self) -> bool {
        matches!(self, PreTokenizer::Llama3)
    }
}

/// A byte-level BPE encoder built from a GGUF vocabulary.
pub struct Bpe {
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    ranks: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pattern: Regex,
    ignore_merges: bool,
    /// Literal-match tokens, longest first so `<|im_start|>` wins over any
    /// shorter marker sharing its prefix.
    specials: Vec<(String, u32)>,
    eos: Option<u32>,
    bos: Option<u32>,
    eog: Vec<u32>, // Every token that ends a turn, the declared EOS included.
}

impl Bpe {
    pub fn from_vocab(vocab: &Vocab) -> Result<Bpe> {
        if vocab.model != "gpt2" {
            bail!(
                "tokenizer model '{}' is not byte-level BPE; only 'gpt2' vocabularies are supported",
                vocab.model
            );
        }
        if vocab.merges.is_empty() {
            bail!("vocabulary carries no BPE merges");
        }

        let encoder: HashMap<String, u32> = vocab
            .tokens
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();

        let ranks = vocab
            .merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.clone(), b.clone()), rank as u32))
            .collect();

        let mut specials: Vec<(String, u32)> = vocab
            .special_tokens()
            .map(|(id, text)| (text.to_string(), id))
            .collect();
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));

        let mut eog: Vec<u32> = vocab
            .special_tokens()
            .filter(|(_, text)| END_OF_TURN.contains(text))
            .map(|(id, _)| id)
            .chain(vocab.eos)
            .collect();
        eog.sort_unstable();
        eog.dedup();

        let (byte_to_char, char_to_byte) = byte_char_maps();
        let pre = PreTokenizer::from_name(vocab.pre.as_deref());

        Ok(Bpe {
            encoder,
            decoder: vocab.tokens.clone(),
            ranks,
            byte_to_char,
            char_to_byte,
            pattern: Regex::new(pre.pattern()).context("compile pre-tokenizer pattern")?,
            ignore_merges: pre.ignores_merges(),
            specials,
            eos: vocab.eos,
            bos: vocab.bos,
            eog,
        })
    }

    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    /// Whether this token ends the turn.
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.binary_search(&id).is_ok()
    }

    /// Every turn-ending token, ascending.
    pub fn eog(&self) -> &[u32] {
        &self.eog
    }

    /// The vocabulary's text for one token, as stored.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.decoder.get(id as usize).map(String::as_str)
    }

    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    /// Encode text, matching special tokens literally and BPE-merging the rest.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            match self.next_special(rest) {
                Some((at, len, id)) => {
                    self.encode_ordinary(&rest[..at], &mut ids)?;
                    ids.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_ordinary(rest, &mut ids)?;
                    break;
                }
            }
        }

        Ok(ids)
    }

    /// The earliest special token in `text`, as (byte offset, length, id).
    fn next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        self.specials
            .iter()
            .filter_map(|(marker, id)| text.find(marker.as_str()).map(|at| (at, marker.len(), *id)))
            .min_by_key(|&(at, len, _)| (at, usize::MAX - len)) // longest wins
    }

    fn encode_ordinary(&self, text: &str, ids: &mut Vec<u32>) -> Result<()> {
        for piece in self.pattern.find_iter(text) {
            let piece = piece.context("pre-tokenizer match")?.as_str();
            let mapped: String = piece
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();

            if self.ignore_merges
                && let Some(&id) = self.encoder.get(&mapped)
            {
                ids.push(id);
                continue;
            }

            for symbol in self.merge(&mapped) {
                let id = self
                    .encoder
                    .get(&symbol)
                    .with_context(|| format!("token {symbol:?} is not in the vocabulary"))?;
                ids.push(*id);
            }
        }

        Ok(())
    }

    /// Decode ids to their raw byte stream. A token can end mid-character, so
    /// a streaming caller must buffer these and emit complete UTF-8 only.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|&id| self.decoder.get(id as usize))
            .flat_map(|token| token.chars())
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Merge one pre-token into BPE symbols, lowest-ranked adjacent pair first.
    fn merge(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() < 2 {
            return word;
        }

        loop {
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

/// The reversible byte-to-unicode table: printable bytes map to themselves, the
/// rest to a contiguous run above 0xFF.
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
    use crate::vocab::TokenType;

    /// Every mapped byte, plus a couple of merges and one control token.
    fn toy_vocab() -> Vocab {
        let (byte_to_char, _) = byte_char_maps();
        let mut tokens: Vec<String> = (0..256).map(|b| byte_to_char[b].to_string()).collect();
        let mut token_types = vec![TokenType::Normal; tokens.len()];

        tokens.push("hi".into());
        token_types.push(TokenType::Normal);
        tokens.push("\u{0120}th".into()); // a leading space plus "th"
        token_types.push(TokenType::Normal);
        tokens.push("<|im_end|>".into());
        token_types.push(TokenType::Control);
        let eos = (tokens.len() - 1) as u32;

        Vocab {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens,
            token_types,
            scores: Vec::new(),
            merges: vec![("h".into(), "i".into()), ("\u{0120}".into(), "th".into())],
            bos: None,
            eos: Some(eos),
            pad: None,
            unk: None,
            sep: None,
            add_bos: false,
            add_eos: false,
            chat_template: None,
        }
    }

    #[test]
    fn merges_by_rank_and_round_trips() {
        let bpe = Bpe::from_vocab(&toy_vocab()).unwrap();
        for text in [
            "hi there",
            "plain",
            "  spaces\tand\ttabs",
            "caf\u{e9} \u{4e2d}\u{6587}",
        ] {
            let ids = bpe.encode(text).unwrap();
            assert_eq!(bpe.decode(&ids), text, "round trip of {text:?}");
        }
        // "hi" is a single merged token, not two byte tokens.
        assert_eq!(bpe.encode("hi").unwrap().len(), 1);
    }

    #[test]
    fn matches_control_tokens_literally() {
        let bpe = Bpe::from_vocab(&toy_vocab()).unwrap();
        let ids = bpe.encode("hi<|im_end|>").unwrap();
        assert_eq!(ids.last().copied(), bpe.eos());
        assert_eq!(ids.len(), 2);
        assert_eq!(bpe.decode(&ids), "hi<|im_end|>");
    }

    #[test]
    fn qwen_splits_digits_singly() {
        let bpe = Bpe::from_vocab(&toy_vocab()).unwrap();
        // The Qwen pattern emits one token per digit, so 123 is three pieces.
        assert_eq!(bpe.encode("123").unwrap().len(), 3);
    }

    #[test]
    fn rejects_non_bpe_vocabularies() {
        let mut vocab = toy_vocab();
        vocab.model = "llama".into();
        assert!(Bpe::from_vocab(&vocab).is_err());
    }
}
