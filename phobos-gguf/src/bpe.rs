use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_inference::bpe::{ByteBpe, END_OF_TURN, PreTokenizer, Specials};

use crate::vocab::Vocab;

/// A byte-level BPE encoder built from a GGUF vocabulary.
///
/// The merging itself is [`ByteBpe`]; what a GGUF file adds is the vocabulary,
/// its control tokens, and which of them end a turn.
pub struct Bpe {
    bpe: ByteBpe,
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    specials: Specials<u32>,
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

        let specials = Specials::new(
            vocab
                .special_tokens()
                .map(|(id, text)| (text.to_string(), id)),
        );

        let mut eog: Vec<u32> = vocab
            .special_tokens()
            .filter(|(_, text)| END_OF_TURN.contains(text))
            .map(|(id, _)| id)
            .chain(vocab.eos)
            .collect();
        eog.sort_unstable();
        eog.dedup();

        let pre = PreTokenizer::from_name(vocab.pre.as_deref());

        Ok(Bpe {
            bpe: ByteBpe::new(pre, vocab.merges.iter().cloned())?,
            encoder,
            decoder: vocab.tokens.clone(),
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
        self.bpe.encode(text, &self.specials, |symbol| {
            self.encoder.get(symbol).copied()
        })
    }

    /// Decode ids to their raw byte stream. A token can end mid-character, so
    /// a streaming caller must buffer these and emit complete UTF-8 only.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        self.bpe.decode_bytes(
            ids.iter()
                .filter_map(|&id| self.decoder.get(id as usize).map(String::as_str)),
        )
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::TokenType;

    /// Every mapped byte, plus a couple of merges and one control token.
    fn toy_vocab() -> Vocab {
        // The byte tokens are the first 256 of any byte-level vocabulary, in
        // the order the shared byte-to-character map assigns them.
        let bytes = ByteBpe::new(PreTokenizer::Gpt2, []).unwrap();
        let mut tokens: Vec<String> = (0u8..=255)
            .map(|b| bytes.byte_char(b).to_string())
            .collect();
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
