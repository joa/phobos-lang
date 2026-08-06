use anyhow::{Context, Result, bail};

use crate::meta::Metadata;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_code(code: i64) -> TokenType {
        match code {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }

    /// A marker like `<|im_start|>`, matched literally before any BPE splitting.
    pub fn is_special(self) -> bool {
        matches!(self, TokenType::Control | TokenType::UserDefined)
    }
}

#[derive(Clone, Debug)]
pub struct Vocab {
    /// Tokenizer family: "gpt2" (byte-level BPE), "llama" (SPM), "bert", ...
    pub model: String,
    /// Pre-tokenizer variant, e.g. "qwen35". Selects the text-splitting regex.
    pub pre: Option<String>,
    pub tokens: Vec<String>,
    /// Per-token classification, parallel to `tokens`. Empty when the file
    /// omits `tokenizer.ggml.token_type`.
    pub token_types: Vec<TokenType>,
    /// Scores used by SentencePiece models; empty for BPE.
    pub scores: Vec<f32>,
    /// BPE merge rules in priority order.
    pub merges: Vec<(String, String)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub sep: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    /// The Jinja chat template, when the file ships one.
    pub chat_template: Option<String>,
}

impl Vocab {
    pub fn from_metadata(metadata: &Metadata) -> Result<Vocab> {
        let model = metadata
            .string("tokenizer.ggml.model")
            .context("GGUF file carries no tokenizer (tokenizer.ggml.model is missing)")?
            .to_string();
        let tokens = metadata.strings("tokenizer.ggml.tokens")?.to_vec();

        let token_types = match metadata.get("tokenizer.ggml.token_type") {
            Some(_) => {
                let codes = metadata.ints("tokenizer.ggml.token_type")?;
                if codes.len() != tokens.len() {
                    bail!(
                        "tokenizer.ggml.token_type has {} entries but there are {} tokens",
                        codes.len(),
                        tokens.len()
                    );
                }
                codes.into_iter().map(TokenType::from_code).collect()
            }
            None => Vec::new(),
        };

        let scores = match metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
        {
            Some(crate::meta::Array::F32(v)) => v.clone(),
            _ => Vec::new(),
        };

        let merges = match metadata.get("tokenizer.ggml.merges") {
            Some(_) => metadata
                .strings("tokenizer.ggml.merges")?
                .iter()
                .map(|line| split_merge(line))
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };

        let token_id = |key: &str| {
            metadata
                .get(key)
                .and_then(|v| v.as_int())
                .and_then(|v| u32::try_from(v).ok())
        };

        Ok(Vocab {
            model,
            pre: metadata
                .get("tokenizer.ggml.pre")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            tokens,
            token_types,
            scores,
            merges,
            bos: token_id("tokenizer.ggml.bos_token_id"),
            eos: token_id("tokenizer.ggml.eos_token_id"),
            pad: token_id("tokenizer.ggml.padding_token_id"),
            unk: token_id("tokenizer.ggml.unknown_token_id"),
            sep: token_id("tokenizer.ggml.seperator_token_id"),
            add_bos: metadata
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            add_eos: metadata
                .get("tokenizer.ggml.add_eos_token")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            chat_template: metadata
                .get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn token_type(&self, id: usize) -> TokenType {
        self.token_types
            .get(id)
            .copied()
            .unwrap_or(TokenType::Normal)
    }

    /// Ids of tokens that must be matched literally, with their text.
    pub fn special_tokens(&self) -> impl Iterator<Item = (u32, &str)> {
        self.tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| self.token_type(*id).is_special())
            .map(|(id, text)| (id as u32, text.as_str()))
    }
}

fn split_merge(line: &str) -> Result<(String, String)> {
    let (left, right) = line
        .split_once(' ')
        .with_context(|| format!("malformed BPE merge entry {line:?}"))?;
    Ok((left.to_string(), right.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::{Array, Value};

    fn metadata_with(entries: Vec<(&str, Value)>) -> Metadata {
        let mut m = Metadata::default();
        for (k, v) in entries {
            m.insert(k.to_string(), v);
        }
        m
    }

    #[test]
    fn reads_a_bpe_vocabulary() {
        let m = metadata_with(vec![
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            ("tokenizer.ggml.pre", Value::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(Array::String(vec![
                    "a".into(),
                    "b".into(),
                    "<|im_end|>".into(),
                ])),
            ),
            (
                "tokenizer.ggml.token_type",
                Value::Array(Array::I32(vec![1, 1, 3])),
            ),
            (
                "tokenizer.ggml.merges",
                Value::Array(Array::String(vec!["a b".into()])),
            ),
            ("tokenizer.ggml.eos_token_id", Value::U32(2)),
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ]);

        let vocab = Vocab::from_metadata(&m).unwrap();
        assert_eq!(vocab.model, "gpt2");
        assert_eq!(vocab.pre.as_deref(), Some("qwen35"));
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.merges, vec![("a".to_string(), "b".to_string())]);
        assert_eq!(vocab.eos, Some(2));
        assert!(!vocab.add_bos);
        assert_eq!(vocab.token_type(2), TokenType::Control);
        assert_eq!(
            vocab.special_tokens().collect::<Vec<_>>(),
            vec![(2, "<|im_end|>")]
        );
    }

    /// MiniCPM5's shape: the declared EOS is not what its template ends turns
    /// with, so a generation stopping only on the declared id never stops.
    #[test]
    fn a_turn_ender_need_not_be_the_declared_eos() {
        let m = metadata_with(vec![
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(Array::String(vec![
                    "<s>".into(),
                    "</s>".into(),
                    "hi".into(),
                    "<|im_end|>".into(),
                ])),
            ),
            (
                "tokenizer.ggml.token_type",
                Value::Array(Array::I32(vec![3, 3, 1, 3])),
            ),
            (
                "tokenizer.ggml.merges",
                Value::Array(Array::String(vec!["h i".into()])),
            ),
            ("tokenizer.ggml.bos_token_id", Value::U32(0)),
            ("tokenizer.ggml.eos_token_id", Value::U32(1)),
        ]);

        let bpe = crate::Bpe::from_vocab(&Vocab::from_metadata(&m).unwrap()).unwrap();
        assert_eq!(bpe.eog(), &[1, 3]);
        assert!(bpe.is_eog(1), "the declared eos still stops");
        assert!(bpe.is_eog(3), "and so does the marker the template uses");
        assert!(!bpe.is_eog(2), "ordinary text does not");
    }

    #[test]
    fn rejects_mismatched_token_types() {
        let m = metadata_with(vec![
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(Array::String(vec!["a".into()])),
            ),
            (
                "tokenizer.ggml.token_type",
                Value::Array(Array::I32(vec![1, 1])),
            ),
        ]);
        assert!(Vocab::from_metadata(&m).is_err());
    }
}
