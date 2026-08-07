use std::path::Path;

use anyhow::{Context, Result};
use phobos_inference::{Model, ModelInfo, Session, Tokenizer};

use crate::backend::Backend;
use crate::model::State;
use crate::{Bpe, Decoder, Gguf};

fn make_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(crate::backend::DeviceBackend::new()?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(crate::backend::HostBackend::new()))
    }
}

pub fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "phobos GPU"
    } else {
        "host reference"
    }
}

/// Positions one pass of the prompt covers.
///
/// A pass sizes its intermediates per row, so an unbounded one would size them
/// to the prompt: a feed-forward's two halves alone are 37 KB a position.
/// Batching also stops paying somewhere above this, since attention's cost per
/// position grows with the block, and `bench` measures the peak at 512. The
/// multiple of 64 keeps every batch after the first on the aligned attention
/// path.
const PROMPT_BATCH: usize = 512;

pub struct GgufModel {
    decoder: Decoder,
    bpe: Bpe,
    backend: Box<dyn Backend>,
    info: ModelInfo,
}

impl GgufModel {
    pub fn load(path: &Path) -> Result<GgufModel> {
        let gguf = Gguf::open(path)?;
        let architecture = gguf.architecture()?.to_string();
        let vocab = gguf.vocab()?;
        let chat_template = vocab.chat_template.clone();
        let bpe = Bpe::from_vocab(&vocab)
            .with_context(|| format!("build tokenizer for a '{architecture}' model"))?;
        let decoder =
            Decoder::load(&gguf).with_context(|| format!("load '{architecture}' weights"))?;
        let info = ModelInfo {
            label: format!("GGUF, {}", decoder.architecture()),
            backend: backend_name(),
            vocab_size: decoder.vocab(),
            context_limit: decoder.context_length(),
            chat_template,
        };
        Ok(GgufModel {
            decoder,
            bpe,
            backend: make_backend()?,
            info,
        })
    }
}

impl Model for GgufModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        &self.bpe
    }

    fn session(&self) -> Result<Box<dyn Session + '_>> {
        Ok(Box::new(GgufSession {
            state: self.decoder.new_state(),
            model: self,
        }))
    }
}

pub struct GgufSession<'a> {
    model: &'a GgufModel,
    state: State,
}

impl Session for GgufSession<'_> {
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        let ids = to_u32(ids);
        let mut logits = Vec::new();
        // A prompt arrives as one call and is split here, where the reason for
        // the split lives; a single generated token is one batch of one.
        for batch in ids.chunks(PROMPT_BATCH) {
            logits =
                self.model
                    .decoder
                    .forward(&mut self.state, batch, self.model.backend.as_ref())?;
        }
        Ok(logits)
    }

    fn len(&self) -> usize {
        self.state.len()
    }
}

impl Drop for GgufSession<'_> {
    fn drop(&mut self) {
        // A [`crate::backend::Buf`] is a handle, not an owner, so dropping the
        // state alone would strand its caches on the device.
        self.state.release(self.model.backend.as_ref());
    }
}

impl Tokenizer for Bpe {
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(Bpe::encode(self, text)?
            .into_iter()
            .map(i64::from)
            .collect())
    }

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        Bpe::decode_bytes(self, &to_u32(ids))
    }

    fn is_eog(&self, id: i64) -> bool {
        u32::try_from(id).is_ok_and(|id| Bpe::is_eog(self, id))
    }

    fn bos_text(&self) -> Option<&str> {
        self.bos().and_then(|id| self.token(id))
    }
}

fn to_u32(ids: &[i64]) -> Vec<u32> {
    ids.iter().map(|&id| id as u32).collect()
}
