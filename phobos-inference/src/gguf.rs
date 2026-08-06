use std::path::Path;

use anyhow::{Context, Result};
use phobos_gguf::compute::Backend;
#[cfg(not(feature = "cuda"))]
use phobos_gguf::compute::HostBackend;
use phobos_gguf::model::State;
use phobos_gguf::{Bpe, Decoder, Gguf};

pub type GgufState = State;

fn make_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(crate::device::DeviceBackend::new()?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(HostBackend::new()))
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

pub struct GgufRuntime {
    decoder: Decoder,
    bpe: Bpe,
    backend: Box<dyn Backend>,
    chat_template: Option<String>,
}

impl GgufRuntime {
    pub fn load(path: &Path) -> Result<GgufRuntime> {
        let gguf = Gguf::open(path)?;
        let architecture = gguf.architecture()?.to_string();
        let vocab = gguf.vocab()?;
        let chat_template = vocab.chat_template.clone();
        let bpe = Bpe::from_vocab(&vocab)
            .with_context(|| format!("build tokenizer for a '{architecture}' model"))?;
        let decoder =
            Decoder::load(&gguf).with_context(|| format!("load '{architecture}' weights"))?;
        Ok(GgufRuntime {
            decoder,
            bpe,
            backend: make_backend()?,
            chat_template,
        })
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn label(&self) -> &str {
        self.decoder.architecture()
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.vocab()
    }

    pub fn context_limit(&self) -> usize {
        self.decoder.context_length()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(self.bpe.encode(text)?.into_iter().map(i64::from).collect())
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        self.bpe.decode(&to_u32(ids))
    }

    pub fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        self.bpe.decode_bytes(&to_u32(ids))
    }

    pub fn bos_text(&self) -> Option<&str> {
        self.bpe.bos().and_then(|id| self.bpe.token(id))
    }

    pub fn is_eog(&self, id: i64) -> bool {
        u32::try_from(id).is_ok_and(|id| self.bpe.is_eog(id))
    }

    /// Run the prompt and return its last logits plus fresh generation state.
    pub fn start(&self, ids: &[i64]) -> Result<(Vec<f32>, GgufState)> {
        let mut state = self.decoder.new_state();
        let mut logits = Vec::new();
        for batch in to_u32(ids).chunks(PROMPT_BATCH) {
            logits = self
                .decoder
                .forward(&mut state, batch, self.backend.as_ref())?;
        }
        Ok((logits, state))
    }

    /// Append one token and return the logits for the position after it.
    pub fn advance(&self, state: &mut GgufState, token: i64) -> Result<Vec<f32>> {
        self.decoder
            .forward(state, &to_u32(&[token]), self.backend.as_ref())
    }

    /// Returns a finished generation's caches to the device.
    ///
    /// Dropping the/ state instead leaks them; see [`GgufState::release`].
    pub fn finish(&self, mut state: GgufState) {
        state.release(self.backend.as_ref());
    }
}

fn to_u32(ids: &[i64]) -> Vec<u32> {
    ids.iter().map(|&id| id as u32).collect()
}
