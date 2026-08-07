//! What the runtime needs of a model, and nothing about how one is built.
//!
//! `phobos-gguf` and `phobos-onnx` implement these. Everything above them, the
//! sampler, the chat rendering and the server, is written against the traits
//! and never names a front end, which is what lets the front ends depend on
//! this crate rather than the other way round.

use anyhow::Result;

/// What a loaded model reports about itself.
pub struct ModelInfo {
    /// How the model identifies itself, e.g. `"GGUF, qwen35"`. This is the name
    /// the server reports from `/v1/models`.
    pub label: String,
    /// Where the arithmetic happens, e.g. `"phobos GPU"`. For the loading
    /// banner; nothing dispatches on it.
    pub backend: &'static str,
    pub vocab_size: usize,
    /// Positions the model can attend over. A generation stops here.
    pub context_limit: usize,
    /// The chat template the model carries, when it carries one. A GGUF file
    /// has one in its metadata; an ONNX export has nothing to put one in.
    pub chat_template: Option<String>,
}

/// A model that has been loaded and can generate.
pub trait Model {
    fn info(&self) -> &ModelInfo;

    fn tokenizer(&self) -> &dyn Tokenizer;

    /// A fresh generation.
    ///
    /// The session borrows the model, so whatever it allocated on the device
    /// goes back when it drops. That is not a detail a caller can opt out of:
    /// a dropped GGUF state used to strand a few hundred megabytes of key and
    /// value cache per request, and four separate call sites carried a comment
    /// saying to call a `finish` method instead.
    fn session(&self) -> Result<Box<dyn Session + '_>>;
}

/// One generation in progress: the prompt it has consumed and whatever cache
/// that left behind.
pub trait Session {
    /// Run `ids` and return the logits for the position after the last one.
    ///
    /// The prompt is one call and each generated token another, so a caller
    /// never has to keep a model and a state in step by hand. An implementation
    /// is free to split a long call into batches; that is a property of the
    /// backend, not of the interface.
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>>;

    /// Tokens consumed so far, prompt included.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Text to token ids and back.
///
/// A GGUF file carries its tokenizer, so the one a GGUF model returns is the
/// one it was trained with. An ONNX export carries nothing of the sort, so the
/// one an ONNX model returns is whatever the loader was built to pair with it.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>>;

    /// Decode to raw bytes. A token can hold a partial UTF-8 sequence, so a
    /// streaming caller has to buffer these and emit only complete characters;
    /// see `generate`.
    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8>;

    /// Whether `id` ends a turn, by any of the markers the model uses.
    fn is_eog(&self, id: i64) -> bool;

    /// The token a prompt opens with, when the model wants one.
    fn bos_text(&self) -> Option<&str> {
        None
    }

    fn decode(&self, ids: &[i64]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}
