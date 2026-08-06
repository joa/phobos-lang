use anyhow::{Result, bail};

use crate::backend::Backend;
use crate::{Gguf, llama, qwen35};

pub enum Decoder {
    Llama(Box<llama::Model>),
    Qwen35(Box<qwen35::Model>),
}

/// What a loaded model will ask of a backend, so a device path can decide
/// whether it fits before the first pass starts uploading.
///
/// An estimate, and deliberately one of the weights alone: a pass's
/// intermediates and a delta net's recurrent state are the caller's to allow
/// for. Nothing here is measured against the file's size on disk, which is a
/// poor guide, see `dense_bytes`.
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    /// Every weight the forward pass uploads. They are constants: uploaded
    /// once, kept for the backend's lifetime, never released.
    pub weight_bytes: usize,
    /// Of [`Footprint::weight_bytes`], what goes up as f32 because the file did
    /// not store it Q8_0. Nothing else is left quantized, so a file in a format
    /// this crate dequantizes wants four bytes a weight on the device however
    /// few it took on disk.
    pub dense_bytes: usize,
    /// What both attention caches across every block add per position. They
    /// grow by doubling and the growth holds the old pair while it copies, so a
    /// run's peak reaches about three times this times the sequence length.
    pub kv_bytes_per_token: usize,
}

pub enum State {
    Llama(Box<llama::State>),
    Qwen35(Box<qwen35::State>),
}

impl Decoder {
    pub fn load(gguf: &Gguf) -> Result<Decoder> {
        Ok(match gguf.architecture()? {
            "llama" => Decoder::Llama(Box::new(llama::Model::load(gguf)?)),
            "qwen35" => Decoder::Qwen35(Box::new(qwen35::Model::load(gguf)?)),
            other => bail!("no forward pass is implemented for the '{other}' architecture"),
        })
    }

    pub fn architecture(&self) -> &'static str {
        match self {
            Decoder::Llama(_) => "llama",
            Decoder::Qwen35(_) => "qwen35",
        }
    }

    pub fn vocab(&self) -> usize {
        match self {
            Decoder::Llama(m) => m.config.vocab,
            Decoder::Qwen35(m) => m.config.vocab,
        }
    }

    /// The trained context length.
    pub fn context_length(&self) -> usize {
        match self {
            Decoder::Llama(m) => m.config.context_length,
            Decoder::Qwen35(m) => m.config.context_length,
        }
    }

    /// What this model will occupy on a backend, with the caches sized for a
    /// context of `positions`.
    pub fn footprint(&self, positions: usize) -> Footprint {
        let (uploads, kv_bytes_per_token) = match self {
            Decoder::Llama(m) => (m.footprint(positions), m.kv_bytes_per_token()),
            Decoder::Qwen35(m) => (m.footprint(positions), m.kv_bytes_per_token()),
        };
        Footprint {
            weight_bytes: uploads.bytes(),
            dense_bytes: uploads.dense_bytes(),
            kv_bytes_per_token,
        }
    }

    /// Hyperparameter summary.
    pub fn summary(&self) -> String {
        match self {
            Decoder::Llama(m) => format!("{:?}", m.config),
            Decoder::Qwen35(m) => format!("{:?}", m.config),
        }
    }

    pub fn new_state(&self) -> State {
        match self {
            Decoder::Llama(m) => State::Llama(Box::new(m.new_state())),
            Decoder::Qwen35(m) => State::Qwen35(Box::new(m.new_state())),
        }
    }

    /// Perform a forward pass.
    ///
    /// Run `tokens`, advancing `state`, and return the final position's logits.
    pub fn forward(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<Vec<f32>> {
        match (self, state) {
            (Decoder::Llama(m), State::Llama(s)) => m.forward(s, tokens, backend),
            (Decoder::Qwen35(m), State::Qwen35(s)) => m.forward(s, tokens, backend),
            _ => bail!("generation state does not belong to the loaded architecture"),
        }
    }
}

impl State {
    /// Tokens consumed so far.
    pub fn len(&self) -> usize {
        match self {
            State::Llama(s) => s.len(),
            State::Qwen35(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hands every device allocation the state holds back to the backend.
    ///
    /// Dropping a state instead strands its caches: a [`crate::backend::Buf`]
    /// is a handle, not an owner.
    pub fn release(&mut self, backend: &dyn Backend) {
        match self {
            State::Llama(s) => s.release(backend),
            State::Qwen35(s) => s.release(backend),
        }
    }
}
