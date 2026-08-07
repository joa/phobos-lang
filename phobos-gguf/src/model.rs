use anyhow::{Result, bail};

use crate::backend::Backend;
use crate::{Gguf, llama, qwen35};

pub enum Decoder {
    Llama(Box<llama::Model>),
    Qwen35(Box<qwen35::Model>),
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
