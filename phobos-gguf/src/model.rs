use anyhow::{Result, bail};

use crate::backend::{Backend, Buf};
use crate::{Gguf, llama, qwen35};

pub enum Decoder {
    Llama(Box<llama::Model>),
    Qwen35(Box<qwen35::Model>),
}

/// The device-only buffers one architecture's `forward_to_logits` leaves
/// live, shared by `llama` and `qwen35` since both end a pass the same way:
/// residual stream, normalized copy, the last row split off it, and the
/// logits the LM head projected it into.
pub(crate) struct ForwardBufs {
    pub(crate) x: Buf,
    pub(crate) normed: Buf,
    pub(crate) last: Buf,
    pub(crate) logits: Buf,
}

impl ForwardBufs {
    pub(crate) fn release(self, backend: &dyn Backend) {
        for buf in [self.x, self.normed, self.last, self.logits] {
            backend.release(buf);
        }
    }
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
    /// Weights that are not resident at all: expert sets the backend streams
    /// from the file, as the file holds them. What the host has to hold, and
    /// what a device cache is filled from.
    pub streamed_bytes: usize,
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
            "qwen35" | "qwen35moe" => Decoder::Qwen35(Box::new(qwen35::Model::load(gguf)?)),
            other => bail!("no forward pass is implemented for the '{other}' architecture"),
        })
    }

    pub fn architecture(&self) -> &'static str {
        match self {
            Decoder::Llama(_) => "llama",
            Decoder::Qwen35(m) => m.config.arch,
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
            streamed_bytes: uploads.streamed_bytes(),
            kv_bytes_per_token,
        }
    }

    /// The shape of the network, for a caller that displays it.
    ///
    /// A llama model is attention the whole way down. A qwen35 interleaves,
    /// and which blocks are which is the interesting part: only the attention
    /// ones pay per position, so the ratio is what makes a long context fit.
    pub fn layout(&self) -> phobos_inference::Architecture {
        match self {
            Decoder::Llama(m) => phobos_inference::Architecture {
                d_model: m.config.d_model,
                d_ff: m.config.d_ff,
                n_head: m.config.n_head,
                n_head_kv: m.config.n_head_kv,
                head_dim: m.config.head_dim,
                blocks: vec![phobos_inference::BlockKind::Attention; m.config.n_block],
            },
            Decoder::Qwen35(m) => phobos_inference::Architecture {
                d_model: m.config.d_model,
                d_ff: m.config.d_ff,
                n_head: m.config.n_head,
                n_head_kv: m.config.n_head_kv,
                head_dim: m.config.head_dim,
                blocks: (0..m.config.n_block)
                    .map(|i| {
                        if m.config.is_attention_block(i) {
                            phobos_inference::BlockKind::Attention
                        } else {
                            phobos_inference::BlockKind::Recurrent
                        }
                    })
                    .collect(),
            },
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

    /// [`Decoder::forward`] reporting what a mixture-of-experts model's
    /// routers chose; see [`qwen35::RouteTrace`]. Refused for a model with
    /// no routers.
    pub fn forward_traced(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<(Vec<f32>, qwen35::RouteTrace)> {
        match (self, state) {
            (Decoder::Qwen35(m), State::Qwen35(s)) => m.forward_traced(s, tokens, backend),
            (Decoder::Llama(_), State::Llama(_)) => bail!("a llama model has no routers to trace"),
            _ => bail!("generation state does not belong to the loaded architecture"),
        }
    }

    /// [`Decoder::forward`] for a caller that only wants the winning token
    /// id, as greedy decoding does. See [`Backend::argmax`].
    pub fn forward_greedy(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<i64> {
        match (self, state) {
            (Decoder::Llama(m), State::Llama(s)) => m.forward_greedy(s, tokens, backend),
            (Decoder::Qwen35(m), State::Qwen35(s)) => m.forward_greedy(s, tokens, backend),
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

    /// Forget everything past `positions`, and say whether it could. See the
    /// two architectures' own notes: one rewinds, one only extends.
    pub fn truncate(&mut self, positions: usize) -> bool {
        match self {
            State::Llama(s) => s.truncate(positions),
            State::Qwen35(s) => s.truncate(positions),
        }
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
