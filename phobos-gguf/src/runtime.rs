use std::path::Path;

use anyhow::{Context, Result, bail};
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

/// Positions one pass of the prompt covers. A pass sizes its intermediates
/// per row (a feed-forward's two halves alone are 37 KB a position), and
/// `bench` measures batching's payoff peaking around here.
const PROMPT_BATCH: usize = 512;

/// Device bytes [`check_fits`] leaves for everything that is not a weight: a
/// pass's intermediates, quantized-activation scratch, split-K partials, a
/// delta net's recurrent state, and the driver's own context. Generous on
/// purpose, to catch a model that cannot possibly fit rather than adjudicate
/// the last hundred megabytes.
const RESERVE_BYTES: usize = 768 << 20;

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
        let backend = make_backend()?;
        check_fits(backend.as_ref(), &decoder)?;
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
            backend,
            info,
        })
    }
}

/// Refuse a model whose weights cannot fit the device, before the first pass
/// discovers it a third of the way through uploading them.
///
/// Weights are the part worth checking: they are constants, so every one of
/// them is resident at once and none is ever released. Everything else is
/// bounded by [`RESERVE_BYTES`], and the caches are only reported, since their
/// cost depends on how long a sequence gets rather than on the model.
fn check_fits(backend: &dyn Backend, decoder: &Decoder) -> Result<()> {
    let Some((free_bytes, total_bytes)) = backend.device_memory() else {
        return Ok(());
    };

    // Sized at a pass's row count, not the trained context: the rope table
    // this folds in grows lazily like the KV cache, so gating on the full
    // context length would reserve for a table the run may never reach.
    let footprint = decoder.footprint(PROMPT_BATCH);
    let want_bytes = footprint.weight_bytes + RESERVE_BYTES;
    if want_bytes <= free_bytes {
        return Ok(());
    }

    // A file this crate has to dequantize costs four bytes a weight on the
    // device whatever it took on disk, which is the usual reason the figure is
    // a surprise. Worth saying once it accounts for a quarter of the total.
    let dense_note = if footprint.dense_bytes * 4 > footprint.weight_bytes {
        format!(
            "\n{} of that is weights this crate does not keep quantized, held as f32 on the device",
            gibibytes(footprint.dense_bytes)
        )
    } else {
        String::new()
    };

    bail!(
        "this model does not fit in device memory.\n\
         its weights need {}, a pass needs about {} more, and {} of the card's {} are free.{}\n\
         each token of context adds a further {} KiB of key/value cache.\n\
         phobos keeps every weight resident and has no host offload, so this needs a smaller \
         model, a Q8_0 or smaller quantization, or a larger card.",
        gibibytes(footprint.weight_bytes),
        gibibytes(RESERVE_BYTES),
        gibibytes(free_bytes),
        gibibytes(total_bytes),
        dense_note,
        footprint.kv_bytes_per_token / (1 << 10),
    )
}

fn gibibytes(bytes: usize) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
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

    fn extend_greedy(&mut self, ids: &[i64]) -> Result<i64> {
        let ids = to_u32(ids);
        let mut id = 0i64;
        // Same split as `extend`. Only the last batch's result is kept;
        // greedy decoding calls this with one token, so in practice the
        // loop runs once.
        for batch in ids.chunks(PROMPT_BATCH) {
            id = self
                .model
                .decoder
                .forward_greedy(&mut self.state, batch, self.model.backend.as_ref())?;
        }
        Ok(id)
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
