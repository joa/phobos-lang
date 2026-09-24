use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_inference::{DeviceMemory, Model, ModelInfo, Session, Tokenizer};

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
/// per row, and batching's payoff peaks around here.
pub(crate) const PROMPT_BATCH: usize = 512;

/// Device bytes [`check_fits`] leaves for everything that is not a weight: a
/// pass's intermediates, quantized-activation scratch, split-K partials, a
/// delta net's recurrent state, and the driver's own context. Generous on
/// purpose, to catch a model that cannot possibly fit rather than adjudicate
/// the last hundred megabytes.
pub(crate) const RESERVE_BYTES: usize = 768 << 20;

pub struct GgufModel {
    decoder: Decoder,
    bpe: Bpe,
    backend: Box<dyn Backend>,
    info: ModelInfo,
    /// Sized once at load, at the same batch [`check_fits`] gates on, so the
    /// weight figure a caller displays is the one the fit was decided by.
    footprint: crate::model::Footprint,
}

/// What a caller can choose about a load beyond the file.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// Device bytes for the expert cache of a model whose experts stream,
    /// instead of everything the resident weights leave. Ignored by a model
    /// with no experts and by a backend with no cache.
    pub expert_cache_bytes: Option<u64>,
}

impl GgufModel {
    pub fn load(path: &Path) -> Result<GgufModel> {
        GgufModel::load_with(path, LoadOptions::default())
    }

    pub fn load_with(path: &Path, options: LoadOptions) -> Result<GgufModel> {
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
        if let Some(bytes) = options.expert_cache_bytes {
            let bytes = usize::try_from(bytes).context("expert cache size")?;
            backend.limit_expert_cache(bytes)?;
        }
        let info = ModelInfo {
            label: format!("GGUF, {}", decoder.architecture()),
            backend: backend_name(),
            vocab_size: decoder.vocab(),
            context_limit: decoder.context_length(),
            chat_template,
        };
        let footprint = decoder.footprint(PROMPT_BATCH);
        Ok(GgufModel {
            decoder,
            bpe,
            backend,
            info,
            footprint,
        })
    }
}

/// Refuse a model whose weights cannot fit the device, before the first pass
/// discovers it a third of the way through uploading them.
///
/// Weights are the part worth checking: they are constants, so every one of
/// them is resident at once and none is ever released. Everything else is
/// bounded by [`RESERVE_BYTES`], and the caches are only reported, since their
/// cost depends on how long a sequence gets rather than on the model. Streamed
/// experts are not weights in this sense: they are never all resident, and
/// whatever room is left after the resident ones is what their cache gets.
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
    // device whatever it took on disk. Surfaced only once it accounts for a
    // quarter of the total.
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

    fn footprint(&self) -> Option<phobos_inference::Footprint> {
        Some(phobos_inference::Footprint {
            weight_bytes: self.footprint.weight_bytes as u64,
            dense_bytes: self.footprint.dense_bytes as u64,
            streamed_bytes: self.footprint.streamed_bytes as u64,
            kv_bytes_per_token: self.footprint.kv_bytes_per_token as u64,
        })
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        let (free_bytes, total_bytes) = self.backend.device_memory()?;
        Some(DeviceMemory {
            free_bytes: free_bytes as u64,
            total_bytes: total_bytes as u64,
        })
    }

    fn architecture(&self) -> Option<phobos_inference::Architecture> {
        Some(self.decoder.layout())
    }

    fn device_info(&self) -> Option<phobos_inference::DeviceInfo> {
        self.backend.device_info()
    }

    fn cache_stats(&self) -> Option<phobos_inference::CacheStats> {
        self.backend.cache_stats()
    }

    /// A short prompt and one decode step through a throwaway session. A
    /// weight reaches the device the first time a pass uses it, and a kernel
    /// is loaded the first time a pass launches it. A prompt pass launches
    /// kernels a decode step never does, and the other way round, so both
    /// run. Streamed experts are not uploaded: only the ones routed to are
    /// copied.
    fn warm_up(&self) -> Result<()> {
        // Wide enough that the prompt pass takes the same paths a real
        // prompt does, the grouped expert GEMMs among them.
        const PROMPT_TOKENS: usize = 128;
        // The host backend reads its weights where the file left them, and a
        // pass there costs seconds for nothing.
        if !cfg!(feature = "cuda") {
            return Ok(());
        }
        let mut session = self.session()?;
        session.extend(&[0; PROMPT_TOKENS])?;
        session.extend(&[0])?;
        Ok(())
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
        // A prompt arrives as one call and is split into batches here; a
        // single generated token is one batch of one.
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
        // greedy decoding calls this with one token, so the loop runs once.
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

    fn prompt_batch(&self) -> Option<usize> {
        Some(PROMPT_BATCH)
    }

    fn truncate(&mut self, positions: usize) -> Option<usize> {
        // A failed restore leaves the state for release, which is what the
        // caller does with a session that cannot be rewound.
        self.state.truncate(positions, self.model.backend.as_ref()).ok().flatten()
    }

    fn checkpoint(&mut self) -> Result<()> {
        self.state.checkpoint(self.model.backend.as_ref())
    }

    fn cache_bytes(&self) -> Option<u64> {
        let per_token = self.model.footprint.kv_bytes_per_token as u64;
        Some(per_token * kv_capacity(self.state.len()) as u64)
    }
}

/// Positions the attention caches hold for a sequence of `len`.
///
/// They grow by doubling from a floor, so what is reserved is mostly headroom
/// just after a growth. Reporting `len` instead would understate the cache by
/// up to half of it. See `layers::KvCache::reserve`, which this mirrors; the
/// recurrent state a delta net carries is fixed in size and not counted here.
fn kv_capacity(len: usize) -> usize {
    len.next_power_of_two().max(64)
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
