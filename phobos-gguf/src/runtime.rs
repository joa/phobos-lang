use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use phobos_inference::{DeviceMemory, Model, ModelInfo, Session, Tokenizer};

use crate::backend::Backend;
use crate::model::State;
use crate::tune::{self, Tuner};
use crate::{Bpe, Decoder, Gguf, Quant};

/// The backend for a model whose weights are stored in `quants`. The device
/// one compiles kernels for those formats only.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn make_backend(quants: &[Quant]) -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(crate::backend::DeviceBackend::new(quants)?))
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

/// Positions one prompt pass covers when the batch is not tuned, and what
/// load-time sizing assumes. A pass sizes its intermediates per row.
pub(crate) const PROMPT_BATCH: usize = tune::DEFAULT_BATCH;

/// Card memory a step up to a wider prompt pass leaves free, beyond the
/// wider pass's own estimate.
const ROOM_RESERVE_BYTES: usize = 512 << 20;

/// Device bytes [`check_fits`] leaves for everything that is not a weight: a
/// pass's intermediates, quantized-activation scratch, split-K partials, a
/// delta net's recurrent state, and the driver's context. Deliberately
/// generous: it catches a model that cannot fit, not the last hundred
/// megabytes.
pub(crate) const RESERVE_BYTES: usize = 768 << 20;

pub struct GgufModel {
    decoder: Decoder,
    bpe: Bpe,
    backend: Box<dyn Backend>,
    info: ModelInfo,
    /// Computed once at load, at the batch [`check_fits`] uses, so the weight
    /// figure a caller displays matches the fit decision.
    footprint: crate::model::Footprint,
    batching: Batching,
}

/// How a prompt is cut into passes.
enum Batching {
    Fixed(usize),
    Tuned(RefCell<Tuned>),
}

struct Tuned {
    tuner: Tuner,
    /// Where the picks are kept, `None` with the kernel cache off.
    record: Option<PathBuf>,
    /// Pass lengths this process has run. A length's first pass grows
    /// scratch that only grows, so it is not a sample.
    used: BTreeSet<usize>,
}

impl Batching {
    /// `pinned`, else `PHOBOS_PROMPT_BATCH` if set, else a tuner on a device
    /// backend unless `PHOBOS_BATCH_TUNE` is off. The host reference is not
    /// worth tuning, and has no card to key a record on.
    fn new(gguf: &Gguf, decoder: &Decoder, backend: &dyn Backend, pinned: Option<usize>) -> Result<Batching> {
        let rows_max = if decoder.has_experts() { backend.moe_max_rows() } else { usize::MAX };
        let from_env = || std::env::var("PHOBOS_PROMPT_BATCH").ok().and_then(|v| v.trim().parse().ok());
        if let Some(rows) = pinned.or_else(from_env).filter(|&rows: &usize| rows > 0) {
            ensure!(
                rows <= rows_max,
                "a prompt batch of {rows} is more than this model's mixture-of-experts pass takes, {rows_max}"
            );
            return Ok(Batching::Fixed(rows));
        }
        let device = backend.device_info().filter(|_| phobos_base::env::flag_on("PHOBOS_BATCH_TUNE"));
        let Some(device) = device else {
            return Ok(Batching::Fixed(PROMPT_BATCH));
        };
        let mut tuner = Tuner::new(rows_max.min(tune::MAX_BATCH));
        let record = tune::record_path(gguf.header(), &device);
        if let Some(path) = &record {
            tuner.load(path);
        }
        Ok(Batching::Tuned(RefCell::new(Tuned { tuner, record, used: BTreeSet::new() })))
    }
}

/// What a caller can choose about a load beyond the file.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// Device bytes for the expert cache of a model whose experts stream,
    /// instead of everything the resident weights leave. Ignored by a model
    /// with no experts and by a backend with no cache.
    pub expert_cache_bytes: Option<u64>,
    /// The most tokens of context to serve, instead of what fits on the
    /// device after the weights and the reserve. Never above the trained
    /// length.
    pub context_tokens: Option<usize>,
    /// The most positions one prompt pass covers. `None` tunes it on the
    /// device, per depth of the context, unless the environment pins it.
    pub prompt_batch: Option<usize>,
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
        let backend = make_backend(&gguf.quants())?;
        check_fits(backend.as_ref(), &decoder)?;
        let batching = Batching::new(&gguf, &decoder, backend.as_ref(), options.prompt_batch)?;
        if let Some(bytes) = options.expert_cache_bytes {
            let bytes = usize::try_from(bytes).context("expert cache size")?;
            backend.limit_expert_cache(bytes)?;
        }
        let footprint = decoder.footprint(PROMPT_BATCH);
        let trained = decoder.context_length();
        let context_limit = match options.context_tokens {
            Some(tokens) => tokens.clamp(1, trained),
            None => fitting_context(backend.as_ref(), &footprint, trained),
        };
        let info = ModelInfo {
            label: format!("GGUF, {}", decoder.architecture()),
            backend: backend_name(),
            vocab_size: decoder.vocab(),
            context_limit,
            chat_template,
        };
        Ok(GgufModel {
            decoder,
            bpe,
            backend,
            info,
            footprint,
            batching,
        })
    }

    /// The length of the next pass from `start`, `remaining` positions left.
    fn pass_rows(&self, start: usize, remaining: usize) -> usize {
        let rows = match &self.batching {
            Batching::Fixed(rows) => *rows,
            Batching::Tuned(tuned) => tuned.borrow_mut().tuner.size(start, remaining),
        };
        rows.min(remaining)
    }

    /// Runs `pass`, a pass of `rows` from `start`, and gives its time to the
    /// tuner. A pass that compiled a kernel, or the first of its length,
    /// paid a one-time cost and is not timed.
    fn timed<T>(&self, start: usize, rows: usize, pass: impl FnOnce() -> Result<T>) -> Result<T> {
        let Batching::Tuned(tuned) = &self.batching else {
            return pass();
        };
        if rows < tune::MIN_BATCH {
            return pass();
        }
        let compiled = || self.backend.cache_stats().map(|s| s.kernels_compiled);
        let before = compiled();
        let began = Instant::now();
        let out = pass()?;
        let secs = began.elapsed().as_secs_f64();
        let mut tuned = tuned.borrow_mut();
        if tuned.used.insert(rows) || compiled() != before {
            return Ok(out);
        }
        // What a wider pass is estimated to add: the pool's idle bytes after
        // this one are about its intermediates, and twice that per row
        // covers what the pool does not hold.
        let idle_bytes = self.backend.cache_stats().map_or(0, |s| s.buffer_idle_bytes as usize);
        let room = |size: usize| {
            let Some((free_bytes, _)) = self.backend.card_memory() else {
                return true;
            };
            let wider_bytes = size.saturating_sub(rows) * 2 * (idle_bytes / rows);
            free_bytes >= wider_bytes + ROOM_RESERVE_BYTES
        };
        if tuned.tuner.observe(rows, start, secs, &room) {
            phobos_base::phdebug!("prompt batch {}", tuned.tuner.describe(start));
        }
        if tuned.tuner.take_dirty()
            && let Some(path) = &tuned.record
            && let Err(e) = tuned.tuner.save(path)
        {
            phobos_base::phdebug!("prompt batch: could not write {}: {e}", path.display());
        }
        Ok(out)
    }
}

/// The longest context whose caches fit on the device after the weights and
/// [`RESERVE_BYTES`], at most the trained length.
///
/// The caches grow by doubling, so the answer is a power of two: any length
/// up to it reserves no more than it. A model with no per-token cache, or a
/// backend that cannot read device memory, gets the trained length.
fn fitting_context(backend: &dyn Backend, footprint: &crate::model::Footprint, trained: usize) -> usize {
    let Some((free_bytes, _)) = backend.device_memory() else {
        return trained;
    };
    if footprint.kv_bytes_per_token == 0 {
        return trained;
    }
    let room_bytes = free_bytes
        .saturating_sub(footprint.weight_bytes)
        .saturating_sub(RESERVE_BYTES);
    let tokens = (room_bytes / footprint.kv_bytes_per_token).max(1);
    let fits = 1usize << tokens.ilog2();
    fits.min(trained)
}

/// Refuses a model whose weights cannot fit the device, before the first
/// pass finds out partway through uploading.
///
/// Weights are the part worth checking: they are constants, all resident at
/// once and never released. Everything else is bounded by [`RESERVE_BYTES`].
/// The caches are only reported, since their cost depends on sequence
/// length, not the model. Streamed experts do not count: they are never all
/// resident, and their cache gets whatever room the resident weights leave.
fn check_fits(backend: &dyn Backend, decoder: &Decoder) -> Result<()> {
    let Some((free_bytes, total_bytes)) = backend.device_memory() else {
        return Ok(());
    };

    // Sized at a pass's row count, not the trained context. The rope table
    // included here grows lazily like the KV cache, so the full context
    // length would reserve for a table the run may never reach.
    let footprint = decoder.footprint(PROMPT_BATCH);
    let want_bytes = footprint.weight_bytes + RESERVE_BYTES;
    if want_bytes <= free_bytes {
        return Ok(());
    }

    // A weight this crate dequantizes costs four bytes per element on the
    // device, whatever it took on disk. Mention it only when it is over a
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
        let (free_bytes, total_bytes) = self.backend.card_memory()?;
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

    fn sensors(&self) -> Option<phobos_inference::SensorSource> {
        self.backend.sensors()
    }

    fn cache_stats(&self) -> Option<phobos_inference::CacheStats> {
        self.backend.cache_stats()
    }

    /// Runs a short prompt and one decode step through a throwaway session.
    /// A weight reaches the device the first time a pass uses it, and a
    /// kernel loads the first time a pass launches it. Prompt passes and
    /// decode steps launch different kernels, so both run. Streamed experts
    /// are not uploaded; only the routed ones are copied.
    fn warm_up(&self) -> Result<()> {
        // Wide enough that the prompt pass takes the same paths a real
        // prompt does, the grouped expert GEMMs among them.
        const PROMPT_TOKENS: usize = 128;
        // The host backend reads weights in place from the file, and a pass
        // there costs seconds for nothing.
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
        let model = self.model;
        let mut logits = Vec::new();
        // A prompt arrives as one call and is split into passes here; a
        // single generated token is one pass of one.
        let mut at = 0;
        while at < ids.len() {
            let start = self.state.len();
            let rows = model.pass_rows(start, ids.len() - at);
            let batch = &ids[at..at + rows];
            logits = model.timed(start, rows, || model.decoder.forward(&mut self.state, batch, model.backend.as_ref()))?;
            at += rows;
        }
        Ok(logits)
    }

    fn extend_greedy(&mut self, ids: &[i64]) -> Result<i64> {
        let ids = to_u32(ids);
        let model = self.model;
        let mut id = 0i64;
        // Same split as `extend`. Only the last pass's result is kept;
        // greedy decoding calls this with one token, so the loop runs once.
        let mut at = 0;
        while at < ids.len() {
            let start = self.state.len();
            let rows = model.pass_rows(start, ids.len() - at);
            let batch = &ids[at..at + rows];
            id = model.timed(start, rows, || model.decoder.forward_greedy(&mut self.state, batch, model.backend.as_ref()))?;
            at += rows;
        }
        Ok(id)
    }

    fn len(&self) -> usize {
        self.state.len()
    }

    /// The widest pass the tuner may choose, so the caller's batches never
    /// cut a pass short.
    fn prompt_batch(&self) -> Option<usize> {
        Some(match &self.model.batching {
            Batching::Fixed(rows) => *rows,
            Batching::Tuned(tuned) => tuned.borrow().tuner.limit(),
        })
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
/// They grow by doubling from a floor, so just after a growth most of the
/// reservation is headroom. Reporting `len` would understate the cache by up
/// to half. Mirrors `layers::KvCache::reserve`. A delta net's fixed-size
/// recurrent state is not counted.
fn kv_capacity(len: usize) -> usize {
    len.next_power_of_two().max(64)
}

impl Drop for GgufSession<'_> {
    fn drop(&mut self) {
        // A [`crate::backend::Buf`] is a handle, not an owner, so dropping the
        // state alone would leak its caches on the device.
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
