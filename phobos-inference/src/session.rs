use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_onnx::eval::fold_graph;
use phobos_onnx::interp::{self, MatmulBackend, Tensor};
use phobos_onnx::{Graph, load_model, transform};

/// One LM-head model, re-run over the whole prefix each step. Simple, but
/// O(seq) work per token; [`KvSession`] is the cached alternative.
pub struct Session {
    graph: Graph,
    input_name: String,
    output_name: String,
    vocab: usize,
    backend: Box<dyn MatmulBackend>,
}

impl Session {
    /// Load `model.onnx` from `model_dir` and pick a backend.
    pub fn load(model_dir: &Path) -> Result<Session> {
        let bytes = std::fs::read(model_dir.join("model.onnx"))
            .with_context(|| format!("read {}", model_dir.join("model.onnx").display()))?;
        let model = load_model(&bytes)?;
        let graph = model.graph;

        let input_name = graph
            .inputs
            .first()
            .context("model has no input")?
            .name
            .clone();
        let out = graph.outputs.first().context("model has no output")?;
        let output_name = out.name.clone();
        let vocab = last_fixed_dim(out).context("first output has no fixed vocabulary axis")?;

        Ok(Session {
            graph,
            input_name,
            output_name,
            vocab,
            backend: make_backend()?,
        })
    }

    /// One forward pass over `ids`, returning the final position's logits.
    pub fn next_logits(&self, ids: &[i64]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("cannot run inference on an empty prompt");
        }
        // The LM-head export takes a [1, 1, seq] token tensor.
        let dims = vec![1, 1, ids.len() as i64];
        let folded = self.fold(&HashMap::from([(self.input_name.clone(), dims.clone())]))?;
        let inputs = HashMap::from([(self.input_name.clone(), Tensor::i64(dims, ids.to_vec()))]);
        let outputs = interp::run_with(&folded, &inputs, self.backend.as_ref())?;
        let logits = outputs
            .get(&self.output_name)
            .context("missing logits output")?
            .to_f32();
        last_row(&logits, ids.len(), self.vocab)
    }

    fn fold(&self, shapes: &HashMap<String, Vec<i64>>) -> Result<Graph> {
        Ok(transform::fuse_layernorm(&fold_graph(&self.graph, shapes)?))
    }
}

/// The prompt goes through `decoder` and each single-token step through
/// `decoder_with_past`, the per-layer key/value cache threaded between them.
/// A step is then O(1) in sequence length.
pub struct KvSession {
    prompt_graph: Graph,
    step_graph: Graph,
    /// Decoded once and reused every token: re-decoding 500 MB per step
    /// otherwise dominates a single-token decode.
    step_weights: HashMap<String, Tensor>,
    vocab: usize,
    n_layer: usize,
    backend: Box<dyn MatmulBackend>,
}

/// `[key0, value0, key1, value1, ...]`, each `[1, heads, len, head_dim]`.
pub struct KvCache {
    past: Vec<Tensor>,
    len: usize,
}

impl KvSession {
    /// Load `decoder.onnx` and `decoder_with_past.onnx` from `dir`.
    pub fn load(dir: &Path) -> Result<KvSession> {
        let prompt_graph = load_model(&std::fs::read(dir.join("decoder.onnx"))?)?.graph;
        let step_graph = load_model(&std::fs::read(dir.join("decoder_with_past.onnx"))?)?.graph;

        let logits = prompt_graph
            .outputs
            .first()
            .context("decoder has no output")?;
        let vocab = last_fixed_dim(logits).context("decoder logits have no fixed vocab axis")?;
        let n_layer = prompt_graph
            .outputs
            .iter()
            .filter(|o| o.name.ends_with(".key"))
            .count();
        if n_layer == 0 {
            bail!("decoder exposes no present.*.key outputs; not a with-cache export");
        }

        let step_weights = interp::decode_initializers(&step_graph)?;
        Ok(KvSession {
            prompt_graph,
            step_graph,
            step_weights,
            vocab,
            n_layer,
            backend: make_backend()?,
        })
    }

    /// A pass over the whole prompt, returning its last logits and the
    /// initialized cache.
    fn prompt(&self, ids: &[i64]) -> Result<(Vec<f32>, KvCache)> {
        if ids.is_empty() {
            bail!("cannot run inference on an empty prompt");
        }
        let n = ids.len();
        let dims = vec![1, n as i64];
        let folded = transform::fuse_layernorm(&fold_graph(
            &self.prompt_graph,
            &HashMap::from([("input_ids".to_string(), dims.clone())]),
        )?);
        let inputs = HashMap::from([("input_ids".to_string(), Tensor::i64(dims, ids.to_vec()))]);
        let out = interp::run_with(&folded, &inputs, self.backend.as_ref())?;

        let logits = last_row(
            &out.get("logits").context("no logits")?.to_f32(),
            n,
            self.vocab,
        )?;
        Ok((logits, self.collect_cache(&out, n)?))
    }

    /// One step over `token` and the cache, returning its logits and the
    /// grown cache.
    fn step(&self, token: i64, cache: &KvCache) -> Result<(Vec<f32>, KvCache)> {
        let mut shapes = HashMap::from([("input_ids".to_string(), vec![1, 1])]);
        let mut inputs = HashMap::from([(
            "input_ids".to_string(),
            Tensor::i64(vec![1, 1], vec![token]),
        )]);
        for (i, t) in cache.past.iter().enumerate() {
            let name = past_name(i);
            shapes.insert(name.clone(), t.dims.clone());
            inputs.insert(name, t.clone());
        }
        let folded = transform::fuse_layernorm(&fold_graph(&self.step_graph, &shapes)?);
        let out =
            interp::run_with_env(&folded, &inputs, &self.step_weights, self.backend.as_ref())?;

        let logits = last_row(
            &out.get("logits").context("no logits")?.to_f32(),
            1,
            self.vocab,
        )?;
        Ok((logits, self.collect_cache(&out, cache.len + 1)?))
    }

    /// The `present.*` outputs gathered into a fresh cache.
    fn collect_cache(&self, out: &HashMap<String, Tensor>, len: usize) -> Result<KvCache> {
        let mut past = Vec::with_capacity(2 * self.n_layer);
        for l in 0..self.n_layer {
            for kind in ["key", "value"] {
                let name = format!("present.{l}.{kind}");
                past.push(
                    out.get(&name)
                        .with_context(|| format!("missing {name}"))?
                        .clone(),
                );
            }
        }
        Ok(KvCache { past, len })
    }
}

fn past_name(flat_index: usize) -> String {
    let (layer, kind) = (
        flat_index / 2,
        if flat_index.is_multiple_of(2) {
            "key"
        } else {
            "value"
        },
    );
    format!("past.{layer}.{kind}")
}

pub enum Engine {
    Full(Session),
    Kv(KvSession),
}

pub enum GenState {
    Full(Vec<i64>),
    Kv(KvCache),
}

impl GenState {
    /// The prompt plus every token generated since.
    pub fn len(&self) -> usize {
        match self {
            GenState::Full(ids) => ids.len(),
            GenState::Kv(cache) => cache.len,
        }
    }
}

impl Engine {
    /// The KV-cached engine when both cache models are in `kv_dir`, and the
    /// full-recompute model in `model_dir` otherwise.
    pub fn load(kv_dir: &Path, model_dir: &Path) -> Result<Engine> {
        if kv_dir.join("decoder.onnx").is_file() && kv_dir.join("decoder_with_past.onnx").is_file()
        {
            Ok(Engine::Kv(KvSession::load(kv_dir)?))
        } else {
            Ok(Engine::Full(Session::load(model_dir)?))
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Engine::Full(s) => s.vocab,
            Engine::Kv(s) => s.vocab,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Engine::Full(_) => "full-recompute",
            Engine::Kv(_) => "KV-cached",
        }
    }

    /// Run the prompt and return its last logits plus fresh state.
    pub fn start(&self, ids: &[i64]) -> Result<(Vec<f32>, GenState)> {
        match self {
            Engine::Full(s) => Ok((s.next_logits(ids)?, GenState::Full(ids.to_vec()))),
            Engine::Kv(s) => {
                let (logits, cache) = s.prompt(ids)?;
                Ok((logits, GenState::Kv(cache)))
            }
        }
    }

    /// Append `token` and return the logits for the position after it.
    pub fn advance(&self, state: &mut GenState, token: i64) -> Result<Vec<f32>> {
        match (self, state) {
            (Engine::Full(s), GenState::Full(ids)) => {
                ids.push(token);
                s.next_logits(ids)
            }
            (Engine::Kv(s), GenState::Kv(cache)) => {
                let (logits, next) = s.step(token, cache)?;
                *cache = next;
                Ok(logits)
            }
            _ => bail!("engine and generation state are mismatched"),
        }
    }
}

/// The `vocab`-length logit row for the final position of a
/// `[.., seq, vocab]` output.
fn last_row(logits: &[f32], seq: usize, vocab: usize) -> Result<Vec<f32>> {
    let start = (seq - 1) * vocab;
    logits
        .get(start..start + vocab)
        .context("logits shorter than expected")
        .map(<[f32]>::to_vec)
}

/// The last dimension of a value's shape, when it is fixed.
fn last_fixed_dim(vi: &phobos_onnx::ir::ValueInfo) -> Option<usize> {
    match vi.shape.0.as_ref()?.last()? {
        phobos_onnx::ir::Dim::Fixed(n) => Some(*n as usize),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn make_backend() -> Result<Box<dyn MatmulBackend>> {
    Ok(Box::new(phobos_onnx::runner::GpuBackend::new()?))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn make_backend() -> Result<Box<dyn MatmulBackend>> {
    Ok(Box::new(interp::HostBackend))
}

pub fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "phobos GPU"
    } else {
        "host interpreter"
    }
}
