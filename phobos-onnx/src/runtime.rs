use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_inference::{Model, ModelInfo, Session, Tokenizer};

use crate::backend::{MatmulBackend, Tensor, host};
use crate::eval::fold_graph;
use crate::tokenizer::Gpt2Tokenizer;
use crate::{Graph, load_model, transform};

/// One LM-head model, re-run over the whole prefix each step. Simple, but
/// O(seq) work per token; [`KvGraph`] is the cached alternative.
struct FullGraph {
    graph: Graph,
    input_name: String,
    output_name: String,
    vocab: usize,
    backend: Box<dyn MatmulBackend>,
}

impl FullGraph {
    pub fn load(model_dir: &Path) -> Result<FullGraph> {
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

        Ok(FullGraph {
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
        let outputs = host::run_with(&folded, &inputs, self.backend.as_ref())?;
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
struct KvGraph {
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
struct KvCache {
    past: Vec<Tensor>,
    len: usize,
}

impl KvGraph {
    /// Load `decoder.onnx` and `decoder_with_past.onnx` from `dir`.
    pub fn load(dir: &Path) -> Result<KvGraph> {
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

        let step_weights = host::decode_initializers(&step_graph)?;
        Ok(KvGraph {
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
        let out = host::run_with(&folded, &inputs, self.backend.as_ref())?;

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
        let out = host::run_with_env(&folded, &inputs, &self.step_weights, self.backend.as_ref())?;

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

enum Engine {
    Full(FullGraph),
    Kv(KvGraph),
}

/// A generation's state, whichever engine produced it.
enum GenState {
    /// Every id so far: the full-recompute engine re-runs the lot each step.
    Full(Vec<i64>),
    /// Populated by the first `extend`, which is the prompt pass.
    Kv(Option<KvCache>),
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
fn last_fixed_dim(vi: &crate::ir::ValueInfo) -> Option<usize> {
    match vi.shape.0.as_ref()?.last()? {
        crate::ir::Dim::Fixed(n) => Some(*n as usize),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
fn make_backend() -> Result<Box<dyn MatmulBackend>> {
    Ok(Box::new(crate::backend::device::GpuBackend::new()?))
}

#[cfg(not(feature = "cuda"))]
fn make_backend() -> Result<Box<dyn MatmulBackend>> {
    Ok(Box::new(crate::backend::HostBackend))
}

/// Where an ONNX model's arithmetic happens, for the loading banner.
pub fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "phobos GPU"
    } else {
        "host interpreter"
    }
}

/// GPT-2 has 1024 learned position embeddings and cannot exceed them.
const GPT2_CONTEXT: usize = 1024;

/// An exported GPT-2, paired with the tokenizer it was exported against.
///
/// The pairing is the loader's, not the format's: an ONNX file carries no
/// vocabulary, so nothing in the graph says which tokenizer produced the ids it
/// expects.
pub struct OnnxModel {
    engine: Engine,
    tokenizer: Gpt2Tokenizer,
    info: ModelInfo,
}

impl OnnxModel {
    /// Load whichever export `dir` holds.
    ///
    /// A `decoder.onnx` and a `decoder_with_past.onnx` side by side are a
    /// with-cache export and give the KV-cached engine; a `model.onnx` is a
    /// full-recompute LM-head export. Nothing in the files says which is
    /// intended, so the directory's contents are what decides.
    pub fn load(dir: &Path) -> Result<OnnxModel> {
        let engine = if dir.join("decoder.onnx").is_file()
            && dir.join("decoder_with_past.onnx").is_file()
        {
            Engine::Kv(KvGraph::load(dir)?)
        } else if dir.join("model.onnx").is_file() {
            Engine::Full(FullGraph::load(dir)?)
        } else {
            bail!(
                "{} holds no ONNX export: expected decoder.onnx and \
                 decoder_with_past.onnx, or model.onnx",
                dir.display()
            );
        };
        let (vocab_size, engine_name) = match &engine {
            Engine::Full(g) => (g.vocab, "full-recompute"),
            Engine::Kv(g) => (g.vocab, "KV-cached"),
        };
        Ok(OnnxModel {
            engine,
            tokenizer: Gpt2Tokenizer::gpt2()?,
            info: ModelInfo {
                label: format!("ONNX, {engine_name}"),
                backend: backend_name(),
                vocab_size,
                context_limit: GPT2_CONTEXT,
                chat_template: None,
            },
        })
    }
}

impl Model for OnnxModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        &self.tokenizer
    }

    fn session(&self) -> Result<Box<dyn Session + '_>> {
        let state = match &self.engine {
            Engine::Full(_) => GenState::Full(Vec::new()),
            Engine::Kv(_) => GenState::Kv(None),
        };
        Ok(Box::new(OnnxSession {
            engine: &self.engine,
            state,
        }))
    }
}

pub struct OnnxSession<'a> {
    engine: &'a Engine,
    state: GenState,
}

impl Session for OnnxSession<'_> {
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        match (self.engine, &mut self.state) {
            (Engine::Full(graph), GenState::Full(seen)) => {
                seen.extend_from_slice(ids);
                graph.next_logits(seen)
            }
            // The prompt goes through `decoder` and every token after it
            // through `decoder_with_past`, one at a time: the step graph takes
            // a single position.
            (Engine::Kv(graph), GenState::Kv(cache @ None)) => {
                let (logits, fresh) = graph.prompt(ids)?;
                *cache = Some(fresh);
                Ok(logits)
            }
            (Engine::Kv(graph), GenState::Kv(Some(cache))) => {
                let mut logits = Vec::new();
                for &id in ids {
                    let (next_logits, grown) = graph.step(id, cache)?;
                    *cache = grown;
                    logits = next_logits;
                }
                Ok(logits)
            }
            _ => bail!("engine and generation state are mismatched"),
        }
    }

    fn len(&self) -> usize {
        match &self.state {
            GenState::Full(ids) => ids.len(),
            GenState::Kv(cache) => cache.as_ref().map_or(0, |c| c.len),
        }
    }
}
