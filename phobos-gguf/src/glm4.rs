// GLM-4: llama's grouped-query attention with biased query, key and value
// projections, gate and up stored stacked in one tensor, and a second RMS
// norm on each mixer's output before it joins the residual stream.

use anyhow::{Context, Result, bail, ensure};

use crate::Gguf;
use crate::backend::{Attn, Backend, Buf, HPlane, Plane, QAct, Rope, read_vec};
use crate::layers::{Ffn, Gain, KvCache, Linear, RopeTable, Uploads, load_vector};
use crate::llama::{State, check_no_rope_scaling, dense_plane, embed_rows, head_logits, neox_order};
use crate::model::ForwardBufs;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_block: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub context_length: usize,
    pub rms_eps: f32,

    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    /// Whether the file rotates pairs half a rotary width apart, as the
    /// backend does, rather than consecutive ones.
    ///
    /// A file with multimodal rotary sections (GLM-4V) uses ggml's MROPE,
    /// which rotates the NeoX way. Its sections give the height and width
    /// positions of an image their own frequencies, but a text position is
    /// the same in every section, so for text it is plain NeoX rotary.
    /// Without sections ggml rotates consecutive pairs.
    pub neox: bool,
}

impl Config {
    pub fn from_gguf(gguf: &Gguf) -> Result<Config> {
        let arch = gguf.architecture()?;
        ensure!(arch == "glm4", "expected a glm4 model, found architecture '{arch}'");
        let m = gguf.metadata();

        check_no_rope_scaling(gguf)?;

        let d_model = m.arch_count("embedding_length")?;
        let n_head = m.arch_count("attention.head_count")?;
        ensure!(n_head > 0, "a glm4 model needs at least one head");
        let head_dim = match m.arch_get("attention.key_length") {
            Some(_) => m.arch_count("attention.key_length")?,
            None => d_model / n_head,
        };
        let sections = m
            .arch_get("rope.dimension_sections")
            .and_then(|v| v.as_array())
            .and_then(|a| a.to_int_vec())
            .unwrap_or_default();

        Ok(Config {
            n_block: m.arch_count("block_count")?,
            d_model,
            d_ff: m.arch_count("feed_forward_length")?,
            vocab: gguf
                .tensor("token_embd.weight")
                .context("model has no token_embd.weight")?
                .row_major_dims()[0] as usize,
            context_length: m.arch_count("context_length")?,
            rms_eps: m.arch_float("attention.layer_norm_rms_epsilon")?,
            n_head,
            n_head_kv: m.arch_count("attention.head_count_kv").unwrap_or(n_head),
            head_dim,
            rope_dim: match m.arch_get("rope.dimension_count") {
                Some(_) => m.arch_count("rope.dimension_count")?,
                None => head_dim,
            },
            rope_freq_base: m.arch_float("rope.freq_base").unwrap_or(10000.0),
            // llama.cpp's `use_mrope`.
            neox: sections.len() >= 2 && sections[0] > 0 && sections[1] > 0,
        })
    }
}

/// The query and key projections stacked into one, and the value with them
/// when it shares their format. A GLM-4 file often keeps the value wider,
/// and stacking mixed formats would go dense.
struct Attention {
    /// Query, key and maybe value, in that order, and their biases end to end
    /// over its whole width.
    qkv: Linear,
    qkv_bias: Gain,
    /// The value and its bias when `qkv` does not hold them.
    v: Option<(Linear, Gain)>,
    output: Linear,
}

impl Attention {
    fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<Attention> {
        let (d, width, kv_width) = (cfg.d_model, cfg.n_head * cfg.head_dim, cfg.n_head_kv * cfg.head_dim);
        let q = Linear::load(gguf, &format!("{prefix}.attn_q.weight"), d, width)?;
        let k = Linear::load(gguf, &format!("{prefix}.attn_k.weight"), d, kv_width)?;
        let v = Linear::load(gguf, &format!("{prefix}.attn_v.weight"), d, kv_width)?;
        let mut q_bias = load_vector(gguf, &format!("{prefix}.attn_q.bias"), width)?;
        let mut k_bias = load_vector(gguf, &format!("{prefix}.attn_k.bias"), kv_width)?;
        let v_bias = load_vector(gguf, &format!("{prefix}.attn_v.bias"), kv_width)?;

        // Consecutive pairs move into the backend's layout, the bias with
        // its weight. See `neox_order`.
        let (q, k) = match cfg.neox {
            true => (q, k),
            false => {
                let q_order = neox_order(cfg.n_head, cfg.head_dim, cfg.rope_dim);
                let k_order = neox_order(cfg.n_head_kv, cfg.head_dim, cfg.rope_dim);
                q_bias = q_order.iter().map(|&j| q_bias[j]).collect();
                k_bias = k_order.iter().map(|&j| k_bias[j]).collect();
                (q.reorder_outputs("neox", &q_order)?, k.reorder_outputs("neox", &k_order)?)
            }
        };

        let (qkv, v) = match Linear::stacks(&[&q, &k, &v]) {
            true => (Linear::fuse(&[&q, &k, &v])?, None),
            false if Linear::stacks(&[&q, &k]) => (Linear::fuse(&[&q, &k])?, Some(v)),
            false => bail!("'{prefix}' query and key are held in different formats"),
        };
        let mut bias = [q_bias, k_bias].concat();
        let v = match v {
            Some(v) => Some((v, Gain::derived(format!("{prefix}.attn_v.bias"), v_bias))),
            None => {
                bias.extend(v_bias);
                None
            }
        };
        // `fuse` pads the width to its tile; the padding takes no bias.
        bias.resize(qkv.out_dim, 0.0);

        Ok(Attention {
            qkv_bias: Gain::derived(format!("{prefix}.attn_qkv.bias"), bias),
            qkv,
            v,
            output: Linear::load(gguf, &format!("{prefix}.attn_output.weight"), width, d)?,
        })
    }
}

struct Block {
    attn_norm: Gain,
    attn: Attention,
    post_attn_norm: Gain,
    ffn_norm: Gain,
    ffn: Ffn,
    post_ffn_norm: Gain,
}

pub struct Model {
    pub config: Config,
    /// `token_embd.weight`, kept quantized and read on the host.
    embed: Linear,
    head: Linear,
    blocks: Vec<Block>,
    output_norm: Gain,
    rope: RopeTable,
}

impl Model {
    pub fn load(gguf: &Gguf) -> Result<Model> {
        let config = Config::from_gguf(gguf)?;
        ensure!(
            config.head_dim > 0 && config.rope_dim <= config.head_dim && config.rope_dim.is_multiple_of(2),
            "rope dimension {} does not fit head dimension {}",
            config.rope_dim,
            config.head_dim
        );
        ensure!(
            config.n_head_kv > 0 && config.n_head.is_multiple_of(config.n_head_kv),
            "{} query heads do not group evenly over {} key/value heads",
            config.n_head,
            config.n_head_kv
        );

        let d = config.d_model;
        let embed = Linear::load(gguf, "token_embd.weight", d, config.vocab)?;
        let head = match gguf.tensor("output.weight") {
            Some(_) => Linear::load(gguf, "output.weight", d, config.vocab)?,
            None => Linear::load(gguf, "token_embd.weight", d, config.vocab)?,
        };

        let blocks = (0..config.n_block)
            .map(|index| {
                let prefix = format!("blk.{index}");
                let gain = |name: &str| Gain::load(gguf, &format!("{prefix}.{name}.weight"), d);
                Ok(Block {
                    attn_norm: gain("attn_norm")?,
                    attn: Attention::load(gguf, &prefix, &config)?,
                    post_attn_norm: gain("post_attention_norm")?,
                    ffn_norm: gain("ffn_norm")?,
                    ffn: Ffn::load_stacked(gguf, &prefix, d, config.d_ff)?,
                    post_ffn_norm: gain("post_ffw_norm")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Model {
            rope: RopeTable::new(config.rope_dim, config.rope_freq_base),
            output_norm: Gain::load(gguf, "output_norm.weight", d)?,
            config,
            embed,
            head,
            blocks,
        })
    }

    /// Every constant the forward pass uploads, at a context of `positions`.
    /// The embedding is read on the host and left out.
    pub(crate) fn footprint(&self, positions: usize) -> Uploads {
        let mut into = Uploads::default();
        self.head.footprint(&mut into);
        for block in &self.blocks {
            let attn = &block.attn;
            attn.qkv.footprint(&mut into);
            attn.output.footprint(&mut into);
            if let Some((v, v_bias)) = &attn.v {
                v.footprint(&mut into);
                v_bias.footprint(&mut into);
            }
            for gain in [
                &block.attn_norm,
                &attn.qkv_bias,
                &block.post_attn_norm,
                &block.ffn_norm,
                &block.post_ffn_norm,
            ] {
                gain.footprint(&mut into);
            }
            block.ffn.footprint(&mut into);
        }
        self.output_norm.footprint(&mut into);
        self.rope.footprint(&mut into, positions);
        into
    }

    /// Device bytes both caches of every block take per position.
    pub fn kv_bytes_per_token(&self) -> usize {
        let width = self.config.n_head_kv * self.config.head_dim;
        self.config.n_block * 2 * width * size_of::<u16>()
    }

    pub fn new_state(&self) -> State {
        State::new(self.config.n_block)
    }

    /// Runs `tokens`, advancing `state`, and returns the final position's
    /// logits.
    pub fn forward(&self, state: &mut State, tokens: &[u32], backend: &dyn Backend) -> Result<Vec<f32>> {
        let bufs = self.forward_to_logits(state, tokens, backend)?;
        let out = read_vec(backend, bufs.logits, self.config.vocab)?;
        bufs.release(backend);
        Ok(out)
    }

    /// [`Model::forward`] reading back only the winning token id.
    pub fn forward_greedy(&self, state: &mut State, tokens: &[u32], backend: &dyn Backend) -> Result<i64> {
        let bufs = self.forward_to_logits(state, tokens, backend)?;
        let id = backend.argmax(bufs.logits, self.config.vocab)?;
        bufs.release(backend);
        Ok(id)
    }

    fn forward_to_logits(&self, state: &mut State, tokens: &[u32], backend: &dyn Backend) -> Result<ForwardBufs> {
        ensure!(!tokens.is_empty(), "cannot run a forward pass over zero tokens");
        let cfg = &self.config;
        let (d, rows, eps) = (cfg.d_model, tokens.len(), cfg.rms_eps);

        let x = backend.upload(&embed_rows(&self.embed, tokens, d, cfg.vocab)?)?;
        let normed = backend.alloc(rows * d)?;
        // A mixer's output, held apart from the residual stream until its
        // post-norm has run.
        let mixed = backend.alloc(rows * d)?;

        backend.begin_pass(rows)?;

        for (block, cache) in self.blocks.iter().zip(&mut state.caches) {
            let gain = block.attn_norm.buf(backend)?;
            let act = backend.rms_norm_q(x, rows, d, gain, eps, normed)?;
            self.attention(&block.attn, normed, act, rows, state.pos, cache, backend, mixed)?;
            backend.rms_norm_add(mixed, rows, d, block.post_attn_norm.buf(backend)?, eps, normed, x)?;

            let gain = block.ffn_norm.buf(backend)?;
            let input = block.ffn.input().share_norm(backend, x, rows, gain, eps, normed)?;
            block.ffn.forward_into(backend, input, rows, mixed)?;
            input.release(backend);
            backend.rms_norm_add(mixed, rows, d, block.post_ffn_norm.buf(backend)?, eps, normed, x)?;
        }
        backend.release(mixed);

        state.pos += rows;
        head_logits(backend, x, rows, &self.output_norm, eps, &self.head, normed)
    }

    /// Attention over `x`, its normalized input, written into `out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        attn: &Attention,
        x: Buf,
        act: QAct,
        rows: usize,
        start_pos: usize,
        cache: &mut KvCache,
        backend: &dyn Backend,
        out: Buf,
    ) -> Result<()> {
        let cfg = &self.config;
        let spec = Attn {
            rows,
            start_pos,
            n_head: cfg.n_head,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
        };
        let (width, kv_width) = (cfg.n_head * cfg.head_dim, spec.kv_width());

        let qkv = attn.qkv.forward_act(backend, x, Some(act), rows)?;
        backend.add_rows(qkv, rows, attn.qkv.out_dim, attn.qkv_bias.buf(backend)?)?;
        // The parts sit side by side in each position's row.
        let part = |offset| Plane {
            buf: qkv,
            offset,
            pitch: attn.qkv.out_dim,
        };
        let mut scratch = vec![qkv];
        let value = match &attn.v {
            Some((v, v_bias)) => {
                let buf = v.forward_act(backend, x, Some(act), rows)?;
                backend.add_rows(buf, rows, kv_width, v_bias.buf(backend)?)?;
                scratch.push(buf);
                dense_plane(buf, kv_width)
            }
            None => part(width + kv_width),
        };

        let table = self.rope.buf(backend, spec.total())?;
        let rope = |heads| Rope {
            heads,
            head_dim: cfg.head_dim,
            rope_dim: cfg.rope_dim,
            start_pos,
        };
        // A decode step's query is the front of the projection and rotates
        // in place. Past one row the parts interleave and `rope_gather`
        // reads the query's strided window.
        let q = if rows == 1 {
            backend.rope(qkv, rows, table, rope(cfg.n_head))?;
            qkv
        } else {
            let buf = backend.alloc(rows * width)?;
            backend.rope_gather(part(0), rows, table, rope(cfg.n_head), buf)?;
            scratch.push(buf);
            buf
        };
        let k = backend.alloc(rows * kv_width)?;
        backend.rope_gather(part(width), rows, table, rope(cfg.n_head_kv), k)?;
        scratch.push(k);

        // A cached position holds every head together, so each store is one
        // contiguous row per position.
        let (keys, values) = cache.reserve(backend, spec.total(), kv_width)?;
        let landing = |buf| HPlane {
            buf,
            offset: start_pos * kv_width,
            pitch: kv_width,
        };
        backend.store_2d_pair(
            (value, landing(values)),
            (dense_plane(k, kv_width), landing(keys)),
            rows,
            kv_width,
        )?;

        let heads = backend.alloc(rows * width)?;
        backend.attention(q, keys, values, spec, heads)?;
        attn.output.project_into(backend, heads, rows, out)?;

        for buf in scratch.into_iter().chain([heads]) {
            backend.release(buf);
        }
        Ok(())
    }
}
