use anyhow::{Context, Result, ensure};

use crate::Gguf;
use crate::compute::{Attn, Backend, Buf, Plane, QAct, Rope, read_vec};
use crate::layers::{Ffn, Gain, KvCache, Linear, RopeTable};

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

    pub head_dim: usize, // width of one attention head. not necessarily d_model / n_head
    pub rope_dim: usize,
    pub rope_freq_base: f32,
}

impl Config {
    pub fn from_gguf(gguf: &Gguf) -> Result<Config> {
        let arch = gguf.architecture()?;
        ensure!(
            arch == "llama",
            "expected a llama model, found architecture '{arch}'"
        );
        let m = gguf.metadata();

        check_no_rope_scaling(gguf)?;

        let d_model = m.arch_count("embedding_length")?;
        let n_head = m.arch_count("attention.head_count")?;
        ensure!(n_head > 0, "a llama model needs at least one head");

        let head_dim = match m.arch_get("attention.key_length") {
            Some(_) => m.arch_count("attention.key_length")?, // use if specified
            None => d_model / n_head,                         // fallback otherwise
        };
        let vocab = gguf
            .tensor("token_embd.weight")
            .context("model has no token_embd.weight")?
            .row_major_dims()[0] as usize;

        Ok(Config {
            n_block: m.arch_count("block_count")?,
            d_model,
            d_ff: m.arch_count("feed_forward_length")?,
            vocab,
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
        })
    }
}

/// Refuse a file whose rotary frequencies are rescaled. Extending a model past
/// its trained window rescales the angles rather than changing the graph, so
/// such a file would load, run, and be quietly wrong at long context.
fn check_no_rope_scaling(gguf: &Gguf) -> Result<()> {
    let m = gguf.metadata();
    if let Some(kind) = m.arch_get("rope.scaling.type").and_then(|v| v.as_str()) {
        ensure!(
            kind == "none",
            "rope scaling '{kind}' is not implemented for the llama architecture"
        );
    }

    for key in ["rope.scaling.factor", "rope.scale_linear"] {
        if let Some(factor) = m.arch_get(key).and_then(|v| v.as_float()) {
            ensure!(
                factor == 1.0,
                "'{key}' is {factor}; rope scaling is not implemented for the llama architecture"
            );
        }
    }

    ensure!(
        gguf.tensor("rope_freqs.weight").is_none()
            && gguf.tensor("blk.0.rope_freqs.weight").is_none(),
        "model carries a rope_freqs table; rope scaling is not implemented for the llama architecture"
    );

    Ok(())
}

/// The permutation that turns ggml's `llama` rotary layout into the one
/// [`Backend::rope`] implements.
///
/// NeoX pairs element `i` with `i + rope_dim / 2`, which is what the backend
/// implements; `llama` pairs consecutive elements, `2i` with `2i + 1`.
/// Reordering a head's channels to `[0, 2, 4, ..., 1, 3, 5, ...]` maps the
/// second onto the first: destination pair `i` holds source elements `2i` and
/// `2i + 1`, and both conventions give pair `i` the same angle.
///
/// Applied to the query and key weights once at load, so it costs nothing per
/// token. Every use those two have is a dot product of one against the other,
/// which a permutation of both leaves alone. Channels past `rope_dim` pass
/// through in place.
fn neox_order(heads: usize, head_dim: usize, rope_dim: usize) -> Vec<usize> {
    let half = rope_dim / 2;
    (0..heads * head_dim)
        .map(|j| {
            let (head, at) = (j / head_dim, j % head_dim);
            let source = match at {
                _ if at >= rope_dim => at,
                _ if at < half => 2 * at,
                _ => 2 * (at - half) + 1,
            };
            head * head_dim + source
        })
        .collect()
}

fn dense_plane(buf: Buf, width: usize) -> Plane {
    Plane {
        buf,
        offset: 0,
        pitch: width,
    }
}

/// Grouped-query attention: one fused query/key/value projection, rotary
/// embeddings, causal softmax attention over a growing cache, and an output
/// projection.
struct Attention {
    /// Query, key and value stacked in that order. They read the same
    /// normalized row, so one launch over a wide output replaces three, two of
    /// them only `n_head_kv * head_dim` wide.
    qkv: Linear,
    output: Linear,
}

impl Attention {
    fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<Attention> {
        let (d, width, kv_width) = (
            cfg.d_model,
            cfg.n_head * cfg.head_dim,
            cfg.n_head_kv * cfg.head_dim,
        );
        // The query and the key rotate, so they move into the backend's rotary
        // layout. See `neox_order`.
        let q = Linear::load(gguf, &format!("{prefix}.attn_q.weight"), d, width)?
            .reorder_outputs("neox", &neox_order(cfg.n_head, cfg.head_dim, cfg.rope_dim))?;
        let k = Linear::load(gguf, &format!("{prefix}.attn_k.weight"), d, kv_width)?
            .reorder_outputs(
                "neox",
                &neox_order(cfg.n_head_kv, cfg.head_dim, cfg.rope_dim),
            )?;
        let v = Linear::load(gguf, &format!("{prefix}.attn_v.weight"), d, kv_width)?;

        Ok(Attention {
            qkv: Linear::fuse(&[&q, &k, &v])?,
            output: Linear::load(gguf, &format!("{prefix}.attn_output.weight"), width, d)?,
        })
    }
}

struct Block {
    attn_norm: Gain,
    attn: Attention,
    ffn_norm: Gain,
    ffn: Ffn,
}

pub struct Model {
    pub config: Config,
    /// `token_embd.weight`, kept quantized.
    embed: Linear,
    /// `output.weight`, or the embedding again when the file ties them, kept
    /// quantized.
    head: Linear,
    blocks: Vec<Block>,
    output_norm: Gain,
    rope: RopeTable,
}

impl Model {
    /// Read every weight this architecture needs, leaving quantized ones so
    pub fn load(gguf: &Gguf) -> Result<Model> {
        let config = Config::from_gguf(gguf)?;
        ensure!(
            config.head_dim > 0
                && config.rope_dim <= config.head_dim
                && config.rope_dim.is_multiple_of(2),
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

        let embed = Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?;

        // same as llama.cpp
        let head = match gguf.tensor("output.weight") {
            Some(_) => Linear::load(gguf, "output.weight", config.d_model, config.vocab)?,
            None => Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?,
        };

        let mut blocks = Vec::with_capacity(config.n_block);
        for index in 0..config.n_block {
            let prefix = format!("blk.{index}");
            blocks.push(Block {
                attn_norm: Gain::load(gguf, &format!("{prefix}.attn_norm.weight"), config.d_model)?,
                attn: Attention::load(gguf, &prefix, &config)?,
                ffn_norm: Gain::load(gguf, &format!("{prefix}.ffn_norm.weight"), config.d_model)?,
                ffn: Ffn::load(gguf, &prefix, config.d_model, config.d_ff)?,
            });
        }

        Ok(Model {
            rope: RopeTable::new(config.rope_dim, config.rope_freq_base),
            output_norm: Gain::load(gguf, "output_norm.weight", config.d_model)?,
            config,
            embed,
            head,
            blocks,
        })
    }

    /// Fresh generation state.
    pub fn new_state(&self) -> State {
        State {
            pos: 0,
            caches: (0..self.config.n_block)
                .map(|_| KvCache::default())
                .collect(),
        }
    }

    /// Run `tokens`, advancing `state`, and return the final position's logits.
    /// Only the last row is projected through the LM head.
    pub fn forward(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty(),
            "cannot run a forward pass over zero tokens"
        );

        let cfg = &self.config;
        let (d, rows) = (cfg.d_model, tokens.len());

        let mut host_x = vec![0.0f32; rows * d];
        for (t, &token) in tokens.iter().enumerate() {
            let id = token as usize;
            ensure!(
                id < cfg.vocab,
                "token id {id} is outside the {}-entry vocabulary",
                cfg.vocab
            );
            self.embed.row_into(id, &mut host_x[t * d..(t + 1) * d])?;
        }

        let x = backend.upload(&host_x)?;
        let normed = backend.alloc(rows * d)?;

        // Everything from here to the logits is device-only
        backend.begin_pass()?;

        // For debugging purposes; prints rms magnitude
        // of activation vectors per token at the end
        // of each transformer block
        let trace = std::env::var_os("PHOBOS_TRACE").is_some();

        for (index, (block, cache)) in self.blocks.iter().zip(&mut state.caches).enumerate() {
            // The normalization leaves the quantized copy behind too, which the
            // projection reading it would otherwise redo.
            let act = backend.rms_norm_q(
                x,
                rows,
                d,
                block.attn_norm.buf(backend)?,
                cfg.rms_eps,
                normed,
            )?;
            self.attention(&block.attn, normed, act, rows, state.pos, cache, backend, x)?;

            let act = backend.rms_norm_q(
                x,
                rows,
                d,
                block.ffn_norm.buf(backend)?,
                cfg.rms_eps,
                normed,
            )?;
            block.ffn.forward(backend, normed, act, rows, x)?;

            if trace {
                let seen = read_vec(backend, x, rows * d)?;
                let per_row: Vec<String> = seen
                    .chunks_exact(d)
                    .map(|r| {
                        format!(
                            "{:>10.6}",
                            (r.iter().map(|&v| v * v).sum::<f32>() / d as f32).sqrt()
                        )
                    })
                    .collect();

                eprintln!("  blk {index:>2} rows [{}]", per_row.join(" "));
            }
        }

        state.pos += rows;

        backend.rms_norm(
            x,
            rows,
            d,
            self.output_norm.buf(backend)?,
            cfg.rms_eps,
            normed,
        )?;

        // Only the final position goes through the LM head, the largest weight
        // in the model; the other normalized rows are dead.
        let last = backend.alloc(d)?;
        backend.copy(normed, (rows - 1) * d, last, 0, d)?;

        let logits = backend.alloc(cfg.vocab)?;
        self.head.project_into(backend, last, 1, logits)?;

        backend.end_pass()?;

        let out = read_vec(backend, logits, cfg.vocab)?;
        for buf in [x, normed, last, logits] {
            backend.release(buf);
        }

        Ok(out)
    }

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
        dest: Buf,
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

        let qkv = attn.qkv.forward_act(backend, x, act, rows)?;

        // The three parts (QKV) sit side by side in fused order.
        // Each is a window of one position's row of the result.
        let part = |offset| Plane {
            buf: qkv,
            offset,
            pitch: attn.qkv.out_dim,
        };

        // The caches hold every head of one position together, so a cached
        // position is a contiguous row and the value plane goes from the
        // projection straight into the cache.
        let (keys, values) = cache.reserve(backend, spec.total(), kv_width)?;
        let landing = Plane {
            buf: values,
            offset: start_pos * kv_width,
            pitch: kv_width,
        };

        backend.copy_2d(part(width + kv_width), landing, rows, kv_width)?;

        // A decode step's query is the front of the projection and already
        // contiguous, so it rotates and attends where it lies. Past one row the
        // three parts interleave and it has to be pulled out.
        let mut scratch = Vec::new();
        let q = if rows == 1 {
            qkv
        } else {
            let buf = backend.alloc(rows * width)?;
            backend.copy_2d(part(0), dense_plane(buf, width), rows, width)?;
            scratch.push(buf);
            buf
        };

        // The key never can: the rotation is in place and takes no offset, so
        // it needs a buffer starting at its own first element.
        let k = backend.alloc(rows * kv_width)?;
        backend.copy_2d(part(width), dense_plane(k, kv_width), rows, kv_width)?;
        scratch.push(k);

        let table = self.rope.buf(backend, spec.total())?;

        for (buf, heads) in [(q, cfg.n_head), (k, cfg.n_head_kv)] {
            backend.rope(
                buf,
                rows,
                table,
                Rope {
                    heads,
                    head_dim: cfg.head_dim,
                    rope_dim: cfg.rope_dim,
                    start_pos,
                },
            )?;
        }

        backend.copy(k, 0, keys, start_pos * kv_width, rows * kv_width)?;

        let mixed = backend.alloc(rows * width)?;
        backend.attention(q, keys, values, spec, mixed)?;
        attn.output.add_into(backend, mixed, rows, dest)?;

        for buf in [qkv, mixed].into_iter().chain(scratch) {
            backend.release(buf);
        }
        Ok(())
    }
}

/// State is one key/value cache per block.
pub struct State {
    pos: usize,
    caches: Vec<KvCache>,
}

impl State {
    /// Tokens consumed so far.
    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Hands every device allocation the state holds back to the backend.
    ///
    /// Dropping a state instead strands its caches: a [`Buf`] is a handle, not
    /// an owner.
    pub fn release(&mut self, backend: &dyn Backend) {
        for cache in &mut self.caches {
            cache.release(backend);
        }
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotary_order_maps_consecutive_pairs_onto_split_ones() {
        // One head of eight: a NeoX rotation pairs (0, 4), (1, 5), (2, 6),
        // (3, 7), which have to become the consecutive pairs llama rotates.
        let order = neox_order(1, 8, 8);
        assert_eq!(order, vec![0, 2, 4, 6, 1, 3, 5, 7]);
        for i in 0..4 {
            assert_eq!((order[i], order[i + 4]), (2 * i, 2 * i + 1));
        }
    }

    #[test]
    fn the_rotary_order_leaves_unrotated_channels_and_later_heads_in_place() {
        // Six channels of a head rotate and the last two pass through.
        assert_eq!(neox_order(1, 8, 6), vec![0, 2, 4, 1, 3, 5, 6, 7]);
        // Each head is permuted within itself.
        let order = neox_order(2, 4, 4);
        assert_eq!(order, vec![0, 2, 1, 3, 4, 6, 5, 7]);
    }
}
