use anyhow::{Context, Result, bail, ensure};

use crate::Gguf;
use crate::compute::{Attn, Backend, Buf, DeltaMix, Plane, QAct, Rope, read_vec};
use crate::layers::{Ffn, Gain, KvCache, Linear, RopeTable, check_dims};

#[derive(Clone, Debug)]
pub struct Config {
    /// Blocks that take part in ordinary decoding, excluding the trailing
    /// multi-token-prediction block.
    pub n_block: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub context_length: usize,
    pub rms_eps: f32,

    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// Blocks where `(index + 1) % interval == 0` use softmax attention.
    pub full_attention_interval: usize,
    /// Rotary embeddings cover only the first `rope_dim` of each head.
    pub rope_dim: usize,
    pub rope_freq_base: f32,

    pub ssm_heads: usize,
    pub ssm_inner: usize,
    pub ssm_head_dim: usize,
    pub conv_kernel: usize,
}

impl Config {
    pub fn from_gguf(gguf: &Gguf) -> Result<Config> {
        let arch = gguf.architecture()?;
        ensure!(
            arch == "qwen35",
            "expected a qwen35 model, found architecture '{arch}'"
        );

        let m = gguf.metadata();

        let n_block_total = m.arch_count("block_count")?;

        // The trailing nextn blocks predict extra tokens for speculative
        // decoding and take no part in a plain autoregressive pass.
        let nextn = m
            .arch_get("nextn_predict_layers")
            .and_then(|v| v.as_int())
            .unwrap_or(0);

        let n_block = n_block_total
            .checked_sub(usize::try_from(nextn).unwrap_or(0))
            .context("nextn_predict_layers exceeds block_count")?;

        let ssm_heads = m.arch_count("ssm.group_count")?;
        let ssm_inner = m.arch_count("ssm.inner_size")?;
        ensure!(
            ssm_heads > 0 && ssm_inner.is_multiple_of(ssm_heads),
            "ssm inner size {ssm_inner} does not split into {ssm_heads} heads"
        );

        let vocab = gguf
            .tensor("token_embd.weight")
            .context("model has no token_embd.weight")?
            .row_major_dims()[0] as usize;

        Ok(Config {
            n_block,
            d_model: m.arch_count("embedding_length")?,
            d_ff: m.arch_count("feed_forward_length")?,
            vocab,
            context_length: m.arch_count("context_length")?,
            rms_eps: m.arch_float("attention.layer_norm_rms_epsilon")?,
            n_head: m.arch_count("attention.head_count")?,
            n_head_kv: m.arch_count("attention.head_count_kv")?,
            head_dim: m.arch_count("attention.key_length")?,
            full_attention_interval: m.arch_count("full_attention_interval")?,
            rope_dim: m.arch_count("rope.dimension_count")?,
            rope_freq_base: m.arch_float("rope.freq_base")?,
            ssm_heads,
            ssm_inner,
            ssm_head_dim: ssm_inner / ssm_heads,
            conv_kernel: m.arch_count("ssm.conv_kernel")?,
        })
    }

    pub fn is_attention_block(&self, index: usize) -> bool {
        self.full_attention_interval > 0 && (index + 1).is_multiple_of(self.full_attention_interval)
    }
}

/// Points where the tensor extents alone do not pin the architecture down.
///
/// [`Variants::REFERENCE`] is llama.cpp's `qwen35` impl and our default.
/// 
/// The alternatives are kept because the same ambiguities
/// recur in every GGUF architecture, and sweeping them (`examples/sweep.rs`)
/// against a repeated-phrase probe is faster than reading a graph builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variants {
    /// `attn_qkv` groups query/key/value per head rather than as three
    /// contiguous blocks.
    pub qkv_interleaved: bool,
    /// `attn_q` emits all queries then all gates, rather than pairing them
    /// inside each head's slice.
    pub attn_gate_contiguous: bool,
    /// The gate precedes the query wherever they are split.
    pub attn_gate_first: bool,
    /// Normalize before applying the output gate rather than after.
    pub norm_before_gate: bool,
    /// `ssm_a` holds `log(A)`, as the HuggingFace checkpoint does. The GGUF
    /// converter instead stores `-exp(A_log)` and llama.cpp multiplies by it
    /// directly, so this is false for GGUF files.
    pub decay_from_log: bool,
    /// L2-normalize delta-rule queries and keys.
    pub l2_normalize_qk: bool,
    /// `ssm_alpha` supplies the delta-rule write strength and `ssm_beta` the
    /// decay, rather than the other way round.
    pub swap_alpha_beta: bool,
    /// The convolution's taps run newest-first.
    pub conv_reversed: bool,
    /// The delta-rule readout contracts the value axis instead of the key axis.
    pub query_contracts_value: bool,
}

impl Variants {
    pub const REFERENCE: Variants = Variants {
        qkv_interleaved: false,
        attn_gate_contiguous: false,
        attn_gate_first: false,
        norm_before_gate: true,
        decay_from_log: false,
        l2_normalize_qk: true,
        swap_alpha_beta: false,
        conv_reversed: false,
        query_contracts_value: false,
    };

    /// The combinations the sweep explores.
    /// 
    /// The query/gate split, the decay formula and the delta-rule contraction axis are settled and held fixed;
    /// the gated-norm order, the L2 normalization, the convolution tap order and the fused-projection layout
    /// are worth re-checking against a new checkpoint.
    pub fn all() -> Vec<Variants> {
        (0..16u32)
            .map(|bits| Variants {
                qkv_interleaved: bits & 1 != 0,
                norm_before_gate: bits & 2 != 0,
                conv_reversed: bits & 4 != 0,
                attn_gate_contiguous: false,
                attn_gate_first: false,
                decay_from_log: false,
                l2_normalize_qk: bits & 8 == 0,
                swap_alpha_beta: false,
                query_contracts_value: false,
            })
            .collect()
    }
}

impl Default for Variants {
    fn default() -> Variants {
        Variants::REFERENCE
    }
}

/// Gated grouped-query attention with per-head QK normalization and partial
/// rotary embeddings.
struct Attention {
    /// Emits `head_dim * 2` per head: the query and its output gate.
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    q_norm: Gain,
    k_norm: Gain,
}

impl Attention {
    fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<Attention> {
        let d = cfg.d_model;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;

        Ok(Attention {
            q: Linear::load(
                gguf,
                &format!("{prefix}.attn_q.weight"),
                d,
                cfg.n_head * cfg.head_dim * 2,
            )?,
            k: Linear::load(gguf, &format!("{prefix}.attn_k.weight"), d, kv_dim)?,
            v: Linear::load(gguf, &format!("{prefix}.attn_v.weight"), d, kv_dim)?,
            output: Linear::load(
                gguf,
                &format!("{prefix}.attn_output.weight"),
                cfg.n_head * cfg.head_dim,
                d,
            )?,
            q_norm: Gain::load(gguf, &format!("{prefix}.attn_q_norm.weight"), cfg.head_dim)?,
            k_norm: Gain::load(gguf, &format!("{prefix}.attn_k_norm.weight"), cfg.head_dim)?,
        })
    }
}

/// A GatedDeltaNet block: a causal depthwise convolution over the fused q/k/v
/// stream, then the gated delta rule, then a gated RMSNorm.
struct DeltaNet {
    /// The query-key-value, gate, decay and beta projections stacked. They read
    /// the same normalized row, and the decay and beta are sixteen wide apiece.
    /// See [`Linear::fuse`].
    projections: Linear,

    /// Where each part starts in the stacked output, and how wide it is.
    parts: [(usize, usize); 4],

    /// `[kernel, channels]` row-major, transposed from the file so one tap
    /// across a run of channels is contiguous, which is how both backends read
    /// it. The file groups each channel's taps instead, making every load of a
    /// channel tile a stride.
    conv_taps: Gain,

    /// Log-space decay rate per head.
    a_log: Gain,
    
    dt_bias: Gain,
    norm: Gain,
    out: Linear,
}

impl DeltaNet {
    fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<DeltaNet> {
        let d = cfg.d_model;
        let channels = 3 * cfg.ssm_inner;

        let conv_name = format!("{prefix}.ssm_conv1d.weight");
        let conv_info = gguf
            .tensor(&conv_name)
            .with_context(|| format!("missing tensor '{conv_name}'"))?;
        check_dims(conv_info, &[cfg.conv_kernel as u64, channels as u64])?;
        let per_channel = gguf.dequantize(&conv_name)?;
        let mut taps = vec![0.0f32; per_channel.len()];
        for c in 0..channels {
            for k in 0..cfg.conv_kernel {
                taps[k * channels + c] = per_channel[c * cfg.conv_kernel + k];
            }
        }

        let qkv = Linear::load(gguf, &format!("{prefix}.attn_qkv.weight"), d, channels)?;
        let gate = Linear::load(
            gguf,
            &format!("{prefix}.attn_gate.weight"),
            d,
            cfg.ssm_inner,
        )?;
        let alpha = Linear::load(
            gguf,
            &format!("{prefix}.ssm_alpha.weight"),
            d,
            cfg.ssm_heads,
        )?;
        let beta = Linear::load(gguf, &format!("{prefix}.ssm_beta.weight"), d, cfg.ssm_heads)?;
        let mut parts = [(0usize, 0usize); 4];
        let mut at = 0;
        for (slot, part) in parts.iter_mut().zip([&qkv, &gate, &alpha, &beta]) {
            *slot = (at, part.out_dim);
            at += part.out_dim;
        }

        Ok(DeltaNet {
            projections: Linear::fuse(&[&qkv, &gate, &alpha, &beta])?,
            parts,
            conv_taps: Gain::derived(format!("{conv_name}.taps"), taps),
            a_log: Gain::load(gguf, &format!("{prefix}.ssm_a"), cfg.ssm_heads)?,
            dt_bias: Gain::load(gguf, &format!("{prefix}.ssm_dt.bias"), cfg.ssm_heads)?,
            norm: Gain::load(gguf, &format!("{prefix}.ssm_norm.weight"), cfg.ssm_head_dim)?,
            out: Linear::load(gguf, &format!("{prefix}.ssm_out.weight"), cfg.ssm_inner, d)?,
        })
    }

    /// The convolution taps as the backend reads them, newest-first if the
    /// architecture runs them that way. The reversal goes up under its own key,
    /// and only the sweep asks for it: no GGUF file stores them reversed.
    fn taps(&self, backend: &dyn Backend, channels: usize, reversed: bool) -> Result<Buf> {
        if !reversed {
            return self.conv_taps.buf(backend);
        }

        let taps = &self.conv_taps.data;
        let kernel = taps.len() / channels;
        let mut flipped = vec![0.0f32; taps.len()];
        for k in 0..kernel {
            let src = (kernel - 1 - k) * channels;
            flipped[k * channels..][..channels].copy_from_slice(&taps[src..][..channels]);
        }
        
        Gain::derived(format!("{}.reversed", self.conv_taps.key), flipped).buf(backend)
    }

    /// The per-head decay rate. A GGUF file stores `-exp(A_log)` and llama.cpp
    /// multiplies by it directly; the HuggingFace checkpoint stores `A_log`, so
    /// that reading has to exponentiate.
    fn rate(&self, backend: &dyn Backend, from_log: bool) -> Result<Buf> {
        if !from_log {
            return self.a_log.buf(backend);
        }

        let rates = self.a_log.data.iter().map(|&v| -v.exp()).collect();
        Gain::derived(format!("{}.rate", self.a_log.key), rates).buf(backend)
    }
}

enum Mixer {
    Attention(Box<Attention>),
    DeltaNet(Box<DeltaNet>),
}

struct Block {
    attn_norm: Gain,
    /// Despite the name this gates the FFN input, as in a standard pre-norm
    /// transformer block.
    post_attn_norm: Gain,
    mixer: Mixer,
    ffn: Ffn,
}

// Qwen3.5 Model
pub struct Model {
    pub config: Config,
    /// `token_embd.weight`, serving as both the tied LM head and the embedding
    /// table.
    ///
    /// Kept quantized. This way embedding lookup becomes one contiguous row.
    head: Linear,
    blocks: Vec<Block>,
    output_norm: Gain,
    rope: RopeTable,
}

impl Model {
    /// Read every weight this architecture needs, leaving quantized ones so.
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

        let head = Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?;

        let mut blocks = Vec::with_capacity(config.n_block);
        for index in 0..config.n_block {
            let prefix = format!("blk.{index}");
            let mixer = if config.is_attention_block(index) {
                Mixer::Attention(Box::new(Attention::load(gguf, &prefix, &config)?))
            } else {
                Mixer::DeltaNet(Box::new(DeltaNet::load(gguf, &prefix, &config)?))
            };
            blocks.push(Block {
                attn_norm: Gain::load(gguf, &format!("{prefix}.attn_norm.weight"), config.d_model)?,
                post_attn_norm: Gain::load(
                    gguf,
                    &format!("{prefix}.post_attention_norm.weight"),
                    config.d_model,
                )?,
                mixer,
                ffn: Ffn::load(gguf, &prefix, config.d_model, config.d_ff)?,
            });
        }

        let output_norm = Gain::load(gguf, "output_norm.weight", config.d_model)?;
        ensure!(
            gguf.tensor("output.weight").is_none(),
            "model carries a separate output.weight; the tied LM head assumed here is wrong for it"
        );

        Ok(Model {
            rope: RopeTable::new(config.rope_dim, config.rope_freq_base),
            config,
            head,
            blocks,
            output_norm,
        })
    }

    /// Fresh generation state.
    pub fn new_state(&self) -> State {
        let cfg = &self.config;
        let layers = (0..cfg.n_block)
            .map(|index| {
                if cfg.is_attention_block(index) {
                    LayerState::Attention(KvCache::default())
                } else {
                    LayerState::DeltaNet {
                        carry: None,
                        recurrent: None,
                    }
                }
            })
            .collect();
        State { pos: 0, layers }
    }

    /// Run `tokens`, advancing `state`, and return the final position's logits.
    /// Only the last row is projected through the LM head.
    pub fn forward(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<Vec<f32>> {
        self.forward_with(state, tokens, backend, Variants::REFERENCE)
    }

    /// [`Model::forward`] with the architecture choices spelled out, for the
    /// sweep that resolves them.
    pub fn forward_with(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
        variants: Variants,
    ) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty(),
            "cannot run a forward pass over zero tokens"
        );
        let cfg = &self.config;
        let d = cfg.d_model;
        let rows = tokens.len();

        let mut host_x = vec![0.0f32; rows * d];
        for (t, &token) in tokens.iter().enumerate() {
            let id = token as usize;
            ensure!(
                id < cfg.vocab,
                "token id {id} is outside the {}-entry vocabulary",
                cfg.vocab
            );
            self.head.row_into(id, &mut host_x[t * d..(t + 1) * d])?;
        }

        let x = backend.upload(&host_x)?;
        let normed = backend.alloc(rows * d)?;

        // Everything from here to the logits is device-only, which lets a
        // backend take the whole pass as one unit.
        backend.begin_pass()?;

        let trace = std::env::var_os("PHOBOS_TRACE").is_some();
        for (index, (block, layer_state)) in self.blocks.iter().zip(&mut state.layers).enumerate() {
            // The normalization leaves the quantized copy behind too, which the
            // projections reading it would otherwise redo.
            let act = backend.rms_norm_q(
                x,
                rows,
                d,
                block.attn_norm.buf(backend)?,
                cfg.rms_eps,
                normed,
            )?;

            // Both mixers end by adding their output projection into the
            // residual stream, which the projection does itself.
            match (&block.mixer, layer_state) {
                (Mixer::Attention(attn), LayerState::Attention(cache)) => self.attention(
                    attn, normed, act, rows, state.pos, cache, backend, variants, x,
                )?,
                (Mixer::DeltaNet(delta), LayerState::DeltaNet { carry, recurrent }) => self
                    .delta_net(
                        delta, normed, act, rows, carry, recurrent, backend, variants, x,
                    )?,
                _ => bail!("generation state does not match the model's block layout"),
            }

            let act = backend.rms_norm_q(
                x,
                rows,
                d,
                block.post_attn_norm.buf(backend)?,
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
        variants: Variants,
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
        let (dim, width, kv_width) = (cfg.head_dim, cfg.n_head * cfg.head_dim, spec.kv_width());

        let q_gate_buf = attn.q.forward_act(backend, x, act, rows)?;
        let k_buf = attn.k.forward_act(backend, x, act, rows)?;
        let v_buf = attn.v.forward_act(backend, x, act, rows)?;

        // Split the query from its output gate. Pairing the two inside each
        // head's slice or stacking them as whole blocks changes only where the
        // copy starts and how far apart its rows are.
        let (blocks, block) = if variants.attn_gate_contiguous {
            (rows, width)
        } else {
            (rows * cfg.n_head, dim)
        };

        let (q_at, gate_at) = if variants.attn_gate_first {
            (block, 0)
        } else {
            (0, block)
        };
        
        let q = backend.alloc(rows * width)?;
        let gate = backend.alloc(rows * width)?;
        for (offset, dst) in [(q_at, q), (gate_at, gate)] {
            backend.copy_2d(
                Plane {
                    buf: q_gate_buf,
                    offset,
                    pitch: 2 * block,
                },
                Plane {
                    buf: dst,
                    offset: 0,
                    pitch: block,
                },
                blocks,
                block,
            )?;
        }

        // QK normalization is per head, before the rotary embedding. Both are
        // already a run of `head_dim` heads, the row the norm wants.
        let q_normed = backend.alloc(rows * width)?;
        let k_normed = backend.alloc(rows * kv_width)?;
        let eps = cfg.rms_eps;
        backend.rms_norm(
            q,
            rows * cfg.n_head,
            dim,
            attn.q_norm.buf(backend)?,
            eps,
            q_normed,
        )?;
        backend.rms_norm(
            k_buf,
            rows * cfg.n_head_kv,
            dim,
            attn.k_norm.buf(backend)?,
            eps,
            k_normed,
        )?;

        let table = self.rope.buf(backend, spec.total())?;
        for (buf, heads) in [(q_normed, cfg.n_head), (k_normed, cfg.n_head_kv)] {
            backend.rope(
                buf,
                rows,
                table,
                Rope {
                    heads,
                    head_dim: dim,
                    rope_dim: cfg.rope_dim,
                    start_pos,
                },
            )?;
        }

        // The caches hold every head of one position together, so appending is
        // one contiguous copy rather than one per head, and the kernel reads a
        // head as a column window.
        let (keys, values) = cache.reserve(backend, spec.total(), kv_width)?;
        backend.copy(k_normed, 0, keys, start_pos * kv_width, rows * kv_width)?;
        backend.copy(v_buf, 0, values, start_pos * kv_width, rows * kv_width)?;

        let mixed = backend.alloc(rows * width)?;
        backend.attention(q_normed, keys, values, spec, mixed)?;
        backend.gate_into(mixed, gate)?;

        attn.output.add_into(backend, mixed, rows, dest)?;
        for buf in [q_gate_buf, k_buf, v_buf, q, gate, q_normed, k_normed, mixed] {
            backend.release(buf);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_net(
        &self,
        delta: &DeltaNet,
        x: Buf,
        act: QAct,
        rows: usize,
        carry: &mut Option<(Buf, usize)>,
        recurrent: &mut Option<Buf>,
        backend: &dyn Backend,
        variants: Variants,
        dest: Buf,
    ) -> Result<()> {
        let cfg = &self.config;
        let (heads, head_dim) = (cfg.ssm_heads, cfg.ssm_head_dim);
        let inner = cfg.ssm_inner;

        ensure!(
            !variants.query_contracts_value,
            "the delta-rule readout that contracts the value axis has no backend op; it is not the reference layout and only the sweep ever asked for it"
        );

        // Both fused layouts are the same strided read, so which one the file
        // uses is an offset and a stride, not a case in the kernel.
        let (planes, head_stride) = if variants.qkv_interleaved {
            ([0, head_dim, 2 * head_dim], 3 * head_dim)
        } else {
            ([0, inner, 2 * inner], head_dim)
        };

        let mix = DeltaMix {
            rows,
            heads,
            head_dim,
            kernel: cfg.conv_kernel,
            planes,
            head_stride,
            normalize: variants.l2_normalize_qk,
            // The readout carries the same 1/sqrt(d) scale as softmax attention.
            query_scale: (head_dim as f32).sqrt().recip(),
        };

        let (channels, carried) = (mix.channels(), mix.pad() * mix.channels());

        // One projection for all four, off the quantized copy the caller's
        // normalization left behind. Each part is then a window of the result:
        // at a single row an offset, past one row a strided copy apiece.
        let stacked = delta.projections.forward_act(backend, x, act, rows)?;
        let width = delta.projections.out_dim;

        // The query/key/value plane is not among these: it is the widest of the
        // four and the convolution's stream is its only reader, so it goes
        // there directly below.
        let mut planes = [(stacked, 0usize); 3];
        let mut extracted = Vec::new();
        for (plane, &(at, part)) in planes.iter_mut().zip(&delta.parts[1..]) {
            *plane = if rows == 1 {
                (stacked, at)
            } else {
                let buf = backend.alloc(rows * part)?;
                backend.copy_2d(
                    Plane {
                        buf: stacked,
                        offset: at,
                        pitch: width,
                    },
                    Plane {
                        buf,
                        offset: 0,
                        pitch: part,
                    },
                    rows,
                    part,
                )?;
                extracted.push(buf);
                (buf, 0)
            };
        }
        let [(z_buf, z_at), (alpha_buf, alpha_at), (beta_buf, beta_at)] = planes;

        // The convolution reads one padded stream, so the positions carried
        // from the previous call go in front of this call's projection and the
        // kernel has no boundary case.
        //
        // What is carried is the previous call's whole stream, and this call's
        // is a fresh allocation. Keeping the tail in a buffer of its own needed
        // a third copy to refill it, and refilling in place is not open either:
        // shifting a buffer into itself races a launch's programs against each
        // other whenever the shift is shorter than the buffer, which is every
        // decode step.
        let history = backend.alloc(mix.history_len())?;
        match *carry {
            Some((previous, len)) => {
                // The tail, wherever it falls: the last call may have had a
                // different number of rows, a prompt followed by decoding.
                backend.copy(previous, len - carried, history, 0, carried)?;
                backend.release(previous);
            }
            None => {
                let zeros = backend.zeroed(carried)?;
                backend.copy(zeros, 0, history, 0, carried)?;
                backend.release(zeros);
            }
        }

        // Straight from the projection into the stream: pulling the plane out
        // first was two passes over the widest of the four, a quarter of a
        // gigabyte a pass across the blocks at 512 positions. A decode step's
        // one row is already contiguous and the strided kernel is the wrong
        // shape for it.
        let (qkv_at, _) = delta.parts[0];
        if rows == 1 {
            backend.copy(stacked, qkv_at, history, carried, channels)?;
        } else {
            backend.copy_2d(
                Plane {
                    buf: stacked,
                    offset: qkv_at,
                    pitch: width,
                },
                Plane {
                    buf: history,
                    offset: carried,
                    pitch: channels,
                },
                rows,
                channels,
            )?;
        }
        *carry = Some((history, mix.history_len()));

        // The delta rule's five operands are one allocation, so each is a
        // window of it and producing them is two launches writing in place.
        let packed = backend.alloc(mix.packed_len())?;
        let taps = delta.taps(backend, channels, variants.conv_reversed)?;
        backend.delta_conv(history, taps, mix, packed)?;

        let (decay_in, beta_in) = if variants.swap_alpha_beta {
            ((beta_buf, beta_at), (alpha_buf, alpha_at))
        } else {
            ((alpha_buf, alpha_at), (beta_buf, beta_at))
        };
        backend.delta_gates(
            decay_in.0,
            decay_in.1,
            beta_in.0,
            beta_in.1,
            delta.rate(backend, variants.decay_from_log)?,
            delta.dt_bias.buf(backend)?,
            mix,
            packed,
        )?;

        // The recurrent state is why this op exists on the backend at all: a
        // [head_dim, head_dim] matrix per head, so keeping it on the host means
        // moving a megabyte in and out per block per token.
        let state = match *recurrent {
            Some(buf) => buf,
            None => {
                let buf = backend.zeroed(heads * head_dim * head_dim)?;
                *recurrent = Some(buf);
                buf
            }
        };

        let n = rows * inner;
        let mixed_buf = backend.alloc(n)?;
        backend.delta_rule(packed, rows, heads, head_dim, state, mixed_buf)?;

        // Gated RMSNorm: the gate multiplies before normalization, as in the
        // Mamba2-style RMSNormGated this architecture inherits. swiglu is
        // silu(gate) * up, the multiply this wants.
        let scratch = backend.alloc(n)?;
        let gain = delta.norm.buf(backend)?;
        let (gated, gated_act) = if variants.norm_before_gate {
            // Into a destination that is not the source: the gate reads one
            // element per thread, but the norm's reduction reads the whole row.
            let readout = backend.rms_norm_gated(
                mixed_buf,
                rows * heads,
                head_dim,
                gain,
                cfg.rms_eps,
                z_buf,
                z_at,
                scratch,
            )?;
            (scratch, Some(readout))
        } else {
            backend.swiglu(z_buf, z_at, mixed_buf, 0, scratch, rows * inner)?;
            backend.rms_norm(
                scratch,
                rows * heads,
                head_dim,
                gain,
                cfg.rms_eps,
                mixed_buf,
            )?;
            (mixed_buf, None)
        };

        match gated_act {
            Some(act) => delta.out.add_into_act(backend, act, rows, dest)?,
            None => delta.out.add_into(backend, gated, rows, dest)?,
        }
        for buf in [stacked, packed, scratch, mixed_buf]
            .into_iter()
            .chain(extracted)
        {
            backend.release(buf);
        }
        Ok(())
    }
}

/// Per-block generation state: a K/V cache for attention blocks, and the
/// convolution window plus recurrent matrix for GatedDeltaNet blocks.
enum LayerState {
    Attention(KvCache),
    DeltaNet {
        /// Both are allocated on first use, since [`Model::new_state`] has no
        /// backend to allocate from. Zero-filled, the identity for the
        /// convolution's history and the recurrence alike.
        ///
        /// The carry is the previous call's whole convolution stream and its
        /// length: a prompt and a decode step leave streams of different
        /// lengths, and only the last few positions of either are wanted.
        carry: Option<(Buf, usize)>,
        recurrent: Option<Buf>,
    },
}

pub struct State {
    pos: usize,
    layers: Vec<LayerState>,
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
    /// Dropping a state instead strands its caches, since a [`Buf`] is a
    /// handle, not an owner.
    pub fn release(&mut self, backend: &dyn Backend) {
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention(cache) => cache.release(backend),
                LayerState::DeltaNet { carry, recurrent } => {
                    if let Some((buf, _)) = carry.take() {
                        backend.release(buf);
                    }
                    if let Some(buf) = recurrent.take() {
                        backend.release(buf);
                    }
                }
            }
        }
        self.pos = 0;
    }
}
