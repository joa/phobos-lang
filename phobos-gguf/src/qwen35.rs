use anyhow::{Context, Result, ensure};

use crate::Gguf;
use crate::backend::{
    Attn, Backend, Buf, DeltaMix, FusedMix, FusedProject, HPlane, Plane, ProjRun, QAct, Rope,
};
use crate::layers::{Ffn, Gain, KvCache, Linear, RopeTable, Uploads};

mod delta_net;
mod forward;
mod variants;

use delta_net::{DeltaNet, Proj};
pub use variants::Variants;

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

    /// Value heads: `ssm_inner / ssm_head_dim`.
    pub ssm_heads: usize,
    pub ssm_inner: usize,
    /// Per-head state dimension, `ssm.state_size`.
    pub ssm_head_dim: usize,
    /// Query/key heads, `ssm.group_count`; equal to `ssm_heads` outside a
    /// grouped-query deltanet.
    pub ssm_kv_heads: usize,
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

        let ssm_kv_heads = m.arch_count("ssm.group_count")?;
        let ssm_inner = m.arch_count("ssm.inner_size")?;
        let ssm_head_dim = m.arch_count("ssm.state_size")?;
        ensure!(
            ssm_head_dim > 0 && ssm_inner.is_multiple_of(ssm_head_dim),
            "ssm inner {ssm_inner} does not split into a state size of {ssm_head_dim}"
        );
        // Two readings of the value head count, cross-checked: `time_step_rank`
        // is llama.cpp's, `inner / state_size` derives it. A GQA deltanet's
        // key/query heads, `group_count`, may be fewer.
        let ssm_heads = ssm_inner / ssm_head_dim;
        ensure!(
            m.arch_count("ssm.time_step_rank")? == ssm_heads,
            "ssm.time_step_rank disagrees with inner {ssm_inner} / state {ssm_head_dim}"
        );
        ensure!(
            ssm_kv_heads > 0 && ssm_heads.is_multiple_of(ssm_kv_heads),
            "{ssm_heads} value heads do not group evenly over {ssm_kv_heads} key/query heads"
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
            ssm_head_dim,
            ssm_kv_heads,
            conv_kernel: m.arch_count("ssm.conv_kernel")?,
        })
    }

    pub fn is_attention_block(&self, index: usize) -> bool {
        self.full_attention_interval > 0 && (index + 1).is_multiple_of(self.full_attention_interval)
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

enum Mixer {
    Attention(Box<Attention>),
    DeltaNet(Box<DeltaNet>),
}

impl Mixer {
    /// The constants this mixer uploads. A delta net's derived readings, the
    /// reversed taps and the exponentiated decay rate, are a few kilobytes and
    /// go up under keys of their own; they are left out.
    fn footprint(&self, into: &mut Uploads) {
        match self {
            Mixer::Attention(attn) => {
                for weight in [&attn.q, &attn.k, &attn.v, &attn.output] {
                    weight.footprint(into);
                }
                attn.q_norm.footprint(into);
                attn.k_norm.footprint(into);
            }
            Mixer::DeltaNet(delta) => {
                match &delta.proj {
                    Proj::Fused { linear, .. } => linear.footprint(into),
                    Proj::Split { qkv, gate, alpha, beta } => {
                        for weight in [qkv, gate, alpha, beta] {
                            weight.footprint(into);
                        }
                    }
                }
                delta.out.footprint(into);
                for gain in [&delta.conv_taps, &delta.a_log, &delta.dt_bias, &delta.norm] {
                    gain.footprint(into);
                }
            }
        }
    }
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
    /// `token_embd.weight`, kept quantized.
    embed: Linear,
    /// `output.weight`, or the embedding again when the file ties them.
    head: Linear,
    blocks: Vec<Block>,
    output_norm: Gain,
    rope: RopeTable,
}

/// Copies a projection's qkv window into the delta net's history cache: a
/// decode step's one row is already contiguous, and the strided kernel is the
/// wrong shape for it.
fn copy_qkv_into_history(
    backend: &dyn Backend,
    src: Plane,
    history: Buf,
    carried: usize,
    rows: usize,
    channels: usize,
) -> Result<()> {
    if rows == 1 {
        backend.copy(src.buf, src.offset, history, carried, channels)
    } else {
        backend.copy_2d(
            src,
            Plane { buf: history, offset: carried, pitch: channels },
            rows,
            channels,
        )
    }
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

        let embed = Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?;
        let head = match gguf.tensor("output.weight") {
            Some(_) => Linear::load(gguf, "output.weight", config.d_model, config.vocab)?,
            None => Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?,
        };

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

        Ok(Model {
            rope: RopeTable::new(config.rope_dim, config.rope_freq_base),
            config,
            embed,
            head,
            blocks,
            output_norm,
        })
    }

    /// Every constant the forward pass uploads, at a context of `positions`.
    /// `embed` is missing on purpose: the pass reads a token's row out of it
    /// on the host, and only `head` reaches the backend.
    pub(crate) fn footprint(&self, positions: usize) -> Uploads {
        let mut into = Uploads::default();
        self.head.footprint(&mut into);
        for block in &self.blocks {
            block.attn_norm.footprint(&mut into);
            block.post_attn_norm.footprint(&mut into);
            block.mixer.footprint(&mut into);
            block.ffn.footprint(&mut into);
        }
        self.output_norm.footprint(&mut into);
        self.rope.footprint(&mut into, positions);
        into
    }

    /// Device bytes the caches take per position. Only the attention blocks
    /// have any: a delta net carries a recurrent state whose size is fixed.
    pub(crate) fn kv_bytes_per_token(&self) -> usize {
        let attention_blocks = (0..self.config.n_block)
            .filter(|&index| self.config.is_attention_block(index))
            .count();
        let width = self.config.n_head_kv * self.config.head_dim;
        attention_blocks * 2 * width * size_of::<u16>()
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

    // forward() and friends live in qwen35/forward.rs.

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

        // Split the query from its output gate; the layout only changes where
        // the copy starts and how far apart its rows are.
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
        for (src, dst) in [(k_normed, keys), (v_buf, values)] {
            backend.store_2d(
                Plane {
                    buf: src,
                    offset: 0,
                    pitch: kv_width,
                },
                HPlane {
                    buf: dst,
                    offset: start_pos * kv_width,
                    pitch: kv_width,
                },
                rows,
                kv_width,
            )?;
        }

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
        resid: Buf,
        normed: Buf,
        gain: Buf,
        rows: usize,
        carry: &mut Option<(Buf, usize)>,
        recurrent: &mut Option<Buf>,
        backend: &dyn Backend,
        variants: Variants,
    ) -> Result<()> {
        let cfg = &self.config;
        let (heads, head_dim, kv_heads) = (cfg.ssm_heads, cfg.ssm_head_dim, cfg.ssm_kv_heads);
        let inner = cfg.ssm_inner;

        ensure!(
            !variants.query_contracts_value,
            "the delta-rule readout that contracts the value axis has no backend op; it is not the reference layout and only the sweep ever asked for it"
        );
        ensure!(
            !variants.qkv_interleaved || kv_heads == heads,
            "the interleaved qkv layout assumes uniform query/key/value heads; it is not the reference layout and only the sweep ever asked for it on a grouped-query deltanet"
        );

        // Both fused layouts are the same strided read, just a different
        // offset and stride. Query and key are `kv_heads` wide apiece outside
        // the interleaved layout; value is always the full `inner`.
        let (planes, head_stride) = if variants.qkv_interleaved {
            ([0, head_dim, 2 * head_dim], 3 * head_dim)
        } else {
            let kv_width = kv_heads * head_dim;
            ([0, kv_width, 2 * kv_width], head_dim)
        };

        let mix = DeltaMix {
            rows,
            heads,
            head_dim,
            kv_heads,
            kernel: cfg.conv_kernel,
            planes,
            head_stride,
            normalize: variants.l2_normalize_qk,
            // The readout carries the same 1/sqrt(d) scale as softmax attention.
            query_scale: (head_dim as f32).sqrt().recip(),
        };

        let (channels, carried) = (mix.channels(), mix.pad() * mix.channels());

        // History is a fresh alloc each call, with the previous call's tail
        // copied to the front so the convolution reads one padded stream and
        // needs no boundary case.
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
        *carry = Some((history, mix.history_len()));

        // The delta rule's five operands are one allocation; each is a window
        // of it, written in place by the projection.
        let packed = backend.alloc(mix.packed_len())?;
        let taps = delta.taps(backend, channels, variants.conv_reversed)?;

        // Which of the two gate projections decays and which weights the
        // write is a layout question, shared by both paths below.
        let gates = |alpha, beta| match variants.swap_alpha_beta {
            true => (beta, alpha),
            false => (alpha, beta),
        };

        // The convolution and gates as the tail of a fused projection, given
        // where the decay and write strength land. The fused kernel bakes in
        // a single head count; a grouped-query deltanet falls back to the
        // unfused delta_conv/delta_gates path, which expands `kv_heads` into
        // `heads`.
        let fused_mix = |decay_at: (Buf, usize), beta_at: (Buf, usize)| -> Result<Option<FusedMix>> {
            if rows != 1 || kv_heads != heads {
                return Ok(None);
            }
            let (decay, beta) = gates(decay_at, beta_at);
            Ok(Some(FusedMix {
                spec: mix,
                history,
                taps,
                decay,
                beta,
                rate: delta.rate(backend, variants.decay_from_log)?,
                dt_bias: delta.dt_bias.buf(backend)?,
                packed,
            }))
        };

        // [`Proj::Fused`] is a single launch producing history, gate, and
        // gate operands together; [`Proj::Split`] is the four separate
        // tensors a raw file holds, one fused kernel where the backend has a
        // stage for each format and four launches where it does not.
        let (z, alpha_op, beta_op, mix_done, mut release) = match &delta.proj {
            Proj::Fused { linear, parts } => {
                let width = linear.out_dim;
                let stacked = backend.alloc(rows * width)?;

                // One projection, output split into two destinations: qkv
                // goes straight into the convolution's history buffer, the
                // other three parts stay in `stacked` and are read as windows
                // below.
                let (qkv_at, _) = parts[0];
                let (rest_at, rest_end) = (parts[1].0, parts[3].0 + parts[3].1);
                let runs = [
                    ProjRun { weight: 0, row_off: qkv_at, width: channels, dst: history, dst_off: carried },
                    ProjRun {
                        weight: 0,
                        row_off: rest_at,
                        width: rest_end - rest_at,
                        dst: stacked,
                        dst_off: rest_at,
                    },
                ];
                let fused_mix = fused_mix((stacked, parts[2].0), (stacked, parts[3].0))?;

                let fused =
                    linear.project_fused(backend, resid, gain, cfg.rms_eps, rows, &runs, fused_mix)?;
                if !fused.project {
                    // The normalization leaves the quantized copy behind,
                    // which the projection reading it would otherwise redo.
                    let act = backend.rms_norm_q(resid, rows, cfg.d_model, gain, cfg.rms_eps, normed)?;
                    linear.project_into_act(backend, normed, act, rows, stacked)?;
                    copy_qkv_into_history(
                        backend,
                        Plane { buf: stacked, offset: qkv_at, pitch: width },
                        history,
                        carried,
                        rows,
                        channels,
                    )?;
                }

                // Each of the other three parts is a window of the
                // projection: at a single row an offset, past one row a
                // strided copy apiece.
                let mut planes = [(stacked, 0usize); 3];
                let mut extracted = Vec::new();
                for (plane, &(at, part)) in planes.iter_mut().zip(&parts[1..]) {
                    *plane = if rows == 1 {
                        (stacked, at)
                    } else {
                        let buf = backend.alloc(rows * part)?;
                        backend.copy_2d(
                            Plane { buf: stacked, offset: at, pitch: width },
                            Plane { buf, offset: 0, pitch: part },
                            rows,
                            part,
                        )?;
                        extracted.push(buf);
                        (buf, 0)
                    };
                }
                let [z, alpha_op, beta_op] = planes;
                extracted.push(stacked);
                (z, alpha_op, beta_op, fused.mix, extracted)
            }
            Proj::Split { qkv, gate, alpha, beta } => {
                // The three gate operands in one buffer, as the fused layout
                // has them, so the kernel's tail can read them the same way.
                let (gate_w, alpha_w, beta_w) = (gate.out_dim, alpha.out_dim, beta.out_dim);
                let (alpha_at, beta_at) = (gate_w, gate_w + alpha_w);
                let mut fused = None;
                if rows == 1 {
                    let parts = [qkv, gate, alpha, beta];
                    let weights: Option<Vec<_>> = parts
                        .iter()
                        .map(|part| Ok(part.proj_weight(backend)?.map(|w| (w, part.out_dim))))
                        .collect::<Result<_>>()?;
                    if let Some(weights) = weights {
                        let stacked = backend.alloc(gate_w + alpha_w + beta_w)?;
                        let runs = [
                            ProjRun { weight: 0, row_off: 0, width: channels, dst: history, dst_off: carried },
                            ProjRun { weight: 1, row_off: 0, width: gate_w, dst: stacked, dst_off: 0 },
                            ProjRun { weight: 2, row_off: 0, width: alpha_w, dst: stacked, dst_off: alpha_at },
                            ProjRun { weight: 3, row_off: 0, width: beta_w, dst: stacked, dst_off: beta_at },
                        ];
                        let done = backend.fused_project(FusedProject {
                            x: resid,
                            d_model: cfg.d_model,
                            gain,
                            eps: cfg.rms_eps,
                            weights: &weights,
                            runs: &runs,
                            mix: fused_mix((stacked, alpha_at), (stacked, beta_at))?,
                        })?;
                        match done.project {
                            true => fused = Some((stacked, done.mix)),
                            false => backend.release(stacked),
                        }
                    }
                }
                match fused {
                    Some((stacked, mix_done)) => {
                        ((stacked, 0), (stacked, alpha_at), (stacked, beta_at), mix_done, vec![stacked])
                    }
                    None => {
                        let act = backend.rms_norm_q(resid, rows, cfg.d_model, gain, cfg.rms_eps, normed)?;
                        let qkv_buf = qkv.forward_act(backend, normed, act, rows)?;
                        copy_qkv_into_history(
                            backend,
                            Plane { buf: qkv_buf, offset: 0, pitch: channels },
                            history,
                            carried,
                            rows,
                            channels,
                        )?;
                        backend.release(qkv_buf);

                        let gate_buf = gate.forward_act(backend, normed, act, rows)?;
                        let alpha_buf = alpha.forward_act(backend, normed, act, rows)?;
                        let beta_buf = beta.forward_act(backend, normed, act, rows)?;
                        (
                            (gate_buf, 0),
                            (alpha_buf, 0),
                            (beta_buf, 0),
                            false,
                            vec![gate_buf, alpha_buf, beta_buf],
                        )
                    }
                }
            }
        };
        let (z_buf, z_at) = z;

        if !mix_done {
            backend.delta_conv(history, taps, mix, packed)?;

            let (decay, beta) = gates(alpha_op, beta_op);
            backend.delta_gates(
                decay.0,
                decay.1,
                beta.0,
                beta.1,
                delta.rate(backend, variants.decay_from_log)?,
                delta.dt_bias.buf(backend)?,
                mix,
                packed,
            )?;
        }

        // The recurrent state is why this op exists on the backend at all: a
        // [head_dim, head_dim] matrix per head, so keeping it on the host means
        // moving a megabyte in and out per block per token.
        let state = match *recurrent {
            Some(buf) => buf,
            None => {
                let buf = backend.zeroed_state(heads * head_dim * head_dim)?;
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
        let readout_gain = delta.norm.buf(backend)?;
        let (gated, gated_act) = if variants.norm_before_gate {
            // Into a destination that is not the source: the gate reads one
            // element per thread, but the norm's reduction reads the whole row.
            let readout = backend.rms_norm_gated(
                mixed_buf,
                rows * heads,
                head_dim,
                readout_gain,
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
                readout_gain,
                cfg.rms_eps,
                mixed_buf,
            )?;
            (mixed_buf, None)
        };

        match gated_act {
            Some(act) => delta.out.add_into_act(backend, gated, act, rows, resid)?,
            None => delta.out.add_into(backend, gated, rows, resid)?,
        }
        release.extend([packed, scratch, mixed_buf]);
        for buf in release {
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
        /// Allocated on first use; `carry` is the previous call's whole
        /// convolution stream plus its length, since a prompt and a decode
        /// step leave streams of different lengths.
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
