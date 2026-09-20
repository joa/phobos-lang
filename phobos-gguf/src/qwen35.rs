use anyhow::{Context, Result, ensure};

use crate::Gguf;
use crate::backend::{Backend, Buf, DeltaMix, FusedMix, FusedProject, Plane, ProjRun};
use crate::layers::{Ffn, Gain, KvCache, Linear, MoeFfn, RopeTable, Shared, Uploads};

mod attention;
mod delta_net;
mod forward;
#[cfg(test)]
mod tests;
mod variants;

use attention::Attention;
use delta_net::{DeltaNet, Proj};
pub use forward::RouteTrace;
pub use variants::Variants;

/// The routed feed-forward of the `qwen35moe` architecture.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub n_expert: usize,
    /// Experts a token goes through.
    pub n_used: usize,
    /// Width of one expert.
    pub d_expert: usize,
    /// Width of the shared expert every token goes through.
    pub d_shared: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// `qwen35`, or `qwen35moe` for the same trunk with routed experts.
    pub arch: &'static str,
    /// Blocks that take part in ordinary decoding, excluding the trailing
    /// multi-token-prediction block.
    pub n_block: usize,
    pub d_model: usize,
    /// Width of the feed-forward, or of one expert in the routed one.
    pub d_ff: usize,
    pub moe: Option<MoeConfig>,
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
        let arch = match gguf.architecture()? {
            "qwen35" => "qwen35",
            "qwen35moe" => "qwen35moe",
            other => anyhow::bail!("expected a qwen35 model, found architecture '{other}'"),
        };

        let m = gguf.metadata();

        let moe = match arch {
            "qwen35moe" => Some(MoeConfig {
                n_expert: m.arch_count("expert_count")?,
                n_used: m.arch_count("expert_used_count")?,
                d_expert: m.arch_count("expert_feed_forward_length")?,
                d_shared: m.arch_count("expert_shared_feed_forward_length")?,
            }),
            _ => None,
        };
        let d_ff = match moe {
            Some(moe) => moe.d_expert,
            None => m.arch_count("feed_forward_length")?,
        };

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
            arch,
            n_block,
            d_model: m.arch_count("embedding_length")?,
            d_ff,
            moe,
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
                    Proj::Split { qkv, gate, gates } => {
                        qkv.footprint(into);
                        gate.footprint(into);
                        gates.footprint(into);
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

/// The feed-forward half of a block: one SwiGLU, or a router's choice among
/// many plus a shared one.
enum FeedForward {
    Dense(Ffn),
    Moe(MoeFfn),
}

impl FeedForward {
    fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<FeedForward> {
        Ok(match cfg.moe {
            Some(moe) => FeedForward::Moe(MoeFfn::load(
                gguf,
                prefix,
                cfg.d_model,
                moe.n_expert,
                moe.n_used,
                moe.d_expert,
                moe.d_shared,
            )?),
            None => FeedForward::Dense(Ffn::load(gguf, prefix, cfg.d_model, cfg.d_ff)?),
        })
    }

    fn footprint(&self, into: &mut Uploads) {
        match self {
            FeedForward::Dense(ffn) => ffn.footprint(into),
            FeedForward::Moe(moe) => moe.footprint(into),
        }
    }
}

struct Block {
    attn_norm: Gain,
    /// Despite the name this gates the FFN input, as in a standard pre-norm
    /// transformer block.
    post_attn_norm: Gain,
    mixer: Mixer,
    ffn: FeedForward,
}

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

/// Refuses a Hadamard-folded file naming a tensor this model does not read
/// through a [`Linear`], which is where the transform is applied.
fn check_folding(gguf: &Gguf) -> Result<()> {
    const LINEAR: [&str; 12] = [
        "attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_alpha",
        "ssm_beta", "ssm_out", "ffn_gate", "ffn_up", "ffn_down",
    ];
    gguf.folding().map_or(Ok(()), |f| f.check_projected(&LINEAR))
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

        check_folding(gguf)?;
        let embed = Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?;
        let head = match gguf.tensor("output.weight") {
            Some(_) => Linear::load(gguf, "output.weight", config.d_model, config.vocab)?,
            None => {
                // A folded table's rows are `H S h`, which the head would
                // contract against an unrotated `h`.
                ensure!(
                    !gguf.folding().is_some_and(|f| f.restores("token_embd.weight")),
                    "a Hadamard-folded token_embd cannot double as the output head"
                );
                Linear::load(gguf, "token_embd.weight", config.d_model, config.vocab)?
            }
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
                ffn: FeedForward::load(gguf, &prefix, &config)?,
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

    /// The pre-projection normalization of `x` into `normed`, shared by
    /// `reader` and every projection beside it; see [`Linear::share_norm`].
    pub(super) fn norm(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        gain: Buf,
        normed: Buf,
        reader: &Linear,
    ) -> Result<Shared> {
        reader.share_norm(backend, x, rows, gain, self.config.rms_eps, normed)
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

        // History is scratch for this call, with the previous call's tail
        // copied to the front so the convolution reads one padded stream and
        // needs no boundary case.
        let history = backend.alloc(mix.history_len())?;
        match *carry {
            Some((previous, len)) => backend.copy(previous, len - carried, history, 0, carried)?,
            None => {
                let zeros = backend.zeroed(carried)?;
                backend.copy(zeros, 0, history, 0, carried)?;
                backend.release(zeros);
            }
        }

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
        // a single head count, so a grouped-query deltanet falls back to the
        // unfused delta_conv/delta_gates path instead.
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
                    let input = self.norm(backend, resid, rows, gain, normed, linear)?;
                    linear.project_into_shared(backend, input, rows, stacked)?;
                    input.release(backend);
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
            Proj::Split { qkv, gate, gates: decay_gates } => {
                // The three gate operands in one buffer, as the fused layout
                // has them, so the kernel's tail can read them the same way.
                let (gate_w, (alpha_w, beta_w)) = (gate.out_dim, decay_gates.widths());
                let (alpha_at, beta_at) = (gate_w, gate_w + alpha_w);
                let mut fused = None;
                if rows == 1
                    && let Some([alpha, beta]) = decay_gates.apart()
                {
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
                        let input = self.norm(backend, resid, rows, gain, normed, qkv)?;
                        let qkv_buf = qkv.forward_shared(backend, input, rows)?;
                        copy_qkv_into_history(
                            backend,
                            Plane { buf: qkv_buf, offset: 0, pitch: channels },
                            history,
                            carried,
                            rows,
                            channels,
                        )?;
                        backend.release(qkv_buf);

                        let gate_buf = gate.forward_shared(backend, input, rows)?;
                        // The gates read the row unfolded, which `normed` is.
                        let act = input.plain_act();
                        input.release(backend);
                        let (alpha_op, beta_op, mut release) = decay_gates.project(backend, normed, act, rows)?;
                        release.push(gate_buf);
                        ((gate_buf, 0), alpha_op, beta_op, false, release)
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

        // The next call reads only the last `pad` positions: they are carried
        // in a buffer of their own that stays put, rather than the whole
        // stream (20 MiB a layer after a 512-row prompt), and with the history
        // released a decode step records the same buffers as the one before
        // it, so the cached pass graph needs no patching for them.
        if carried > 0 {
            let tail = match carry.take() {
                Some((buf, len)) if len == carried => buf,
                other => {
                    if let Some((buf, _)) = other {
                        backend.release(buf);
                    }
                    backend.alloc(carried)?
                }
            };
            backend.copy(history, mix.history_len() - carried, tail, 0, carried)?;
            *carry = Some((tail, carried));
        }
        backend.release(history);

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
            Some(act) => delta.out.add_into_act(backend, gated, Some(act), rows, resid)?,
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
        /// Allocated on first use; `carry` is the last `pad` positions of the
        /// previous call's convolution stream, and their length.
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

    /// Forget everything past `positions`, which this architecture can only do
    /// when there is nothing to forget.
    ///
    /// Its attention blocks would rewind as llama's do, but its delta net
    /// blocks carry a recurrent state summarising every token they have seen.
    /// Nothing in it is indexed by position, so there is no prefix of it to
    /// keep, and rewinding one half of the model but not the other would leave
    /// the two at different points in the sequence. Extending is still fine:
    /// that is what a decode step does.
    pub fn truncate(&mut self, positions: usize) -> bool {
        positions == self.pos
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
