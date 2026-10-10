// The softmax-attention mixer. A descendant module of `qwen35.rs`, so
// `Model`'s private fields stay visible here.

use anyhow::Result;

use crate::Gguf;
use crate::backend::{Attn, AttnPrep, Backend, Buf, HPlane, Plane, Rope};
use crate::layers::{Gain, KvCache, Linear, Shared};

use super::{Config, Model, Variants};

/// Gated grouped-query attention with per-head QK normalization and partial
/// rotary embeddings.
pub(super) struct Attention {
    /// Emits `head_dim * 2` per head: the query and its output gate.
    pub(super) q: Linear,
    pub(super) k: Linear,
    pub(super) v: Linear,
    pub(super) output: Linear,
    pub(super) q_norm: Gain,
    pub(super) k_norm: Gain,
}

impl Attention {
    pub(super) fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<Attention> {
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

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        attn: &Attention,
        input: Shared,
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

        let q_gate_buf = attn.q.forward_shared(backend, input, rows)?;
        let k_buf = attn.k.forward_shared(backend, input, rows)?;
        let v_buf = attn.v.forward_shared(backend, input, rows)?;
        input.release(backend);

        // Split the query from its output gate. The variant only changes
        // where each copy starts and its row stride.
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
        let q_normed = backend.alloc(rows * width)?;
        let k_normed = backend.alloc(rows * kv_width)?;
        let eps = cfg.rms_eps;
        let table = self.rope.buf(backend, spec.total())?;
        let (keys, values) = cache.reserve(backend, spec.total(), kv_width)?;

        // A decode row can take all of the below in one pass per side. A
        // head's query and gate are `head_pitch` apart in either layout.
        let head_pitch = if variants.attn_gate_contiguous { dim } else { 2 * dim };
        let rope = Rope {
            heads: cfg.n_head,
            head_dim: dim,
            rope_dim: cfg.rope_dim,
            start_pos,
        };
        let prepared = rows == 1
            && backend.attn_prep(AttnPrep {
                q: Plane { buf: q_gate_buf, offset: q_at, pitch: head_pitch },
                gate: Plane { buf: q_gate_buf, offset: gate_at, pitch: head_pitch },
                k: k_buf,
                v: v_buf,
                kv_heads: cfg.n_head_kv,
                q_gain: attn.q_norm.buf(backend)?,
                k_gain: attn.k_norm.buf(backend)?,
                eps,
                table,
                rope,
                q_out: q_normed,
                gate_out: gate,
                keys,
                values,
            })?;
        if !prepared {
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

            // QK normalization is per head, before the rotary embedding.
            // Both buffers are already runs of `head_dim`-wide heads, the
            // rows the norm expects.
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

            for (buf, heads) in [(q_normed, cfg.n_head), (k_normed, cfg.n_head_kv)] {
                backend.rope(buf, rows, table, Rope { heads, ..rope })?;
            }

            // The caches hold every head of one position together, so
            // appending is one contiguous copy, and the kernel reads a head
            // as a column window.
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
}
