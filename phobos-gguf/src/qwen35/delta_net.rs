// DeltaNet and its projection layout. A sibling module of `qwen35.rs`, so
// its types need `pub(super)` to stay reachable from the parent.

use anyhow::{Context, Result};

use crate::Gguf;
use crate::backend::{Backend, Buf};
use crate::layers::{Gain, Linear, check_dims};

use super::Config;

/// The query-key-value, gate, decay and beta projections, fused into one
/// launch when [`Linear::should_fuse`] allows it and run as four ordinary
/// projections otherwise. A four-way fuse needs matching quant formats
/// across all four tensors, which per-tensor quantized files rarely have.
pub(super) enum Proj {
    /// Where each part starts in the fused output, and how wide it is.
    Fused { linear: Linear, parts: [(usize, usize); 4] },
    Split { qkv: Linear, gate: Linear, alpha: Linear, beta: Linear },
}

/// A GatedDeltaNet block: a causal depthwise convolution over the fused q/k/v
/// stream, then the gated delta rule, then a gated RMSNorm.
pub(super) struct DeltaNet {
    pub(super) proj: Proj,

    /// `[kernel, channels]` row-major, transposed from the file so one tap
    /// across a run of channels is contiguous, which is how both backends read
    /// it. The file groups each channel's taps instead, making every load of a
    /// channel tile a stride.
    pub(super) conv_taps: Gain,

    /// Log-space decay rate per head.
    pub(super) a_log: Gain,

    pub(super) dt_bias: Gain,
    pub(super) norm: Gain,
    pub(super) out: Linear,
}

impl DeltaNet {
    pub(super) fn load(gguf: &Gguf, prefix: &str, cfg: &Config) -> Result<DeltaNet> {
        let d = cfg.d_model;
        // Query and key at `kv_heads` wide apiece, value at the full `inner`;
        // equal to `3 * ssm_inner` outside a grouped-query deltanet.
        let channels = 2 * cfg.ssm_kv_heads * cfg.ssm_head_dim + cfg.ssm_inner;

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
        let gate = Linear::load(gguf, &format!("{prefix}.attn_gate.weight"), d, cfg.ssm_inner)?;
        let alpha = Linear::load(gguf, &format!("{prefix}.ssm_alpha.weight"), d, cfg.ssm_heads)?;
        let beta = Linear::load(gguf, &format!("{prefix}.ssm_beta.weight"), d, cfg.ssm_heads)?;

        let proj = if Linear::should_fuse(&[&qkv, &gate, &alpha, &beta]) {
            let mut parts = [(0usize, 0usize); 4];
            let mut at = 0;
            for (slot, part) in parts.iter_mut().zip([&qkv, &gate, &alpha, &beta]) {
                *slot = (at, part.out_dim);
                at += part.out_dim;
            }
            Proj::Fused { linear: Linear::fuse(&[&qkv, &gate, &alpha, &beta])?, parts }
        } else {
            Proj::Split { qkv, gate, alpha, beta }
        };

        Ok(DeltaNet {
            proj,
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
    pub(super) fn taps(&self, backend: &dyn Backend, channels: usize, reversed: bool) -> Result<Buf> {
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
    pub(super) fn rate(&self, backend: &dyn Backend, from_log: bool) -> Result<Buf> {
        if !from_log {
            return self.a_log.buf(backend);
        }

        let rates = self.a_log.data.iter().map(|&v| -v.exp()).collect();
        Gain::derived(format!("{}.rate", self.a_log.key), rates).buf(backend)
    }
}
