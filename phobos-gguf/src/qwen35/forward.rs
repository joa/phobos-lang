// The `forward` family. A descendant module of `qwen35.rs`, so `Model`'s
// private fields stay visible here.

use anyhow::{Result, bail, ensure};

use crate::backend::{Backend, read_vec};
use crate::model::ForwardBufs;

use super::{Config, LayerState, Mixer, Model, State, Variants};

impl Model {
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

    /// [`Model::forward`] for a caller that only wants the winning token id,
    /// as greedy decoding does: the LM head still runs, only the vocab-wide
    /// readback that follows it is skipped. See [`Backend::argmax`].
    pub fn forward_greedy(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<i64> {
        let (bufs, cfg) = self.forward_to_logits(state, tokens, backend, Variants::REFERENCE)?;
        let id = backend.argmax(bufs.logits, cfg.vocab)?;
        bufs.release(backend);
        Ok(id)
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
        let (bufs, cfg) = self.forward_to_logits(state, tokens, backend, variants)?;
        let out = read_vec(backend, bufs.logits, cfg.vocab)?;
        bufs.release(backend);
        Ok(out)
    }

    /// The shared body of [`Model::forward_with`] and [`Model::forward_greedy`]:
    /// everything through the LM head projection and the pass's own
    /// `end_pass`, leaving only "how much of the result to read back" to the
    /// two callers above.
    fn forward_to_logits(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
        variants: Variants,
    ) -> Result<(ForwardBufs, &Config)> {
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
            self.embed.row_into(id, &mut host_x[t * d..(t + 1) * d])?;
        }

        let x = backend.upload(&host_x)?;
        let normed = backend.alloc(rows * d)?;

        // Everything from here to the logits is device-only, which lets a
        // backend take the whole pass as one unit.
        backend.begin_pass()?;

        let trace = std::env::var_os("PHOBOS_TRACE").is_some();
        for (index, (block, layer_state)) in self.blocks.iter().zip(&mut state.layers).enumerate() {
            // Both mixers add their output into the residual stream
            // themselves and own running their input normalization, so a
            // fused projection can absorb it.
            let gain = block.attn_norm.buf(backend)?;
            match (&block.mixer, layer_state) {
                (Mixer::Attention(attn), LayerState::Attention(cache)) => {
                    // The normalization leaves the quantized copy behind too,
                    // which the three projections reading it would otherwise
                    // redo.
                    let act = backend.rms_norm_q(x, rows, d, gain, cfg.rms_eps, normed)?;
                    self.attention(
                        attn, normed, act, rows, state.pos, cache, backend, variants, x,
                    )?
                }
                (Mixer::DeltaNet(delta), LayerState::DeltaNet { carry, recurrent }) => self
                    .delta_net(
                        delta, x, normed, gain, rows, carry, recurrent, backend, variants,
                    )?,
                _ => bail!("generation state does not match the model's block layout"),
            }

            // The normalization is passed along rather than run first, so a
            // backend with a fused MLP owns the whole of it.
            let gain = block.post_attn_norm.buf(backend)?;
            if !block
                .ffn
                .forward_fused(backend, x, gain, cfg.rms_eps, rows)?
            {
                let act = backend.rms_norm_q(x, rows, d, gain, cfg.rms_eps, normed)?;
                block.ffn.forward(backend, normed, act, rows, x)?;
            }

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

        Ok((
            ForwardBufs {
                x,
                normed,
                last,
                logits,
            },
            cfg,
        ))
    }
}
