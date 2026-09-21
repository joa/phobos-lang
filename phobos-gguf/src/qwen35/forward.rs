// The `forward` family. A descendant module of `qwen35.rs`, so `Model`'s
// private fields stay visible here.

use anyhow::{Result, bail, ensure};

use crate::backend::{Backend, Buf, read_vec, route};
use crate::model::ForwardBufs;

use super::{Config, FeedForward, LayerState, Mixer, Model, State, Variants};

/// What the routers of a mixture-of-experts model chose during one pass,
/// for a caller studying them: which experts, and which ones a cheap
/// prediction would have named ahead of time.
#[derive(Debug, Default)]
pub struct RouteTrace {
    /// Experts a token went through.
    pub n_used: usize,
    /// `[block][row][n_used]` expert ids, every block's router as it ran.
    pub routes: Vec<u32>,
    /// `[block][row][n_used]`: block `b + 1`'s router evaluated on the
    /// residual as it left block `b`, before block `b + 1`'s own mixer
    /// touched it. The last block predicts nothing; its entries are zero.
    /// A prefetch that acts on this is only as good as its agreement with
    /// `routes`, which is what the trace is for measuring.
    pub lookahead: Vec<u32>,
}

/// The device buffers a traced pass leaves behind for [`RouteTrace`] to be
/// read out of once the pass has ended.
struct TraceBufs {
    /// One `[rows, n_used]` buffer a block.
    routes: Vec<Buf>,
    /// One `[rows, n_expert]` logits buffer a block that predicts.
    lookahead: Vec<Option<Buf>>,
    scratch: Buf,
}

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
        let (bufs, cfg) =
            self.forward_to_logits(state, tokens, backend, Variants::REFERENCE, None)?;
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
        let (bufs, cfg) = self.forward_to_logits(state, tokens, backend, variants, None)?;
        let out = read_vec(backend, bufs.logits, cfg.vocab)?;
        bufs.release(backend);
        Ok(out)
    }

    /// [`Model::forward`] on a mixture-of-experts model, also reporting what
    /// its routers chose. Slower than the plain pass by a router evaluation
    /// a block and the readbacks; for studying the routers, not for serving.
    pub fn forward_traced(
        &self,
        state: &mut State,
        tokens: &[u32],
        backend: &dyn Backend,
    ) -> Result<(Vec<f32>, RouteTrace)> {
        let cfg = &self.config;
        let moe = cfg
            .moe
            .ok_or_else(|| anyhow::anyhow!("a {} model has no routers to trace", cfg.arch))?;
        let rows = tokens.len();
        let mut bufs = TraceBufs {
            routes: (0..cfg.n_block)
                .map(|_| backend.alloc(rows * moe.n_used))
                .collect::<Result<_>>()?,
            lookahead: (0..cfg.n_block)
                .map(|b| {
                    (b + 1 < cfg.n_block)
                        .then(|| backend.alloc(rows * moe.n_expert))
                        .transpose()
                })
                .collect::<Result<_>>()?,
            scratch: backend.alloc(rows * cfg.d_model)?,
        };
        let (out, _) =
            self.forward_to_logits(state, tokens, backend, Variants::REFERENCE, Some(&mut bufs))?;
        let logits = read_vec(backend, out.logits, cfg.vocab)?;
        out.release(backend);

        let mut trace = RouteTrace { n_used: moe.n_used, ..RouteTrace::default() };
        for (routes, lookahead) in bufs.routes.iter().zip(&bufs.lookahead) {
            let chosen = read_vec(backend, *routes, rows * moe.n_used)?;
            trace.routes.extend(chosen.iter().map(|&id| id as u32));
            match lookahead {
                Some(buf) => {
                    let predicted = read_vec(backend, *buf, rows * moe.n_expert)?;
                    for row in predicted.chunks_exact(moe.n_expert) {
                        trace
                            .lookahead
                            .extend(route(row, moe.n_used).iter().map(|&(e, _)| e as u32));
                    }
                }
                None => trace.lookahead.extend(std::iter::repeat_n(0, rows * moe.n_used)),
            }
        }
        for buf in bufs.routes.into_iter().chain(bufs.lookahead.into_iter().flatten()) {
            backend.release(buf);
        }
        backend.release(bufs.scratch);
        Ok((logits, trace))
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
        mut tracing: Option<&mut TraceBufs>,
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

        // A backend that caches streamed experts sizes the cache from what
        // the resident weights leave; told every pass, it sizes once.
        if self.config.moe.is_some() {
            backend.budget_streamed(self.resident_bytes, cfg.n_block)?;
            for block in &self.blocks {
                if let FeedForward::Moe(moe) = &block.ffn {
                    moe.register(backend)?;
                }
            }
        }

        // Everything from here to the logits is device-only, which lets a
        // backend take the whole pass as one unit.
        backend.begin_pass(rows)?;

        let trace = phobos_base::env::flag("PHOBOS_TRACE");
        for (index, (block, layer_state)) in self.blocks.iter().zip(&mut state.layers).enumerate() {
            // Both mixers add their output into the residual stream
            // themselves and own running their input normalization, so a
            // fused projection can absorb it.
            let gain = block.attn_norm.buf(backend)?;
            match (&block.mixer, layer_state) {
                (Mixer::Attention(attn), LayerState::Attention(cache)) => {
                    // The three projections share one normalized, quantized
                    // (and for folded weights, transformed) row.
                    let input = self.norm(backend, x, rows, gain, normed, &attn.q)?;
                    self.attention(attn, input, rows, state.pos, cache, backend, variants, x)?;
                    input.release(backend);
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
            match &block.ffn {
                FeedForward::Dense(ffn) => {
                    if !ffn.forward_fused(backend, x, gain, cfg.rms_eps, rows)? {
                        let input = self.norm(backend, x, rows, gain, normed, ffn.input())?;
                        ffn.forward(backend, input, rows, x)?;
                        input.release(backend);
                    }
                }
                FeedForward::Moe(moe) => {
                    let input = self.norm(backend, x, rows, gain, normed, moe.input())?;
                    let routes = tracing.as_ref().map(|t| t.routes[index]);
                    // The next block's router, for a backend that starts
                    // its misses early.
                    let lookahead = match self.blocks.get(index + 1).map(|next| (next, &next.ffn)) {
                        Some((next, FeedForward::Moe(moe))) => {
                            moe.lookahead(backend, &next.post_attn_norm, cfg.rms_eps)?
                        }
                        _ => None,
                    };
                    moe.forward(backend, input, rows, x, routes, lookahead)?;
                    input.release(backend);
                }
            }

            // The next block's router on the residual as it stands, ahead of
            // that block's mixer: what a prefetch could know now.
            if let Some(tracing) = tracing.as_deref_mut()
                && let Some(logits) = tracing.lookahead[index]
                && let Some(next) = self.blocks.get(index + 1)
                && let FeedForward::Moe(moe) = &next.ffn
            {
                let gain = next.post_attn_norm.buf(backend)?;
                backend.rms_norm(x, rows, d, gain, cfg.rms_eps, tracing.scratch)?;
                moe.router_into(backend, tracing.scratch, rows, logits)?;
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
