// The SwiGLU feed-forward, with its gate and up either stacked or apart.

use anyhow::Result;

use crate::Gguf;
use crate::backend::{Backend, Buf, FusedMlp, FusedMlpRaw, Plane};
use crate::quant::Quant;

use super::{Linear, Shared, Uploads};

/// Gate and up, fused into one launch when [`Linear::should_fuse`] allows it
/// and run as two ordinary projections otherwise.
enum GateUp {
    /// Stacked: they read the same row, so one launch over a doubly wide
    /// output replaces two over half of it.
    Fused(Linear),
    Split { gate: Linear, up: Linear },
}

/// The SwiGLU feed-forward every architecture here ends a block with.
pub(crate) struct Ffn {
    gate_up: GateUp,
    down: Linear,
}

impl Ffn {
    pub(crate) fn load(gguf: &Gguf, prefix: &str, d_model: usize, d_ff: usize) -> Result<Ffn> {
        let gate = Linear::load(gguf, &format!("{prefix}.ffn_gate.weight"), d_model, d_ff)?;
        let up = Linear::load(gguf, &format!("{prefix}.ffn_up.weight"), d_model, d_ff)?;
        let gate_up = if Linear::should_fuse(&[&gate, &up]) {
            GateUp::Fused(Linear::fuse(&[&gate, &up])?)
        } else {
            GateUp::Split { gate, up }
        };
        Ok(Ffn {
            gate_up,
            down: Linear::load(gguf, &format!("{prefix}.ffn_down.weight"), d_ff, d_model)?,
        })
    }

    pub(crate) fn footprint(&self, into: &mut Uploads) {
        match &self.gate_up {
            GateUp::Fused(gate_up) => gate_up.footprint(into),
            GateUp::Split { gate, up } => {
                gate.footprint(into);
                up.footprint(into);
            }
        }
        self.down.footprint(into);
    }

    /// The normalization and the whole of [`Ffn::forward`] as one kernel, if
    /// the backend has one. `false` leaves the caller to take the usual
    /// path.
    pub(crate) fn forward_fused(
        &self,
        backend: &dyn Backend,
        x: Buf,
        gain: Buf,
        eps: f32,
        rows: usize,
    ) -> Result<bool> {
        if rows != 1 || self.folded() {
            return Ok(false);
        }
        // Raw formats keep gate and up apart; a backend with a fused form for
        // them takes the three weights and their formats.
        if let (GateUp::Split { gate, up }, Some((gq, uq)), Some(dq)) =
            (&self.gate_up, self.split_raw_quants(), self.down.raw_quant())
        {
            return backend.fused_mlp_raw(FusedMlpRaw {
                x,
                d_model: gate.in_dim,
                d_ff: self.down.in_dim,
                gain,
                eps,
                gate: (gate.raw(backend)?, gq),
                up: (up.raw(backend)?, uq),
                down: (self.down.raw(backend)?, dq),
            });
        }
        let GateUp::Fused(gate_up) = &self.gate_up else {
            return Ok(false);
        };
        if !gate_up.is_quantized() || !self.down.is_quantized() {
            return Ok(false);
        }
        backend.fused_mlp(FusedMlp {
            x,
            d_model: gate_up.in_dim,
            d_ff: self.down.in_dim,
            gain,
            eps,
            gate_up: gate_up.quantized(backend)?,
            down: self.down.quantized(backend)?,
        })
    }

    /// The weight that reads the block's input, or the first of the two
    /// that do; see [`Linear::share`].
    pub(crate) fn input(&self) -> &Linear {
        match &self.gate_up {
            GateUp::Fused(gate_up) => gate_up,
            GateUp::Split { gate, .. } => gate,
        }
    }

    /// Whether any of the three weights is Hadamard-folded, which the fused
    /// kernels do not transform for.
    fn folded(&self) -> bool {
        let gate_up = match &self.gate_up {
            GateUp::Fused(gate_up) => gate_up.folded(),
            GateUp::Split { gate, up } => gate.folded() || up.folded(),
        };
        gate_up || self.down.folded()
    }

    /// A split gate and up's raw formats, when both are raw.
    fn split_raw_quants(&self) -> Option<(Quant, Quant)> {
        match &self.gate_up {
            GateUp::Split { gate, up } => Some((gate.raw_quant()?, up.raw_quant()?)),
            GateUp::Fused(_) => None,
        }
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))`, added into `dest`, `x` shared
    /// by [`Ffn::input`]. Nothing leaves the backend, so the two wide
    /// intermediates never reach the host.
    pub(crate) fn forward(&self, backend: &dyn Backend, x: Shared, rows: usize, dest: Buf) -> Result<()> {
        let width = self.down.in_dim;
        let joined = backend.alloc(rows * width)?;
        let dense = |buf| Plane { buf, offset: 0, pitch: width };

        // Either a fused projection's two windows or two ordinary
        // projections' own dense buffers, and either way this ends as a
        // (gate, up) pair of planes and the buffers to release once the
        // SwiGLU has read them.
        let (gate_p, up_p, release): (Plane, Plane, [Buf; 2]) = match &self.gate_up {
            GateUp::Fused(gate_up) => {
                let both = gate_up.forward_shared(backend, x, rows)?;
                // Past one row the two halves interleave, so the SwiGLU reads
                // them where they lie rather than pulling them apart first.
                let stacked = |offset| Plane { buf: both, offset, pitch: 2 * width };
                (stacked(0), stacked(width), [both, both])
            }
            GateUp::Split { gate, up } => {
                let gate_buf = gate.forward_shared(backend, x, rows)?;
                let up_buf = up.forward_shared(backend, x, rows)?;
                (dense(gate_buf), dense(up_buf), [gate_buf, up_buf])
            }
        };

        if rows == 1 {
            // One launch for the gate, the product, and the quantized copy.
            let act =
                backend.swiglu_q(gate_p.buf, gate_p.offset, up_p.buf, up_p.offset, joined, width)?;
            release_once(backend, release);
            self.down.add_into_act(backend, joined, Some(act), rows, dest)?;
            backend.release(joined);
            return Ok(());
        }
        backend.swiglu_planes(gate_p, up_p, joined, rows, width)?;
        release_once(backend, release);
        self.down.add_into(backend, joined, rows, dest)?;
        backend.release(joined);
        Ok(())
    }
}

/// Releases `[a, b]`, once each even when `a == b` (the fused case, where
/// gate and up are two windows of the same buffer).
fn release_once(backend: &dyn Backend, [a, b]: [Buf; 2]) {
    backend.release(a);
    if b != a {
        backend.release(b);
    }
}
