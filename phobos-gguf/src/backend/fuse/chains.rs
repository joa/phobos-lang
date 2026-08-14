// The two chains a decode step builds: the MLP and the projection.

use super::*;

/// The decode MLP as a chain: the normalization, the two halves of the gate/up
/// projection, the SwiGLU between them, and the down projection accumulating
/// back into the residual.
///
/// `x` is both the first stage's input and the last stage's target, which is
/// what the aliasing in the emitted kernel means, and the barrier is what makes
/// it safe: every block reads the residual in its own normalization before it
/// arrives, and the down projection adds into it only after.
pub(crate) fn mlp_chain(
    x: Buf,
    gain: Buf,
    gate_up: QBuf,
    down: QBuf,
    d_model: usize,
    d_ff: usize,
    eps: f32,
) -> Chain {
    let mut chain = Chain::default();
    let xv = chain.given(x, d_model);
    let gv = chain.given(gain, d_model);
    let act = chain.quant(d_model);
    chain.push(Stage::norm_q(xv, gv, act, d_model, eps));

    let wgu = chain.weight(gate_up, 2 * d_ff, d_model);
    let units = d_ff / Q8_BLOCK;
    let gate = chain.temp(Q8_BLOCK);
    let up = chain.temp(Q8_BLOCK);
    chain.push(Stage::ProjQ {
        a: act,
        w: wgu,
        out: gate,
        units,
        row_off: 0,
    });
    chain.push(Stage::ProjQ {
        a: act,
        w: wgu,
        out: up,
        units,
        row_off: d_ff,
    });
    let hidden = chain.temp(Q8_BLOCK);
    chain.push(Stage::Swiglu {
        g: gate,
        u: up,
        out: hidden,
    });
    let hq = chain.quant(d_ff);
    chain.push(Stage::QuantQ {
        h: hidden,
        out: hq,
        units,
    });

    let wdn = chain.weight(down, d_model, d_ff);
    chain.push(Stage::ProjAdd {
        a: hq,
        w: wdn,
        y: xv,
        width: d_model,
    });
    chain
}

/// A mixer's input normalization and the projection reading it as a chain, with
/// each run of the projection's outputs written where its consumer wants it, and
/// optionally the delta net's convolution and gates behind them.
///
/// The projection alone costs no barrier: the normalization is redundant, so its
/// quantized row crosses into the projection for free, and the runs are
/// independent nests writing disjoint windows of the caller's buffers. The
/// convolution costs exactly one, for the reason [`super::FusedMix`] gives, and
/// the gates ride in its nest.
///
/// `None` means a shape the pass has no stage for, so the caller keeps its own
/// launches: a run not dividing into whole Q8_0 output blocks, more than one
/// position, or gates split across two projections.
pub(crate) fn project_chain(project: &FusedProject) -> Option<Chain> {
    // One value per buffer, at the widest extent any stage touches. Two values
    // naming one buffer would hide a dependency: the convolution reads the
    // stream position the projection wrote, and the pass sees that only if both
    // stages name the same value.
    let mut extents: Vec<(Buf, usize)> = Vec::new();
    let mut want = |buf: Buf, len: usize| match extents.iter_mut().find(|(b, _)| *b == buf) {
        Some((_, at)) => *at = (*at).max(len),
        None => extents.push((buf, len)),
    };
    for run in project.runs {
        want(run.dst, run.dst_off + run.width);
    }
    if let Some(m) = &project.mix {
        // Two projections feeding the gates would need a value each, and no
        // layout splits them; the stage carries one on purpose.
        if m.decay.0 != m.beta.0 {
            return None;
        }
        let spec = &m.spec;
        want(m.history, spec.history_len());
        want(m.taps, spec.kernel * spec.channels());
        want(m.packed, spec.packed_len());
        want(m.decay.0, m.decay.1 + spec.gates());
        want(m.beta.0, m.beta.1 + spec.gates());
        want(m.rate, spec.heads);
        want(m.dt_bias, spec.heads);
    }

    let mut chain = Chain::default();
    let xv = chain.given(project.x, project.d_model);
    let gv = chain.given(project.gain, project.d_model);
    let vals: Vec<Val> = extents
        .iter()
        .map(|&(buf, len)| chain.given(buf, len))
        .collect();
    let val_of = |buf: Buf| {
        let at = extents.iter().position(|(b, _)| *b == buf)?;
        Some(vals[at])
    };

    let act = chain.quant(project.d_model);
    chain.push(Stage::norm_q(xv, gv, act, project.d_model, project.eps));

    let w = chain.weight(project.w, project.out_dim, project.d_model);
    for run in project.runs {
        if !run.width.is_multiple_of(Q8_BLOCK) || !run.row_off.is_multiple_of(Q8_BLOCK) {
            return None;
        }
        chain.push(Stage::ProjF {
            a: act,
            w,
            out: val_of(run.dst)?,
            out_off: run.dst_off,
            units: run.width / Q8_BLOCK,
            row_off: run.row_off,
        });
    }

    if let Some(m) = &project.mix {
        let spec = &m.spec;
        // One head a unit, and a position at a time, which is the decode shape
        // the whole fused path is for. A prompt pass keeps its own launches.
        if spec.rows != 1 {
            return None;
        }
        // The plane rides the unit index as a stride, so unevenly spaced planes
        // have no stage. No layout uses them, and the device backend's own
        // convolution asserts the same thing.
        let [first, second, third] = spec.planes;
        if third - second != second - first {
            return None;
        }
        chain.push(Stage::Conv {
            history: val_of(m.history)?,
            taps: val_of(m.taps)?,
            out: val_of(m.packed)?,
            planes: spec.planes.len(),
            heads: spec.heads,
            head_dim: spec.head_dim,
            kernel: spec.kernel,
            channels: spec.channels(),
            plane_base: first,
            plane_stride: second - first,
            head_stride: spec.head_stride,
            normalize: spec.normalize,
            scale_bits: spec.query_scale.to_bits(),
        });
        chain.push(Stage::Gates {
            raw: val_of(m.decay.0)?,
            decay_at: m.decay.1,
            beta_at: m.beta.1,
            rate: val_of(m.rate)?,
            bias: val_of(m.dt_bias)?,
            out: val_of(m.packed)?,
            heads: spec.heads,
            units: spec.planes.len() * spec.heads,
            span: spec.span(),
        });
    }
    Some(chain)
}
