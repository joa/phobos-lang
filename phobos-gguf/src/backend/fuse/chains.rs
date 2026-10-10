// The chains a decode step builds: the MLP, the mixer's input projection and
// attention's output projection.

use super::*;

/// The decode MLP as a chain: the normalization, the two halves of the
/// gate/up projection, the SwiGLU, and the down projection accumulating into
/// the residual.
///
/// `x` is both the first input and the last target. This is safe because
/// every block reads the residual before the barrier and the down projection
/// adds into it only after.
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
        blocks: 1,
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

/// Whether a raw format has a `<fmt>_qdot_i8_t` the pass can emit.
fn has_fused_decode(quant: Quant) -> bool {
    matches!(quant, Quant::Q4_K | Quant::Q5_K | Quant::Q6_K)
}

/// [`mlp_chain`] over raw-format weights, with gate and up as two separate
/// weights. A unit is a run of [`RAW_UNIT`] outputs.
///
/// `None` for a format without a `<fmt>_qdot_i8_t`, or a width that
/// [`RAW_UNIT`] does not divide.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mlp_chain_raw(
    x: Buf,
    gain: Buf,
    gate: (RawBuf, Quant),
    up: (RawBuf, Quant),
    down: (RawBuf, Quant),
    d_model: usize,
    d_ff: usize,
    eps: f32,
) -> Option<Chain> {
    if ![gate.1, up.1, down.1].into_iter().all(has_fused_decode) {
        return None;
    }
    if !d_ff.is_multiple_of(RAW_UNIT) || !d_model.is_multiple_of(RAW_UNIT) {
        return None;
    }
    let mut chain = Chain::default();
    let xv = chain.given(x, d_model);
    let gv = chain.given(gain, d_model);
    let act = chain.quant(d_model);
    chain.push(Stage::norm_q(xv, gv, act, d_model, eps));

    let units = d_ff / RAW_UNIT;
    let wg = chain.raw_weight(gate.0, d_ff, d_model, gate.1);
    let wu = chain.raw_weight(up.0, d_ff, d_model, up.1);
    let g = chain.temp(RAW_UNIT);
    let u = chain.temp(RAW_UNIT);
    chain.push(Stage::ProjRaw {
        a: act,
        w: wg,
        out: g,
        units,
        row_off: 0,
    });
    chain.push(Stage::ProjRaw {
        a: act,
        w: wu,
        out: u,
        units,
        row_off: 0,
    });
    let hidden = chain.temp(RAW_UNIT);
    chain.push(Stage::Swiglu {
        g,
        u,
        out: hidden,
    });
    let hq = chain.quant(d_ff);
    chain.push(Stage::QuantQ {
        h: hidden,
        out: hq,
        units,
        blocks: RAW_UNIT / Q8_BLOCK,
    });

    let wd = chain.raw_weight(down.0, d_model, d_ff, down.1);
    chain.push(Stage::ProjAddRaw {
        a: hq,
        w: wd,
        y: xv,
        width: d_model,
    });
    Some(chain)
}

/// A mixer's input normalization and projection as a chain, optionally
/// followed by the delta net's convolution and gates. Each run of the
/// projection's outputs is written where its consumer wants it.
///
/// The projection needs no barrier: every block normalizes redundantly and
/// the runs write disjoint windows. The convolution needs one, see
/// [`super::FusedMix`]. The gates share its nest.
///
/// `None` for an unsupported shape: a run not made of whole Q8_0 blocks,
/// more than one position, or gates split across two projections.
pub(crate) fn project_chain(project: &FusedProject) -> Option<Chain> {
    // One value per buffer, at the widest extent any stage touches. The pass
    // sees a dependency only between stages naming the same value, and the
    // convolution reads what the projection wrote.
    let mut extents: Vec<(Buf, usize)> = Vec::new();
    let mut want = |buf: Buf, len: usize| match extents.iter_mut().find(|(b, _)| *b == buf) {
        Some((_, at)) => *at = (*at).max(len),
        None => extents.push((buf, len)),
    };
    for run in project.runs {
        want(run.dst, run.dst_off + run.width);
    }
    if let Some(m) = &project.mix {
        // The gates stage reads both gates from one value.
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

    // A weight enters the chain once, on the first run that reads it.
    let mut weights: Vec<Option<Val>> = vec![None; project.weights.len()];
    for run in project.runs {
        let &(weight, out_dim) = project.weights.get(run.weight)?;
        let w = match weights[run.weight] {
            Some(w) => w,
            None => {
                let w = match weight {
                    ProjWeight::Q8(q) => chain.weight(q, out_dim, project.d_model),
                    ProjWeight::Raw(r, quant) => {
                        if !has_fused_decode(quant) {
                            return None;
                        }
                        chain.raw_weight(r, out_dim, project.d_model, quant)
                    }
                };
                weights[run.weight] = Some(w);
                w
            }
        };
        let out = val_of(run.dst)?;
        match weight {
            ProjWeight::Q8(_) => {
                if !run.width.is_multiple_of(Q8_BLOCK) || !run.row_off.is_multiple_of(Q8_BLOCK) {
                    return None;
                }
                chain.push(Stage::ProjF {
                    a: act,
                    w,
                    out,
                    out_off: run.dst_off,
                    units: run.width / Q8_BLOCK,
                    row_off: run.row_off,
                });
            }
            ProjWeight::Raw(..) => {
                // Whole units, then the remainder as one narrower unit. The
                // upload pads rows to a whole unit, so the remainder's decode
                // stays in bounds.
                if !run.row_off.is_multiple_of(RAW_UNIT) {
                    return None;
                }
                let (full, rem) = (run.width / RAW_UNIT, run.width % RAW_UNIT);
                if full > 0 {
                    chain.push(Stage::ProjRawF {
                        a: act,
                        w,
                        out,
                        out_off: run.dst_off,
                        units: full,
                        width: RAW_UNIT,
                        row_off: run.row_off,
                    });
                }
                if rem > 0 {
                    chain.push(Stage::ProjRawF {
                        a: act,
                        w,
                        out,
                        out_off: run.dst_off + full * RAW_UNIT,
                        units: 1,
                        width: rem,
                        row_off: run.row_off + full * RAW_UNIT,
                    });
                }
            }
        }
    }

    if let Some(m) = &project.mix {
        let spec = &m.spec;
        // Decode only: one position at a time.
        if spec.rows != 1 {
            return None;
        }
        // The stage addresses planes by a single stride, so they must be
        // evenly spaced.
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
            kv_heads: spec.kv_heads,
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
            units: 2 * spec.kv_heads + spec.heads,
            span: spec.span(),
        });
    }
    Some(chain)
}

/// Attention's output epilogue as a chain: quantize the mixed heads, then
/// the output projection accumulating into the residual. There is no
/// normalization; `x` is the attention kernel's output.
///
/// The quantization and projection have different unit widths, so they land
/// in separate nests with a barrier between them.
///
/// `None` if `width` is not a whole number of Q8_0 blocks.
pub(crate) fn attn_out_chain(x: Buf, w: QBuf, dest: Buf, width: usize, d_model: usize) -> Option<Chain> {
    if !width.is_multiple_of(Q8_BLOCK) {
        return None;
    }
    let mut chain = Chain::default();
    let xv = chain.given(x, width);
    let hq = chain.quant(width);
    chain.push(Stage::QuantQ {
        h: xv,
        out: hq,
        units: width / Q8_BLOCK,
        blocks: 1,
    });
    let wv = chain.weight(w, d_model, width);
    let yv = chain.given(dest, d_model);
    chain.push(Stage::ProjAdd {
        a: hq,
        w: wv,
        y: yv,
        width: d_model,
    });
    Some(chain)
}
