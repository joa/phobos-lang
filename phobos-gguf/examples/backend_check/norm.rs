// The quantizing norms: the normalized rows, and the int8 copy read back
// through a Q8_0 projection.

use super::*;
use raw::{CheckSpread, CheckWithin};

pub(super) fn check_norms_q(
    host: &dyn Backend,
    gpu: &dyn Backend,
    rows: usize,
    eps: f32,
    check_within: &CheckWithin,
    check_spread: &CheckSpread,
    next: &mut dyn FnMut() -> f32,
) -> Result<()> {
    // `rms_norm_q`: the normalized rows, and the int8 copy read back through
    // a Q8_0 projection. At this width and the 27B's.
    for width_q in [1024usize, 5120] {
        let x: Vec<f32> = (0..rows * width_q).map(|_| next() * 4.0).collect();
        let gain: Vec<f32> = (0..width_q).map(|_| next() + 1.5).collect();
        let n = 64;
        let qs: Vec<i8> = (0..width_q * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(width_q / 32) * n).map(|_| next().abs() + 0.01).collect();
        let packed = pack_q8_0(&qs, &scales, width_q, n)?;
        let run = |b: &dyn Backend| -> Result<(Vec<f32>, Vec<f32>)> {
            let xb = b.upload(&x)?;
            let gb = b.upload(&gain)?;
            let out = b.alloc(rows * width_q)?;
            let act = b.rms_norm_q(xb, rows, width_q, gb, eps, out)?;
            let wb = b.constant_quant(&format!("rmsq{width_q}"), &packed)?;
            let proj = b.alloc(rows * n)?;
            b.matmul_quant_act(act, rows, width_q, wb, n, proj)?;
            Ok((read_vec(b, out, rows * width_q)?, read_vec(b, proj, rows * n)?))
        };
        let ((want_o, want_p), (got_o, got_p)) = (run(host)?, run(gpu)?);
        check_within(&format!("rms_norm_q [{rows} x {width_q}]"), 1e-4, &want_o, &got_o);
        // A .5 tie may round to neighbouring int8 values on host and device;
        // see `check_hadamard`.
        check_spread(&format!("rms_norm_q act [{rows} x {width_q} x {n}]"), &want_p, &got_p);
    }

    // `rms_norm_gated`: the delta net's readout, at its head width, with the
    // gate read from past the start of its buffer.
    for width_g in [128usize, 1024] {
        let x: Vec<f32> = (0..rows * width_g).map(|_| next() * 4.0).collect();
        let gain: Vec<f32> = (0..width_g).map(|_| next() + 1.5).collect();
        let gate_at = 4;
        let gate: Vec<f32> = (0..gate_at + rows * width_g).map(|_| next() * 3.0).collect();
        let n = 64;
        let qs: Vec<i8> = (0..width_g * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(width_g / 32) * n).map(|_| next().abs() + 0.01).collect();
        let packed = pack_q8_0(&qs, &scales, width_g, n)?;
        let run = |b: &dyn Backend| -> Result<(Vec<f32>, Vec<f32>)> {
            let xb = b.upload(&x)?;
            let gb = b.upload(&gain)?;
            let zb = b.upload(&gate)?;
            let out = b.alloc(rows * width_g)?;
            let act = b.rms_norm_gated(xb, rows, width_g, gb, eps, zb, gate_at, out)?;
            let wb = b.constant_quant(&format!("rmsg{width_g}"), &packed)?;
            let proj = b.alloc(rows * n)?;
            b.matmul_quant_act(act, rows, width_g, wb, n, proj)?;
            Ok((read_vec(b, out, rows * width_g)?, read_vec(b, proj, rows * n)?))
        };
        let ((want_o, want_p), (got_o, got_p)) = (run(host)?, run(gpu)?);
        check_within(&format!("rms_norm_gated [{rows} x {width_g}]"), 1e-4, &want_o, &got_o);
        check_spread(&format!("rms_norm_gated act [{rows} x {width_g} x {n}]"), &want_p, &got_p);
    }
    Ok(())
}
