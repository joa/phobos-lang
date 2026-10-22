// The Q8_0 projection's handoff from the staged kernel to the tiles below it.

use super::*;
use raw::CheckWithin;

/// The staged kernel takes whole 128-row tiles and passes the rest down:
/// 200 = 128 + 64 + 8 rows gives it, the shallow tile and the row kernel
/// each a share. Runs last, as a check inserted earlier would change every
/// later one's data.
pub(super) fn check_q8_handoff(
    host: &dyn Backend,
    gpu: &dyn Backend,
    check_within: &CheckWithin,
    next: &mut dyn FnMut() -> f32,
) -> Result<()> {
    let (m, k, n) = (200usize, 1024usize, 512usize);
    let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
    let scales: Vec<f32> = (0..(k / 32) * n).map(|_| next().abs() + 0.01).collect();
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let packed = pack_q8_0(&qs, &scales, k, n)?;
    let run = |b: &dyn Backend| -> Result<Vec<f32>> {
        let ab = b.upload(&a)?;
        let wb = b.constant_quant(&format!("q{m}x{k}x{n}"), &packed)?;
        let out = b.alloc(m * n)?;
        b.matmul_quant(ab, m, k, wb, n, out)?;
        read_vec(b, out, m * n)
    };
    check_within(&format!("matmul_quant [{m} x {k} x {n}]"), 1e-3, &run(host)?, &run(gpu)?);
    Ok(())
}
