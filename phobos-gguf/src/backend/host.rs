use std::cell::RefCell;

use anyhow::{Result, ensure};

use super::{
    Attn, Backend, Buf, DeltaMix, HostQuant, L2_EPS, Plane, Q8_BLOCK, QAct, QBuf, Rope,
    check_q8_shape, quantize_row,
};

/// The reference backend: every buffer is a `Vec<f32>` on the host. Defines the
/// semantics the GPU backend must reproduce, and keeps the model runnable with
/// no GPU or MLIR toolchain present.
#[derive(Default)]
pub struct HostBackend {
    slabs: RefCell<Vec<Vec<f32>>>,
    /// Freed handles, reused so a long decode does not grow the slab forever.
    free: RefCell<Vec<usize>>,
    constants: RefCell<std::collections::HashMap<String, Buf>>,
    /// Q8_0 weights, kept quantized.
    quants: RefCell<Vec<HostQuant>>,
    /// Quantized activations, reused slot by slot across passes.
    qacts: RefCell<Vec<(Vec<i8>, Vec<f32>)>>,
    qact_next: std::cell::Cell<usize>,
    q_constants: RefCell<std::collections::HashMap<String, QBuf>>,
}

impl HostBackend {
    pub fn new() -> HostBackend {
        HostBackend::default()
    }

    /// Move a destination out of the slab so it can be written while other
    /// buffers stay borrowed, then put it back. Moving rather than copying
    /// matters: the LM head destination alone is a quarter of a billion floats.
    /// The destination must not alias a source.
    fn writing<R>(&self, dst: Buf, f: impl FnOnce(&[Vec<f32>], &mut Vec<f32>) -> R) -> R {
        let mut taken = std::mem::take(&mut self.slabs.borrow_mut()[dst.0]);
        let result = {
            let slabs = self.slabs.borrow();
            f(&slabs, &mut taken)
        };
        self.slabs.borrow_mut()[dst.0] = taken;
        result
    }
}

impl Backend for HostBackend {
    fn alloc(&self, len: usize) -> Result<Buf> {
        if let Some(index) = self.free.borrow_mut().pop() {
            let mut slabs = self.slabs.borrow_mut();
            slabs[index].clear();
            slabs[index].resize(len, 0.0);
            return Ok(Buf(index));
        }
        let mut slabs = self.slabs.borrow_mut();
        slabs.push(vec![0.0; len]);
        Ok(Buf(slabs.len() - 1))
    }

    fn release(&self, buf: Buf) {
        self.free.borrow_mut().push(buf.0);
    }

    fn upload(&self, data: &[f32]) -> Result<Buf> {
        let buf = self.alloc(data.len())?;
        self.slabs.borrow_mut()[buf.0].copy_from_slice(data);
        Ok(buf)
    }

    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()> {
        let slabs = self.slabs.borrow();
        let src = &slabs[buf.0];
        ensure!(
            src.len() >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            src.len()
        );
        out.copy_from_slice(&src[..out.len()]);
        Ok(())
    }

    fn constant(&self, key: &str, data: &[f32]) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        let buf = self.upload(data)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn constant_q8(
        &self,
        key: &str,
        qs: &[i8],
        scales: &[f32],
        k: usize,
        n: usize,
    ) -> Result<QBuf> {
        if let Some(&buf) = self.q_constants.borrow().get(key) {
            return Ok(buf);
        }
        check_q8_shape(qs, scales, k, n)?;
        let mut quants = self.quants.borrow_mut();
        quants.push((qs.to_vec(), scales.to_vec(), n));
        let buf = QBuf(quants.len() - 1);
        drop(quants);
        self.q_constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn begin_pass(&self) -> Result<()> {
        self.qact_next.set(0);
        Ok(())
    }

    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        let at = self.qact_next.get();
        let mut acts = self.qacts.borrow_mut();
        if at == acts.len() {
            acts.push((Vec::new(), Vec::new()));
        }
        let (qa, da) = &mut acts[at];
        qa.resize(m * k, 0);
        da.resize(m * k / Q8_BLOCK, 0.0);
        let slabs = self.slabs.borrow();
        let left = &slabs[a.0];
        ensure!(left.len() >= m * k, "quantize_act activation is too small");
        for (b, block) in left[..m * k].chunks_exact(Q8_BLOCK).enumerate() {
            da[b] = quantize_row(block, &mut qa[b * Q8_BLOCK..(b + 1) * Q8_BLOCK]);
        }
        self.qact_next.set(at + 1);
        Ok(QAct(at))
    }

    fn matmul_q8_act(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let quants = self.quants.borrow();
        let (qs, scales, stored_n) = quants
            .get(w.0)
            .ok_or_else(|| anyhow::anyhow!("use of an unknown quantized weight handle"))?;
        ensure!(
            *stored_n == n,
            "quantized weight was uploaded with n = {stored_n}, used with n = {n}"
        );
        check_q8_shape(qs, scales, k, n)?;
        let acts = self.qacts.borrow();
        let (qa, da) = acts
            .get(act.0)
            .ok_or_else(|| anyhow::anyhow!("use of an unknown quantized activation handle"))?;
        let blocks = k / Q8_BLOCK;
        ensure!(
            qa.len() >= m * k && da.len() >= m * blocks,
            "quantized activation {} holds {} values, used as {m} x {k}",
            act.0,
            qa.len()
        );
        self.writing(out, |_, dst| {
            ensure!(dst.len() >= m * n, "matmul_q8 destination is too small");
            dst[..m * n].fill(0.0);
            // The weight walks k contiguously for one output, so the
            // contraction is the inner loop and the scale hoists out of each
            // 32-element block.
            for i in 0..m {
                let row = &qa[i * k..(i + 1) * k];
                let row_scales = &da[i * blocks..(i + 1) * blocks];
                for j in 0..n {
                    let weights = &qs[j * k..(j + 1) * k];
                    let mut total = 0.0f32;
                    for (b, (a_block, w_block)) in row
                        .chunks_exact(Q8_BLOCK)
                        .zip(weights.chunks_exact(Q8_BLOCK))
                        .enumerate()
                    {
                        // The integer dot the hardware does one instruction per
                        // four lanes; both scales are constant across it.
                        let partial: i32 = a_block
                            .iter()
                            .zip(w_block)
                            .map(|(&x, &q)| i32::from(x) * i32::from(q))
                            .sum();
                        total += partial as f32 * scales[b * n + j] * row_scales[b];
                    }
                    dst[i * n + j] = total;
                }
            }
            Ok(())
        })
    }

    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.writing(out, |slabs, dst| {
            let left = &slabs[a.0];
            let right = &slabs[w.0];
            ensure!(
                left.len() >= m * k && right.len() >= k * n,
                "matmul operands are too small"
            );
            ensure!(dst.len() >= m * n, "matmul destination is too small");
            dst[..m * n].fill(0.0);
            // Ordered so the inner pass walks the weight and the destination
            // contiguously, which lets it vectorize.
            for i in 0..m {
                let row = &mut dst[i * n..(i + 1) * n];
                for p in 0..k {
                    let scale = left[i * k + p];
                    if scale == 0.0 {
                        continue;
                    }
                    for (d, &r) in row.iter_mut().zip(&right[p * n..(p + 1) * n]) {
                        *d += scale * r;
                    }
                }
            }
            Ok(())
        })
    }

    fn rms_norm(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<()> {
        self.writing(out, |slabs, dst| {
            let src = &slabs[x.0];
            let g = &slabs[gain.0];
            ensure!(
                src.len() >= rows * width && g.len() >= width,
                "rms_norm operands are too small"
            );
            for r in 0..rows {
                let row = &src[r * width..(r + 1) * width];
                let mean_square = row.iter().map(|&v| v * v).sum::<f32>() / width as f32;
                let inv = (mean_square + eps).sqrt().recip();
                for (i, &v) in row.iter().enumerate() {
                    dst[r * width + i] = v * inv * g[i];
                }
            }
            Ok(())
        })
    }

    fn add_into(&self, acc: Buf, add: Buf) -> Result<()> {
        self.writing(acc, |slabs, dst| {
            for (d, &a) in dst.iter_mut().zip(&slabs[add.0]) {
                *d += a;
            }
            Ok(())
        })
    }

    fn swiglu(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<()> {
        self.writing(out, |slabs, dst| {
            let g = &slabs[gate.0][gate_at..];
            let u = &slabs[up.0][up_at..];
            for (i, d) in dst.iter_mut().enumerate().take(len) {
                *d = silu(g[i]) * u[i];
            }
            Ok(())
        })
    }

    fn copy_2d(&self, src: Plane, dst: Plane, rows: usize, width: usize) -> Result<()> {
        self.writing(dst.buf, |slabs, out| {
            let from = &slabs[src.buf.0];
            ensure!(
                from.len() >= src.offset + (rows - 1) * src.pitch + width
                    && out.len() >= dst.offset + (rows - 1) * dst.pitch + width,
                "copy_2d runs off one of its planes"
            );
            for r in 0..rows {
                let (a, b) = (src.offset + r * src.pitch, dst.offset + r * dst.pitch);
                out[b..b + width].copy_from_slice(&from[a..a + width]);
            }
            Ok(())
        })
    }

    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()> {
        let half = spec.rope_dim / 2;
        self.writing(x, |slabs, dst| {
            let angles = &slabs[table.0];
            ensure!(
                dst.len() >= rows * spec.heads * spec.head_dim
                    && angles.len() >= (spec.start_pos + rows) * spec.rope_dim,
                "rope operands are too small"
            );
            for r in 0..rows * spec.heads {
                let pos = spec.start_pos + r / spec.heads;
                let (cos, sin) = angles[pos * spec.rope_dim..][..spec.rope_dim].split_at(half);
                let head = &mut dst[r * spec.head_dim..][..spec.head_dim];
                for i in 0..half {
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cos[i] - b * sin[i];
                    head[i + half] = a * sin[i] + b * cos[i];
                }
            }
            Ok(())
        })
    }

    fn attention(&self, q: Buf, keys: Buf, values: Buf, spec: Attn, out: Buf) -> Result<()> {
        let (dim, width) = (spec.head_dim, spec.kv_width());
        let scale = (dim as f32).sqrt().recip();
        self.writing(out, |slabs, dst| {
            let (queries, k, v) = (&slabs[q.0], &slabs[keys.0], &slabs[values.0]);
            ensure!(
                queries.len() >= spec.rows * spec.n_head * dim
                    && k.len() >= spec.total() * width
                    && v.len() >= spec.total() * width,
                "attention operands are too small"
            );
            let mut scores = vec![0.0f32; spec.total()];
            for t in 0..spec.rows {
                let visible = spec.start_pos + t + 1;
                for h in 0..spec.n_head {
                    let at = (t * spec.n_head + h) * dim;
                    let query = &queries[at..at + dim];
                    let column = (h / spec.group()) * dim;
                    for (j, score) in scores[..visible].iter_mut().enumerate() {
                        let key = &k[j * width + column..][..dim];
                        *score = query.iter().zip(key).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                    }
                    softmax(&mut scores[..visible]);
                    let row = &mut dst[at..at + dim];
                    row.fill(0.0);
                    for (j, &p) in scores[..visible].iter().enumerate() {
                        let value = &v[j * width + column..][..dim];
                        for (o, &val) in row.iter_mut().zip(value) {
                            *o += p * val;
                        }
                    }
                }
            }
            Ok(())
        })
    }

    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()> {
        self.writing(x, |slabs, dst| {
            let g = &slabs[gate.0];
            for (d, &v) in dst.iter_mut().zip(g) {
                *d *= sigmoid(v);
            }
            Ok(())
        })
    }

    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()> {
        let (heads, dim, channels) = (mix.heads, mix.head_dim, mix.channels());
        self.writing(packed, |slabs, dst| {
            let x = &slabs[history.0];
            let w = &slabs[taps.0];
            ensure!(
                x.len() >= mix.history_len()
                    && w.len() >= mix.kernel * channels
                    && dst.len() >= mix.packed_len(),
                "delta_conv operands or destination are too small"
            );
            for (plane, &base) in mix.planes.iter().enumerate() {
                // Only the query carries the readout scale, and the value is
                // never normalized: it is written into the state, not matched
                // against it.
                let scale = if plane == 0 { mix.query_scale } else { 1.0 };
                let normalize = mix.normalize && plane < 2;
                // One row per (position, head), head fastest: both the packed
                // layout and the span the norm covers.
                for t in 0..mix.gates() {
                    let (position, head) = (t / heads, t % heads);
                    let at = plane * mix.span() + t * dim;
                    let column = base + head * mix.head_stride;
                    let row = &mut dst[at..at + dim];
                    for (d, o) in row.iter_mut().enumerate() {
                        let c = column + d;
                        let mut acc = 0.0;
                        for k in 0..mix.kernel {
                            acc += w[k * channels + c] * x[(position + k) * channels + c];
                        }
                        *o = silu(acc);
                    }
                    let gain = if normalize {
                        let square = row.iter().map(|&v| v * v).sum::<f32>();
                        scale / (square + L2_EPS).sqrt()
                    } else {
                        scale
                    };
                    for v in row.iter_mut() {
                        *v *= gain;
                    }
                }
            }
            Ok(())
        })
    }

    fn delta_gates(
        &self,
        decay_in: Buf,
        decay_at: usize,
        beta_in: Buf,
        beta_at: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()> {
        let (heads, span, gates) = (mix.heads, mix.span(), mix.gates());
        self.writing(packed, |slabs, dst| {
            let (a, b) = (&slabs[decay_in.0][decay_at..], &slabs[beta_in.0][beta_at..]);
            let (r, bias) = (&slabs[rate.0], &slabs[dt_bias.0]);
            ensure!(
                a.len() >= gates
                    && b.len() >= gates
                    && r.len() >= heads
                    && bias.len() >= heads
                    && dst.len() >= mix.packed_len(),
                "delta_gates operands or destination are too small"
            );
            for i in 0..gates {
                let h = i % heads;
                dst[3 * span + i] = (r[h] * softplus(a[i] + bias[h])).exp();
                dst[3 * span + gates + i] = sigmoid(b[i]);
            }
            Ok(())
        })
    }

    fn delta_rule(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        let plane = head_dim * head_dim;
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        // The state is read and written by the same call, so it comes out of
        // the slab alongside the destination rather than being borrowed.
        let mut carried = std::mem::take(&mut self.slabs.borrow_mut()[state.0]);
        let result = self.writing(out, |slabs, dst| {
            let all = &slabs[packed.0];
            ensure!(
                all.len() >= 3 * span + 2 * gates
                    && carried.len() >= heads * plane
                    && dst.len() >= span,
                "delta_rule operands, state or destination are too small"
            );
            let (qs, ks, vs) = (&all[..span], &all[span..2 * span], &all[2 * span..3 * span]);
            let decays = &all[3 * span..3 * span + gates];
            let betas = &all[3 * span + gates..3 * span + 2 * gates];
            let mut error = vec![0.0f32; head_dim];
            for t in 0..rows {
                for h in 0..heads {
                    let r = t * heads + h;
                    let at = r * head_dim;
                    let (q_row, k_row, v_row) = (
                        &qs[at..at + head_dim],
                        &ks[at..at + head_dim],
                        &vs[at..at + head_dim],
                    );
                    let s = &mut carried[h * plane..(h + 1) * plane];
                    let (d, b) = (decays[r], betas[r]);

                    error.fill(0.0);
                    for (i, &ki) in k_row.iter().enumerate() {
                        let row = &mut s[i * head_dim..(i + 1) * head_dim];
                        for (e, sij) in error.iter_mut().zip(row.iter_mut()) {
                            *sij *= d;
                            *e += *sij * ki;
                        }
                    }
                    for (e, &vj) in error.iter_mut().zip(v_row) {
                        *e = b * (vj - *e);
                    }
                    let o = &mut dst[at..at + head_dim];
                    o.fill(0.0);
                    for (i, (&ki, &qi)) in k_row.iter().zip(q_row).enumerate() {
                        let row = &mut s[i * head_dim..(i + 1) * head_dim];
                        for ((sij, &e), oj) in row.iter_mut().zip(&error).zip(o.iter_mut()) {
                            *sij += ki * e;
                            *oj += qi * *sij;
                        }
                    }
                }
            }
            Ok(())
        });
        self.slabs.borrow_mut()[state.0] = carried;
        result
    }

    fn copy(
        &self,
        src: Buf,
        src_offset: usize,
        dst: Buf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.writing(dst, |slabs, out| {
            out[dst_offset..dst_offset + len]
                .copy_from_slice(&slabs[src.0][src_offset..src_offset + len]);
            Ok(())
        })
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `log(1 + e^x)`, guarded so large inputs do not overflow the exponential.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Root-mean-square normalization with a per-channel gain, over each
/// `width`-element row of `x` in place.
pub fn rms_norm(x: &mut [f32], width: usize, weight: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let mean_square = row.iter().map(|&v| v * v).sum::<f32>() / width as f32;
        let inv = (mean_square + eps).sqrt().recip();
        for (v, &w) in row.iter_mut().zip(weight) {
            *v = *v * inv * w;
        }
    }
}

/// In-place softmax over a slice, shifted by the maximum for stability.
pub fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { sum.recip() } else { 0.0 };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::super::read_vec;
    use super::*;

    #[test]
    fn host_matmul_matches_by_hand() {
        let backend = HostBackend::new();
        let a = backend.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = backend.upload(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let c = backend.alloc(4).unwrap();
        backend.matmul(a, 2, 3, b, 2, c).unwrap();
        assert_eq!(
            read_vec(&backend, c, 4).unwrap(),
            vec![4.0, 5.0, 10.0, 11.0]
        );
    }

    #[test]
    fn q8_matmul_matches_the_dequantized_matmul() {
        // The quantized path has to agree with multiplying by the dequantized
        // weight, since that is what it replaces.
        let (k, n) = (Q8_BLOCK * 3, 5usize);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 8388608.0 - 1.0
        };

        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(k / Q8_BLOCK) * n)
            .map(|_| next().abs() + 0.01)
            .collect();
        // qs is [n, k]; the dense equivalent is [k, n], so this transposes.
        let mut dense = vec![0.0f32; k * n];
        for j in 0..n {
            for p in 0..k {
                dense[p * n + j] = qs[j * k + p] as f32 * scales[(p / Q8_BLOCK) * n + j];
            }
        }
        // Pre-quantize the activation and hand both paths the dequantized
        // values. matmul_q8 quantizes internally, and requantizing an already
        // quantized row is exact, so this isolates the weight layout from the
        // activation's precision loss.
        let mut a: Vec<f32> = (0..2 * k).map(|_| next()).collect();
        let mut scratch = vec![0i8; Q8_BLOCK];
        for block in a.chunks_exact_mut(Q8_BLOCK) {
            let d = quantize_row(block, &mut scratch);
            for (v, &q) in block.iter_mut().zip(&scratch) {
                *v = q as f32 * d;
            }
        }

        let backend = HostBackend::new();
        let x = backend.upload(&a).unwrap();

        let dense_buf = backend.constant("dense", &dense).unwrap();
        let want = backend.alloc(2 * n).unwrap();
        backend.matmul(x, 2, k, dense_buf, n, want).unwrap();

        let q = backend.constant_q8("q", &qs, &scales, k, n).unwrap();
        let got = backend.alloc(2 * n).unwrap();
        backend.matmul_q8(x, 2, k, q, n, got).unwrap();

        for (w, g) in read_vec(&backend, want, 2 * n)
            .unwrap()
            .iter()
            .zip(&read_vec(&backend, got, 2 * n).unwrap())
        {
            assert!((w - g).abs() / w.abs().max(1.0) < 1e-5, "{w} vs {g}");
        }
    }

    #[test]
    fn q8_constants_upload_once_and_check_their_shape() {
        let backend = HostBackend::new();
        let qs = vec![1i8; Q8_BLOCK * 2];
        let scales = vec![0.5f32; 2];
        let a = backend.constant_q8("w", &qs, &scales, Q8_BLOCK, 2).unwrap();
        let b = backend.constant_q8("w", &qs, &scales, Q8_BLOCK, 2).unwrap();
        assert_eq!(a, b);

        // k must tile into whole blocks, and the scale count follows from it.
        assert!(backend.constant_q8("bad_k", &qs, &scales, 7, 2).is_err());
        assert!(
            backend
                .constant_q8("bad_scales", &qs, &[0.5], Q8_BLOCK, 2)
                .is_err()
        );
    }

    #[test]
    fn released_handles_are_reused() {
        let backend = HostBackend::new();
        let first = backend.alloc(8).unwrap();
        backend.release(first);
        let second = backend.alloc(8).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn constants_upload_once() {
        let backend = HostBackend::new();
        let a = backend.constant("w", &[1.0, 2.0]).unwrap();
        let b = backend.constant("w", &[9.9, 9.9]).unwrap();
        assert_eq!(a, b);
        assert_eq!(read_vec(&backend, a, 2).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn rms_norm_scales_rows_independently() {
        let backend = HostBackend::new();
        let x = backend.upload(&[3.0, 4.0, 30.0, 40.0]).unwrap();
        let g = backend.upload(&[1.0, 1.0]).unwrap();
        let out = backend.alloc(4).unwrap();
        backend.rms_norm(x, 2, 2, g, 0.0, out).unwrap();
        let got = read_vec(&backend, out, 4).unwrap();
        let expected = 3.0 / (12.5f32).sqrt();
        assert!((got[0] - expected).abs() < 1e-6);
        assert!((got[2] - expected).abs() < 1e-6);
    }

    #[test]
    fn swiglu_and_residual() {
        let backend = HostBackend::new();
        // The gate and the up half as one buffer, as the fused projection
        // hands them over.
        let both = backend.upload(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        let u = backend.upload(&[2.0, 3.0]).unwrap();
        let out = backend.alloc(2).unwrap();
        backend.swiglu(both, 0, both, 2, out, 2).unwrap();
        let got = read_vec(&backend, out, 2).unwrap();
        assert_eq!(got[0], 0.0);
        assert!((got[1] - silu(1.0) * 3.0).abs() < 1e-6);

        backend.add_into(out, u).unwrap();
        assert_eq!(read_vec(&backend, out, 2).unwrap()[0], 2.0);
    }

    #[test]
    fn copy_moves_a_window() {
        let backend = HostBackend::new();
        let src = backend.upload(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let dst = backend.alloc(4).unwrap();
        backend.copy(src, 1, dst, 2, 2).unwrap();
        assert_eq!(
            read_vec(&backend, dst, 4).unwrap(),
            vec![0.0, 0.0, 2.0, 3.0]
        );
    }

    #[test]
    fn delta_conv_leaves_an_all_zero_head_at_zero() {
        // One position and one tap, so the convolution is a multiply by one and
        // the query plane is its input through SiLU and the normalization. A
        // silent head has no norm to divide by.
        let backend = HostBackend::new();
        let mix = DeltaMix {
            rows: 1,
            heads: 2,
            head_dim: 2,
            kernel: 1,
            planes: [0, 4, 8],
            head_stride: 2,
            normalize: true,
            query_scale: 1.0,
        };
        let mut stream = vec![0.0f32; mix.channels()];
        stream[0..2].copy_from_slice(&[3.0, 4.0]);
        let history = backend.upload(&stream).unwrap();
        let taps = backend.upload(&vec![1.0; mix.channels()]).unwrap();
        let packed = backend.alloc(mix.packed_len()).unwrap();
        backend.delta_conv(history, taps, mix, packed).unwrap();

        let out = read_vec(&backend, packed, mix.packed_len()).unwrap();
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "loud head normalized to {norm}");
        assert_eq!(&out[2..4], &[0.0, 0.0]);
    }

    #[test]
    fn delta_gates_hold_up_where_the_direct_softplus_overflows() {
        let backend = HostBackend::new();
        let mix = DeltaMix {
            rows: 2,
            heads: 1,
            head_dim: 2,
            kernel: 1,
            planes: [0, 2, 4],
            head_stride: 2,
            normalize: false,
            query_scale: 1.0,
        };
        // exp(200) is infinite in f32, so a softplus written as log(1 + exp(x))
        // returns infinity here and the decay comes out as zero or a NaN.
        let decay_in = backend.upload(&[200.0, 0.0]).unwrap();
        let beta_in = backend.upload(&[0.0, 0.0]).unwrap();
        let rate = backend.upload(&[-1.0]).unwrap();
        let bias = backend.upload(&[0.0]).unwrap();
        let packed = backend.alloc(mix.packed_len()).unwrap();
        backend
            .delta_gates(decay_in, 0, beta_in, 0, rate, bias, mix, packed)
            .unwrap();

        let out = read_vec(&backend, packed, mix.packed_len()).unwrap();
        let at = 3 * mix.span();
        // softplus(200) is 200, so the decay is exp(-200), which underflows to
        // zero. The point is that it is a zero and not a NaN.
        assert_eq!(out[at], 0.0);
        // softplus(0) is ln(2), so the decay is a half.
        assert!((out[at + 1] - 0.5).abs() < 1e-6);
        assert_eq!(&out[at + 2..at + 4], &[0.5, 0.5]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0];
        softmax(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn softplus_stays_finite_for_large_inputs() {
        assert!((softplus(0.0) - 2.0f32.ln()).abs() < 1e-6);
        assert_eq!(softplus(100.0), 100.0);
        assert!(softplus(-100.0).abs() < 1e-6);
    }
}
