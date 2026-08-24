use std::cell::RefCell;

use anyhow::{Result, ensure};

use phobos_base::half::{f16_to_f32, f32_to_f16};

use super::{
    Attn, Backend, Buf, DeltaMix, HBuf, HPlane, HostQuant, L2_EPS, Packed, Plane, Q8_BLOCK, QAct,
    QBuf, Rope, quantize_row,
};

/// The reference backend: every buffer is a `Vec<f32>` on the host. Defines the
/// semantics the GPU backend must reproduce, and keeps the model runnable with
/// no GPU or MLIR toolchain present.
#[derive(Default)]
pub struct HostBackend {
    slabs: RefCell<Vec<Vec<f32>>>,
    /// Freed handles, reused so a long decode does not grow the slab forever.
    free: RefCell<Vec<usize>>,
    /// The f16 slabs, in their own table because [`HBuf`] indexes separately.
    /// The caches are the only thing here; see [`HBuf`].
    halves: RefCell<Vec<Vec<u16>>>,
    free_halves: RefCell<Vec<usize>>,
    constants: RefCell<std::collections::HashMap<String, Buf>>,
    /// Weights kept quantized, in the planes their kernels index.
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
    /// buffers stay borrowed, then put it back. Moving, not copying: the LM
    /// head destination alone is a quarter of a billion floats. Must not
    /// alias a source.
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

    fn alloc_h(&self, len: usize) -> Result<HBuf> {
        if let Some(index) = self.free_halves.borrow_mut().pop() {
            let mut halves = self.halves.borrow_mut();
            halves[index].clear();
            halves[index].resize(len, 0);
            return Ok(HBuf(index));
        }
        let mut halves = self.halves.borrow_mut();
        halves.push(vec![0; len]);
        Ok(HBuf(halves.len() - 1))
    }

    fn release_h(&self, buf: HBuf) {
        self.free_halves.borrow_mut().push(buf.0);
    }

    fn zeroed_h(&self, len: usize) -> Result<HBuf> {
        self.alloc_h(len)
    }

    fn read_h(&self, buf: HBuf, out: &mut [f32]) -> Result<()> {
        let halves = self.halves.borrow();
        let src = &halves[buf.0];
        ensure!(
            src.len() >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            src.len()
        );
        for (o, &h) in out.iter_mut().zip(src) {
            *o = f16_to_f32(h);
        }
        Ok(())
    }

    fn copy_h(
        &self,
        src: HBuf,
        src_offset: usize,
        dst: HBuf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let mut halves = self.halves.borrow_mut();
        ensure!(src.0 != dst.0, "copy_h aliases its source");
        let (from, into) = if src.0 < dst.0 {
            let (a, b) = halves.split_at_mut(dst.0);
            (&a[src.0], &mut b[0])
        } else {
            let (a, b) = halves.split_at_mut(src.0);
            (&b[0], &mut a[dst.0])
        };
        into[dst_offset..dst_offset + len].copy_from_slice(&from[src_offset..src_offset + len]);
        Ok(())
    }

    fn store_2d(&self, src: Plane, dst: HPlane, rows: usize, width: usize) -> Result<()> {
        let slabs = self.slabs.borrow();
        let from = &slabs[src.buf.0];
        let mut halves = self.halves.borrow_mut();
        let into = &mut halves[dst.buf.0];
        ensure!(
            from.len() >= src.offset + (rows - 1) * src.pitch + width
                && into.len() >= dst.offset + (rows - 1) * dst.pitch + width,
            "store_2d runs off one of its planes"
        );
        for r in 0..rows {
            let (a, b) = (src.offset + r * src.pitch, dst.offset + r * dst.pitch);
            for (h, &v) in into[b..b + width].iter_mut().zip(&from[a..a + width]) {
                *h = f32_to_f16(v);
            }
        }
        Ok(())
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

    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        let buf = self.upload(&fill()?)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf> {
        if let Some(&buf) = self.q_constants.borrow().get(key) {
            return Ok(buf);
        }
        let planes = packed.planes()?;
        let mut quants = self.quants.borrow_mut();
        quants.push((planes.qs, planes.scales, packed.n()));
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

    fn matmul_quant_act(
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
        ensure!(
            qs.len() == k * n && scales.len() == (k / Q8_BLOCK) * n,
            "a [{k}, {n}] quantized weight was uploaded with {} quants and {} scales",
            qs.len(),
            scales.len()
        );
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
            ensure!(dst.len() >= m * n, "matmul_quant destination is too small");
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

    fn attention(&self, q: Buf, keys: HBuf, values: HBuf, spec: Attn, out: Buf) -> Result<()> {
        let (dim, width) = (spec.head_dim, spec.kv_width());
        let scale = (dim as f32).sqrt().recip();
        let halves = self.halves.borrow();
        let (k, v) = (&halves[keys.0], &halves[values.0]);
        self.writing(out, |slabs, dst| {
            let queries = &slabs[q.0];
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
                        *score = query
                            .iter()
                            .zip(key)
                            .map(|(&a, &b)| a * f16_to_f32(b))
                            .sum::<f32>()
                            * scale;
                    }
                    softmax(&mut scores[..visible]);
                    let row = &mut dst[at..at + dim];
                    row.fill(0.0);
                    for (j, &p) in scores[..visible].iter().enumerate() {
                        let value = &v[j * width + column..][..dim];
                        for (o, &val) in row.iter_mut().zip(value) {
                            *o += p * f16_to_f32(val);
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
                // Only the query carries the readout scale; the value is
                // never normalized, since it's written into the state rather
                // than matched against it.
                let scale = if plane == 0 { mix.query_scale } else { 1.0 };
                let normalize = mix.normalize && plane < 2;
                // Query/key exist at only kv_heads physical columns; a packed
                // head beyond that reads back via h % kv_heads, matching
                // upstream's ggml_repeat_4d.
                let src_heads = if plane == 2 { heads } else { mix.kv_heads };
                for t in 0..mix.gates() {
                    let (position, head) = (t / heads, t % heads);
                    let at = plane * mix.span() + t * dim;
                    let column = base + (head % src_heads) * mix.head_stride;
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
mod tests;
