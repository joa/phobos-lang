use std::cell::RefCell;

use anyhow::{Result, ensure};

use phobos_base::half::{f16_to_f32, f32_to_f16};

use crate::quant::{Quant, q4_k};

use super::{
    Attn, Backend, Buf, DeltaMix, HADAMARD_BLOCK, HBuf, HPlane, HeadPerm, HostQuant, L2_EPS, Packed,
    Plane, Q8_BLOCK, QAct,
    QBuf, RawBuf, Rope, quantize_row,
};

/// The reference backend, where every buffer is a `Vec<f32>` on the host.
///
/// It defines the semantics the GPU backend must reproduce, and runs the
/// model with no GPU or MLIR toolchain.
#[derive(Default)]
pub struct HostBackend {
    slabs: RefCell<Vec<Vec<f32>>>,
    /// Freed handles, reused by the next allocation.
    free: RefCell<Vec<usize>>,
    /// The f16 slabs, indexed by [`HBuf`]. Only the caches live here.
    halves: RefCell<Vec<Vec<u16>>>,
    free_halves: RefCell<Vec<usize>>,
    constants: RefCell<std::collections::HashMap<String, Buf>>,
    /// Quantized weights as `(quants, scales, n)` planes.
    quants: RefCell<Vec<HostQuant>>,
    /// Quantized activations, reused slot by slot across passes.
    qacts: RefCell<Vec<(Vec<i8>, Vec<f32>)>>,
    qact_next: std::cell::Cell<usize>,
    q_constants: RefCell<std::collections::HashMap<String, QBuf>>,
    /// Expert sets in their file encoding. See `host/moe.rs`.
    experts: RefCell<Vec<std::sync::Arc<crate::experts::ExpertSet>>>,
    expert_keys: RefCell<std::collections::HashMap<String, super::ExpertsBuf>>,
    /// Elements sharing one int8 scale when a raw-weight matmul's activation
    /// is rounded first, 0 for the exact pass, and each such product's error.
    /// See [`HostBackend::round_activations`].
    act_round: std::cell::Cell<usize>,
    act_errors: RefCell<Vec<ActError>>,
    /// Whether the minimum term's factors round to f16 as well, the sums
    /// that overflowed f16, and per Q4_K weight its `dmin * m` per run and
    /// column. See [`HostBackend::round_min_f16`].
    act_min_f16: std::cell::Cell<bool>,
    act_overflows: std::cell::Cell<usize>,
    raw_mins: RefCell<std::collections::HashMap<usize, Vec<f32>>>,
}

/// The error one raw-weight matmul took from rounding its activation: the
/// product's shape and its relative Frobenius error against the exact one.
#[derive(Clone, Copy, Debug)]
pub struct ActError {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub rel: f64,
}

impl HostBackend {
    pub fn new() -> HostBackend {
        HostBackend::default()
    }

    /// Rounds every raw-weight matmul's activation to int8 with one scale per
    /// `run` elements before multiplying, as a device kernel reading an int8
    /// activation does, and keeps each product's error against the exact
    /// one. 0 restores the exact pass. The errors gathered so far are
    /// dropped.
    pub fn round_activations(&self, run: usize) {
        self.act_round.set(run);
        self.act_errors.borrow_mut().clear();
    }

    /// The errors gathered since [`HostBackend::round_activations`].
    pub fn act_errors(&self) -> Vec<ActError> {
        self.act_errors.borrow().clone()
    }

    /// With the activation rounded per 32, also computes a Q4_K product's
    /// minimum term as the staged kernel does when given the activation's
    /// (scale, sum) pairs: `f16(sa * asum) * f16(-(dmin * m))` per row,
    /// column and run, in place of the exact `-(sa * asum) * dmin * m`.
    /// Q4_K is the format the device sends with its pairs. Resets the
    /// overflow count.
    pub fn round_min_f16(&self, on: bool) {
        self.act_min_f16.set(on);
        self.act_overflows.set(0);
    }

    /// How many `sa * asum` since [`HostBackend::round_min_f16`] did not fit
    /// f16, each an infinity in the kernel's output.
    pub fn act_overflows(&self) -> usize {
        self.act_overflows.get()
    }

    /// Swaps the exact minimum term out of `out` and the f16 one in, group
    /// by group: `pairs` is the rounded activation's (scale, sum) per row
    /// and group, `mins` the weight's `dmin * m` as `[groups, n]`.
    fn round_min_term(&self, out: Buf, m: usize, groups: usize, n: usize, pairs: &[(f32, f32)], mins: &[f32]) {
        let mut overflows = 0;
        let mut slabs = self.slabs.borrow_mut();
        let out = &mut slabs[out.0][..m * n];
        for (row, row_pairs) in out.chunks_exact_mut(n).zip(pairs.chunks_exact(groups)) {
            for (&(sa, asum), mins) in row_pairs.iter().zip(mins.chunks_exact(n)) {
                let s = sa * asum;
                let s16 = f16_to_f32(f32_to_f16(s));
                if !s16.is_finite() {
                    overflows += 1;
                }
                for (o, &mw) in row.iter_mut().zip(mins) {
                    let mw16 = f16_to_f32(f32_to_f16(-mw));
                    *o += s16 * mw16 + s * mw;
                }
            }
        }
        self.act_overflows.set(self.act_overflows.get() + overflows);
    }

    /// Moves `dst` out of the slab table so it can be written while the other
    /// buffers stay borrowed, then puts it back.
    ///
    /// `dst` must not also be a source: inside `f` its slot is empty.
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

/// A Q4_K weight's `dmin * m` per run and column, `[k / 32, n]`, the
/// product as the staged kernel's decode forms it.
fn q4k_run_mins(packed: &Packed) -> Vec<f32> {
    let (spec, k, n) = (packed.spec(), packed.k(), packed.n());
    let runs = spec.block / spec.scale_run;
    let mut mins = vec![0.0f32; k / spec.scale_run * n];
    let row_bytes = k / spec.block * spec.block_bytes;
    for (j, row) in packed.blocks().chunks_exact(row_bytes).enumerate() {
        for (b, block) in row.chunks_exact(spec.block_bytes).enumerate() {
            // `d`, `dmin`, then the twelve scale bytes.
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            for run in 0..runs {
                let (_, m) = q4_k::scale_min(run, &block[4..16]);
                mins[(b * runs + run) * n + j] = dmin * f32::from(m);
            }
        }
    }
    mins
}

impl Backend for HostBackend {
    fn matmul_raw(&self, a: Buf, m: usize, k: usize, w: RawBuf, n: usize, out: Buf) -> Result<()> {
        let run = self.act_round.get();
        if run == 0 {
            return self.matmul(a, m, k, Buf(w.0), n, out);
        }
        ensure!(k.is_multiple_of(run), "rounding a [{m}, {k}] activation at one scale per {run}");
        let min_f16 = self.act_min_f16.get();
        ensure!(!min_f16 || run == Q8_BLOCK, "the f16 minimum term goes with one scale per {Q8_BLOCK}, not {run}");
        // The exact product first, then the rounded activation's over it.
        self.matmul(a, m, k, Buf(w.0), n, out)?;
        let exact: Vec<f32> = self.slabs.borrow()[out.0][..m * n].to_vec();
        let rounded = self.alloc(m * k)?;
        let mut pairs = Vec::new();
        self.writing(rounded, |slabs, dst| {
            let mut qs = vec![0i8; run];
            for (block, out) in slabs[a.0][..m * k].chunks_exact(run).zip(dst.chunks_exact_mut(run)) {
                let d = quantize_row(block, &mut qs);
                if min_f16 {
                    pairs.push((d, qs.iter().map(|&q| i32::from(q)).sum::<i32>() as f32));
                }
                for (o, &q) in out.iter_mut().zip(&qs) {
                    *o = f32::from(q) * d;
                }
            }
        });
        self.matmul(rounded, m, k, Buf(w.0), n, out)?;
        self.release(rounded);
        if min_f16 && let Some(mins) = self.raw_mins.borrow().get(&w.0) {
            self.round_min_term(out, m, k / run, n, &pairs, mins);
        }
        let (mut diff, mut norm) = (0.0f64, 0.0f64);
        for (&y, &x) in self.slabs.borrow()[out.0][..m * n].iter().zip(&exact) {
            diff += f64::from(y - x).powi(2);
            norm += f64::from(x).powi(2);
        }
        let rel = if norm > 0.0 { (diff / norm).sqrt() } else { 0.0 };
        self.act_errors.borrow_mut().push(ActError { m, k, n, rel });
        Ok(())
    }

    fn buf_len(&self, buf: Buf) -> Result<usize> {
        Ok(self.slabs.borrow()[buf.0].len())
    }

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

    fn constant_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {
        let buf = self.constant_lazy(key, &|| Ok(packed.dense()))?;
        if packed.quant() == Quant::Q4_K {
            self.raw_mins.borrow_mut().entry(buf.0).or_insert_with(|| q4k_run_mins(packed));
        }
        Ok(RawBuf(buf.0))
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

    fn constant_experts(
        &self,
        key: &str,
        set: &std::sync::Arc<crate::experts::ExpertSet>,
    ) -> Result<super::ExpertsBuf> {
        Ok(self.register_experts(key, set))
    }

    fn moe(&self, req: super::Moe) -> Result<()> {
        self.run_moe(req)
    }

    fn begin_pass(&self, _rows: usize) -> Result<()> {
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
            // The weight is contiguous along k, so k is the inner loop and
            // the scales apply once per block.
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

    fn matmul_rows(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.writing(out, |slabs, dst| {
            let (left, right) = (&slabs[a.0], &slabs[w.0]);
            ensure!(
                left.len() >= m * k && right.len() >= n * k && dst.len() >= m * n,
                "matmul_rows operands are too small"
            );
            for (row, d) in left.chunks_exact(k).zip(dst.chunks_exact_mut(n)).take(m) {
                for (v, col) in d.iter_mut().zip(right.chunks_exact(k)) {
                    *v = row.iter().zip(col).map(|(x, y)| x * y).sum();
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
            // The inner loop walks the weight and destination contiguously,
            // so it vectorizes.
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

    fn add_rows(&self, x: Buf, rows: usize, width: usize, bias: Buf) -> Result<()> {
        self.writing(x, |slabs, dst| {
            let bias = &slabs[bias.0];
            ensure!(
                dst.len() >= rows * width && bias.len() >= width,
                "add_rows: {rows} rows of {width} do not fit their buffers"
            );
            for row in dst[..rows * width].chunks_exact_mut(width) {
                for (d, &b) in row.iter_mut().zip(bias) {
                    *d += b;
                }
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

    fn hadamard(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        signs: Buf,
        perm: Option<HeadPerm>,
        out: Buf,
    ) -> Result<()> {
        ensure!(x != out, "hadamard cannot run in place");
        self.writing(out, |slabs, dst| {
            let (src, signs) = (&slabs[x.0], &slabs[signs.0]);
            ensure!(
                src.len() >= rows * width && dst.len() >= rows * width && signs.len() >= width,
                "hadamard operands are too small for {rows} rows of {width}"
            );
            for (row, o) in src.chunks_exact(width).zip(dst.chunks_exact_mut(width)).take(rows) {
                crate::hadamard::rotate_row(row, &signs[..width], perm, HADAMARD_BLOCK, o);
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
                // Only the query gets the readout scale. Only query and key
                // are normalized; the value is not.
                let scale = if plane == 0 { mix.query_scale } else { 1.0 };
                let normalize = mix.normalize && plane < 2;
                // Query and key have only `kv_heads` heads. Packed head `h`
                // reads head `h % kv_heads`, like upstream's ggml_repeat_4d.
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
        pitch: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()> {
        let (heads, span, gates) = (mix.heads, mix.span(), mix.gates());
        let read_len = (mix.rows - 1) * pitch + heads;
        self.writing(packed, |slabs, dst| {
            let (a, b) = (&slabs[decay_in.0][decay_at..], &slabs[beta_in.0][beta_at..]);
            let (r, bias) = (&slabs[rate.0], &slabs[dt_bias.0]);
            ensure!(
                a.len() >= read_len
                    && b.len() >= read_len
                    && r.len() >= heads
                    && bias.len() >= heads
                    && dst.len() >= mix.packed_len(),
                "delta_gates operands or destination are too small"
            );
            for i in 0..gates {
                let h = i % heads;
                let at = i / heads * pitch + h;
                dst[3 * span + i] = (r[h] * softplus(a[at] + bias[h])).exp();
                dst[3 * span + gates + i] = sigmoid(b[at]);
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
        // The state is updated in place, so it is moved out like the
        // destination.
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

mod moe;

#[cfg(test)]
mod tests;
