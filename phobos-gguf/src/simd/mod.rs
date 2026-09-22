//! The host's K-quant expert feed-forward: a threaded AVX2 dot of Q4_K,
//! Q5_K and Q6_K weight blocks against a Q8 activation, for the experts a
//! pass computes on the host rather than copies to the device.
//!
//! The activation is quantized to eight bits a block of 256 with one scale
//! and the sums of each sixteen quants, so a block's minimums (or Q6_K's
//! offset) fold into one integer term, as llama.cpp's Q8_K does. A weight
//! block is unpacked once, to an aligned byte array either path reads, and
//! dotted against every activation row.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;

use crate::experts::{ExpertSet, ExpertStack};
use crate::quant::Quant;

mod q4_k;
mod q5_k;
mod q6_k;
#[cfg(test)]
mod tests;

/// Elements of one activation block, the K-quant super-block.
pub const BLOCK: usize = 256;
/// Elements each partial sum of a block covers.
const SUM_RUN: usize = 16;
const SUMS: usize = BLOCK / SUM_RUN;

/// `[rows, k]` activations quantized to `i8` a block of 256: a scale a
/// block and the sums of each sixteen quants. Held block-major, the rows
/// of one block side by side, so a weight block's dot over the rows walks
/// memory forward rather than striding a row's width.
#[derive(Default)]
pub struct Q8Act {
    rows: usize,
    k: usize,
    d: Vec<f32>,
    qs: Vec<i8>,
    sums: Vec<i16>,
}

impl Q8Act {
    pub fn quantize(x: &[f32], rows: usize, k: usize) -> Q8Act {
        let mut act = Q8Act::default();
        act.quantize_into(x, rows, k);
        act
    }

    /// Requantizes `x`, `[rows, k]`, into the buffers already held.
    pub fn quantize_into(&mut self, x: &[f32], rows: usize, k: usize) {
        assert!(k.is_multiple_of(BLOCK) && x.len() == rows * k, "a [{rows}, {k}] activation of {} elements", x.len());
        let blocks = rows * k / BLOCK;
        (self.rows, self.k) = (rows, k);
        self.d.resize(blocks, 0.0);
        self.qs.resize(blocks * BLOCK, 0);
        self.sums.resize(blocks * SUMS, 0);
        for (i, block) in x.chunks_exact(BLOCK).enumerate() {
            let (r, b) = (i / self.blocks(), i % self.blocks());
            let at = b * rows + r;
            let absmax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let inv = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
            self.d[at] = absmax / 127.0;
            let qs = &mut self.qs[at * BLOCK..][..BLOCK];
            for (q, &v) in qs.iter_mut().zip(block) {
                *q = (v * inv).round_ties_even() as i8;
            }
            for (sum, run) in self.sums[at * SUMS..][..SUMS].iter_mut().zip(qs.chunks_exact(SUM_RUN)) {
                *sum = run.iter().map(|&q| i16::from(q)).sum();
            }
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn k(&self) -> usize {
        self.k
    }

    fn blocks(&self) -> usize {
        self.k / BLOCK
    }

    /// Row `r`'s block `b`: its scale, quants and sums.
    fn block(&self, r: usize, b: usize) -> (f32, &[i8], &[i16]) {
        let at = b * self.rows + r;
        (self.d[at], &self.qs[at * BLOCK..][..BLOCK], &self.sums[at * SUMS..][..SUMS])
    }
}

/// One block's quants unpacked to a byte apiece, in element order, aligned
/// for the vector loads.
#[repr(C, align(32))]
pub(crate) struct Quants([u8; BLOCK]);

impl Default for Quants {
    fn default() -> Quants {
        Quants([0; BLOCK])
    }
}

/// One format's block arithmetic against a Q8 block, in one instruction
/// set.
pub(crate) trait Block {
    /// Whether the implementation runs only under AVX2.
    const AVX2: bool;
    type Unpacked: Default;

    /// Unpacks one block's bytes.
    ///
    /// # Safety
    /// An AVX2 implementation runs only on a CPU that has it.
    unsafe fn unpack(bytes: &[u8], into: &mut Self::Unpacked);

    /// The block's scale, and the scale of the correction `dot` returns.
    fn scales(u: &Self::Unpacked) -> (f32, f32);

    /// The integer dot with a Q8 block, and the correction the block's
    /// minimums (or offset) owe, from the Q8 sums per sixteen.
    ///
    /// # Safety
    /// As [`Block::unpack`].
    unsafe fn dot(u: &Self::Unpacked, qs: &[i8], sums: &[i16]) -> (i32, i32);
}

/// `yt[j * rows + r] = sum_i w[j, i] x[r, i]` over the weight rows in
/// `weight`, each `act.blocks()` blocks of `block_bytes`.
#[inline(always)]
unsafe fn rows_dot<F: Block>(weight: &[u8], block_bytes: usize, act: &Q8Act, yt: &mut [f32]) {
    let rows = act.rows;
    let mut u = F::Unpacked::default();
    for (row_bytes, out) in weight.chunks_exact(act.blocks() * block_bytes).zip(yt.chunks_exact_mut(rows)) {
        out.fill(0.0);
        for (b, bytes) in row_bytes.chunks_exact(block_bytes).enumerate() {
            // SAFETY: the caller's contract, see `Block::unpack`.
            unsafe { F::unpack(bytes, &mut u) };
            let (d, dmin) = F::scales(&u);
            for (r, acc) in out.iter_mut().enumerate() {
                let (da, qs, sums) = act.block(r, b);
                // SAFETY: as above.
                let (sumi, correction) = unsafe { F::dot(&u, qs, sums) };
                *acc += da * (d * sumi as f32 - dmin * correction as f32);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_dot_avx2<F: Block>(weight: &[u8], block_bytes: usize, act: &Q8Act, yt: &mut [f32]) {
    // SAFETY: the caller checked for AVX2.
    unsafe { rows_dot::<F>(weight, block_bytes, act, yt) }
}

/// `rows_dot` in the instruction set `F` is written for.
fn rows_dot_in<F: Block>(weight: &[u8], block_bytes: usize, act: &Q8Act, yt: &mut [f32]) {
    // SAFETY: an AVX2 implementation is chosen only after `avx2()` said
    // the CPU has it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if F::AVX2 {
            return rows_dot_avx2::<F>(weight, block_bytes, act, yt);
        }
        rows_dot::<F>(weight, block_bytes, act, yt)
    }
}

/// The GEMM in one format and instruction set. Threaded, the weight rows
/// are dealt out in contiguous chunks, each thread's outputs its own;
/// otherwise it runs where it is called, for a caller that is itself one
/// of many tasks.
fn gemm_with<F: Block>(weight: &[u8], n: usize, block_bytes: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) {
    if !threaded {
        return rows_dot_in::<F>(weight, block_bytes, act, yt);
    }
    let rows = act.rows;
    let row_bytes = act.blocks() * block_bytes;
    let chunk = n.div_ceil(pool().current_num_threads() * 2).max(1);
    pool().install(|| {
        yt.par_chunks_mut(chunk * rows)
            .zip(weight.par_chunks(chunk * row_bytes))
            .for_each(|(out, w)| rows_dot_in::<F>(w, block_bytes, act, out))
    });
}

/// Whether the vector path runs here.
pub fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// [`gemm`] over raw blocks, in the instruction set asked for.
fn gemm_raw(quant: Quant, wide: bool, weight: &[u8], n: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) -> Result<()> {
    let block_bytes = quant.spec().block_bytes;
    ensure!(weight.len() == n * act.blocks() * block_bytes && yt.len() == n * act.rows, "a [{n}, {}] weight against {} activation rows", act.k, act.rows);
    macro_rules! by_format {
        ($set:ident) => {
            match quant {
                Quant::Q4_K => gemm_with::<q4_k::$set>(weight, n, block_bytes, act, yt, threaded),
                Quant::Q5_K => gemm_with::<q5_k::$set>(weight, n, block_bytes, act, yt, threaded),
                Quant::Q6_K => gemm_with::<q6_k::$set>(weight, n, block_bytes, act, yt, threaded),
                other => bail!("no host expert kernel for {}", other.name()),
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if wide {
        by_format!(Avx2);
        return Ok(());
    }
    let _ = wide;
    by_format!(Scalar);
    Ok(())
}

/// `yt[j * rows + r] = sum_i w[j, i] x[r, i]` over expert `e`'s `[n, k]`
/// rows of `stack`: the outputs transposed, so each thread's rows are
/// contiguous.
pub fn gemm(stack: &ExpertStack, e: usize, act: &Q8Act, yt: &mut [f32]) -> Result<()> {
    gemm_in(stack, e, act, yt, true)
}

fn gemm_in(stack: &ExpertStack, e: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) -> Result<()> {
    ensure!(stack.k() == act.k, "a [{}, {}] expert against a {}-wide activation", stack.n(), stack.k(), act.k);
    gemm_raw(stack.quant(), avx2(), stack.expert(e), stack.n(), act, yt, threaded)
}

/// Whether [`expert_ffn`] can run a set on this host.
pub fn supports(set: &ExpertSet) -> bool {
    [&set.gate, &set.up, &set.down]
        .iter()
        .all(|stack| matches!(stack.quant(), Quant::Q4_K | Quant::Q5_K | Quant::Q6_K))
}

/// What [`expert_ffn`] works in, kept between calls.
#[derive(Default)]
pub struct Scratch {
    gate: Vec<f32>,
    up: Vec<f32>,
    h: Vec<f32>,
    hact: Q8Act,
    down: Vec<f32>,
}

/// Expert `e` of `set` over the activation's rows, each row's result
/// scaled by its `weights[r]` and added into `out`, `[rows, d_model]`,
/// the three GEMMs over the pool.
pub fn expert_ffn(set: &ExpertSet, e: usize, act: &Q8Act, weights: &[f32], scratch: &mut Scratch, out: &mut [f32]) -> Result<()> {
    ffn_in(set, e, act, weights, scratch, out, true)
}

fn ffn_in(set: &ExpertSet, e: usize, act: &Q8Act, weights: &[f32], scratch: &mut Scratch, out: &mut [f32], threaded: bool) -> Result<()> {
    let (rows, d, d_ff) = (act.rows, set.down.n(), set.gate.n());
    ensure!(weights.len() == rows && out.len() == rows * d, "{} weights and {} outputs for {rows} rows of {d}", weights.len(), out.len());
    scratch.gate.resize(d_ff * rows, 0.0);
    scratch.up.resize(d_ff * rows, 0.0);
    gemm_in(&set.gate, e, act, &mut scratch.gate, threaded)?;
    gemm_in(&set.up, e, act, &mut scratch.up, threaded)?;
    scratch.h.resize(rows * d_ff, 0.0);
    for (j, (g, u)) in scratch.gate.chunks_exact(rows).zip(scratch.up.chunks_exact(rows)).enumerate() {
        for r in 0..rows {
            scratch.h[r * d_ff + j] = g[r] / (1.0 + (-g[r]).exp()) * u[r];
        }
    }
    scratch.hact.quantize_into(&scratch.h, rows, d_ff);
    scratch.down.resize(d * rows, 0.0);
    gemm_in(&set.down, e, &scratch.hact, &mut scratch.down, threaded)?;
    for (i, y) in scratch.down.chunks_exact(rows).enumerate() {
        for (r, &v) in y.iter().enumerate() {
            out[r * d + i] += weights[r] * v;
        }
    }
    Ok(())
}

/// One host expert's share of a pass: the rows of the activation it was
/// chosen for and the router's weight for each.
pub struct Job {
    pub expert: usize,
    pub rows: Vec<usize>,
    pub weights: Vec<f32>,
}

/// A worker's own scratch and its sum of the jobs it ran.
struct Worker {
    gathered: Vec<f32>,
    act: Q8Act,
    scratch: Scratch,
    y: Vec<f32>,
    acc: Vec<f32>,
}

impl Worker {
    fn new(len: usize) -> Worker {
        Worker { gathered: Vec::new(), act: Q8Act::default(), scratch: Scratch::default(), y: Vec::new(), acc: vec![0.0; len] }
    }

    fn run(&mut self, set: &ExpertSet, job: &Job, x: &[f32], d: usize) -> Result<()> {
        let n = job.rows.len();
        self.gathered.clear();
        for &r in &job.rows {
            self.gathered.extend_from_slice(&x[r * d..(r + 1) * d]);
        }
        self.act.quantize_into(&self.gathered, n, d);
        self.y.clear();
        self.y.resize(n * d, 0.0);
        ffn_in(set, job.expert, &self.act, &job.weights, &mut self.scratch, &mut self.y, false)?;
        for (&r, y) in job.rows.iter().zip(self.y.chunks_exact(d)) {
            for (acc, &v) in self.acc[r * d..(r + 1) * d].iter_mut().zip(y) {
                *acc += v;
            }
        }
        Ok(())
    }
}

/// The jobs over `x`, `[rows, d_model]`, the experts spread over the pool
/// and each one's result scaled by its weights and added into `out`.
pub fn experts_ffn(set: &ExpertSet, jobs: &[Job], x: &[f32], rows: usize, out: &mut [f32]) -> Result<()> {
    let d = set.gate.k();
    ensure!(x.len() == rows * d && out.len() == rows * d, "{} inputs and {} outputs for {rows} rows of {d}", x.len(), out.len());
    let sum: Vec<f32> = pool().install(|| {
        jobs.par_iter()
            .try_fold(
                || Worker::new(rows * d),
                |mut worker, job| {
                    worker.run(set, job, x, d)?;
                    Ok(worker)
                },
            )
            .map(|worker: Result<Worker>| worker.map(|w| w.acc))
            .try_reduce(
                || vec![0.0; rows * d],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        *a += b;
                    }
                    Ok(a)
                },
            )
    })?;
    for (o, s) in out.iter_mut().zip(sum) {
        *o += s;
    }
    Ok(())
}

/// Builds the pool with `threads` workers, before its first use. Once it
/// exists the count stands.
pub fn init_pool(threads: usize) -> Result<()> {
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("phobos-host-{i}"))
        .build()?;
    ensure!(POOL.set(built).is_ok(), "the host pool is already running");
    Ok(())
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Threads the host kernels run on.
pub fn threads() -> usize {
    pool().current_num_threads()
}

fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("phobos-host-{i}"))
            .build()
            .expect("a thread pool")
    })
}

/// The vector helpers the formats share.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use std::arch::x86_64::*;

    use super::Quants;

    /// The lanes of `v` summed.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// The dot of unpacked quants with a Q8 block where every run of 32
    /// shares a scale: Q4_K and Q5_K.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_runs32(q: &Quants, scales: &[i16; 8], qs: &[i8]) -> i32 {
        debug_assert!(qs.len() >= super::BLOCK);
        // SAFETY: the caller has AVX2; `q` is aligned and `qs` a block long.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            for (run, &scale) in scales.iter().enumerate() {
                let w = _mm256_load_si256(q.0.as_ptr().add(run * 32).cast());
                let a = _mm256_loadu_si256(qs.as_ptr().add(run * 32).cast());
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi16(scale), _mm256_maddubs_epi16(w, a)));
            }
            hsum(acc)
        }
    }

    /// The same where every run of 16 has its own scale: Q6_K.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_runs16(q: &Quants, scales: &[i16; 16], qs: &[i8]) -> i32 {
        debug_assert!(qs.len() >= super::BLOCK);
        // SAFETY: as `dot_runs32`.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            for (i, pair) in scales.chunks_exact(2).enumerate() {
                let w = _mm256_load_si256(q.0.as_ptr().add(i * 32).cast());
                let a = _mm256_loadu_si256(qs.as_ptr().add(i * 32).cast());
                // The multiply-add's sixteen lanes are the run's first
                // sixteen elements then its last: one scale a half.
                let scale = _mm256_set_m128i(_mm_set1_epi16(pair[1]), _mm_set1_epi16(pair[0]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(scale, _mm256_maddubs_epi16(w, a)));
            }
            hsum(acc)
        }
    }
}
