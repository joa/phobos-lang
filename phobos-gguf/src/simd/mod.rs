//! The host's K-quant expert feed-forward: a threaded AVX2 dot of Q4_K,
//! Q5_K and Q6_K weight blocks against a Q8 activation, for the experts a
//! pass computes on the host rather than copies to the device.
//!
//! The activation is quantized to eight bits with a scale a run of 32, as
//! the device's is, and the sums of each sixteen quants, so a block's
//! minimums (or Q6_K's offset) fold into one integer term a run. A weight
//! row's blocks are unpacked once, to aligned byte arrays either path
//! reads, and dotted against every activation row with each run's
//! eight-lane sums scaled into one float accumulator, so a row costs one
//! horizontal sum. The weights come from a [`Source`]: the file's
//! row-major blocks, or the device's grouped layout in pinned memory,
//! which a long pass wants since the mapping's pages get trimmed.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;

use crate::experts::{ExpertSet, ExpertStack};
use crate::quant::Quant;
use crate::quant::grouped::RAW_GROUP;

mod q4_k;
mod q5_k;
mod q6_k;
#[cfg(test)]
mod tests;

/// Elements of one activation block, the K-quant super-block.
pub const BLOCK: usize = 256;
/// Elements sharing one activation scale.
const RUN: usize = 32;
const RUNS: usize = BLOCK / RUN;
/// Elements each partial sum of a block covers.
const SUM_RUN: usize = 16;
const SUMS: usize = BLOCK / SUM_RUN;

/// `[rows, k]` activations quantized to `i8`: a scale a run of 32 and the
/// sums of each sixteen quants. Held block-major, the rows of one block
/// side by side, so a weight block's dot over the rows walks memory
/// forward rather than striding a row's width.
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
        self.d.resize(blocks * RUNS, 0.0);
        self.qs.resize(blocks * BLOCK, 0);
        self.sums.resize(blocks * SUMS, 0);
        for (i, block) in x.chunks_exact(BLOCK).enumerate() {
            let (r, b) = (i / self.blocks(), i % self.blocks());
            let at = b * rows + r;
            let qs = &mut self.qs[at * BLOCK..][..BLOCK];
            for ((d, run), qrun) in self.d[at * RUNS..][..RUNS].iter_mut().zip(block.chunks_exact(RUN)).zip(qs.chunks_exact_mut(RUN)) {
                let absmax = run.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let inv = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
                *d = absmax / 127.0;
                for (q, &v) in qrun.iter_mut().zip(run) {
                    *q = (v * inv).round_ties_even() as i8;
                }
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

    /// Row `r`'s block `b`: its runs' scales, its quants and its sums.
    fn block(&self, r: usize, b: usize) -> (&[f32], &[i8], &[i16]) {
        let at = b * self.rows + r;
        (&self.d[at * RUNS..][..RUNS], &self.qs[at * BLOCK..][..BLOCK], &self.sums[at * SUMS..][..SUMS])
    }
}

/// One expert's `[n, k]` blocks, wherever they are.
#[derive(Clone, Copy)]
pub enum Weight<'a> {
    /// The file's layout: row after row, each `nb` whole blocks.
    Flat { bytes: &'a [u8], block_bytes: usize },
    /// The device's layout, see `quant::grouped`: rows in eights, a block
    /// of each of the eight side by side, at the format's device stride;
    /// Q6_K's trailing scale is in `scales`, one a block in the same
    /// order.
    Grouped { bytes: &'a [u8], unit: usize, nb: usize, scales: &'a [u16] },
}

impl Weight<'_> {
    /// Block `b` of row `j`, and its scale where the layout keeps it apart.
    #[inline(always)]
    fn block(&self, j: usize, b: usize, nb: usize) -> (&[u8], Option<u16>) {
        match *self {
            Weight::Flat { bytes, block_bytes } => (&bytes[(j * nb + b) * block_bytes..][..block_bytes], None),
            Weight::Grouped { bytes, unit, scales, .. } => {
                let at = ((j / RAW_GROUP) * nb + b) * RAW_GROUP + j % RAW_GROUP;
                (&bytes[at * unit..][..unit], Some(scales[at]))
            }
        }
    }

    fn holds(&self, n: usize, nb: usize) -> bool {
        match *self {
            Weight::Flat { bytes, block_bytes } => bytes.len() >= n * nb * block_bytes,
            Weight::Grouped { bytes, unit, nb: have, scales } => {
                let blocks = n.div_ceil(RAW_GROUP) * RAW_GROUP * nb;
                have == nb && bytes.len() >= blocks * unit && scales.len() >= blocks
            }
        }
    }
}

/// The three stacks of a block's routed feed-forward.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stack {
    Gate,
    Up,
    Down,
}

/// Where a block's experts are read from.
pub trait Source: Sync {
    /// The format, `n` and `k` of one stack.
    fn shape(&self, stack: Stack) -> (Quant, usize, usize);
    /// Expert `e`'s blocks in one stack.
    fn weight(&self, stack: Stack, e: usize) -> Weight<'_>;
}

impl Source for ExpertSet {
    fn shape(&self, stack: Stack) -> (Quant, usize, usize) {
        let s = self.stack(stack);
        (s.quant(), s.n(), s.k())
    }

    fn weight(&self, stack: Stack, e: usize) -> Weight<'_> {
        let s = self.stack(stack);
        Weight::Flat { bytes: s.expert(e), block_bytes: s.quant().spec().block_bytes }
    }
}

impl ExpertSet {
    pub fn stack(&self, stack: Stack) -> &ExpertStack {
        match stack {
            Stack::Gate => &self.gate,
            Stack::Up => &self.up,
            Stack::Down => &self.down,
        }
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

/// A row's running sum in eight lanes: the vector path keeps all eight,
/// the scalar path the first.
#[repr(C, align(32))]
#[derive(Default)]
pub(crate) struct Lanes(pub(crate) [f32; 8]);

/// One format's block arithmetic against a Q8 block, in one instruction
/// set.
pub(crate) trait Block {
    /// Whether the implementation runs only under AVX2.
    const AVX2: bool;
    type Unpacked: Default;

    /// Unpacks one block's bytes, its scale from `d` where the layout
    /// keeps it apart.
    ///
    /// # Safety
    /// An AVX2 implementation runs only on a CPU that has it.
    unsafe fn unpack(bytes: &[u8], d: Option<u16>, into: &mut Self::Unpacked);

    /// The block's dot with a Q8 block whose runs are scaled by `da`, the
    /// minimums' correction taken off, added into `acc`.
    ///
    /// # Safety
    /// As [`Block::unpack`].
    unsafe fn dot(u: &Self::Unpacked, qs: &[i8], sums: &[i16], da: &[f32], acc: &mut Lanes);
}

/// `yt[(j - j0) * rows + r] = sum_i w[j, i] x[r, i]` over the weight rows
/// from `j0` that `yt` has room for.
#[inline(always)]
unsafe fn rows_dot<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32]) {
    let (rows, nb) = (act.rows, act.blocks());
    let mut row: Vec<F::Unpacked> = (0..nb).map(|_| F::Unpacked::default()).collect();
    for (j, out) in (j0..).zip(yt.chunks_exact_mut(rows)) {
        for (b, u) in row.iter_mut().enumerate() {
            let (bytes, d) = weight.block(j, b, nb);
            // SAFETY: the caller's contract, see `Block::unpack`.
            unsafe { F::unpack(bytes, d, u) };
        }
        for (r, acc) in out.iter_mut().enumerate() {
            let mut lanes = Lanes::default();
            for (b, u) in row.iter().enumerate() {
                let (da, qs, q8_sums) = act.block(r, b);
                // SAFETY: as above.
                unsafe { F::dot(u, qs, q8_sums, da, &mut lanes) };
            }
            *acc = lanes.0.iter().sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn rows_dot_avx2<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32]) {
    // SAFETY: the caller checked for AVX2.
    unsafe { rows_dot::<F>(weight, j0, act, yt) }
}

/// `rows_dot` in the instruction set `F` is written for.
fn rows_dot_in<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32]) {
    // SAFETY: an AVX2 implementation is chosen only after `avx2()` said
    // the CPU has it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if F::AVX2 {
            return rows_dot_avx2::<F>(weight, j0, act, yt);
        }
        rows_dot::<F>(weight, j0, act, yt)
    }
}

/// The GEMM in one format and instruction set. Threaded, the weight rows
/// are dealt out in contiguous chunks, each thread's outputs its own;
/// otherwise it runs where it is called, for a caller that is itself one
/// of many tasks.
fn gemm_with<F: Block>(weight: &Weight, n: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) {
    if !threaded {
        return rows_dot_in::<F>(weight, 0, act, yt);
    }
    let rows = act.rows;
    // Two tasks a thread, but never a task under sixteen rows: waking a
    // worker costs more than that many rows of one activation row.
    let chunk = n.div_ceil(pool().current_num_threads() * 2).max(16);
    pool().install(|| {
        yt.par_chunks_mut(chunk * rows)
            .enumerate()
            .for_each(|(c, out)| rows_dot_in::<F>(weight, c * chunk, act, out))
    });
}

/// Whether the vector path runs here.
pub fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// [`gemm`] over a weight, in the instruction set asked for.
fn gemm_raw(quant: Quant, wide: bool, weight: &Weight, n: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) -> Result<()> {
    ensure!(weight.holds(n, act.blocks()) && yt.len() == n * act.rows, "a [{n}, {}] weight against {} activation rows", act.k, act.rows);
    macro_rules! by_format {
        ($set:ident) => {
            match quant {
                Quant::Q4_K => gemm_with::<q4_k::$set>(weight, n, act, yt, threaded),
                Quant::Q5_K => gemm_with::<q5_k::$set>(weight, n, act, yt, threaded),
                Quant::Q6_K => gemm_with::<q6_k::$set>(weight, n, act, yt, threaded),
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
/// rows of one stack: the outputs transposed, so each thread's rows are
/// contiguous.
pub fn gemm(source: &impl Source, stack: Stack, e: usize, act: &Q8Act, yt: &mut [f32]) -> Result<()> {
    gemm_in(source, stack, e, act, yt, true)
}

fn gemm_in(source: &impl Source, stack: Stack, e: usize, act: &Q8Act, yt: &mut [f32], threaded: bool) -> Result<()> {
    let (quant, n, k) = source.shape(stack);
    ensure!(k == act.k, "a [{n}, {k}] expert against a {}-wide activation", act.k);
    gemm_raw(quant, avx2(), &source.weight(stack, e), n, act, yt, threaded)
}

/// Whether [`expert_ffn`] can run a source on this host.
pub fn supports(source: &impl Source) -> bool {
    [Stack::Gate, Stack::Up, Stack::Down]
        .iter()
        .all(|&stack| matches!(source.shape(stack).0, Quant::Q4_K | Quant::Q5_K | Quant::Q6_K))
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

/// Expert `e` over the activation's rows, each row's result scaled by its
/// `weights[r]` and added into `out`, `[rows, d_model]`, the three GEMMs
/// over the pool.
pub fn expert_ffn(source: &impl Source, e: usize, act: &Q8Act, weights: &[f32], scratch: &mut Scratch, out: &mut [f32]) -> Result<()> {
    ffn_in(source, e, act, weights, scratch, out, true)
}

fn ffn_in(source: &impl Source, e: usize, act: &Q8Act, weights: &[f32], scratch: &mut Scratch, out: &mut [f32], threaded: bool) -> Result<()> {
    let (rows, d, d_ff) = (act.rows, source.shape(Stack::Down).1, source.shape(Stack::Gate).1);
    ensure!(weights.len() == rows && out.len() == rows * d, "{} weights and {} outputs for {rows} rows of {d}", weights.len(), out.len());
    scratch.gate.resize(d_ff * rows, 0.0);
    scratch.up.resize(d_ff * rows, 0.0);
    gemm_in(source, Stack::Gate, e, act, &mut scratch.gate, threaded)?;
    gemm_in(source, Stack::Up, e, act, &mut scratch.up, threaded)?;
    scratch.h.resize(rows * d_ff, 0.0);
    for (j, (g, u)) in scratch.gate.chunks_exact(rows).zip(scratch.up.chunks_exact(rows)).enumerate() {
        for r in 0..rows {
            scratch.h[r * d_ff + j] = g[r] / (1.0 + (-g[r]).exp()) * u[r];
        }
    }
    scratch.hact.quantize_into(&scratch.h, rows, d_ff);
    scratch.down.resize(d * rows, 0.0);
    gemm_in(source, Stack::Down, e, &scratch.hact, &mut scratch.down, threaded)?;
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

/// A worker's own scratch.
#[derive(Default)]
struct Worker {
    gathered: Vec<f32>,
    act: Q8Act,
    scratch: Scratch,
}

impl Worker {
    /// One job's rows of output, `[rows, d]`, each scaled by its weight.
    fn run(&mut self, source: &impl Source, job: &Job, x: &[f32], d: usize) -> Result<Vec<f32>> {
        let n = job.rows.len();
        self.gathered.clear();
        for &r in &job.rows {
            self.gathered.extend_from_slice(&x[r * d..(r + 1) * d]);
        }
        self.act.quantize_into(&self.gathered, n, d);
        let mut y = vec![0.0; n * d];
        ffn_in(source, job.expert, &self.act, &job.weights, &mut self.scratch, &mut y, false)?;
        Ok(y)
    }
}

/// The jobs over `x`, `[rows, d_model]`, the experts spread over the pool
/// and each one's result scaled by its weights and added into `out`. Each
/// job's rows come back on their own and are added here: a sum a worker
/// would be the whole output apiece, most of it zero.
pub fn experts_ffn(source: &impl Source, jobs: &[Job], x: &[f32], rows: usize, out: &mut [f32]) -> Result<()> {
    let d = source.shape(Stack::Gate).2;
    ensure!(x.len() == rows * d && out.len() == rows * d, "{} inputs and {} outputs for {rows} rows of {d}", x.len(), out.len());
    let ys: Vec<Vec<f32>> = pool().install(|| jobs.par_iter().map_init(Worker::default, |w, job| w.run(source, job, x, d)).collect::<Result<_>>())?;
    for (job, y) in jobs.iter().zip(ys) {
        for (&r, y) in job.rows.iter().zip(y.chunks_exact(d)) {
            for (o, &v) in out[r * d..(r + 1) * d].iter_mut().zip(y) {
                *o += v;
            }
        }
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

/// Workers the pool starts with: `PHOBOS_HOST_THREADS`, or half the
/// logical CPUs, since two threads on one core share its vector units and
/// measured slower than one (see `ENV.md`).
fn default_threads() -> usize {
    std::env::var("PHOBOS_HOST_THREADS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1)))
}

fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(default_threads())
            .thread_name(|i| format!("phobos-host-{i}"))
            .build()
            .expect("a thread pool")
    })
}

/// The vector helpers the formats share.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use std::arch::x86_64::*;

    use super::{Lanes, Quants};

    /// A half to a float.
    #[inline]
    #[target_feature(enable = "f16c")]
    pub(crate) fn half(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
    }

    /// The dot of unpacked quants with a Q8 block where every run of 32
    /// shares a weight scale, each run's eight lanes scaled by the block's
    /// `d`, the run's index and the activation's run scale into `acc`:
    /// Q4_K and Q5_K.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot_runs32(q: &Quants, scales: &[i16; 8], qs: &[i8], d: f32, da: &[f32], acc: &mut Lanes) {
        debug_assert!(qs.len() >= super::BLOCK && da.len() >= 8);
        // SAFETY: the caller has AVX2 and FMA; `q` and `acc` are aligned
        // and `qs` a block long.
        unsafe {
            let mut sum = _mm256_load_ps(acc.0.as_ptr());
            for (run, &scale) in scales.iter().enumerate() {
                let w = _mm256_load_si256(q.0.as_ptr().add(run * 32).cast());
                let a = _mm256_loadu_si256(qs.as_ptr().add(run * 32).cast());
                let p = _mm256_madd_epi16(_mm256_set1_epi16(scale), _mm256_maddubs_epi16(w, a));
                sum = _mm256_fmadd_ps(_mm256_set1_ps(d * da[run]), _mm256_cvtepi32_ps(p), sum);
            }
            _mm256_store_ps(acc.0.as_mut_ptr(), sum);
        }
    }

    /// The same where every run of 16 has its own weight scale, two to an
    /// activation run: Q6_K.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot_runs16(q: &Quants, scales: &[i16; 16], qs: &[i8], d: f32, da: &[f32], acc: &mut Lanes) {
        debug_assert!(qs.len() >= super::BLOCK && da.len() >= 8);
        // SAFETY: as `dot_runs32`.
        unsafe {
            let mut sum = _mm256_load_ps(acc.0.as_ptr());
            for (i, pair) in scales.chunks_exact(2).enumerate() {
                let w = _mm256_load_si256(q.0.as_ptr().add(i * 32).cast());
                let a = _mm256_loadu_si256(qs.as_ptr().add(i * 32).cast());
                // The multiply-add's sixteen lanes are the run's first
                // sixteen elements then its last: one scale a half.
                let scale = _mm256_set_m128i(_mm_set1_epi16(pair[1]), _mm_set1_epi16(pair[0]));
                let p = _mm256_madd_epi16(scale, _mm256_maddubs_epi16(w, a));
                sum = _mm256_fmadd_ps(_mm256_set1_ps(d * da[i]), _mm256_cvtepi32_ps(p), sum);
            }
            _mm256_store_ps(acc.0.as_mut_ptr(), sum);
        }
    }

    /// Sixteen per-run factors against the Q8 sums per sixteen, a lane an
    /// activation run, scaled by `dmin` and the run's activation scale and
    /// taken off `acc`: the minimums' or the offset's term.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn min_term(factors: &[i16; 16], sums: &[i16], dmin: f32, da: &[f32], acc: &mut Lanes) {
        debug_assert!(sums.len() >= 16 && da.len() >= 8);
        // SAFETY: the caller has AVX2 and FMA; `sums` holds a block's
        // sixteen and `da` its eight.
        unsafe {
            let f = _mm256_loadu_si256(factors.as_ptr().cast());
            let s = _mm256_loadu_si256(sums.as_ptr().cast());
            let m = _mm256_cvtepi32_ps(_mm256_madd_epi16(f, s));
            let scale = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_loadu_ps(da.as_ptr()));
            let sum = _mm256_fnmadd_ps(scale, m, _mm256_load_ps(acc.0.as_ptr()));
            _mm256_store_ps(acc.0.as_mut_ptr(), sum);
        }
    }
}
