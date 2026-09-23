//! The host's K-quant expert feed-forward: a threaded AVX2 dot of Q4_K,
//! Q5_K and Q6_K weight blocks against a Q8 activation, for the experts a
//! pass computes on the host rather than copies to the device.
//!
//! The activation is quantized to eight bits with a scale a run of 32, as
//! the device's is, and the sums of each sixteen quants, so a block's
//! minimums (or Q6_K's offset) fold into one integer term a run. A weight
//! block is unpacked once to an aligned byte array either path reads and
//! dotted against every activation row, each run's eight-lane sums scaled
//! into one float accumulator, so a row costs one horizontal sum. The
//! weights come from a [`Source`]: the file's row-major blocks, or the
//! device's grouped layout in pinned memory.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;

use crate::experts::{ExpertSet, Stack};
use crate::quant::Quant;
use crate::quant::grouped::RAW_GROUP;

mod q4_k;
mod q5_k;
mod q6_k;
#[cfg(test)]
mod tests;

/// Elements of one activation block, the K-quant super-block.
const BLOCK: usize = 256;
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

/// One row's block: its runs' scales, its quants and its sums.
struct Q8Block<'a> {
    d: &'a [f32; RUNS],
    qs: &'a [i8; BLOCK],
    sums: &'a [i16; SUMS],
}

impl Q8Act {
    pub fn quantize(x: &[f32], rows: usize, k: usize) -> Result<Q8Act> {
        let mut act = Q8Act::default();
        act.quantize_into(x, rows, k)?;
        Ok(act)
    }

    /// Requantizes `x`, `[rows, k]`, into the buffers already held.
    pub fn quantize_into(&mut self, x: &[f32], rows: usize, k: usize) -> Result<()> {
        ensure!(k.is_multiple_of(BLOCK) && x.len() == rows * k, "a [{rows}, {k}] activation of {} elements", x.len());
        let blocks = rows * k / BLOCK;
        (self.rows, self.k) = (rows, k);
        self.d.resize(blocks * RUNS, 0.0);
        self.qs.resize(blocks * BLOCK, 0);
        self.sums.resize(blocks * SUMS, 0);
        let nb = k / BLOCK;
        for (i, block) in x.chunks_exact(BLOCK).enumerate() {
            let at = (i % nb) * rows + i / nb;
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
        Ok(())
    }

    fn blocks(&self) -> usize {
        self.k / BLOCK
    }

    fn block(&self, r: usize, b: usize) -> Q8Block<'_> {
        let at = b * self.rows + r;
        Q8Block {
            d: self.d[at * RUNS..][..RUNS].try_into().expect("a block's runs"),
            qs: self.qs[at * BLOCK..][..BLOCK].try_into().expect("a block"),
            sums: self.sums[at * SUMS..][..SUMS].try_into().expect("a block's sums"),
        }
    }
}

/// `silu(gate) * up`.
pub fn swiglu(gate: f32, up: f32) -> f32 {
    gate / (1.0 + (-gate).exp()) * up
}

/// The format, rows and columns of one expert stack.
#[derive(Clone, Copy)]
pub struct Shape {
    pub quant: Quant,
    pub n: usize,
    pub k: usize,
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
    Grouped { bytes: &'a [u8], block_bytes: usize, nb: usize, scales: &'a [u16] },
}

impl Weight<'_> {
    /// Block `b` of row `j`, and its scale where the layout keeps it apart.
    #[inline(always)]
    fn block(&self, j: usize, b: usize, nb: usize) -> (&[u8], Option<u16>) {
        match *self {
            Weight::Flat { bytes, block_bytes } => (&bytes[(j * nb + b) * block_bytes..][..block_bytes], None),
            Weight::Grouped { bytes, block_bytes, scales, .. } => {
                let at = ((j / RAW_GROUP) * nb + b) * RAW_GROUP + j % RAW_GROUP;
                (&bytes[at * block_bytes..][..block_bytes], Some(scales[at]))
            }
        }
    }

    /// Whether the bytes hold `n` rows of `nb` blocks.
    fn covers(&self, n: usize, nb: usize) -> bool {
        match *self {
            Weight::Flat { bytes, block_bytes } => bytes.len() >= n * nb * block_bytes,
            Weight::Grouped { bytes, block_bytes, nb: rows_of, scales } => {
                let blocks = n.div_ceil(RAW_GROUP) * RAW_GROUP * nb;
                rows_of == nb && bytes.len() >= blocks * block_bytes && scales.len() >= blocks
            }
        }
    }
}

/// Where a block's experts are read from.
pub trait Source: Sync {
    fn shape(&self, stack: Stack) -> Shape;
    /// Expert `e`'s blocks in one stack.
    fn weight(&self, stack: Stack, e: usize) -> Weight<'_>;
}

impl Source for ExpertSet {
    fn shape(&self, stack: Stack) -> Shape {
        let s = self.stack(stack);
        Shape { quant: s.quant(), n: s.n(), k: s.k() }
    }

    fn weight(&self, stack: Stack, e: usize) -> Weight<'_> {
        let s = self.stack(stack);
        Weight::Flat { bytes: s.expert(e), block_bytes: s.quant().spec().block_bytes }
    }
}

/// Whether the host kernels can run a source: every stack in a format they
/// have.
pub fn supports(source: &impl Source) -> bool {
    Stack::ALL.iter().all(|&stack| matches!(source.shape(stack).quant, Quant::Q4_K | Quant::Q5_K | Quant::Q6_K))
}

/// One block's quants unpacked to a byte apiece, in element order, aligned
/// for the vector loads.
#[repr(C, align(32))]
struct Quants([u8; BLOCK]);

impl Default for Quants {
    fn default() -> Quants {
        Quants([0; BLOCK])
    }
}

/// A row's running sum in eight lanes: the vector path keeps all eight,
/// the scalar path the first.
#[repr(C, align(32))]
#[derive(Default)]
struct Acc([f32; 8]);

impl Acc {
    fn sum(&self) -> f32 {
        self.0.iter().sum()
    }
}

/// The integer dot of unpacked quants with Q8 quants, without vectors.
fn dot_i32(w: &[u8], a: &[i8]) -> i32 {
    w.iter().zip(a).map(|(&w, &a)| i32::from(w) * i32::from(a)).sum()
}

/// One format's block arithmetic against a Q8 block, in one instruction
/// set.
trait Block {
    /// Whether the implementation runs only under AVX2.
    const AVX2: bool;
    type Unpacked: Default;

    /// Unpacks one block's bytes, its scale from `d` where the layout
    /// keeps it apart.
    ///
    /// # Safety
    /// An AVX2 implementation runs only on a CPU that has it.
    unsafe fn unpack(bytes: &[u8], d: Option<u16>, into: &mut Self::Unpacked);

    /// The block's dot with a Q8 block, the minimums' correction taken
    /// off, added into `acc`.
    ///
    /// # Safety
    /// As [`Block::unpack`].
    unsafe fn dot(u: &Self::Unpacked, block: &Q8Block, acc: &mut Acc);
}

/// The AVX2 side of a format: a unit type whose [`Block`] impl forwards to
/// the format's `#[target_feature]` functions, which the trait's methods
/// cannot carry themselves.
macro_rules! avx2_block {
    ($name:ident, $unpacked:ty, $unpack:path, $dot:path) => {
        pub(in crate::simd) struct $name;

        impl crate::simd::Block for $name {
            const AVX2: bool = true;
            type Unpacked = $unpacked;

            #[inline(always)]
            unsafe fn unpack(bytes: &[u8], d: Option<u16>, u: &mut $unpacked) {
                // SAFETY: the caller has AVX2.
                unsafe { $unpack(bytes, d, u) }
            }

            #[inline(always)]
            unsafe fn dot(u: &$unpacked, block: &crate::simd::Q8Block, acc: &mut crate::simd::Acc) {
                // SAFETY: the caller has AVX2.
                unsafe { $dot(u, block, acc) }
            }
        }
    };
}
pub(crate) use avx2_block;

/// `yt[(j - j0) * rows + r] = sum_i w[j, i] x[r, i]` over the weight rows
/// from `j0` that `yt` has room for. One row is dotted block by block as
/// it is unpacked; more rows unpack a weight row once into `row` and dot
/// each against it.
#[inline(always)]
unsafe fn rows_dot<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32], row: &mut Vec<F::Unpacked>) {
    let (rows, nb) = (act.rows, act.blocks());
    if rows == 1 {
        let mut u = F::Unpacked::default();
        for (j, out) in (j0..).zip(yt.iter_mut()) {
            let mut acc = Acc::default();
            for b in 0..nb {
                let (bytes, d) = weight.block(j, b, nb);
                // SAFETY: the caller's contract, see `Block::unpack`.
                unsafe {
                    F::unpack(bytes, d, &mut u);
                    F::dot(&u, &act.block(0, b), &mut acc);
                }
            }
            *out = acc.sum();
        }
        return;
    }
    row.resize_with(nb, Default::default);
    for (j, out) in (j0..).zip(yt.chunks_exact_mut(rows)) {
        for (b, u) in row.iter_mut().enumerate() {
            let (bytes, d) = weight.block(j, b, nb);
            // SAFETY: as above.
            unsafe { F::unpack(bytes, d, u) };
        }
        for (r, acc_out) in out.iter_mut().enumerate() {
            let mut acc = Acc::default();
            for (b, u) in row.iter().enumerate() {
                // SAFETY: as above.
                unsafe { F::dot(u, &act.block(r, b), &mut acc) };
            }
            *acc_out = acc.sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn rows_dot_avx2<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32], row: &mut Vec<F::Unpacked>) {
    // SAFETY: the caller checked for AVX2.
    unsafe { rows_dot::<F>(weight, j0, act, yt, row) }
}

/// `rows_dot` in the instruction set `F` is written for.
fn rows_dot_for<F: Block>(weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32], row: &mut Vec<F::Unpacked>) {
    // SAFETY: an AVX2 implementation is chosen only after `Isa::detect`
    // said the CPU has it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if F::AVX2 {
            return rows_dot_avx2::<F>(weight, j0, act, yt, row);
        }
        rows_dot::<F>(weight, j0, act, yt, row)
    }
}

/// The instruction set the kernels run in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Isa {
    Scalar,
    Avx2,
}

impl Isa {
    fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
            return Isa::Avx2;
        }
        Isa::Scalar
    }
}

/// Whether the vector path runs here.
pub fn avx2() -> bool {
    Isa::detect() == Isa::Avx2
}

/// The GEMM of `weight`'s rows from `j0` against `act` into `yt`, in the
/// shape's format, on the calling thread: the callers are themselves the
/// pool's tasks.
fn gemm(shape: &Shape, isa: Isa, weight: &Weight, j0: usize, act: &Q8Act, yt: &mut [f32]) -> Result<()> {
    ensure!(shape.k == act.k && weight.covers(shape.n, act.blocks()), "a [{}, {}] weight against a [{}, {}] activation", shape.n, shape.k, act.rows, act.k);
    ensure!(yt.len().is_multiple_of(act.rows) && j0 + yt.len() / act.rows <= shape.n, "{} outputs from row {j0} of {}", yt.len(), shape.n);
    macro_rules! by_format {
        ($set:ident) => {
            match shape.quant {
                Quant::Q4_K => rows_dot_for::<q4_k::$set>(weight, j0, act, yt, &mut Vec::new()),
                Quant::Q5_K => rows_dot_for::<q5_k::$set>(weight, j0, act, yt, &mut Vec::new()),
                Quant::Q6_K => rows_dot_for::<q6_k::$set>(weight, j0, act, yt, &mut Vec::new()),
                other => bail!("no host expert kernel for {}", other.name()),
            }
        };
    }
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => by_format!(Avx2),
        _ => by_format!(Scalar),
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

/// A worker's scratch, kept across its jobs.
#[derive(Default)]
struct Worker {
    gathered: Vec<f32>,
    act: Q8Act,
    gate: Vec<f32>,
    up: Vec<f32>,
    h: Vec<f32>,
    hact: Q8Act,
    down: Vec<f32>,
}

impl Worker {
    /// One job's rows of output, `[rows, d]`, each scaled by its weight,
    /// the three GEMMs on this thread.
    fn run(&mut self, source: &impl Source, job: &Job, x: &[f32], isa: Isa) -> Result<Vec<f32>> {
        let (gate, up, down) = (source.shape(Stack::Gate), source.shape(Stack::Up), source.shape(Stack::Down));
        let (d, d_ff, rows) = (gate.k, gate.n, job.rows.len());
        ensure!(job.weights.len() == rows, "{} weights for {rows} rows", job.weights.len());
        self.gathered.clear();
        for &r in &job.rows {
            self.gathered.extend_from_slice(&x[r * d..(r + 1) * d]);
        }
        self.act.quantize_into(&self.gathered, rows, d)?;
        self.gate.resize(d_ff * rows, 0.0);
        self.up.resize(d_ff * rows, 0.0);
        gemm(&gate, isa, &source.weight(Stack::Gate, job.expert), 0, &self.act, &mut self.gate)?;
        gemm(&up, isa, &source.weight(Stack::Up, job.expert), 0, &self.act, &mut self.up)?;
        // The GEMMs' outputs are `[d_ff, rows]`; the down GEMM wants its
        // activation `[rows, d_ff]`.
        self.h.resize(rows * d_ff, 0.0);
        for (j, (g, u)) in self.gate.chunks_exact(rows).zip(self.up.chunks_exact(rows)).enumerate() {
            for r in 0..rows {
                self.h[r * d_ff + j] = swiglu(g[r], u[r]);
            }
        }
        self.hact.quantize_into(&self.h, rows, d_ff)?;
        self.down.resize(d * rows, 0.0);
        gemm(&down, isa, &source.weight(Stack::Down, job.expert), 0, &self.hact, &mut self.down)?;
        let mut y = vec![0.0; rows * d];
        for (i, col) in self.down.chunks_exact(rows).enumerate() {
            for (r, &v) in col.iter().enumerate() {
                y[r * d + i] = job.weights[r] * v;
            }
        }
        Ok(y)
    }
}

/// The jobs over `x`, `[rows, d]`, the experts spread over the pool and
/// each one's result added into `out`.
pub fn experts_ffn(source: &impl Source, jobs: &[Job], x: &[f32], rows: usize, out: &mut [f32]) -> Result<()> {
    let d = source.shape(Stack::Gate).k;
    ensure!(x.len() == rows * d && out.len() == rows * d, "{} inputs and {} outputs for {rows} rows of {d}", x.len(), out.len());
    let isa = Isa::detect();
    let ys: Vec<Vec<f32>> = pool().install(|| jobs.par_iter().map_init(Worker::default, |w, job| w.run(source, job, x, isa)).collect::<Result<_>>())?;
    for (job, y) in jobs.iter().zip(ys) {
        for (&r, y) in job.rows.iter().zip(y.chunks_exact(d)) {
            for (o, &v) in out[r * d..(r + 1) * d].iter_mut().zip(y) {
                *o += v;
            }
        }
    }
    Ok(())
}

/// Weight rows a task of [`experts_row`] takes: a few kilobytes of one
/// expert, so a handful of misses spread over every worker.
const ROW_CHUNK: usize = 16;

/// The experts `misses` names, each with its weight, over one row `x`:
/// each GEMM spread over the pool by rows and every miss's in one region,
/// for a decode step, where an expert on its own thread would read at
/// that thread's memory rate. Each expert's weighted result is added into
/// `out`.
pub fn experts_row(source: &impl Source, misses: &[(usize, f32)], x: &[f32], out: &mut [f32]) -> Result<()> {
    let (gate, up, down) = (source.shape(Stack::Gate), source.shape(Stack::Up), source.shape(Stack::Down));
    let (d, d_ff) = (gate.k, gate.n);
    ensure!(x.len() == d && out.len() == d, "one row of {d} for {} outputs", out.len());
    let isa = Isa::detect();
    let act = Q8Act::quantize(x, 1, d)?;
    let mut gu: Vec<Vec<f32>> = misses.iter().map(|_| vec![0.0; 2 * d_ff]).collect();
    pool().install(|| {
        gu.par_iter_mut().zip(misses).try_for_each(|(gu, &(e, _))| {
            let (g, u) = gu.split_at_mut(d_ff);
            [(g, &gate, Stack::Gate), (u, &up, Stack::Up)].into_par_iter().try_for_each(|(y, shape, stack)| {
                let weight = source.weight(stack, e);
                y.par_chunks_mut(ROW_CHUNK).enumerate().try_for_each(|(c, chunk)| gemm(shape, isa, &weight, c * ROW_CHUNK, &act, chunk))
            })
        })
    })?;
    let hs = gu
        .iter()
        .map(|gu| {
            let (g, u) = gu.split_at(d_ff);
            let h: Vec<f32> = g.iter().zip(u).map(|(&g, &u)| swiglu(g, u)).collect();
            Q8Act::quantize(&h, 1, d_ff)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut ys: Vec<Vec<f32>> = misses.iter().map(|_| vec![0.0; d]).collect();
    pool().install(|| {
        ys.par_iter_mut().zip(misses).zip(&hs).try_for_each(|((y, &(e, _)), h)| {
            let weight = source.weight(Stack::Down, e);
            y.par_chunks_mut(ROW_CHUNK).enumerate().try_for_each(|(c, chunk)| gemm(&down, isa, &weight, c * ROW_CHUNK, h, chunk))
        })
    })?;
    for (&(_, w), y) in misses.iter().zip(&ys) {
        for (o, &v) in out.iter_mut().zip(y) {
            *o += w * v;
        }
    }
    Ok(())
}

/// Builds the pool with `threads` workers, before its first use, for a
/// benchmark that wants a count of its own. Once it exists the count
/// stands.
pub fn init_pool(threads: usize) -> Result<()> {
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("phobos-host-{i}"))
        .build()?;
    ensure!(POOL.set(built).is_ok(), "the host pool is already running");
    Ok(())
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub fn threads() -> usize {
    pool().current_num_threads()
}

/// Workers the pool starts with: `PHOBOS_HOST_THREADS`, or half the
/// logical CPUs, since two threads on one core share its vector units.
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
mod x86 {
    use std::arch::x86_64::*;

    use super::{Acc, BLOCK, Quants, RUNS, SUMS};

    /// A half to a float.
    #[inline]
    #[target_feature(enable = "f16c")]
    pub(super) fn half(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
    }

    /// The dot of unpacked quants with a Q8 block where every run of 32
    /// shares a weight scale, each run's eight lanes scaled by the block's
    /// `d`, the run's index and the activation's run scale into `acc`:
    /// Q4_K and Q5_K.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_runs32(q: &Quants, scales: &[i16; RUNS], qs: &[i8; BLOCK], d: f32, da: &[f32; RUNS], acc: &mut Acc) {
        // SAFETY: the caller has AVX2 and FMA; `q` and `acc` are aligned.
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
    pub(super) unsafe fn dot_runs16(q: &Quants, scales: &[i16; SUMS], qs: &[i8; BLOCK], d: f32, da: &[f32; RUNS], acc: &mut Acc) {
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
    pub(super) unsafe fn min_term(factors: &[i16; SUMS], sums: &[i16; SUMS], dmin: f32, da: &[f32; RUNS], acc: &mut Acc) {
        // SAFETY: the caller has AVX2 and FMA; `acc` is aligned.
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
