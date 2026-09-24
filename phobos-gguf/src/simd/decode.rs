// A decode step's misses on the host: one row through a few experts, a
// few hundred microseconds of work a block. It runs on the team rather
// than the pool, whose wake and fork-join cost as much as the work at this
// size: every member takes chunks of the gate and up rows of every miss,
// meets the others at a barrier, quantizes its share of the SwiGLUs, and
// after a second barrier takes chunks of the down rows.
//
// The runtime starts the work and goes on recording the next block while
// the team computes; the device waits for the result on a flag in mapped
// memory, which the last member to finish raises.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};

use super::team::{Member, Team};
use super::{Isa, Q8Act, Source, gemm, swiglu};
use crate::experts::Stack;

/// Weight rows a chunk takes: a few kilobytes of one expert, so a handful
/// of misses spread over every member.
const ROW_CHUNK: usize = 16;

/// A buffer the members write disjoint parts of.
struct Parts<T>(*mut T);

// SAFETY: every member writes only the chunks it claimed, and reads only
// what the barrier before the read ordered after its writes.
unsafe impl<T> Sync for Parts<T> {}
unsafe impl<T> Send for Parts<T> {}

impl<T> Parts<T> {
    /// # Safety
    /// No other member touches `at..at + len` until the next barrier.
    #[allow(clippy::mut_from_ref)]
    unsafe fn part(&self, at: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(at), len) }
    }
}

/// Chunks handed out one at a time from a counter, so a member that
/// finishes early takes more.
fn claim(next: &AtomicUsize, total: usize) -> impl Iterator<Item = usize> + '_ {
    std::iter::from_fn(move || Some(next.fetch_add(1, Ordering::Relaxed)).filter(|&i| i < total))
}

/// One row's misses, and the buffers the members share.
struct RowWork<S> {
    source: S,
    misses: Vec<(usize, f32)>,
    act: Q8Act,
    isa: Isa,
    // Owned here and reached through the pointers below.
    #[allow(dead_code)]
    gu: Vec<f32>,
    #[allow(dead_code)]
    hs: Vec<Q8Act>,
    ys: Vec<f32>,
    gu_at: Parts<f32>,
    hs_at: Parts<Q8Act>,
    ys_at: Parts<f32>,
    first: AtomicUsize,
    second: AtomicUsize,
    /// Members done, for the last to know it is last.
    left: AtomicUsize,
    failed: Mutex<Option<anyhow::Error>>,
}

impl<S: Source> RowWork<S> {
    fn new(source: S, misses: Vec<(usize, f32)>, x: &[f32]) -> Result<RowWork<S>> {
        let (gate, down) = (source.shape(Stack::Gate), source.shape(Stack::Down));
        let (d, d_ff) = (gate.k, gate.n);
        ensure!(x.len() == d && down.n == d, "one row of {d} for {} inputs", x.len());
        let m = misses.len();
        let mut gu = vec![0.0f32; m * 2 * d_ff];
        let mut hs: Vec<Q8Act> = (0..m).map(|_| Q8Act::default()).collect();
        let mut ys = vec![0.0f32; m * d];
        let (gu_at, hs_at, ys_at) = (Parts(gu.as_mut_ptr()), Parts(hs.as_mut_ptr()), Parts(ys.as_mut_ptr()));
        Ok(RowWork {
            act: Q8Act::quantize(x, 1, d)?,
            isa: Isa::detect(),
            source,
            misses,
            gu,
            hs,
            ys,
            gu_at,
            hs_at,
            ys_at,
            first: AtomicUsize::new(0),
            second: AtomicUsize::new(0),
            left: AtomicUsize::new(0),
            failed: Mutex::new(None),
        })
    }

    fn fail(&self, e: anyhow::Error) {
        self.failed.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get_or_insert(e);
    }

    /// One member's share. Every member reaches both barriers whatever
    /// fails, or the rest would wait for it.
    fn member(&self, member: &Member) {
        let (gate, up, down) = (self.source.shape(Stack::Gate), self.source.shape(Stack::Up), self.source.shape(Stack::Down));
        let (d, d_ff, m) = (gate.k, gate.n, self.misses.len());
        let (up_chunks, down_chunks) = (d_ff.div_ceil(ROW_CHUNK), d.div_ceil(ROW_CHUNK));
        for i in claim(&self.first, m * 2 * up_chunks) {
            let (miss, half, c) = (i / (2 * up_chunks), i / up_chunks % 2, i % up_chunks);
            let (shape, stack) = if half == 0 { (&gate, Stack::Gate) } else { (&up, Stack::Up) };
            let j0 = c * ROW_CHUNK;
            // SAFETY: chunk `i` is this member's alone.
            let y = unsafe { self.gu_at.part((2 * miss + half) * d_ff + j0, ROW_CHUNK.min(d_ff - j0)) };
            if let Err(e) = gemm(shape, self.isa, &self.source.weight(stack, self.misses[miss].0), j0, &self.act, y) {
                self.fail(e);
            }
        }
        member.barrier();
        for miss in (member.index..m).step_by(member.size) {
            // SAFETY: the gate and up rows are written, and miss `miss`'s
            // activation is this member's alone.
            let (g, u) = unsafe { self.gu_at.part(2 * miss * d_ff, 2 * d_ff) }.split_at(d_ff);
            let h: Vec<f32> = g.iter().zip(u.iter()).map(|(&g, &u)| swiglu(g, u)).collect();
            if let Err(e) = unsafe { &mut self.hs_at.part(miss, 1)[0] }.quantize_into(&h, 1, d_ff) {
                self.fail(e);
            }
        }
        member.barrier();
        for i in claim(&self.second, m * down_chunks) {
            let (miss, c) = (i / down_chunks, i % down_chunks);
            let j0 = c * ROW_CHUNK;
            // SAFETY: the activations are written; chunk `i` is this
            // member's alone.
            let (h, y) = unsafe { (&self.hs_at.part(miss, 1)[0], self.ys_at.part(miss * d + j0, ROW_CHUNK.min(d - j0))) };
            if let Err(e) = gemm(&down, self.isa, &self.source.weight(Stack::Down, self.misses[miss].0), j0, h, y) {
                self.fail(e);
            }
        }
    }

    /// The misses' weighted sum into `out`, once every member is done.
    fn sum_into(&self, out: &mut [f32]) {
        out.fill(0.0);
        for (&(_, w), y) in self.misses.iter().zip(self.ys.chunks_exact(out.len())) {
            for (o, &v) in out.iter_mut().zip(y) {
                *o += w * v;
            }
        }
    }

    fn take_error(&self) -> Option<anyhow::Error> {
        self.failed.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
    }
}

/// The experts `misses` names, each with its weight, over one row `x`, the
/// weighted results added into `out`.
pub fn experts_row(source: &impl Source, misses: &[(usize, f32)], x: &[f32], out: &mut [f32]) -> Result<()> {
    ensure!(out.len() == x.len(), "one row of {} for {} outputs", x.len(), out.len());
    if misses.is_empty() {
        return Ok(());
    }
    let work = RowWork::new(source, misses.to_vec(), x)?;
    team().run(&|member| work.member(member));
    if let Some(e) = work.take_error() {
        return Err(e);
    }
    let mut sum = vec![0.0; out.len()];
    work.sum_into(&mut sum);
    for (o, v) in out.iter_mut().zip(sum) {
        *o += v;
    }
    Ok(())
}

/// Where a started row's result goes: `out` gets the misses' weighted sum,
/// then `ready` is set to 1.
pub struct RowOut {
    pub out: *mut f32,
    pub len: usize,
    pub ready: *const AtomicU32,
}

// SAFETY: see `start_experts_row`'s contract.
unsafe impl Send for RowOut {}
unsafe impl Sync for RowOut {}

/// What a started row left behind for [`StartedRow::join`].
#[derive(Default)]
struct Outcome {
    failed: Mutex<Option<anyhow::Error>>,
    nanos: AtomicU64,
}

/// A row [`start_experts_row`] left running on the team.
#[must_use = "a started row is joined for its error"]
pub struct StartedRow {
    outcome: Arc<Outcome>,
}

impl StartedRow {
    /// Waits for the row and returns the nanoseconds it took, or what went
    /// wrong.
    pub fn join(self) -> Result<u64> {
        team().join();
        if let Some(e) = self.outcome.failed.lock().map_err(|_| anyhow!("a team member panicked"))?.take() {
            return Err(e);
        }
        Ok(self.outcome.nanos.load(Ordering::Relaxed))
    }
}

/// [`experts_row`] started on the team's workers, returning at once. The
/// sum is written to `to.out` in place of what was there, and `to.ready`
/// is set once it is, whatever went wrong on the way.
///
/// # Safety
/// `to.out` holds `to.len` floats and `to.ready` stays valid, and the
/// caller touches neither, until the flag is set; `source` borrows nothing
/// that ends before then.
pub unsafe fn start_experts_row<S: Source + Send + 'static>(source: S, misses: Vec<(usize, f32)>, x: &[f32], to: RowOut) -> Result<StartedRow> {
    let work = RowWork::new(source, misses, x)?;
    let outcome = Arc::new(Outcome::default());
    let (left_behind, started) = (Arc::clone(&outcome), Instant::now());
    let job = move |member: &Member| {
        // Raised from a guard, so a member that fails or panics still
        // leaves the device a result to take.
        struct Leave<'a, S: Source> {
            work: &'a RowWork<S>,
            to: &'a RowOut,
            member: &'a Member<'a>,
            outcome: &'a Outcome,
            started: Instant,
        }
        impl<S: Source> Drop for Leave<'_, S> {
            fn drop(&mut self) {
                if self.work.left.fetch_add(1, Ordering::AcqRel) + 1 != self.member.size {
                    return;
                }
                // SAFETY: the caller's contract; every member is done with
                // the shared buffers.
                self.work.sum_into(unsafe { std::slice::from_raw_parts_mut(self.to.out, self.to.len) });
                if let Some(e) = self.work.take_error() {
                    self.outcome.failed.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get_or_insert(e);
                }
                self.outcome.nanos.store(self.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                unsafe { &*self.to.ready }.store(1, Ordering::Release);
            }
        }
        let _leave = Leave { work: &work, to: &to, member, outcome: &left_behind, started };
        work.member(member);
    };
    team().start(Box::new(job));
    Ok(StartedRow { outcome })
}

static TEAM: OnceLock<Team> = OnceLock::new();

/// Builds the team with `size` members, before its first use, for a
/// benchmark that wants a size of its own. Once it exists the size stands.
pub fn init_team(size: usize) -> Result<()> {
    ensure!(TEAM.set(Team::new(size)).is_ok(), "the host team is already running");
    Ok(())
}

fn team() -> &'static Team {
    TEAM.get_or_init(|| Team::new(super::default_threads()))
}
