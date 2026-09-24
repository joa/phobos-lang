// A fixed team of threads for work that arrives every few hundred
// microseconds and takes about as long, a decode step's misses a block.
//
// A pool that parks its workers between jobs pays a wake a job, which on
// this scale is most of the job. The team's workers spin for a while after
// each job instead, so the next one a block later finds them running, and
// park only once the jobs stop coming. A job is one closure every member
// runs, with a barrier the members can meet at: either the caller's, run
// with the caller taking part, or started for the workers alone while the
// caller goes on with something else.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a worker spins for the next job before it parks.
const SPIN: Duration = Duration::from_millis(3);

type Job = dyn Fn(&Member) + Sync;
/// A job the team holds on to past the call that started it.
type Started = dyn Fn(&Member) + Sync + Send;

struct Shared {
    size: usize,
    /// Bumped once a job; a worker runs the job when it sees it move.
    epoch: AtomicU64,
    /// The current job, valid while `epoch` says it is, as a pointer whose
    /// lifetime the team vouches for.
    job: UnsafeCell<Option<*const Job>>,
    /// Whether the caller is a member of the current job.
    with_caller: AtomicBool,
    /// Workers done with the current job.
    finished: AtomicUsize,
    /// Members at the barrier, and its generation.
    arrived: AtomicUsize,
    generation: AtomicUsize,
    parked: Mutex<u64>,
    wake: Condvar,
}

// SAFETY: `job` is written only while every worker is waiting for the
// epoch to move, and read only after it has.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// One member's view of a job: which it is and how many there are.
pub struct Member<'a> {
    pub index: usize,
    pub size: usize,
    shared: &'a Shared,
}

impl Member<'_> {
    /// Waits until every member has reached it.
    pub fn barrier(&self) {
        let generation = self.shared.generation.load(Ordering::Acquire);
        if self.shared.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.size {
            self.shared.arrived.store(0, Ordering::Relaxed);
            self.shared.generation.fetch_add(1, Ordering::Release);
            return;
        }
        while self.shared.generation.load(Ordering::Acquire) == generation {
            std::hint::spin_loop();
        }
    }
}

pub struct Team {
    shared: Arc<Shared>,
    /// A started job, kept alive until it is joined.
    started: Mutex<Option<Box<Started>>>,
}

impl Team {
    /// A team of `size` members: the caller of [`Team::run`] and `size - 1`
    /// workers.
    pub fn new(size: usize) -> Team {
        let size = size.max(1);
        let shared = Arc::new(Shared {
            size,
            epoch: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            with_caller: AtomicBool::new(true),
            finished: AtomicUsize::new(0),
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            parked: Mutex::new(0),
            wake: Condvar::new(),
        });
        for index in 1..size {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("phobos-team-{index}"))
                .spawn(move || work(&shared, index))
                .expect("spawning a team worker");
        }
        Team { shared, started: Mutex::new(None) }
    }

    /// Runs `job` on every member, the caller included, and returns when
    /// all are done. A started job is joined first.
    pub fn run(&self, job: &(dyn Fn(&Member) + Sync)) {
        let mut started = self.started.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.finish(&mut started);
        // SAFETY: `wait` clears the job below, before its borrow ends.
        self.dispatch(unsafe { std::mem::transmute::<*const (dyn Fn(&Member) + Sync + '_), *const Job>(job) }, true);
        job(&Member { index: 0, size: self.shared.size, shared: &self.shared });
        self.wait();
    }

    /// Starts `job` on the workers alone and returns at once; it is joined
    /// by [`Team::join`] or the next job. A team with no workers runs it
    /// here.
    pub fn start(&self, job: Box<Started>) {
        let mut started = self.started.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.finish(&mut started);
        if self.shared.size == 1 {
            job(&Member { index: 0, size: 1, shared: &self.shared });
            return;
        }
        let running: &Job = &*job;
        self.dispatch(running, false);
        *started = Some(job);
    }

    /// Waits for a started job, if there is one.
    pub fn join(&self) {
        let mut started = self.started.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.finish(&mut started);
    }

    fn finish(&self, started: &mut Option<Box<Started>>) {
        if started.is_some() {
            self.wait();
            *started = None;
        }
    }

    fn dispatch(&self, job: *const Job, with_caller: bool) {
        let shared = &*self.shared;
        // SAFETY: no worker is running a job, see `wait`.
        unsafe { *shared.job.get() = Some(job) };
        shared.with_caller.store(with_caller, Ordering::Relaxed);
        shared.finished.store(0, Ordering::Relaxed);
        {
            let mut epoch = shared.parked.lock().expect("the team's lock");
            *epoch = shared.epoch.fetch_add(1, Ordering::Release) + 1;
        }
        shared.wake.notify_all();
    }

    /// Spins until every worker is done with the job, then clears it.
    fn wait(&self) {
        let shared = &*self.shared;
        while shared.finished.load(Ordering::Acquire) + 1 < shared.size {
            std::hint::spin_loop();
        }
        // SAFETY: no worker is running the job.
        unsafe { *shared.job.get() = None };
    }
}

fn work(shared: &Shared, index: usize) {
    let mut seen = 0u64;
    loop {
        let mut since = Instant::now();
        let mut spins = 0u32;
        let epoch = loop {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen {
                break epoch;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && since.elapsed() > SPIN {
                let mut parked = shared.parked.lock().expect("the team's lock");
                while *parked == seen {
                    parked = shared.wake.wait(parked).expect("the team's lock");
                }
                since = Instant::now();
            }
        };
        seen = epoch;
        let (index, size) = if shared.with_caller.load(Ordering::Relaxed) { (index, shared.size) } else { (index - 1, shared.size - 1) };
        // SAFETY: the job was set before the epoch moved and is cleared
        // only after this worker reports finished. A panic is caught so the
        // report still comes; the job sees to what it owed on its own.
        if let Some(job) = unsafe { *shared.job.get() } {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)(&Member { index, size, shared }) }));
        }
        shared.finished.fetch_add(1, Ordering::Release);
    }
}
