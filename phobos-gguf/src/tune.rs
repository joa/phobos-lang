//! The prompt pass's batch size, tuned on the passes a run makes anyway.
//!
//! The context is cut into depth buckets, since the size that suits a
//! prompt's first positions need not suit its ten-thousandth: attention's
//! share of a pass grows with depth, and past a score budget it changes
//! kernel. That budget is on rows times depth, so a narrower pass keeps the
//! faster kernel deeper and the cost over sizes is not convex: a bucket
//! measures every size against its pick and settles once none is left. A
//! sample is one pass of exactly the size being measured, so a ragged tail
//! never counts, and a pass takes the first size still to measure that the
//! prompt can fill.

use std::collections::{BTreeMap, BTreeSet};

mod record;
#[cfg(test)]
mod tests;

pub(crate) use record::record_path;

/// The size every bucket starts from.
pub(crate) const DEFAULT_BATCH: usize = 512;
pub(crate) const MIN_BATCH: usize = 128;
pub(crate) const MAX_BATCH: usize = 4096;

/// Samples a size needs before it is compared.
const SAMPLES: usize = 3;
/// Samples kept per size; older ones are dropped.
const KEEP: usize = 8;
/// How much faster a trial has to be to take over.
const MARGIN: f64 = 0.03;
/// A trial pass this many times slower per token than the pick has fallen
/// off a kernel or set the card paging, and cannot win, so the size is
/// dropped after one pass rather than three.
const CLIFF: f64 = 2.0;

/// The bucket of a pass starting at `start`: positions below 2048 are one,
/// then one per doubling.
pub(crate) fn bucket(start: usize) -> usize {
    (start / 1024).max(1).ilog2() as usize
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    /// The pass's middle position, which attention's cost follows.
    mid: f64,
    secs_per_token: f64,
}

#[derive(Clone, Debug, Default)]
struct Bucket {
    pick: usize,
    /// Sizes still to measure against `pick`, nearest it first. `None` until
    /// the pick has its own samples.
    queue: Option<Vec<usize>>,
    samples: BTreeMap<usize, Vec<Sample>>,
    /// Sizes that lost to the pick, or paged. Never tried again.
    lost: BTreeSet<usize>,
    settled: bool,
    /// The medians a loaded record carried, written back until this process
    /// measures the bucket itself.
    recorded_us: String,
}

impl Bucket {
    fn new(pick: usize) -> Bucket {
        Bucket { pick, ..Bucket::default() }
    }

    /// The size a pass with `remaining` positions left takes.
    fn next(&self, remaining: usize) -> usize {
        self.queue.iter().flatten().copied().find(|&size| size <= remaining).unwrap_or(self.pick)
    }

    fn queued(&self, size: usize) -> bool {
        self.queue.as_ref().is_some_and(|queue| queue.contains(&size))
    }

    /// Every size not yet measured, nearest the pick first, and wider before
    /// narrower at one distance, since a wider pass reads each weight fewer
    /// times. A wider size needs the card to hold its intermediates.
    fn order(&self, limit: usize, room: &dyn Fn(usize) -> bool) -> Vec<usize> {
        let pick = self.pick;
        let mut sizes: Vec<usize> = (MIN_BATCH.ilog2()..=limit.ilog2())
            .map(|bits| 1usize << bits)
            .filter(|&size| size != pick && !self.lost.contains(&size) && (size < pick || room(size)))
            .collect();
        sizes.sort_by_key(|&size| (size.ilog2().abs_diff(pick.ilog2()), size < pick));
        sizes
    }

    /// Takes `size` off the queue, settling the bucket if it was the last.
    fn dequeue(&mut self, size: usize) {
        if let Some(queue) = &mut self.queue {
            queue.retain(|&s| s != size);
            self.settled = queue.is_empty();
        }
    }

    fn count(&self, size: usize) -> usize {
        self.samples.get(&size).map_or(0, Vec::len)
    }

    /// Each size's median seconds per token, its samples first moved to the
    /// bucket's mean depth along a slope shared by every size, so a size
    /// measured deeper in the bucket is not marked down for the depth.
    fn adjusted(&self) -> BTreeMap<usize, f64> {
        let mean = |values: &mut dyn Iterator<Item = f64>| {
            let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
            sum / n.max(1) as f64
        };
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for group in self.samples.values() {
            let mx = mean(&mut group.iter().map(|s| s.mid));
            let my = mean(&mut group.iter().map(|s| s.secs_per_token));
            for s in group {
                sxy += (s.mid - mx) * (s.secs_per_token - my);
                sxx += (s.mid - mx) * (s.mid - mx);
            }
        }
        // A pass never gets cheaper per token deeper into the context.
        let slope = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { 0.0 };
        let center = mean(&mut self.samples.values().flatten().map(|s| s.mid));
        self.samples
            .iter()
            .filter(|(_, group)| !group.is_empty())
            .map(|(&size, group)| {
                let values: Vec<f64> = group
                    .iter()
                    .map(|s| s.secs_per_token - slope * (s.mid - center))
                    .collect();
                (size, median(values))
            })
            .collect()
    }

    /// Moves the search on once a size in play has its samples: the pick's
    /// own open the queue, and a queued size's decide it against the pick.
    /// Returns whether the pick, the queue or the settled flag changed.
    fn step(&mut self, limit: usize, room: &dyn Fn(usize) -> bool) -> bool {
        if self.settled || self.count(self.pick) < SAMPLES {
            return false;
        }
        let Some(queue) = &self.queue else {
            let queue = self.order(limit, room);
            self.settled = queue.is_empty();
            self.queue = Some(queue);
            return true;
        };
        let Some(trial) = queue.iter().copied().find(|&size| self.count(size) >= SAMPLES) else {
            return false;
        };
        let adjusted = self.adjusted();
        if adjusted[&trial] < adjusted[&self.pick] * (1.0 - MARGIN) {
            self.lost.insert(self.pick);
            self.pick = trial;
            let queue = self.order(limit, room);
            self.settled = queue.is_empty();
            self.queue = Some(queue);
        } else {
            self.lost.insert(trial);
            self.dequeue(trial);
        }
        true
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 }
}

/// One model's batch size per depth bucket.
#[derive(Clone, Debug)]
pub(crate) struct Tuner {
    buckets: Vec<Bucket>,
    /// The widest pass the model can take.
    limit: usize,
    /// Whether a pick, a loss or a settlement changed since the last
    /// [`Tuner::take_dirty`].
    dirty: bool,
}

impl Tuner {
    pub(crate) fn new(limit: usize) -> Tuner {
        Tuner {
            buckets: Vec::new(),
            limit: limit.clamp(MIN_BATCH, MAX_BATCH),
            dirty: false,
        }
    }

    /// The widest pass any bucket may choose.
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }

    /// The size a pass starting at `start`, with `remaining` positions left
    /// to feed, should take. A size still to measure runs only where it
    /// fills a whole pass, since a shorter one measures nothing; the pick
    /// runs otherwise. A bucket first used here starts from the pick of the
    /// nearest shallower one.
    pub(crate) fn size(&mut self, start: usize, remaining: usize) -> usize {
        self.bucket_mut(bucket(start)).next(remaining)
    }

    fn bucket_mut(&mut self, index: usize) -> &mut Bucket {
        while self.buckets.len() <= index {
            let seed = self.buckets.last().map_or(DEFAULT_BATCH, |b| b.pick);
            self.buckets.push(Bucket::new(seed.clamp(MIN_BATCH, self.limit)));
        }
        &mut self.buckets[index]
    }

    /// Records a pass of `size` rows from `start` that took `secs`. `room`
    /// says whether the card has memory for a pass of a wider size. Returns
    /// whether the bucket's search moved.
    pub(crate) fn observe(&mut self, size: usize, start: usize, secs: f64, room: &dyn Fn(usize) -> bool) -> bool {
        let limit = self.limit;
        let bucket = self.bucket_mut(bucket(start));
        if bucket.settled || (size != bucket.pick && !bucket.queued(size)) {
            return false;
        }
        let sample = Sample {
            mid: start as f64 + size as f64 / 2.0,
            secs_per_token: secs / size as f64,
        };
        let pick_median = (bucket.count(bucket.pick) > 0).then(|| bucket.adjusted()[&bucket.pick]);
        let moved = if size != bucket.pick && pick_median.is_some_and(|pick| sample.secs_per_token > CLIFF * pick) {
            bucket.lost.insert(size);
            bucket.dequeue(size);
            true
        } else {
            let group = bucket.samples.entry(size).or_default();
            group.push(sample);
            if group.len() > KEEP {
                group.remove(0);
            }
            bucket.step(limit, room)
        };
        self.dirty |= moved;
        moved
    }

    /// A one-line account of a bucket, for the log.
    pub(crate) fn describe(&self, start: usize) -> String {
        let index = bucket(start);
        let Some(bucket) = self.buckets.get(index) else {
            return String::new();
        };
        let low = if index == 0 { 0 } else { 1024 << index };
        let state = match (bucket.settled, &bucket.queue) {
            (true, _) => "settled".to_string(),
            (false, Some(queue)) => format!("to measure {queue:?}"),
            (false, None) => "measuring".to_string(),
        };
        let medians: Vec<String> = bucket
            .adjusted()
            .iter()
            .map(|(size, secs)| format!("{size}: {:.1} us/token", secs * 1e6))
            .collect();
        format!("from position {low}: {} ({state}) [{}]", bucket.pick, medians.join(", "))
    }

    pub(crate) fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}
