// Kernel-only timing with CUDA events, and the statistics every GEMM row
// reports.

use cust::event::{Event, EventFlags};
use cust::stream::Stream;
use std::time::Instant;

/// Samples of one measurement, in milliseconds, with the scope they cover.
#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub samples_ms: Vec<f64>,
}

impl Stats {
    fn sorted(&self) -> Vec<f64> {
        let mut s = self.samples_ms.clone();
        s.sort_by(f64::total_cmp);
        s
    }

    /// The q-quantile by nearest rank, q in [0, 1].
    pub fn quantile(&self, q: f64) -> f64 {
        let s = self.sorted();
        if s.is_empty() {
            return f64::NAN;
        }
        let i = ((s.len() - 1) as f64 * q).round() as usize;
        s[i]
    }

    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    pub fn min(&self) -> f64 {
        self.quantile(0.0)
    }

    /// Median absolute deviation from the median, in ms.
    pub fn mad(&self) -> f64 {
        let m = self.median();
        Stats {
            samples_ms: self.samples_ms.iter().map(|x| (x - m).abs()).collect(),
        }
        .median()
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "n": self.samples_ms.len(),
            "median_ms": self.median(),
            "min_ms": self.min(),
            "p10_ms": self.quantile(0.1),
            "p90_ms": self.quantile(0.9),
            "max_ms": self.quantile(1.0),
            "mad_ms": self.mad(),
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "median {:.4} ms (p10 {:.4}, p90 {:.4}, min {:.4}, n {})",
            self.median(),
            self.quantile(0.1),
            self.quantile(0.9),
            self.min(),
            self.samples_ms.len()
        )
    }
}

/// One timed work item: `prepare` runs on the stream outside the events
/// (restoring C, say), `launch` enqueues the work the events bracket. Neither
/// may synchronize the stream.
pub(crate) struct Timed<'a> {
    pub prepare: Box<dyn FnMut() -> anyhow::Result<()> + 'a>,
    pub launch: Box<dyn FnMut() -> anyhow::Result<()> + 'a>,
}

/// Times `launch` alone between two events, after `prepare`.
pub(crate) fn time_once(stream: &Stream, t: &mut Timed<'_>, start: &Event, end: &Event) -> anyhow::Result<f64> {
    (t.prepare)()?;
    start.record(stream)?;
    (t.launch)()?;
    end.record(stream)?;
    end.synchronize()?;
    Ok(f64::from(end.elapsed_time_f32(start)?))
}

/// Warms every item, then times `rounds` rounds of one launch each, items
/// interleaved within a round so they share clock and thermal state.
pub(crate) fn interleaved(
    stream: &Stream,
    items: &mut [Timed<'_>],
    warmup: u32,
    rounds: u32,
) -> anyhow::Result<Vec<Stats>> {
    let (start, end) = (Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?);
    for _ in 0..warmup {
        for t in items.iter_mut() {
            time_once(stream, t, &start, &end)?;
        }
    }
    let mut out = vec![Stats::default(); items.len()];
    for _ in 0..rounds.max(1) {
        for (t, s) in items.iter_mut().zip(&mut out) {
            s.samples_ms.push(time_once(stream, t, &start, &end)?);
        }
    }
    Ok(out)
}

/// Host wall clock over prepare, launch and a stream synchronize: what a
/// caller that waits on every GEMM sees. Labelled end-to-end wherever it is
/// reported, and never mixed with [`interleaved`]'s kernel-only times.
pub(crate) fn end_to_end(stream: &Stream, t: &mut Timed<'_>, reps: u32) -> anyhow::Result<Stats> {
    let mut s = Stats::default();
    for _ in 0..reps.max(1) {
        let now = Instant::now();
        (t.prepare)()?;
        (t.launch)()?;
        stream.synchronize()?;
        s.samples_ms.push(now.elapsed().as_secs_f64() * 1e3);
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_by_nearest_rank() {
        let s = Stats {
            samples_ms: vec![5.0, 1.0, 3.0, 2.0, 4.0],
        };
        assert_eq!(s.median(), 3.0);
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.quantile(1.0), 5.0);
        assert_eq!(s.mad(), 1.0);
    }
}
