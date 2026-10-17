// What the dashboard remembers between frames.
//
// A snapshot is the engine's state. This is the screen's state: eased gauge
// values, peaks, the sky, and the frame counter.

use phobos_inference::telemetry::{SensorTotal, Snapshot, Total};

use super::mark::Mark;
use super::sky::Sky;

pub struct View {
    /// Frames of splash left to draw. Counts down to zero and stays there.
    pub splash: u64,
    /// Frames since the dashboard opened. Every animation is a function of
    /// this.
    pub frame: u64,
    pub sky: Sky,
    /// The mark at the height last asked for, kept since tracing it is the
    /// costliest thing a frame does.
    mark: Option<Mark>,
    /// Eased readings, each the fraction of its gauge that is filled.
    pub vram: f64,
    pub context: f64,
    /// Eased rates, in tokens per second.
    pub prefill: f64,
    pub decode: f64,
    /// The best each rate has reached, which the live figure is colored
    /// against. Only reset by hand.
    pub best_prefill: f64,
    pub best_decode: f64,
    /// The totals as they stood at the last reset, so the mean rate shown
    /// covers the same stretch as the peak.
    pub base_prefill: Total,
    pub base_decode: Total,
    /// The highest power draw and temperature read since the last reset, and
    /// the sensor totals as they stood then, for the means.
    pub best_power: f64,
    pub best_temperature: f64,
    pub base_sensors: SensorTotal,
}

impl Default for View {
    fn default() -> View {
        View::new()
    }
}

impl View {
    pub fn new() -> View {
        View {
            splash: super::splash::FRAMES,
            frame: 0,
            sky: Sky::new(),
            mark: None,
            vram: 0.0,
            context: 0.0,
            prefill: 0.0,
            decode: 0.0,
            best_prefill: 0.0,
            best_decode: 0.0,
            base_prefill: Total::default(),
            base_decode: Total::default(),
            best_power: 0.0,
            best_temperature: 0.0,
            base_sensors: SensorTotal::default(),
        }
    }

    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.splash = self.splash.saturating_sub(1);
    }

    /// The mark drawn `rows` rows high.
    pub fn mark(&mut self, rows: u16) -> &Mark {
        if self.mark.as_ref().is_none_or(|m| m.rows != rows) {
            self.mark = Some(Mark::new(rows));
        }
        self.mark.as_ref().expect("just built")
    }

    pub fn replay_splash(&mut self) {
        self.splash = super::splash::FRAMES;
    }

    /// Start the peaks and the means over from `snap`.
    pub fn reset_peaks(&mut self, snap: &Snapshot) {
        self.best_prefill = 0.0;
        self.best_decode = 0.0;
        self.base_prefill = snap.prefill_total;
        self.base_decode = snap.decode_total;
        self.best_power = 0.0;
        self.best_temperature = 0.0;
        self.base_sensors = snap.sensor_total;
    }
}
