// What the dashboard remembers between frames.
//
// A snapshot is the engine's state. This is the screen's state: eased gauge
// values, peaks, the rain, and the frame counter.

use super::anim::Rain;

pub struct View {
    /// Frames of splash left to draw. Counts down to zero and stays there.
    pub splash: u64,
    /// Frames since the dashboard opened. Every animation is a function of
    /// this.
    pub frame: u64,
    pub rain: Rain,
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
            rain: Rain::new(),
            vram: 0.0,
            context: 0.0,
            prefill: 0.0,
            decode: 0.0,
            best_prefill: 0.0,
            best_decode: 0.0,
        }
    }

    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.splash = self.splash.saturating_sub(1);
        self.rain.tick(self.frame);
    }

    pub fn replay_splash(&mut self) {
        self.splash = super::splash::FRAMES;
    }

    pub fn reset_peaks(&mut self) {
        self.best_prefill = 0.0;
        self.best_decode = 0.0;
    }
}
