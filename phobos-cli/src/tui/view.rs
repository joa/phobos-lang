// What the dashboard remembers between frames.
//
// A snapshot says what the engine is doing now; this says what the screen is
// doing, which is not the same thing. A gauge eases towards its reading over
// a few frames and the rain falls whether or not a token was produced, so
// both need somewhere to live that outlasts a single draw.

use super::anim::Rain;

pub struct View {
    /// Frames of splash left to draw. Counts down to nothing and stays
    /// there, so the picture is shown once and then gets out of the way.
    pub splash: u64,
    /// Frames since the dashboard opened. Every animation is a function of
    /// this, so a frame rendered twice looks the same both times.
    pub frame: u64,
    pub rain: Rain,
    /// Eased readings, each the fraction of its gauge that is filled.
    pub vram: f64,
    pub context: f64,
    /// Eased rates, in tokens a second.
    pub prefill: f64,
    pub decode: f64,
    /// The best each rate has reached, which is what the live figure is
    /// colored against. Reset by hand: a peak from a warmed card is worth
    /// keeping, and one from a cold first token is not.
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

    /// Show it again, for anyone who wants another look.
    pub fn replay_splash(&mut self) {
        self.splash = super::splash::FRAMES;
    }

    pub fn reset_peaks(&mut self) {
        self.best_prefill = 0.0;
        self.best_decode = 0.0;
    }
}
