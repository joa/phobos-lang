//! The night sky, the website's dark palette (`website/src/styles/tokens.css`)
//! as plain RGB, for everything that draws on a terminal: the dashboard and
//! the compiler's diagnostics.

pub type Rgb = (u8, u8, u8);

/// The sky, `--sky-0`.
pub const SKY: Rgb = (5, 7, 15);
/// The brand blue, `--brand`.
pub const BRAND: Rgb = (106, 184, 255);
/// The brand lit, `--brand-hi`.
pub const BRAND_HI: Rgb = (169, 214, 255);
/// The brand, at rest.
pub const BRAND_DIM: Rgb = (52, 92, 140);
/// Barely lit, for hints and what an animation fades into.
pub const FAINT: Rgb = (48, 64, 98);
/// The nebula violet, `--nebula`.
pub const NEBULA: Rgb = (160, 140, 255);
/// A gold star, the warm hue between the brand and Mars.
pub const GOLD: Rgb = (255, 198, 107);
/// Mars, `--mars`.
pub const MARS: Rgb = (255, 138, 107);
/// Starlight, `--star`.
pub const STAR: Rgb = (230, 236, 248);
/// Dust, `--dust`.
pub const DUST: Rgb = (141, 154, 179);
/// The sky's hairline, `--sky-line`.
pub const SKY_LINE: Rgb = (28, 42, 72);
