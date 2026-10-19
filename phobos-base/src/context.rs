use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Context {
    /// Whether to print the output of different compiler phases.
    pub print_phases: bool,

    pub gpu_config: GpuConfig,

    /// Values for @autotune search dims.
    pub shape_overrides: HashMap<String, i64>,

    /// Bit width index values lower to.
    pub index_bitwidth: u32,

    /// The functions every source can call without defining them.
    pub stdlib: Stdlib,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            print_phases: false,
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::default()),
            shape_overrides: HashMap::new(),
            index_bitwidth: 32,
            stdlib: Stdlib::Builtin,
        }
    }
}

/// Where the standard library a source compiles against comes from.
#[derive(Debug, Clone, Default)]
pub enum Stdlib {
    /// The one the compiler ships.
    #[default]
    Builtin,
    /// None: a source calls only its own functions and the built-ins.
    Off,
    /// A source of `fn` definitions in place of the shipped one; `name` is
    /// what an error in it is reported against.
    Custom { name: String, text: Arc<str> },
}

#[derive(Debug, Clone)]
pub enum GpuConfig {
    Nvidia(NvidiaGpuConfig),
}

impl GpuConfig {
    /// The target's name, as `sm_86`.
    pub fn chip(&self) -> &str {
        match self {
            GpuConfig::Nvidia(nv) => nv.chip(),
        }
    }

    /// The chip's compute capability as a number (sm_75 is 75, sm_90a is 90).
    /// It selects the target's instruction vocabulary. Anything that depends
    /// on the number belongs behind that vocabulary, not here.
    ///
    /// TODO(joa): how to map this across vendors
    pub fn compute_capability(&self) -> u32 {
        match self {
            GpuConfig::Nvidia(nv) => nv
                .chip
                .trim_start_matches("sm_")
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0),
        }
    }
}

/// The NVIDIA targets a release carries compiled kernels for.
pub const SUPPORTED_CHIPS: [&str; 6] = ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"];

/// The newest of [`SUPPORTED_CHIPS`] at or below compute capability
/// `capability`, since PTX for an older target JITs forward. Below every
/// supported chip, the oldest, which the driver then refuses.
pub fn best_supported_chip(capability: u32) -> &'static str {
    SUPPORTED_CHIPS
        .iter()
        .copied()
        .rev()
        .find(|chip| GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(*chip)).compute_capability() <= capability)
        .unwrap_or(SUPPORTED_CHIPS[0])
}

#[derive(Debug, Clone)]
pub struct NvidiaGpuConfig {
    chip: String,
    features: String,
    target_triple: String,
}

impl NvidiaGpuConfig {
    pub fn with_chip(chip: impl Into<String>) -> Self {
        NvidiaGpuConfig {
            chip: chip.into(),
            ..Default::default()
        }
    }

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub fn features(&self) -> &str {
        &self.features
    }

    pub fn target_triple(&self) -> &str {
        &self.target_triple
    }
}

impl Default for NvidiaGpuConfig {
    fn default() -> Self {
        NvidiaGpuConfig {
            chip: "sm_75".to_string(),
            features: "+ptx90".to_string(),
            target_triple: "nvptx64-nvidia-cuda".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{GpuConfig, NvidiaGpuConfig};

    #[test]
    fn a_card_gets_the_newest_chip_it_can_run() {
        use crate::context::best_supported_chip;
        assert_eq!(best_supported_chip(75), "sm_75");
        assert_eq!(best_supported_chip(87), "sm_86");
        assert_eq!(best_supported_chip(89), "sm_89");
        assert_eq!(best_supported_chip(100), "sm_90");
        assert_eq!(best_supported_chip(121), "sm_120");
        assert_eq!(best_supported_chip(61), "sm_75");
    }

    #[test]
    fn a_chip_name_reads_as_a_compute_capability() {
        for (chip, cc) in [("sm_75", 75), ("sm_90a", 90), ("sm_61", 61)] {
            let cfg = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip));
            assert_eq!(cfg.compute_capability(), cc, "{chip}");
        }
    }
}
